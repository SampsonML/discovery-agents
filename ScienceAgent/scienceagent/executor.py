"""
SimulationExecutor: bridges the experiment protocol to FieldSampler in PhysicsSchool.

Protocol mapping
----------------
The discovery protocol exposes two scalar particle properties, p1 and p2.
We map them to FieldSampler parameters as follows:

  p1  → source_coupling of particle 1  (controls field amplitude)
  p2  → particle_inertia of particle 2  (controls how strongly it accelerates)

Both particles have particle_source=1 and particle_force=1 so that the field
operator alone governs the force law the agent must discover.  Particle 1 is
held fixed (infinite effective inertia) by zeroing its acceleration each step.
"""

import json
import numpy as np
import sys
import os
from contextlib import contextmanager

# Make PhysicsSchool importable when running from repo root
_repo_root = os.path.join(os.path.dirname(__file__), "..", "..", "PhysicsSchool")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from physchool.worlds.field_sampler import FieldSampler


class _NoisyExecutorMixin:
    """
    Adds optional Gaussian observation noise on recorded particle positions.

    Subclasses must call _init_noise(noise_std, noise_seed) in their __init__.
    Velocities are never noised. When noise_std == 0 the noise path is fully
    bypassed and the RNG is not touched, so behavior is bit-identical to the
    pre-noise implementation.

    The evaluator should wrap its ground-truth executor.run() call in
    `with executor.noise_disabled(): ...` so the metric compares against
    clean trajectories.
    """

    def _init_noise(self, noise_std: float = 0.0, noise_seed: int = None):
        self.noise_std = float(noise_std or 0.0)
        self.noise_seed = noise_seed
        self._noise_rng = np.random.default_rng(noise_seed)

    def _noisy_positions(self, positions):
        """Add Gaussian noise to a positions array, or return it unchanged if noise is off."""
        if self.noise_std <= 0.0:
            return positions
        arr = np.asarray(positions, dtype=np.float64)
        return arr + self._noise_rng.normal(0.0, self.noise_std, size=arr.shape)

    @contextmanager
    def noise_disabled(self):
        """Temporarily zero out noise_std (used by the evaluator for ground truth)."""
        saved = self.noise_std
        self.noise_std = 0.0
        try:
            yield
        finally:
            self.noise_std = saved


# this is hardcoded for a 2-particle simple system
class SimulationExecutor(_NoisyExecutorMixin):
    """
    Runs experiments defined by the discovery protocol against a FieldSampler world.

    Args:
        operators: Operator list passed to FieldSampler (the "unknown" physics).
        temporal_order: 0=constraint, 1=diffusion, 2=wave.
        grid_size: Simulation grid resolution.
        domain_size: Physical size of the periodic domain.
        dt: Integration timestep.
        noise_std: Std-dev of Gaussian observation noise added to reported particle
            positions only (velocities stay clean). 0.0 disables noise.
        noise_seed: Optional RNG seed for reproducible noise.
    """

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(64, 64),
        domain_size=20.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt
        self._init_noise(noise_std, noise_seed)

    def run(self, experiments: list[dict]) -> list[dict]:
        """
        Run a batch of experiments.

        Args:
            experiments: List of dicts with keys:
                p1, p2, pos2, velocity2, measurement_times
                (duration is inferred as max(measurement_times))

        Returns:
            List of result dicts with keys:
                measurement_times, pos1, pos2, velocity1, velocity2
        """
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        """Parse a JSON string, run experiments, return result as JSON string."""
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    def _run_one(self, exp: dict) -> dict:
        p1 = float(exp["p1"])
        p2 = float(exp["p2"])
        pos2 = list(exp["pos2"])
        velocity2 = list(exp["velocity2"])
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)

        # Particle 1 is fixed at origin; particle 2 is mobile.
        # Positions are placed relative to domain centre so the domain is centred at 0.
        centre = self.domain_size / 2.0
        init_positions = np.array(
            [
                [centre, centre],  # p1 at origin (domain centre)
                [centre + pos2[0], centre + pos2[1]],  # p2 offset from origin
            ],
            dtype=np.float64,
        )
        init_velocities = np.array(
            [
                [0.0, 0.0],  # p1 held fixed
                velocity2,
            ],
            dtype=np.float64,
        )

        # p1 controls field amplitude (source strength); p2 controls inertia.
        # Note: FieldSampler._paint_sources uses self.source_coupling as the
        # per-particle values array (not particle_source), so we pass our
        # per-particle source strengths through source_coupling.
        sim = FieldSampler(
            particle_inertia=np.array([1, p2]),  # p1 inertia ≫ 0 → effectively fixed
            particle_source=np.array(
                [p1, 1.0]
            ),  # stored but currently unused by step()
            particle_force=np.array([0.0, 1.0]),  # p1 feels no force (fixed)
            initial_positions=init_positions,
            initial_velocities=init_velocities,
            n_particles=2,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=np.array([p1, 1.0]),  # per-particle, drives _paint_sources
            force_coupling=1.0,
            periodic_boundaries=True,
        )

        pos1_traj, pos2_traj = [], []
        vel1_traj, vel2_traj = [], []
        recorded = set()

        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            # Record at (or just past) each requested measurement time
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    p1_pos = sim.positions[0] - centre
                    p2_pos = sim.positions[1] - centre
                    pos1_traj.append(self._noisy_positions(p1_pos).tolist())
                    pos2_traj.append(self._noisy_positions(p2_pos).tolist())
                    vel1_traj.append(sim.velocities[0].tolist())
                    vel2_traj.append(sim.velocities[1].tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()

        return {
            "measurement_times": measurement_times,
            "pos1": pos1_traj,
            "pos2": pos2_traj,
            "velocity1": vel1_traj,
            "velocity2": vel2_traj,
        }


# hardcoded for 11 particle gravity system
class CircleExecutor(_NoisyExecutorMixin):
    """
    Runs 11-particle circle world experiments for the discovery agent.

    Layout: particle 0 at center, particles 1-10 equally spaced on a ring.
    The hidden physics is a fractional Laplacian with alpha=0.75.

    Experiment format:
        {
            "ring_radius": float,                   # ring radius (default 5.0)
            "initial_tangential_velocity": float,   # CCW tangential speed for ring (default 0.0)
            "measurement_times": [float, ...]
        }

    Returns:
        {
            "measurement_times": [...],
            "positions":  [[[x,y], ...], ...],   # shape (T, 11, 2), relative to domain center
            "velocities": [[[vx,vy], ...], ...]  # shape (T, 11, 2)
        }
    """

    N_RING = 10
    N_TOTAL = 11
    ALPHA = 0.75

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(128, 128),
        domain_size=50.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
    ):
        self.operators = operators or [
            {
                "type": "fractional_laplacian",
                "params": {"strength": 1.0, "alpha": self.ALPHA},
            }
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt
        self._init_noise(noise_std, noise_seed)

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    def _run_one(self, exp: dict) -> dict:
        ring_radius = float(exp.get("ring_radius", 5.0))
        v_tang = float(exp.get("initial_tangential_velocity", 0.0))
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)

        centre = self.domain_size / 2.0
        angles = np.linspace(0, 2 * np.pi, self.N_RING, endpoint=False)

        ring_pos = np.column_stack(
            [
                centre + ring_radius * np.cos(angles),
                centre + ring_radius * np.sin(angles),
            ]
        )
        positions = np.vstack([[[centre, centre]], ring_pos])

        # Tangential velocities: CCW perpendicular to radial direction
        ring_vel = np.column_stack(
            [
                -v_tang * np.sin(angles),
                v_tang * np.cos(angles),
            ]
        )
        velocities = np.vstack([[[0.0, 0.0]], ring_vel])

        masses = np.ones(self.N_TOTAL)
        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            n_particles=self.N_TOTAL,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=masses,
            force_coupling=1.0,
            periodic_boundaries=False,
        )

        pos_traj, vel_traj = [], []
        recorded = set()

        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    pos_traj.append(
                        self._noisy_positions(sim.positions - centre).tolist()
                    )
                    vel_traj.append(sim.velocities.tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()

        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,  # (T, 11, 2) relative to domain center
            "velocities": vel_traj,  # (T, 11, 2)
        }


# need to see whats going on here, something is not right
class SpeciesExecutor(_NoisyExecutorMixin):
    """
    Runs 6-particle species world experiments for the discovery agent.

    Hidden structure: two species with different source couplings.
      Species A (particles 0, 1, 2): source_coupling = 1.0
      Species B (particles 3, 4, 5): source_coupling = 3.0

    All particles have equal mass (inertia = 1) and equal force coupling.
    The field is a standard Laplacian (n=0), so the *only* hidden variable
    is the per-particle source strength.

    Experiment format:
        {
            "positions":  [[x, y], ...],        # 6 initial positions (relative to center)
            "velocities": [[vx, vy], ...],      # 6 initial velocities
            "measurement_times": [float, ...]
        }

    Returns:
        {
            "measurement_times": [...],
            "positions":  [[[x,y], ...], ...],   # shape (T, 6, 2), relative to domain center
            "velocities": [[[vx,vy], ...], ...]  # shape (T, 6, 2)
        }
    """

    N_PARTICLES = 6
    SPECIES_A = [0, 1, 2]
    SPECIES_B = [3, 4, 5]
    SOURCE_A = 1.0
    SOURCE_B = 3.0

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(64, 64),
        domain_size=20.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt
        self._init_noise(noise_std, noise_seed)

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    def _run_one(self, exp: dict) -> dict:
        positions_rel = np.array(exp["positions"], dtype=np.float64)
        velocities = np.array(exp["velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)

        assert positions_rel.shape == (
            self.N_PARTICLES,
            2,
        ), f"Expected {self.N_PARTICLES} positions, got {positions_rel.shape[0]}"
        assert velocities.shape == (
            self.N_PARTICLES,
            2,
        ), f"Expected {self.N_PARTICLES} velocities, got {velocities.shape[0]}"

        centre = self.domain_size / 2.0
        positions = positions_rel + centre  # shift to domain coords

        masses = np.ones(self.N_PARTICLES)
        source_coupling = np.ones(self.N_PARTICLES)
        source_coupling[self.SPECIES_B] = self.SOURCE_B

        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            n_particles=self.N_PARTICLES,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=source_coupling,
            force_coupling=1.0,
            periodic_boundaries=True,
        )

        pos_traj, vel_traj = [], []
        recorded = set()

        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    pos_traj.append(
                        self._noisy_positions(sim.positions - centre).tolist()
                    )
                    vel_traj.append(sim.velocities.tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()

        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,  # (T, 6, 2) relative to domain center
            "velocities": vel_traj,  # (T, 6, 2)
        }


class ThreeSpeciesExecutor(_NoisyExecutorMixin):
    """
    30 background particles of 3 hidden species + 5 neutral probe particles.

    Species A (particles 0-9):   source_coupling = 1.0
    Species B (particles 10-19): source_coupling = 3.0
    Species C (particles 20-29): source_coupling = -2.0
    Probes   (particles 30-34): source_coupling = 0.0 (feel field, don't source it)

    Background particles start in a fixed random configuration (seed=42)
    with zero initial velocity. Agent controls only the 5 probe positions
    and velocities.

    Experiment format:
        {
            "probe_positions":  [[x, y], ...],    # 5 positions relative to center
            "probe_velocities": [[vx, vy], ...],  # 5 initial velocities
            "measurement_times": [float, ...]
        }

    Returns:
        {
            "measurement_times": [...],
            "positions":  [[[x,y], ...], ...],   # shape (T, 35, 2), relative to center
            "velocities": [[[vx,vy], ...], ...], # shape (T, 35, 2)
            "background_initial_positions": [[x,y], ...]  # (30, 2) relative to center
        }
    """

    N_BACKGROUND = 30
    N_PROBES = 5
    N_TOTAL = 35
    SPECIES_A = list(range(0, 10))
    SPECIES_B = list(range(10, 20))
    SPECIES_C = list(range(20, 30))
    PROBES = list(range(30, 35))
    SOURCE_A = 1.0
    SOURCE_B = 3.0
    SOURCE_C = -2.0

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(128, 128),
        domain_size=50.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt
        self._init_noise(noise_std, noise_seed)

        # Fixed background positions (reproducible)
        rng = np.random.RandomState(42)
        self._bg_positions_rel = rng.uniform(-10, 10, (self.N_BACKGROUND, 2))
        self._bg_velocities = np.zeros((self.N_BACKGROUND, 2))

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    def _run_one(self, exp: dict) -> dict:
        probe_pos_rel = np.array(exp["probe_positions"], dtype=np.float64)
        probe_vel = np.array(exp["probe_velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)

        assert probe_pos_rel.shape == (
            self.N_PROBES,
            2,
        ), f"Expected {self.N_PROBES} probe positions, got {probe_pos_rel.shape[0]}"
        assert probe_vel.shape == (
            self.N_PROBES,
            2,
        ), f"Expected {self.N_PROBES} probe velocities, got {probe_vel.shape[0]}"

        centre = self.domain_size / 2.0
        positions = np.vstack(
            [
                self._bg_positions_rel + centre,
                probe_pos_rel + centre,
            ]
        )
        velocities = np.vstack([self._bg_velocities, probe_vel])

        masses = np.ones(self.N_TOTAL)
        source_coupling = np.zeros(self.N_TOTAL)
        source_coupling[self.SPECIES_A] = self.SOURCE_A
        source_coupling[self.SPECIES_B] = self.SOURCE_B
        source_coupling[self.SPECIES_C] = self.SOURCE_C
        # Probes: source_coupling stays 0

        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            n_particles=self.N_TOTAL,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=source_coupling,
            force_coupling=1.0,
            periodic_boundaries=False,
        )

        pos_traj, vel_traj = [], []
        recorded = set()

        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    pos_traj.append(
                        self._noisy_positions(sim.positions - centre).tolist()
                    )
                    vel_traj.append(sim.velocities.tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()

        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,  # (T, 35, 2) relative to domain center
            "velocities": vel_traj,  # (T, 35, 2)
            "background_initial_positions": self._bg_positions_rel.tolist(),
        }


class DarkMatterExecutor(_NoisyExecutorMixin):
    """
    20 visible background + 10 invisible dark matter + 5 neutral probes = 35 total.

    Internal layout (simulation indices):
        Visible   (0-19):  source_coupling = 1.0, reported to agent
        Dark      (20-29): source_coupling = 5.0, NOT reported to agent
        Probes    (30-34): source_coupling = 0.0, reported to agent

    The agent sees 25 particles (indices 0-19 visible + 20-24 probes in agent
    numbering). Dark matter particles are completely hidden: their positions
    and velocities are never returned.

    Agent-facing index mapping:
        agent 0-19  → sim 0-19   (visible background)
        agent 20-24 → sim 30-34  (probes)

    Visible particles are spread in a wide cloud (radius ~10).
    Dark matter particles are clustered tightly (radius ~3) — a hidden halo.

    Experiment format:
        {
            "probe_positions":  [[x, y], ...],    # 5 positions relative to center
            "probe_velocities": [[vx, vy], ...],  # 5 initial velocities
            "measurement_times": [float, ...]
        }

    Returns:
        {
            "measurement_times": [...],
            "positions":  [[[x,y], ...], ...],   # shape (T, 25, 2) — visible + probes only
            "velocities": [[[vx,vy], ...], ...], # shape (T, 25, 2)
            "background_initial_positions": [[x,y], ...]  # (20, 2) visible only
        }
    """

    N_VISIBLE = 20
    N_DARK = 10
    N_PROBES = 5
    N_TOTAL = 35  # simulated internally
    N_AGENT = 25  # returned to agent (visible + probes)

    VISIBLE = list(range(0, 20))
    DARK = list(range(20, 30))
    PROBES = list(range(30, 35))

    SOURCE_VISIBLE = 1.0
    SOURCE_DARK = 5.0

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(128, 128),
        domain_size=50.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt
        self._init_noise(noise_std, noise_seed)

        rng = np.random.RandomState(123)
        # Visible: orbiting at larger radii (ring between r=8 and r=15)
        vis_angles = rng.uniform(0, 2 * np.pi, self.N_VISIBLE)
        vis_radii = rng.uniform(8, 15, self.N_VISIBLE)
        self._visible_positions_rel = np.column_stack(
            [
                vis_radii * np.cos(vis_angles),
                vis_radii * np.sin(vis_angles),
            ]
        )
        # Dark matter: tightly clustered near center (σ = 1.0)
        self._dark_positions_rel = rng.normal(0, 1.0, (self.N_DARK, 2))

        # Give visible particles tangential velocities for approximate orbits
        # around the dark halo center (origin).
        # In 2D Laplacian gravity: v_circ = sqrt(M_enclosed / (2π))
        # Compute enclosed mass per particle: dark matter inside r_i + other
        # visible particles inside r_i.
        vis_r = np.linalg.norm(self._visible_positions_rel, axis=1)
        dark_r = np.linalg.norm(self._dark_positions_rel, axis=1)

        M_enclosed = np.zeros(self.N_VISIBLE)
        for i in range(self.N_VISIBLE):
            ri = vis_r[i]
            M_enclosed[i] = (
                np.sum(dark_r < ri) * self.SOURCE_DARK
                + np.sum(vis_r < ri) * self.SOURCE_VISIBLE
                - self.SOURCE_VISIBLE  # exclude self
            )

        v_circ = np.sqrt(np.maximum(M_enclosed, 0.0) / (2 * np.pi))
        r_safe = np.maximum(vis_r, 1e-6)
        # Unit tangential direction (CCW): (-y, x) / r
        tangent = (
            np.column_stack(
                [
                    -self._visible_positions_rel[:, 1],
                    self._visible_positions_rel[:, 0],
                ]
            )
            / r_safe[:, None]
        )
        self._visible_velocities = v_circ[:, None] * tangent

        self._dark_velocities = np.zeros((self.N_DARK, 2))

        # Indices of visible + probe particles (what gets returned)
        self._agent_indices = self.VISIBLE + self.PROBES

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    def _run_one(self, exp: dict) -> dict:
        probe_pos_rel = np.array(exp["probe_positions"], dtype=np.float64)
        probe_vel = np.array(exp["probe_velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 10.0)
        # Allow evaluator to flip visible orbit direction (+1 CCW, -1 CW)
        vis_vel_sign = float(exp.get("visible_velocity_sign", 1.0))

        assert probe_pos_rel.shape == (
            self.N_PROBES,
            2,
        ), f"Expected {self.N_PROBES} probe positions, got {probe_pos_rel.shape[0]}"
        assert probe_vel.shape == (
            self.N_PROBES,
            2,
        ), f"Expected {self.N_PROBES} probe velocities, got {probe_vel.shape[0]}"

        centre = self.domain_size / 2.0

        # Full internal state: visible + dark + probes
        positions = np.vstack(
            [
                self._visible_positions_rel + centre,
                self._dark_positions_rel + centre,
                probe_pos_rel + centre,
            ]
        )
        velocities = np.vstack(
            [
                vis_vel_sign * self._visible_velocities,
                self._dark_velocities,
                probe_vel,
            ]
        )

        masses = np.ones(self.N_TOTAL)
        source_coupling = np.zeros(self.N_TOTAL)
        source_coupling[self.VISIBLE] = self.SOURCE_VISIBLE
        source_coupling[self.DARK] = self.SOURCE_DARK
        # Probes stay at 0

        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            n_particles=self.N_TOTAL,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=source_coupling,
            force_coupling=1.0,
            periodic_boundaries=False,
        )

        pos_traj, vel_traj = [], []
        recorded = set()

        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    # Only return visible + probe particles
                    all_pos = sim.positions - centre
                    all_vel = sim.velocities
                    agent_pos = all_pos[self._agent_indices]
                    pos_traj.append(self._noisy_positions(agent_pos).tolist())
                    vel_traj.append(all_vel[self._agent_indices].tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()

        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,  # (T, 25, 2) — visible + probes only
            "velocities": vel_traj,  # (T, 25, 2)
            "background_initial_positions": self._visible_positions_rel.tolist(),
        }

    def run_full(self, experiments: list[dict]) -> list[dict]:
        """Run with FULL output (all 35 particles + field). For evaluation/plotting only."""
        return [self._run_one_full(exp) for exp in experiments]

    def _run_one_full(self, exp: dict) -> dict:
        probe_pos_rel = np.array(exp["probe_positions"], dtype=np.float64)
        probe_vel = np.array(exp["probe_velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 10.0)
        vis_vel_sign = float(exp.get("visible_velocity_sign", 1.0))

        centre = self.domain_size / 2.0
        positions = np.vstack(
            [
                self._visible_positions_rel + centre,
                self._dark_positions_rel + centre,
                probe_pos_rel + centre,
            ]
        )
        velocities = np.vstack(
            [
                vis_vel_sign * self._visible_velocities,
                self._dark_velocities,
                probe_vel,
            ]
        )

        masses = np.ones(self.N_TOTAL)
        source_coupling = np.zeros(self.N_TOTAL)
        source_coupling[self.VISIBLE] = self.SOURCE_VISIBLE
        source_coupling[self.DARK] = self.SOURCE_DARK

        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            n_particles=self.N_TOTAL,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=source_coupling,
            force_coupling=1.0,
            periodic_boundaries=False,
        )

        pos_traj, vel_traj = [], []
        field_snapshots = []
        recorded = set()

        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    pos_traj.append((sim.positions - centre).tolist())
                    vel_traj.append(sim.velocities.tolist())
                    field_snapshots.append(np.asarray(sim.field).tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()

        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,  # (T, 35, 2) — ALL particles
            "velocities": vel_traj,  # (T, 35, 2)
            "field_snapshots": field_snapshots,  # (T, grid, grid)
            "dark_initial_positions": self._dark_positions_rel.tolist(),
            "background_initial_positions": self._visible_positions_rel.tolist(),
        }
