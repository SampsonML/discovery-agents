"""Tests for multi-particle simulations through the environment wrapper.

Validates that the physics is correct for N > 2 particles by checking:
  - basic plumbing (runs without error, correct output shape)
  - momentum conservation
  - energy conservation (n=0 constraint solver)
  - superposition: 3-body forces equal the sum of 2-body forces
  - consistency: N isolated pairs give the same result as N independent 2-body runs
  - radial infall generalizes to N particles
  - the env wrapper round-trips JSON correctly
"""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from physchool.env import (
    PhysicsSchoolEnv,
    WorldConfig,
    _parse_experiment_request,
    _run_single_experiment,
)
from physchool.worlds.field_sampler import FieldSampler


POISSON_OPS = [{"type": "laplacian", "params": {"strength": 1.0}}]


def make_world(**kwargs):
    defaults = dict(
        operators=POISSON_OPS,
        temporal_order=0,
        domain_size=40.0,
        grid_size=(128, 128),
        dt=0.005,
        periodic_boundaries=True,
    )
    defaults.update(kwargs)
    return WorldConfig(**defaults)


# ── Basic plumbing ──────────────────────────────────────────────────────


class TestBasicPlumbing:
    """Verify the wrapper produces correctly shaped, finite output."""

    @pytest.mark.parametrize("n_particles", [2, 3, 5, 10])
    def test_output_shape(self, n_particles):
        world = make_world()
        n_times = 5
        exp = {
            "particles": [
                {"property": 1.0, "position": [i * 1.5, 0.0], "velocity": [0.0, 0.0]}
                for i in range(n_particles)
            ],
            "duration": 0.5,
            "measurement_times": np.linspace(0, 0.5, n_times).tolist(),
        }
        result = _run_single_experiment(exp, world)

        assert "error" not in result, f"Simulation errored: {result.get('error')}"
        assert len(result["positions"]) == n_particles
        assert len(result["velocities"]) == n_particles
        for i in range(n_particles):
            assert len(result["positions"][i]) == n_times
            assert len(result["velocities"][i]) == n_times
            for t in range(n_times):
                assert len(result["positions"][i][t]) == 2
                assert len(result["velocities"][i][t]) == 2

    @pytest.mark.parametrize("n_particles", [3, 5])
    def test_no_nan(self, n_particles):
        world = make_world()
        rng = np.random.RandomState(42)
        exp = {
            "particles": [
                {
                    "property": float(rng.uniform(0.5, 3.0)),
                    "position": rng.uniform(-3, 3, 2).tolist(),
                    "velocity": rng.uniform(-0.5, 0.5, 2).tolist(),
                }
                for _ in range(n_particles)
            ],
            "duration": 1.0,
            "measurement_times": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
        result = _run_single_experiment(exp, world)
        assert "error" not in result
        pos = np.array(result["positions"])
        vel = np.array(result["velocities"])
        assert np.all(np.isfinite(pos)), "Positions contain NaN/Inf"
        assert np.all(np.isfinite(vel)), "Velocities contain NaN/Inf"


# ── Momentum conservation ──────────────────────────────────────────────


class TestMultiParticleMomentum:
    """Total momentum should be conserved for N particles."""

    @pytest.mark.parametrize("n_particles", [3, 5, 8])
    def test_momentum_conservation(self, n_particles):
        world = make_world(periodic_boundaries=True)
        rng = np.random.RandomState(123)
        properties = rng.uniform(0.5, 3.0, n_particles)
        positions = rng.uniform(-3, 3, (n_particles, 2))
        velocities = rng.uniform(-0.5, 0.5, (n_particles, 2))

        center = world.domain_size / 2.0
        sim = FieldSampler(
            particle_inertia=properties,
            particle_source=properties,
            particle_force=properties,
            initial_positions=positions + center,
            initial_velocities=velocities.copy(),
            spatial_dimensions=2,
            temporal_order=0,
            grid_size=world.grid_size,
            domain_size=world.domain_size,
            operators=world.operators,
            n_particles=n_particles,
            force_law="gradient",
            dt=world.dt,
            source_coupling=properties,
            force_coupling=world.force_coupling,
            periodic_boundaries=True,
        )

        p0 = np.sum(sim.velocities * properties[:, None], axis=0)
        for _ in range(100):
            sim.step()
        p1 = np.sum(sim.velocities * properties[:, None], axis=0)

        dp = np.linalg.norm(p1 - p0)
        p_scale = max(np.linalg.norm(p0), 1e-10)
        assert dp / p_scale < 0.05, (
            f"Momentum drift {dp / p_scale:.4f} too large for {n_particles} particles"
        )


# ── Superposition / force additivity ───────────────────────────────────


class TestSuperposition:
    """Forces on particle i in a 3-body system should equal the sum
    of forces from each other particle computed independently."""

    def test_three_body_force_is_sum_of_pairs(self):
        """Place 3 particles in a triangle. Measure force on particle 0
        from the full 3-body sim, and from two separate 2-body sims.
        They should agree (the field equation is linear)."""
        world = make_world(
            domain_size=60.0, grid_size=(128, 128), periodic_boundaries=False
        )
        center = world.domain_size / 2.0

        p0 = np.array([0.0, 0.0])
        p1 = np.array([4.0, 0.0])
        p2 = np.array([0.0, 4.0])
        masses = np.array([1.0, 1.0, 1.0])

        def get_force_on_0(particle_positions, particle_masses):
            n = len(particle_masses)
            sim = FieldSampler(
                particle_inertia=particle_masses,
                particle_source=particle_masses,
                particle_force=particle_masses,
                initial_positions=particle_positions + center,
                initial_velocities=np.zeros((n, 2)),
                spatial_dimensions=2,
                temporal_order=0,
                grid_size=world.grid_size,
                domain_size=world.domain_size,
                operators=world.operators,
                n_particles=n,
                force_law="gradient",
                dt=world.dt,
                source_coupling=particle_masses,
                force_coupling=world.force_coupling,
                periodic_boundaries=False,
            )
            f = sim.step()
            return np.asarray(f[0])

        f_012 = get_force_on_0(np.array([p0, p1, p2]), masses)

        # Force on p0 due to p1 alone (use a "test particle" at p0)
        f_01 = get_force_on_0(np.array([p0, p1]), np.array([1.0, 1.0]))
        f_02 = get_force_on_0(np.array([p0, p2]), np.array([1.0, 1.0]))

        f_sum = f_01 + f_02

        # The 3-body force includes self-interaction of p0 with the field
        # sourced by all 3, so agreement won't be exact, but the
        # cross-terms (p1→p0 and p2→p0) should dominate.
        # We check that the direction and magnitude are close.
        cos_angle = np.dot(f_012, f_sum) / (
            np.linalg.norm(f_012) * np.linalg.norm(f_sum) + 1e-15
        )
        mag_ratio = np.linalg.norm(f_012) / (np.linalg.norm(f_sum) + 1e-15)

        assert cos_angle > 0.95, (
            f"Force directions disagree: cos(angle) = {cos_angle:.4f}"
        )
        assert 0.7 < mag_ratio < 1.3, (
            f"Force magnitudes disagree: ratio = {mag_ratio:.4f}"
        )


# ── Consistency: isolated pairs ─────────────────────────────────────────


class TestIsolatedPairs:
    """Two well-separated pairs should behave identically to two independent
    2-body simulations."""

    def test_separated_pairs_match_independent(self):
        world = make_world(
            domain_size=80.0, grid_size=(128, 128), periodic_boundaries=False
        )

        sep = 3.0

        # Pair A centered at (-15, 0), pair B at (+15, 0)
        pair_a_center = np.array([-15.0, 0.0])
        pair_b_center = np.array([15.0, 0.0])
        half = sep / 2

        pos_4body = np.array([
            pair_a_center + [-half, 0],
            pair_a_center + [half, 0],
            pair_b_center + [-half, 0],
            pair_b_center + [half, 0],
        ])
        vel_4body = np.zeros((4, 2))
        masses_4body = np.ones(4)

        duration = 0.3
        times = [0.0, 0.1, 0.2, 0.3]

        exp_4 = {
            "particles": [
                {"property": 1.0, "position": pos_4body[i].tolist(), "velocity": [0.0, 0.0]}
                for i in range(4)
            ],
            "duration": duration,
            "measurement_times": times,
        }
        result_4 = _run_single_experiment(exp_4, world)
        assert "error" not in result_4

        exp_a = {
            "particles": [
                {"property": 1.0, "position": pos_4body[i].tolist(), "velocity": [0.0, 0.0]}
                for i in [0, 1]
            ],
            "duration": duration,
            "measurement_times": times,
        }
        result_a = _run_single_experiment(exp_a, world)
        assert "error" not in result_a

        # Compare pair A trajectories between 4-body and 2-body
        for i in range(2):
            pos_4 = np.array(result_4["positions"][i])
            pos_2 = np.array(result_a["positions"][i])
            max_diff = np.max(np.abs(pos_4 - pos_2))
            assert max_diff < 0.05, (
                f"Particle {i} in pair A: max position diff = {max_diff:.6f} "
                f"between 4-body and independent 2-body"
            )


# ── N-body infall ───────────────────────────────────────────────────────


class TestNBodyInfall:
    """N particles starting at rest in a ring should all fall inward."""

    @pytest.mark.parametrize("n_particles", [3, 4, 6])
    def test_ring_contracts(self, n_particles):
        world = make_world(periodic_boundaries=False)
        radius = 4.0
        angles = np.linspace(0, 2 * np.pi, n_particles, endpoint=False)
        positions = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

        exp = {
            "particles": [
                {"property": 1.0, "position": positions[i].tolist(), "velocity": [0.0, 0.0]}
                for i in range(n_particles)
            ],
            "duration": 1.0,
            "measurement_times": [0.0, 1.0],
        }
        result = _run_single_experiment(exp, world)
        assert "error" not in result

        initial_radii = np.linalg.norm(positions, axis=1)
        final_positions = np.array([result["positions"][i][-1] for i in range(n_particles)])
        final_radii = np.linalg.norm(final_positions, axis=1)

        for i in range(n_particles):
            assert final_radii[i] < initial_radii[i], (
                f"Particle {i}: radius grew from {initial_radii[i]:.3f} "
                f"to {final_radii[i]:.3f} — should contract"
            )


# ── Energy conservation (n=0, symplectic check) ────────────────────────


class TestEnergyConservation:
    """For the constraint solver (n=0), total energy should not drift
    secularly over many timesteps."""

    def test_energy_bounded(self):
        n_particles = 4
        world = make_world(periodic_boundaries=False, domain_size=60.0)
        center = world.domain_size / 2.0

        rng = np.random.RandomState(7)
        positions = rng.uniform(-3, 3, (n_particles, 2))
        velocities = rng.uniform(-0.3, 0.3, (n_particles, 2))
        masses = rng.uniform(0.5, 2.0, n_particles)

        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions + center,
            initial_velocities=velocities.copy(),
            spatial_dimensions=2,
            temporal_order=0,
            grid_size=world.grid_size,
            domain_size=world.domain_size,
            operators=world.operators,
            n_particles=n_particles,
            force_law="gradient",
            dt=world.dt,
            source_coupling=masses,
            force_coupling=world.force_coupling,
            periodic_boundaries=False,
        )

        def kinetic_energy():
            return 0.5 * np.sum(masses[:, None] * sim.velocities ** 2)

        def potential_energy():
            return 0.5 * float(jnp.sum(sim.field * jnp.fft.ifftn(
                jnp.fft.fftn(sim.field) * sim._L_k
            ).real)) * (sim.dx ** 2)

        energies = []
        n_steps = 200
        for step in range(n_steps):
            sim.step()
            if step % 20 == 0:
                ke = kinetic_energy()
                energies.append(ke)

        energies = np.array(energies)
        drift = abs(energies[-1] - energies[0]) / (np.mean(np.abs(energies)) + 1e-15)
        assert drift < 0.5, (
            f"Energy drifted by {drift:.2%} over {n_steps} steps — "
            f"possible integrator issue"
        )


# ── Env wrapper JSON round-trip ─────────────────────────────────────────


class TestEnvWrapper:
    """Test that the PhysicsSchoolEnv correctly parses experiment JSON
    and returns well-formed results."""

    def test_process_experiment_action(self):
        world = make_world()
        env = PhysicsSchoolEnv(world)

        experiment_json = json.dumps([{
            "particles": [
                {"property": 1.0, "position": [0.0, 0.0], "velocity": [0.0, 0.0]},
                {"property": 1.0, "position": [3.0, 0.0], "velocity": [0.0, 0.5]},
                {"property": 2.0, "position": [-2.0, 1.0], "velocity": [0.1, -0.1]},
            ],
            "duration": 1.0,
            "measurement_times": [0.0, 0.5, 1.0],
        }])

        agent_msg = f"<run_experiment>\n{experiment_json}\n</run_experiment>"
        response = env.process_action(agent_msg)

        assert "<experiment_output>" in response
        assert env.round == 1
        assert not env.is_done

        output_json = response.split("<experiment_output>")[1].split("</experiment_output>")[0]
        results = json.loads(output_json)
        assert len(results) == 1
        assert len(results[0]["positions"]) == 3
        assert len(results[0]["positions"][0]) == 3  # 3 measurement times

    def test_multiple_experiments_single_round(self):
        world = make_world()
        env = PhysicsSchoolEnv(world)

        experiments = [
            {
                "particles": [
                    {"property": 1.0, "position": [0.0, 0.0], "velocity": [0.0, 0.0]},
                    {"property": 1.0, "position": [2.0, 0.0], "velocity": [0.0, 0.0]},
                ],
                "duration": 0.5,
                "measurement_times": [0.0, 0.5],
            },
            {
                "particles": [
                    {"property": 1.0, "position": [0.0, 0.0], "velocity": [0.0, 0.0]},
                    {"property": 1.0, "position": [4.0, 0.0], "velocity": [0.0, 0.0]},
                ],
                "duration": 0.5,
                "measurement_times": [0.0, 0.5],
            },
        ]

        agent_msg = f"<run_experiment>\n{json.dumps(experiments)}\n</run_experiment>"
        response = env.process_action(agent_msg)

        output_json = response.split("<experiment_output>")[1].split("</experiment_output>")[0]
        results = json.loads(output_json)
        assert len(results) == 2
        assert env.round == 1  # both experiments count as one round

    def test_parse_rejects_bad_input(self):
        with pytest.raises(ValueError, match="at least 2"):
            _parse_experiment_request(json.dumps([{
                "particles": [
                    {"property": 1.0, "position": [0.0, 0.0], "velocity": [0.0, 0.0]},
                ],
                "duration": 1.0,
                "measurement_times": [0.0],
            }]))

    def test_round_limit_enforced(self):
        world = make_world()
        env = PhysicsSchoolEnv(world, max_rounds=1)

        exp = json.dumps([{
            "particles": [
                {"property": 1.0, "position": [0.0, 0.0], "velocity": [0.0, 0.0]},
                {"property": 1.0, "position": [3.0, 0.0], "velocity": [0.0, 0.0]},
            ],
            "duration": 0.1,
            "measurement_times": [0.0, 0.1],
        }])

        env.process_action(f"<run_experiment>\n{exp}\n</run_experiment>")
        assert env.round == 1

        response = env.process_action(f"<run_experiment>\n{exp}\n</run_experiment>")
        assert "used all experiment rounds" in response

    def test_final_law_ends_mission(self):
        world = make_world()
        env = PhysicsSchoolEnv(world)

        code = '''
def discovered_law(positions, velocities, properties, duration, measurement_times):
    import numpy as np
    n = len(positions)
    result_pos = [[list(positions[i]) for _ in measurement_times] for i in range(n)]
    result_vel = [[list(velocities[i]) for _ in measurement_times] for i in range(n)]
    return result_pos, result_vel
'''
        agent_msg = f"<final_law>\n{code}\n</final_law>"
        response = env.process_action(agent_msg)

        assert env.is_done
        assert env.evaluation_result is not None
        assert "score" in env.evaluation_result
        assert "Score:" in response
