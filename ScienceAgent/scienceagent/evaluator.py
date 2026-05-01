"""
Evaluator: compares the agent's discovered_law against ground-truth trajectories.
For now, just using raw MSE of the predicted vs PhysicsSchool particle paths
"""

import functools
import re
import numpy as np
from typing import Callable, Optional

from scienceagent.executor import SimulationExecutor, SpeciesExecutor, ThreeSpeciesExecutor, DarkMatterExecutor


MAX_FIT_PARAMETERS = 3
FIT_MAXITER = 15
FIT_TIME_BUDGET_S = 60.0  # wall-clock cap on a single scipy.optimize.minimize call

# Wall-clock cap for a single discovered_law call. Stops a pathological law
# (Python for-loop with tiny dt, stiff ODE, infinite loop) from hanging the
# scoring loop or the fit objective indefinitely.
LAW_CALL_TIMEOUT_S = 10.0


class _FitTimeBudgetExceeded(Exception):
    """Raised inside the optimizer objective when the wall-clock budget is hit."""


class _LawCallTimeout(Exception):
    """Raised when a single discovered_law call exceeds LAW_CALL_TIMEOUT_S."""


def _wrap_with_timeout(fn: Callable, timeout_s: float = LAW_CALL_TIMEOUT_S) -> Callable:
    """
    Wrap `fn` so any single call exceeding `timeout_s` raises _LawCallTimeout.
    Uses SIGALRM (Unix-only, main-thread-only). Falls back to the unwrapped
    function on Windows or off-main-thread, where SIGALRM is unavailable.
    """
    import signal
    import threading

    if not hasattr(signal, "SIGALRM"):
        return fn
    if threading.current_thread() is not threading.main_thread():
        return fn

    def _handler(signum, frame):
        raise _LawCallTimeout(f"discovered_law exceeded {timeout_s:g}s")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    return wrapper


# Default held-out test cases: (p1, p2, pos2, velocity2, measurement_times)
_DEFAULT_TEST_CASES = [
    {"p1": 1.0,  "p2": 1.0,  "pos2": [3.0, 0.0],  "velocity2": [0.0, 0.5],  "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
    {"p1": 2.0,  "p2": 1.0,  "pos2": [5.0, 0.0],  "velocity2": [0.0, 0.0],  "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
    {"p1": 1.0,  "p2": 2.0,  "pos2": [-4.0, 2.0], "velocity2": [0.3, -0.3], "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
]


class Evaluator:
    """
    Evaluates a discovered law function against the simulator.

    Args:
        executor: The same SimulationExecutor used during discovery (same world).
        test_cases: List of experiment dicts. Defaults to _DEFAULT_TEST_CASES.
    """

    def __init__(
        self,
        executor: SimulationExecutor,
        test_cases: list[dict] = None,
    ):
        self.executor = executor
        self.test_cases = test_cases or _DEFAULT_TEST_CASES

    def evaluate(
        self,
        law_source: str,
        verbose: bool = True,
        training_trajectories: Optional[list] = None,
    ) -> dict:
        """
        Execute the agent's law and compare against ground truth.

        Args:
            law_source: Python source string containing the `discovered_law` function.
            verbose: If True, print per-case results.
            training_trajectories: Optional list of training samples (as
                produced by `_extract_training_trajectories`). When the law
                source also defines `fit_parameters()`, scipy.optimize tunes
                the declared free parameters against these trajectories
                before the MSE scoring below runs.

        Returns:
            dict with keys:
                mean_pos_error  — mean Euclidean position error across all (case, time) pairs
                max_pos_error   — max Euclidean position error
                per_case        — list of per-case mean position errors
                passed          — bool, True if mean_pos_error < 0.1 (10% of typical scale)
                fit             — info dict from the parameter fit, or None
        """
        discovered_law = _compile_law(law_source)
        discovered_law, fit_info = _maybe_fit(
            law_source, discovered_law, training_trajectories, _two_particle_loss, verbose
        )

        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []  # collected for plotting

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_pos1 = np.asarray(gt["pos1"])
            gt_pos2 = np.asarray(gt["pos2"])
            try:
                # Call discovered_law at each measurement time to build a full trajectory
                pred_traj = []
                case_errors = []
                vel = list(case["velocity2"])
                for j, t in enumerate(case["measurement_times"]):
                    p2_out, v2_out = discovered_law(
                        pos1=[0.0, 0.0],
                        pos2=case["pos2"],
                        p1=case["p1"],
                        p2=case["p2"],
                        velocity2=case["velocity2"],
                        duration=t,
                    )
                    p2_out = np.asarray(p2_out)
                    # If the law returns a trajectory array, take the last point
                    if p2_out.ndim == 2:
                        p2_out = p2_out[-1]
                    pred_traj.append(p2_out.tolist())
                    err = float(np.linalg.norm(p2_out - np.asarray(gt_pos2[j])))
                    case_errors.append(err)

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                all_errors.extend(case_errors)
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "p1": case["p1"], "p2": case["p2"],
                    "gt1": gt_pos1.tolist(),
                    "gt": gt_pos2.tolist(),
                    "pred": pred_traj,
                    "error": mean_err,
                })

                if verbose:
                    print(f"  Case {i+1}: mean_pos_error = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR — {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "p1": case["p1"], "p2": case["p2"],
                    "gt1": gt_pos1.tolist(),
                    "gt": gt_pos2.tolist(),
                    "pred": None,
                    "error": float("inf"),
                })

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.1

        if verbose:
            print(f"\n  Mean position error: {mean_total:.4f}")
            print(f"  Max  position error: {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
            "fit": fit_info,
        }


# seperate circle world for now as particle amounts are hard-coded into the evals
_CIRCLE_TEST_CASES = [
    # Ring with tangential velocity — tests orbital / spiral dynamics
    {"ring_radius": 5.0, "initial_tangential_velocity": 0.3,
     "measurement_times": [2.0, 4.0, 6.0, 8.0, 10.0]},
]


class CircleEvaluator:
    """
    Evaluates a discovered law for the 11-particle circle world.

    The discovered_law signature is:
        discovered_law(positions, velocities, duration) -> positions_final

    where:
        positions  — list of 11 [x, y] coords relative to center at t=0
        velocities — list of 11 [vx, vy] at t=0
        duration   — float, time to simulate
        return     — list/array of 11 [x, y] positions at t=duration

    The evaluator calls discovered_law once per measurement time (duration=t),
    computes Euclidean error per particle, and averages across all particles,
    times, and test cases.
    """

    def __init__(self, executor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _CIRCLE_TEST_CASES

    def evaluate(
        self,
        law_source: str,
        verbose: bool = True,
        training_trajectories: Optional[list] = None,
    ) -> dict:
        discovered_law = _compile_law(law_source)
        discovered_law, fit_info = _maybe_fit(
            law_source, discovered_law, training_trajectories, _circle_loss, verbose
        )
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])   # (T, 11, 2)
            gt_velocities = np.asarray(gt["velocities"]) # (T, 11, 2)
            init_pos = gt_positions[0] if len(gt_positions) > 0 else None

            # Initial conditions: positions and velocities at t=0
            # Run the simulator one step from t=0 to get t=0 state
            zero_result = self.executor.run([{**case, "measurement_times": [case["measurement_times"][0]]}])
            # Actually just reconstruct from ring_radius and v_tang
            ring_radius = float(case.get("ring_radius", 5.0))
            v_tang = float(case.get("initial_tangential_velocity", 0.0))
            angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            ring_pos = np.column_stack([ring_radius * np.cos(angles), ring_radius * np.sin(angles)])
            init_positions = np.vstack([[[0.0, 0.0]], ring_pos]).tolist()
            ring_vel = np.column_stack([-v_tang * np.sin(angles), v_tang * np.cos(angles)])
            init_velocities = np.vstack([[[0.0, 0.0]], ring_vel]).tolist()

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=t,
                    )
                    pos_out = np.asarray(pos_out)   # (11, 2)
                    pred_traj.append(pos_out.tolist())

                    # Per-particle Euclidean error
                    errs = np.linalg.norm(pos_out - gt_positions[j], axis=-1)  # (11,)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append({
                    "case": i + 1,
                    "ring_radius": case["ring_radius"],
                    "v_tang": case["initial_tangential_velocity"],
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist(),
                    "pred": pred_traj,
                    "error": mean_err,
                })
                if verbose:
                    print(f"  Case {i+1} (r={case['ring_radius']}, v_t={case['initial_tangential_velocity']}): "
                          f"mean_pos_error = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR — {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append({
                    "case": i + 1,
                    "ring_radius": case["ring_radius"],
                    "v_tang": case["initial_tangential_velocity"],
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist() if gt_positions is not None else [],
                    "pred": None,
                    "error": float("inf"),
                })

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.5   # looser threshold: 11-particle problem is harder

        if verbose:
            print(f"\n  Mean position error (all particles): {mean_total:.4f}")
            print(f"  Max  position error:                 {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
            "fit": fit_info,
        }


# this is not working at all right now
_SPECIES_TEST_CASES = [
    # Asymmetric layout: species differences should cause divergent trajectories
    {
        "positions": [[0, 0], [4, 0], [-4, 0], [0, 4], [0, -4], [3, 3]],
        "velocities": [[0, 0], [0, 0.3], [0, -0.3], [0.3, 0], [-0.3, 0], [0, 0]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
    # Closer particles — stronger forces reveal species difference faster
    {
        "positions": [[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2], [2, 2]],
        "velocities": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    },
]


class SpeciesEvaluator:
    """
    Evaluates a discovered law for the 6-particle species world.

    The discovered_law signature is:
        discovered_law(positions, velocities, duration) -> positions_final

    where:
        positions  -- list of 6 [x, y] coords relative to center at t=0
        velocities -- list of 6 [vx, vy] at t=0
        duration   -- float, time to simulate
        return     -- list/array of 6 [x, y] positions at t=duration
    """

    def __init__(self, executor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _SPECIES_TEST_CASES

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        discovered_law = _compile_law(law_source)
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])    # (T, 6, 2)
            init_positions = case["positions"]
            init_velocities = case["velocities"]

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=t,
                    )
                    pos_out = np.asarray(pos_out)   # (6, 2)
                    pred_traj.append(pos_out.tolist())

                    errs = np.linalg.norm(pos_out - gt_positions[j], axis=-1)  # (6,)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist(),
                    "pred": pred_traj,
                    "error": mean_err,
                })
                if verbose:
                    print(f"  Case {i+1}: mean_pos_error = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist(),
                    "pred": None,
                    "error": float("inf"),
                })

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.3

        if verbose:
            print(f"\n  Mean position error (all particles): {mean_total:.4f}")
            print(f"  Max  position error:                 {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }

_THREE_SPECIES_TEST_CASES = [
    {
        "probe_positions": [[5, 0], [0, 5], [-5, 0], [0, -5], [7, 7]],
        "probe_velocities": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
    {
        "probe_positions": [[3, 3], [-3, 3], [-3, -3], [3, -3], [0, 0]],
        "probe_velocities": [[0.2, 0], [0, 0.2], [-0.2, 0], [0, -0.2], [0, 0]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
]


class ThreeSpeciesEvaluator:
    """
    Evaluates a discovered law for the 35-particle three-species world.

    The discovered_law signature is:
        discovered_law(positions, velocities, duration) -> positions_final

    where:
        positions  -- list of 35 [x, y] coords relative to center at t=0
        velocities -- list of 35 [vx, vy] at t=0
        duration   -- float, time to simulate
        return     -- list/array of 35 [x, y] positions at t=duration
    """

    def __init__(self, executor: ThreeSpeciesExecutor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _THREE_SPECIES_TEST_CASES

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        discovered_law = _compile_law(law_source)
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])    # (T, 35, 2)
            bg_init = np.asarray(gt["background_initial_positions"])  # (30, 2)

            # Reconstruct initial conditions for all 35 particles
            probe_pos = np.asarray(case["probe_positions"])    # (5, 2)
            probe_vel = np.asarray(case["probe_velocities"])   # (5, 2)
            init_positions = np.vstack([bg_init, probe_pos]).tolist()
            init_velocities = np.vstack([
                np.zeros((self.executor.N_BACKGROUND, 2)), probe_vel
            ]).tolist()

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=t,
                    )
                    pos_out = np.asarray(pos_out)   # (35, 2)
                    pred_traj.append(pos_out.tolist())

                    errs = np.linalg.norm(pos_out - gt_positions[j], axis=-1)  # (35,)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist(),
                    "pred": pred_traj,
                    "error": mean_err,
                })
                if verbose:
                    print(f"  Case {i+1}: mean_pos_error = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist(),
                    "pred": None,
                    "error": float("inf"),
                })

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.5  # looser threshold: 35-particle problem is harder

        if verbose:
            print(f"\n  Mean position error (all particles): {mean_total:.4f}")
            print(f"  Max  position error:                 {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }


_DARK_MATTER_TEST_CASES = [
    {
        # Probes at large radii with tangential velocities (CCW visible orbits)
        "probe_positions": [[12, 0], [0, 14], [-11, 0], [0, -13], [10, 10]],
        "probe_velocities": [[0, 2.0], [-2.0, 0], [0, -1.5], [2.0, 0], [-1.5, 1.5]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    {
        # Probes at moderate-large radii, CW visible orbits
        "probe_positions": [[9, 5], [-7, 10], [-10, -6], [6, -11], [0, 15]],
        "probe_velocities": [[0.5, -2.0], [2.0, 0.5], [-0.5, 2.0], [-2.0, -0.5], [2.5, 0]],
        "visible_velocity_sign": -1.0,  # CW orbits
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
]


class DarkMatterEvaluator:
    """
    Evaluates a discovered law for the dark matter world.

    The agent's law operates on 25 particles (20 visible + 5 probes).
    Scoring is based ONLY on the 5 probe particles (agent indices 20-24),
    whose exact initial positions and velocities the agent knows.

    The discovered_law signature is:
        discovered_law(positions, velocities, duration) -> positions_final

    where:
        positions  -- list of 25 [x, y] coords relative to center at t=0
        velocities -- list of 25 [vx, vy] at t=0
        duration   -- float, time to simulate
        return     -- list/array of 25 [x, y] positions at t=duration
    """

    def __init__(self, executor: DarkMatterExecutor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _DARK_MATTER_TEST_CASES

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        discovered_law = _compile_law(law_source)

        # Run through the NORMAL executor (agent-facing, 25-particle output)
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)
        # Also run full simulation for plotting (all 35 + field)
        full_truths = self.executor.run_full(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        n_vis = self.executor.N_VISIBLE  # 20
        # Probe indices in agent-facing output (25 particles: 0-19 visible, 20-24 probes)
        probe_slice = slice(n_vis, n_vis + self.executor.N_PROBES)  # 20:25

        for i, (case, gt, gt_full) in enumerate(
            zip(self.test_cases, ground_truths, full_truths)
        ):
            gt_positions = np.asarray(gt["positions"])   # (T, 25, 2)
            bg_init = np.asarray(gt["background_initial_positions"])  # (20, 2)

            # Reconstruct agent-visible initial conditions
            vis_vel_sign = float(case.get("visible_velocity_sign", 1.0))
            vis_vel = vis_vel_sign * self.executor._visible_velocities  # (20, 2)
            probe_pos = np.asarray(case["probe_positions"])
            probe_vel = np.asarray(case["probe_velocities"])
            init_positions = np.vstack([bg_init, probe_pos]).tolist()
            init_velocities = np.vstack([vis_vel, probe_vel]).tolist()

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=t,
                    )
                    pos_out = np.asarray(pos_out)   # (25, 2)
                    pred_traj.append(pos_out.tolist())

                    # Score only on the 5 probe particles
                    errs = np.linalg.norm(
                        pos_out[probe_slice] - gt_positions[j, probe_slice], axis=-1
                    )
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist(),       # (T, 25, 2) agent-visible
                    "gt_full": gt_full["positions"],   # (T, 35, 2) all particles
                    "field_snapshots": gt_full["field_snapshots"],
                    "dark_initial": gt_full["dark_initial_positions"],
                    "pred": pred_traj,
                    "error": mean_err,
                })
                if verbose:
                    print(f"  Case {i+1}: mean_probe_error = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist(),
                    "gt_full": gt_full["positions"],
                    "field_snapshots": gt_full["field_snapshots"],
                    "dark_initial": gt_full["dark_initial_positions"],
                    "pred": None,
                    "error": float("inf"),
                })

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.5

        if verbose:
            print(f"\n  Mean position error (probes only): {mean_total:.4f}")
            print(f"  Max  position error:               {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }


# -----------------
# utility functions
def clean_law_source(source: str) -> str:
    """Strip markdown fences and prose before the first code line."""
    import re as _re
    source = _re.sub(r"^```[a-zA-Z]*\n?", "", source.strip(), flags=_re.MULTILINE)
    source = source.replace("```", "")
    lines = source.splitlines()
    code_start = next(
        (i for i, l in enumerate(lines) if l.startswith("def ") or l.startswith("import ") or l.startswith("from ")),
        0,
    )
    return "\n".join(lines[code_start:])


def _compile_law(source: str) -> Callable:
    """Compile and return the discovered_law function from a source string.

    The returned callable is wrapped with a per-call wall-clock timeout so a
    pathological law (tiny-dt for-loop, stiff ODE, infinite loop) cannot hang
    scoring or the fit objective. Every consumer — direct scoring loops and
    loss_fn calls inside the optimizer — inherits the cap automatically.
    """
    source = clean_law_source(source)
    namespace = {}
    exec(compile(source, "<discovered_law>", "exec"), namespace)
    if "discovered_law" not in namespace:
        raise ValueError("Source does not define a function named `discovered_law`")
    return _wrap_with_timeout(namespace["discovered_law"])


def _compile_fit_parameters(source: str) -> Optional[Callable]:
    """
    Compile and return the optional fit_parameters() function if the law
    source defines one. Returns None when absent.
    """
    source = clean_law_source(source)
    namespace = {}
    try:
        exec(compile(source, "<fit_parameters>", "exec"), namespace)
    except Exception:
        return None
    return namespace.get("fit_parameters")


def _extract_training_trajectories(conversation_log: list) -> list:
    """
    Walk a DiscoveryAgent.conversation_log and collect (input_case, output)
    pairs from all successful experiment rounds. Each returned dict has the
    experiment's input keys plus the executor's output arrays.

    Used as the fit set for evaluator-side parameter optimisation: the agent
    already paid the simulator cost during discovery, so we reuse those
    trajectories rather than generating fresh ones.
    """
    training = []
    if not conversation_log:
        return training
    for entry in conversation_log:
        if entry.get("action") != "experiment":
            continue
        inputs = entry.get("experiment_input")
        outputs = entry.get("experiment_output")
        if not inputs or not outputs:
            continue
        for inp, out in zip(inputs, outputs):
            if out is None or not isinstance(out, dict):
                continue
            training.append({"input": inp, "output": out})
    return training


def _two_particle_loss(law: Callable, training: list) -> float:
    """Mean-squared position error on 2-particle training trajectories."""
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        obs_pos2 = np.asarray(out.get("pos2", []))
        times = out.get("measurement_times", case.get("measurement_times", []))
        if obs_pos2.ndim != 2 or len(times) == 0:
            continue
        for t, obs in zip(times, obs_pos2):
            pred, _ = law(
                pos1=[0.0, 0.0],
                pos2=case["pos2"],
                p1=case["p1"],
                p2=case["p2"],
                velocity2=case["velocity2"],
                duration=float(t),
            )
            pred = np.asarray(pred)
            if pred.ndim == 2:
                pred = pred[-1]
            diff = pred - np.asarray(obs)
            total_sq += float(np.dot(diff, diff))
            count += 1
    if count == 0:
        return float("inf")
    return total_sq / count


def _circle_loss(law: Callable, training: list) -> float:
    """Mean-squared position error on circle-world training trajectories."""
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        obs_positions = np.asarray(out.get("positions", []))
        times = out.get("measurement_times", case.get("measurement_times", []))
        if obs_positions.ndim != 3 or len(times) == 0:
            continue
        ring_radius = float(case.get("ring_radius", 5.0))
        v_tang = float(case.get("initial_tangential_velocity", 0.0))
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        ring_pos = np.column_stack([ring_radius * np.cos(angles), ring_radius * np.sin(angles)])
        init_positions = np.vstack([[[0.0, 0.0]], ring_pos]).tolist()
        ring_vel = np.column_stack([-v_tang * np.sin(angles), v_tang * np.cos(angles)])
        init_velocities = np.vstack([[[0.0, 0.0]], ring_vel]).tolist()
        for t, obs in zip(times, obs_positions):
            pred = law(
                positions=init_positions,
                velocities=init_velocities,
                duration=float(t),
            )
            pred = np.asarray(pred)
            if pred.shape != obs.shape:
                return float("inf")
            diff = pred - obs
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


def _three_species_loss(law: Callable, training: list) -> float:
    """Mean-squared position error on three-species training trajectories.

    The discovered law operates on all 35 particles (30 fixed background
    sources of three hidden species + 5 neutral probes). Loss aggregates
    squared distance across every particle, every measurement time, every
    sample — matching ThreeSpeciesEvaluator's all-particles convention.

    Each training sample provides the full 35-particle initial state in
    `input["init_positions"]` and `input["init_velocities"]`; the
    measurement times and observed positions come from `output`. The
    helper `_one_three_species_sample` in mse_fitting.py builds these
    samples from CSV t=0 + measurement rows.
    """
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        init_positions = case.get("init_positions")
        init_velocities = case.get("init_velocities")
        times = out.get("measurement_times", case.get("measurement_times", []))
        obs_positions = np.asarray(out.get("positions", []))
        if init_positions is None or init_velocities is None:
            continue
        if obs_positions.ndim != 3 or len(times) == 0:
            continue
        for j, t in enumerate(times):
            try:
                pred = np.asarray(
                    law(positions=init_positions,
                        velocities=init_velocities,
                        duration=float(t))
                )
            except Exception:
                return float("inf")
            if pred.shape != obs_positions[j].shape:
                return float("inf")
            diff = pred - obs_positions[j]
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


# Probe slice in the dark-matter agent-facing 25-particle output. Mirrors
# DarkMatterEvaluator: scoring is restricted to the 5 probes whose initial
# conditions the agent set, since the visible-particle initial velocities
# are unknown to the agent and dark matter is hidden.
_DARK_MATTER_PROBE_SLICE = slice(20, 25)


def _dark_matter_loss(law: Callable, training: list) -> float:
    """Mean-squared probe-position error on dark-matter training trajectories.

    The discovered law operates on 25 agent-visible particles (20 visible
    + 5 probes). Loss aggregates squared distance across the 5 probe
    particles only, matching DarkMatterEvaluator's probe-only scoring.
    """
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        init_positions = case.get("init_positions")
        init_velocities = case.get("init_velocities")
        times = out.get("measurement_times", case.get("measurement_times", []))
        obs_positions = np.asarray(out.get("positions", []))
        if init_positions is None or init_velocities is None:
            continue
        if obs_positions.ndim != 3 or len(times) == 0:
            continue
        for j, t in enumerate(times):
            try:
                pred = np.asarray(
                    law(positions=init_positions,
                        velocities=init_velocities,
                        duration=float(t))
                )
            except Exception:
                return float("inf")
            if pred.shape != obs_positions[j].shape:
                return float("inf")
            diff = pred[_DARK_MATTER_PROBE_SLICE] - obs_positions[j, _DARK_MATTER_PROBE_SLICE]
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


def _validate_fit_spec(spec) -> list:
    """
    Normalise the user-returned fit_parameters spec into a list of
    (name, init, (lo, hi)) tuples, raising ValueError on malformed input.
    """
    if not isinstance(spec, dict):
        raise ValueError("fit_parameters() must return a dict")
    if len(spec) > MAX_FIT_PARAMETERS:
        raise ValueError(
            f"fit_parameters() declares {len(spec)} parameters; "
            f"max allowed is {MAX_FIT_PARAMETERS}"
        )
    out = []
    for name, entry in spec.items():
        if not isinstance(entry, dict):
            raise ValueError(f"fit_parameters()['{name}'] must be a dict with 'init' and 'bounds'")
        if "init" not in entry or "bounds" not in entry:
            raise ValueError(f"fit_parameters()['{name}'] must provide both 'init' and 'bounds'")
        bounds = entry["bounds"]
        if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
            raise ValueError(f"fit_parameters()['{name}']['bounds'] must be a 2-element sequence")
        lo, hi = float(bounds[0]), float(bounds[1])
        if not lo < hi:
            raise ValueError(f"fit_parameters()['{name}']: lower bound must be below upper bound")
        init = float(entry["init"])
        if not lo <= init <= hi:
            # Clamp init into the declared bounds rather than erroring.
            init = min(max(init, lo), hi)
        out.append((name, init, (lo, hi)))
    return out


def _fit_law_parameters(
    discovered_law: Callable,
    fit_spec_list: list,
    training: list,
    loss_fn: Callable,
    maxiter: int = FIT_MAXITER,
    time_budget_s: float = FIT_TIME_BUDGET_S,
) -> dict:
    """
    Run scipy.optimize.minimize (L-BFGS-B, bounded) with a wall-clock budget.
    If `time_budget_s` is exceeded mid-fit, returns the best parameters seen
    so far rather than raising — slow agent code never blocks a benchmark cell
    indefinitely. Same code path is used for mid-round and final-eval fitting,
    so the budget covers both.
    """
    if not fit_spec_list:
        return {}
    import time
    from scipy.optimize import minimize

    names  = [s[0] for s in fit_spec_list]
    x0     = [s[1] for s in fit_spec_list]
    bounds = [s[2] for s in fit_spec_list]

    state = {
        "best_x": list(x0),
        "best_loss": float("inf"),
        "deadline": time.monotonic() + time_budget_s,
    }

    def _objective(x):
        if time.monotonic() > state["deadline"]:
            raise _FitTimeBudgetExceeded()
        kwargs = dict(zip(names, x.tolist()))
        bound_law = functools.partial(discovered_law, **kwargs)
        try:
            loss = loss_fn(bound_law, training)
        except Exception:
            return 1e12
        if loss < state["best_loss"]:
            state["best_loss"] = loss
            state["best_x"] = list(x)
        return loss

    try:
        result = minimize(
            _objective, x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        return dict(zip(names, result.x.tolist()))
    except _FitTimeBudgetExceeded:
        print(
            f"  [fit time budget {time_budget_s:.0f}s exceeded; "
            f"using best-so-far params (loss={state['best_loss']:.4g})]"
        )
        return dict(zip(names, state["best_x"]))


def _maybe_fit(
    law_source: str,
    discovered_law: Callable,
    training_trajectories: Optional[list],
    loss_fn: Callable,
    verbose: bool,
) -> tuple:
    """
    If law_source defines fit_parameters() AND training_trajectories is
    non-empty, run scipy.optimize and return (bound_law, fit_info).
    Otherwise return (discovered_law, None).

    fit_info dict keys:
        declared_params  — dict {name: {init, bounds}} before fitting
        fitted_params    — dict {name: value} after fitting (or init if fit failed)
        loss_before      — training-set loss at init values
        loss_after       — training-set loss after fitting
        error            — error string if something went wrong, else None
    """
    fit_fn = _compile_fit_parameters(law_source)
    if fit_fn is None:
        return discovered_law, None
    if not training_trajectories:
        if verbose:
            print("  [fit skipped: no training trajectories available]")
        return discovered_law, {"error": "no_training_trajectories"}

    try:
        raw_spec = fit_fn()
        fit_spec_list = _validate_fit_spec(raw_spec)
    except Exception as e:
        if verbose:
            print(f"  [fit skipped: invalid fit_parameters() — {e}]")
        return discovered_law, {"error": f"invalid_spec: {e}"}

    if not fit_spec_list:
        return discovered_law, None

    declared = {name: {"init": init, "bounds": list(bounds)}
                for name, init, bounds in fit_spec_list}

    init_kwargs = {name: init for name, init, _ in fit_spec_list}
    init_law = functools.partial(discovered_law, **init_kwargs)
    try:
        loss_before = loss_fn(init_law, training_trajectories)
    except Exception:
        loss_before = float("inf")

    try:
        fitted = _fit_law_parameters(
            discovered_law, fit_spec_list, training_trajectories, loss_fn
        )
    except Exception as e:
        if verbose:
            print(f"  [fit failed: {e}; falling back to init values]")
        return (
            functools.partial(discovered_law, **init_kwargs),
            {
                "declared_params": declared,
                "fitted_params": init_kwargs,
                "loss_before": loss_before,
                "loss_after": loss_before,
                "error": f"optimizer_failure: {e}",
            },
        )

    bound_law = functools.partial(discovered_law, **fitted)
    try:
        loss_after = loss_fn(bound_law, training_trajectories)
    except Exception:
        loss_after = float("inf")

    if verbose:
        pretty = ", ".join(f"{k}={v:.4g}" for k, v in fitted.items())
        print(f"  Fitted parameters: {pretty}")
        print(f"  Training-set loss: {loss_before:.4g} → {loss_after:.4g}")

    return bound_law, {
        "declared_params": declared,
        "fitted_params": fitted,
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "error": None,
    }


# ---------------------------------------------------------------
# Explanation judge: scores the agent's prose description of the
# physical system against the world's ground-truth optimal_explanation.
# Uses a fixed strong LLM judge (default claude-opus-4-6) for
# reproducibility across agent models.

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert physicist grading how well a student's prose description of a "
    "simulated physical system matches the ground-truth description. You are precise, "
    "fair, and reward semantic correctness over surface phrasing — paraphrases and "
    "equivalent formulations (e.g. 'inverse-square-like' ≈ '∇²φ' in 2D) should receive "
    "credit, but missing or wrong physical content should not."
)

_GENERIC_SCORING_GUIDE = """\
10 — captures every essential element correctly, with correct quantitative or relational claims where applicable.
 7–9 — captures the operator and qualitative structure but misses or muddles a quantitative detail or one structural feature.
 4–6 — partially correct: identifies the general physics regime but misses key structural features (e.g. fails to identify multiple species).
 1–3 — incorrect operator or fundamentally wrong physical picture, with only superficial correctness.
   0 — empty, irrelevant, or completely wrong."""


_JUDGE_USER_TEMPLATE = """Compare the student's description against the ground-truth description of the physical system.

<ground_truth>
{ground_truth}
</ground_truth>

<student>
{student}
</student>

Score the student description on a 0–10 integer scale based on how well it captures:
  1. The correct field equation / governing operator (e.g. Laplacian, fractional Laplacian, Helmholtz, diffusion, wave).
  2. The temporal character (static vs. time-evolving; instantaneous vs. retarded).
  3. The force law / coupling structure (how particles couple to the field, including p1/p2 roles).
  4. Any structural features unique to this world: hidden species and their relative coupling strengths and signs, neutral probes, hidden/dark sources, screening lengths, etc.

Use the world-specific rubric below to calibrate the bands. A 10/10 represents the best explanation achievable given the experimental capabilities — reward semantically-equivalent phrasings and numeric estimates within the tolerance specified by the rubric.

<scoring_rubric>
{rubric}
</scoring_rubric>

Respond with 1–3 sentences of justification, then your final integer score inside <score>...</score> tags. Example: "<score>7</score>"."""


class ExplanationJudge:
    """
    LLM-judge-based scorer comparing an agent's prose explanation of a discovered
    physical system against a ground-truth optimal_explanation.

    The judge is independent from the trajectory evaluator and returns a scalar
    score in [0, 1].
    """

    def __init__(self, judge_model: str = "claude-opus-4-6", max_tokens: int = 1024):
        self.judge_model = judge_model
        self.max_tokens = max_tokens

    def score(
        self,
        agent_explanation: Optional[str],
        optimal_explanation: str,
        rubric: Optional[str] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Args:
            agent_explanation: The agent's prose explanation (may be None or empty).
            optimal_explanation: The ground-truth explanation from the world config.
            rubric: Per-world scoring rubric calibrated to that problem's
                experimental capabilities. If None/empty, falls back to a
                generic 5-band guide.
            verbose: If True, print the score and judge reasoning.

        Returns:
            dict with keys:
                score        — float in [0, 1]
                raw_score    — int in [0, 10] (or None if unparseable)
                reasoning    — full judge reply text
                error        — error message if the judge call failed, else None
        """
        if not agent_explanation or not agent_explanation.strip():
            result = {
                "score": 0.0,
                "raw_score": 0,
                "reasoning": "No <explanation> tag was submitted by the agent.",
                "error": None,
            }
            if verbose:
                print(f"  Explanation score: 0.00  (no explanation submitted)")
            return result

        if not optimal_explanation:
            result = {
                "score": None,
                "raw_score": None,
                "reasoning": "No optimal_explanation defined for this world.",
                "error": "missing_ground_truth",
            }
            if verbose:
                print("  Explanation score: skipped (no ground truth defined)")
            return result

        prompt = _JUDGE_USER_TEMPLATE.format(
            ground_truth=optimal_explanation.strip(),
            student=agent_explanation.strip(),
            rubric=(rubric.strip() if rubric and rubric.strip() else _GENERIC_SCORING_GUIDE),
        )

        try:
            from scienceagent import llm_client
            reply = llm_client.complete(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                system=_JUDGE_SYSTEM_PROMPT,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            result = {
                "score": None,
                "raw_score": None,
                "reasoning": "",
                "error": f"Judge call failed: {e}",
            }
            if verbose:
                print(f"  Explanation score: ERROR — {e}")
            return result

        raw_score = _parse_judge_score(reply)
        if raw_score is None:
            result = {
                "score": None,
                "raw_score": None,
                "reasoning": reply,
                "error": "Could not parse <score> tag from judge reply.",
            }
            if verbose:
                print("  Explanation score: ERROR — unparseable judge reply")
            return result

        score = max(0.0, min(1.0, raw_score / 10.0))
        result = {
            "score": float(score),
            "raw_score": int(raw_score),
            "reasoning": reply,
            "error": None,
        }
        if verbose:
            print(f"  Explanation score: {score:.2f}  (raw {raw_score}/10)")
            print(f"  Judge reasoning: {reply.strip()}")
        return result


def _parse_judge_score(reply: str) -> Optional[float]:
    """Extract the integer score inside <score>...</score> tags."""
    if not reply:
        return None
    match = re.search(r"<score>\s*(\d+(?:\.\d+)?)\s*</score>", reply, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
