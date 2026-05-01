"""
Tests for the optional Gaussian observation noise on particle positions.

Covers:
  - noise_std=0 is bit-identical to the pre-noise behavior (hard bypass)
  - noise_std>0 perturbs positions but leaves velocities clean
  - fresh noise on every call (agent cannot average it out trivially)
  - noise_disabled() context manager restores noise_std afterwards
  - the trajectory evaluator's ground-truth run is unaffected by noise_std
"""

import numpy as np
import pytest

from scienceagent.executor import (
    SimulationExecutor,
    SpeciesExecutor,
    CircleExecutor,
    ThreeSpeciesExecutor,
    DarkMatterExecutor,
)
from scienceagent.evaluator import Evaluator

_GRAVITY_OPS = [{"type": "laplacian", "params": {"strength": 1.0}}]
_EXP = {
    "p1": 1.0,
    "p2": 1.0,
    "pos2": [3.0, 0.0],
    "velocity2": [0.0, 0.0],
    "measurement_times": [1.0, 2.0, 3.0],
}


def _make_executor(noise_std=0.0, noise_seed=None):
    return SimulationExecutor(
        operators=_GRAVITY_OPS,
        temporal_order=0,
        grid_size=(64, 64),
        domain_size=20.0,
        dt=0.005,
        noise_std=noise_std,
        noise_seed=noise_seed,
    )


def test_noise_std_zero_matches_clean_run():
    """noise_std=0 must be bit-identical to the pre-noise code path."""
    clean = _make_executor(noise_std=0.0)
    clean_result = clean.run([_EXP])[0]

    # Re-run with a different seed but still noise_std=0 — should still match
    bypassed = _make_executor(noise_std=0.0, noise_seed=12345)
    bypassed_result = bypassed.run([_EXP])[0]

    np.testing.assert_array_equal(
        np.asarray(clean_result["pos2"]), np.asarray(bypassed_result["pos2"])
    )
    np.testing.assert_array_equal(
        np.asarray(clean_result["pos1"]), np.asarray(bypassed_result["pos1"])
    )


def test_noise_perturbs_positions_but_not_velocities():
    clean = _make_executor(noise_std=0.0).run([_EXP])[0]
    noisy = _make_executor(noise_std=0.1, noise_seed=0).run([_EXP])[0]

    clean_pos2 = np.asarray(clean["pos2"])
    noisy_pos2 = np.asarray(noisy["pos2"])
    assert not np.allclose(clean_pos2, noisy_pos2), "noise should perturb positions"

    # Perturbations should be within a few sigma
    diff = noisy_pos2 - clean_pos2
    assert np.abs(diff).max() < 1.0, "noise should be small relative to scale"

    # Velocities must stay clean
    np.testing.assert_array_equal(
        np.asarray(clean["velocity2"]), np.asarray(noisy["velocity2"])
    )
    np.testing.assert_array_equal(
        np.asarray(clean["velocity1"]), np.asarray(noisy["velocity1"])
    )


def test_noise_is_fresh_across_calls():
    """Two consecutive calls with the same executor should yield different noise."""
    ex = _make_executor(noise_std=0.1, noise_seed=42)
    a = np.asarray(ex.run([_EXP])[0]["pos2"])
    b = np.asarray(ex.run([_EXP])[0]["pos2"])
    assert not np.array_equal(
        a, b
    ), "fresh noise should be sampled on every call so the agent cannot average it out"


def test_noise_seed_is_reproducible_across_executors():
    a = _make_executor(noise_std=0.1, noise_seed=7).run([_EXP])[0]["pos2"]
    b = _make_executor(noise_std=0.1, noise_seed=7).run([_EXP])[0]["pos2"]
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_noise_disabled_context_manager():
    ex = _make_executor(noise_std=0.1, noise_seed=99)
    assert ex.noise_std == 0.1
    with ex.noise_disabled():
        assert ex.noise_std == 0.0
        result = ex.run([_EXP])[0]
        clean = _make_executor(noise_std=0.0).run([_EXP])[0]
        np.testing.assert_array_equal(
            np.asarray(result["pos2"]), np.asarray(clean["pos2"])
        )
    assert ex.noise_std == 0.1, "noise_std should be restored after the context"


def test_noise_disabled_restores_on_exception():
    ex = _make_executor(noise_std=0.05)
    with pytest.raises(RuntimeError):
        with ex.noise_disabled():
            assert ex.noise_std == 0.0
            raise RuntimeError("boom")
    assert ex.noise_std == 0.05


def test_evaluator_ground_truth_is_noise_free():
    """The trajectory evaluator must compare against clean ground truth."""
    noisy_executor = _make_executor(noise_std=0.5, noise_seed=1)
    clean_executor = _make_executor(noise_std=0.0)

    test_cases = [_EXP]
    evaluator_noisy = Evaluator(noisy_executor, test_cases=test_cases)
    evaluator_clean = Evaluator(clean_executor, test_cases=test_cases)

    # A trivial law that just returns the input position unchanged — useful as a
    # fixed reference: any difference between the two evaluators must come from
    # the ground-truth trajectories, not from the predicted side.
    law_source = (
        "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n"
        "    return pos2, velocity2\n"
    )

    res_noisy = evaluator_noisy.evaluate(law_source, verbose=False)
    res_clean = evaluator_clean.evaluate(law_source, verbose=False)

    assert res_noisy["mean_pos_error"] == pytest.approx(res_clean["mean_pos_error"])
    # noise_std restored after the evaluator returns
    assert noisy_executor.noise_std == 0.5


@pytest.mark.parametrize(
    "cls",
    [
        SimulationExecutor,
        SpeciesExecutor,
        CircleExecutor,
        ThreeSpeciesExecutor,
        DarkMatterExecutor,
    ],
)
def test_all_executors_accept_noise_kwargs(cls):
    """Smoke test: every executor class accepts noise_std/noise_seed in __init__."""
    ex = cls(noise_std=0.1, noise_seed=0)
    assert ex.noise_std == 0.1
    assert hasattr(ex, "noise_disabled")
