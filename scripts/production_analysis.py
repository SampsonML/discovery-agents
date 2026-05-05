#!/usr/bin/env python
"""
Standalone analysis & plotting for a results directory.

Walks a results directory of per-trial JSONs (as produced by
`scripts/yml_benchmark.py`) and writes the same `summary.txt`,
`summary_per_model.txt`, and plots — but into a separate `analysis/`
subdirectory rather than the results directory itself.

Usage:
    python scripts/production_analysis.py results/yml_bench/production_r2
    python scripts/production_analysis.py results/yml_bench/production_r2 --out my_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

# Filenames are produced by yml_benchmark.py as `<world>[_noise<σ>]_seed<N>.json`.
_SEED_RE = re.compile(r"_seed(\d+)\.json$")

_BOOTSTRAP_RESAMPLES = 5000
_BOOTSTRAP_SEED = 0  # fixed so summary.txt is reproducible across re-aggregation

# Errors aggregate as geometric means; values below this are dropped (log undefined).
_GEOM_MIN = 1e-14
# A (model, world) pair "passes" when these are both met across its seeds.
_PASS_ERR_THRESHOLD = 0.5     # geom. mean of mean_pos_error must be < this
_PASS_SCORE_THRESHOLD = 0.7   # arithmetic mean of explanation score must be >= this

# Per-world pass thresholds: a world is "passed at k" iff at least one of the
# first k seeds (seed indices 0..k-1) achieved a trial-pass. A missing seed in
# that window counts as a fail for that slot but does not preclude other seeds
# in the same window from passing the world. Monotonically non-decreasing in k.
_PASS_K_VALUES = (1, 2, 3, 4, 5)
# Pool size for expected_passed@k MC sampling. Hard-pinned: production runs
# always use 5 seeds (seed indices 0..4). Missing seeds in the pool count as
# fails, identical to the deterministic @k metric.
_SEED_POOL_SIZE = 5
# Monte Carlo draws per (model, world, k) when estimating expected_passed@k.
_EXPECTED_PASSED_SAMPLES = 1000


def aggregate(results_dir: Path, analysis_dir: Path) -> Path:
    """Walk results_dir for per-trial JSONs and write summary.txt into analysis_dir."""
    by_trial: dict[tuple[str, str], list[tuple[float | None, float | None]]] = {}
    by_noise: dict[tuple[str, str, float], list[tuple[float | None, float | None]]] = {}
    # Seed-indexed view used for worlds_passed@k. Last-write-wins per seed if
    # duplicate filenames ever appear.
    by_seed: dict[
        tuple[str, str], dict[int, tuple[float | None, float | None]]
    ] = {}

    for json_path in sorted(results_dir.rglob("*.json")):
        if json_path.name == "config.json":
            continue
        try:
            with open(json_path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        model = d.get("model")
        world = d.get("world")
        ev = d.get("evaluation") or {}
        if not model or not world or not ev:
            continue
        mpe = ev.get("mean_pos_error")
        expl = ev.get("explanation") or {}
        score = expl.get("score") if isinstance(expl, dict) else None
        by_trial.setdefault((model, world), []).append((mpe, score))
        noise = d.get("noise_std")
        if isinstance(noise, (int, float)) and math.isfinite(noise):
            by_noise.setdefault((model, world, float(noise)), []).append((mpe, score))
        m_seed = _SEED_RE.search(json_path.name)
        if m_seed is not None:
            by_seed.setdefault((model, world), {})[int(m_seed.group(1))] = (
                mpe, score
            )

    title = results_dir.name
    lines = []
    lines.append(f"Benchmark summary  ({title})")
    lines.append("=" * 116)
    header = (
        f"{'model':<50} {'world':<15} {'n':>3} "
        f"{'expl_score [95% CI]':>22} {'geom_pos_err [95% CI]':>22}"
    )
    lines.append(header)
    lines.append("-" * 116)

    for (model, world), values in sorted(
        by_trial.items(), key=lambda kv: (_model_sort_key(kv[0][0]), kv[0][1])
    ):
        scores = [
            float(s)
            for _, s in values
            if isinstance(s, (int, float)) and math.isfinite(s)
        ]
        errs = [
            float(e)
            for e, _ in values
            if isinstance(e, (int, float)) and math.isfinite(e)
        ]
        n = len(values)
        score_str = _fmt_mean_bootstrap(scores)
        err_str = _fmt_geom_mean_bootstrap(errs)
        lines.append(f"{model:<50} {world:<15} {n:>3} {score_str:>22} {err_str:>22}")

    lines.append("-" * 116)
    lines.append("expl_score: explanation judge score in [0, 1], higher is better. Arithmetic mean.")
    lines.append(
        "geom_pos_err: geometric mean of trajectory mean_pos_error across seeds "
        f"(values < {_GEOM_MIN:.0e} dropped); lower is better."
    )
    lines.append(
        f"Format: point estimate [2.5%, 97.5%] from {_BOOTSTRAP_RESAMPLES} bootstrap "
        f"resamples (seed={_BOOTSTRAP_SEED}). n=1 → no CI. 'n/a' = no successful runs."
    )

    out_path = analysis_dir / "summary.txt"
    out_path.write_text("\n".join(lines) + "\n")
    _make_plots(by_trial, by_seed, analysis_dir, title)
    _write_per_model_summary(by_trial, by_seed, analysis_dir, title)
    _make_noise_ablation_plot(by_noise, analysis_dir, title)
    return out_path


def _per_model_pooled(by_trial):
    """Pool every trial equally; return {model: (errs, scores)} lists."""
    pooled: dict[str, tuple[list[float], list[float]]] = {}
    for (model, _world), values in by_trial.items():
        errs, scores = pooled.setdefault(model, ([], []))
        for e, s in values:
            if isinstance(e, (int, float)) and math.isfinite(e):
                errs.append(float(e))
            if isinstance(s, (int, float)) and math.isfinite(s):
                scores.append(float(s))
    return pooled


def _write_per_model_summary(by_trial, by_seed, analysis_dir: Path, title: str) -> Path:
    pooled = _per_model_pooled(by_trial)
    models = sorted({m for m, _ in by_trial.keys()}, key=_model_sort_key)
    worlds = sorted({w for _, w in by_trial.keys()})
    n_worlds = len(worlds)
    passed = _passed_per_model(by_trial, models, worlds)
    n_trials = _trials_per_model(by_trial, models, worlds)
    worlds_at_k = {
        k: _worlds_passed_at_k(by_seed, models, worlds, k) for k in _PASS_K_VALUES
    }
    expected_at_k = {
        k: _expected_worlds_passed_at_k(by_seed, models, worlds, k)
        for k in _PASS_K_VALUES
    }
    k_col_w = max(6, len(f"{n_worlds}/{n_worlds}") + 2)
    e_col_w = max(14, len(f"{n_worlds}.00±0.00/{n_worlds}") + 1)
    k_headers = " ".join(f"{f'@k={k}':>{k_col_w}}" for k in _PASS_K_VALUES)
    e_headers = " ".join(f"{f'E@k={k}':>{e_col_w}}" for k in _PASS_K_VALUES)
    width = 124 + (k_col_w + 1) * len(_PASS_K_VALUES) + (e_col_w + 1) * len(_PASS_K_VALUES)
    lines = []
    lines.append(f"Per-model summary  ({title})")
    lines.append("=" * width)
    header = (
        f"{'model':<50} {'n_trials':>9} {'passed':>10} "
        f"{'expl_score [95% CI]':>22} {'geom_pos_err [95% CI]':>26} "
        f"{k_headers} {e_headers}"
    )
    lines.append(header)
    lines.append("-" * width)
    for model in sorted(pooled, key=_model_sort_key):
        errs, scores = pooled[model]
        n = max(len(errs), len(scores))
        passed_str = f"{passed.get(model, 0)}/{n_trials.get(model, n)}"
        k_cells = " ".join(
            f"{f'{worlds_at_k[k].get(model, 0)}/{n_worlds}':>{k_col_w}}"
            for k in _PASS_K_VALUES
        )
        e_cells = " ".join(
            f"{f'{expected_at_k[k].get(model, (0.0, 0.0))[0]:.2f}±{expected_at_k[k].get(model, (0.0, 0.0))[1]:.2f}/{n_worlds}':>{e_col_w}}"
            for k in _PASS_K_VALUES
        )
        lines.append(
            f"{model:<50} {n:>9} {passed_str:>10} "
            f"{_fmt_mean_bootstrap(scores):>22} {_fmt_geom_mean_bootstrap(errs):>26} "
            f"{k_cells} {e_cells}"
        )
    lines.append("-" * width)
    lines.append("Pooled across all worlds and seeds (every trial counts equally).")
    lines.append(
        f"passed: number of trials (summed over worlds & seeds) with mean_pos_error < "
        f"{_PASS_ERR_THRESHOLD} AND explanation_score >= {_PASS_SCORE_THRESHOLD}."
    )
    lines.append(
        f"@k=K: number of worlds (out of {n_worlds}) where at least one of seeds "
        "0..K-1 produced a trial-pass for that model. Missing seeds count as a fail "
        "for their slot. Monotonically non-decreasing in K."
    )
    lines.append(
        f"E@k=K: expected number of worlds passed when K seed positions are sampled "
        f"uniformly without replacement from a {_SEED_POOL_SIZE}-seed pool, averaged "
        f"over {_EXPECTED_PASSED_SAMPLES} Monte Carlo draws (RNG seed={_BOOTSTRAP_SEED}). "
        "Format: mean±SEM/N, where SEM is the standard error of the mean across draws."
    )
    lines.append(
        "geom_pos_err: geometric mean of mean_pos_error "
        f"(values < {_GEOM_MIN:.0e} dropped); lower is better."
    )
    lines.append(
        f"Format: point estimate [2.5%, 97.5%] from {_BOOTSTRAP_RESAMPLES} bootstrap "
        f"resamples (seed={_BOOTSTRAP_SEED}). n=1 → no CI. 'n/a' = no successful runs."
    )
    out_path = analysis_dir / "summary_per_model.txt"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def _model_color_map(models, plt):
    """Stable {model: rgba} color mapping shared across every plot.

    Indexes into `tab10` by the canonical sort order, so the same model gets the
    same color in every figure regardless of which subset is currently plotted.
    Qwen 3.2 (Qwen3-235B) is forced to black to disambiguate it from Opus.
    """
    cmap = plt.get_cmap("tab10")
    colors = {
        m: cmap(i % 10) for i, m in enumerate(sorted(models, key=_model_sort_key))
    }
    for m in colors:
        if "qwen3-235b" in m.lower().replace(".", "-"):
            colors[m] = (0.0, 0.0, 0.0, 1.0)
    return colors


def _world_label(world: str) -> str:
    """Display name for a world: drop the `_easy` suffix on coulomb, underscores → spaces."""
    if world == "coulomb_easy":
        return "coulomb"
    return world.replace("_", " ")


def _short(model: str) -> str:
    """Trim long provider-prefixed model names for legend display."""
    short = model.rsplit("/", 1)[-1]
    if "Qwen3-235B" in short:
        return "Qwen 3.2 Instruct"
    if "Qwen3.5" in short:
        return "Qwen 3.5"
    return short


# Models excluded from plots (substring match against the full model string).
# Text summaries still include them.
_PLOT_EXCLUDED_MODELS = ("Kimi",)


def _is_plot_excluded(model: str) -> bool:
    return any(tag in model for tag in _PLOT_EXCLUDED_MODELS)


# Model release dates (ISO 8601). Used by the release-date scatter plot;
# models not in this dict are silently skipped from that plot.
_MODEL_RELEASE_DATES = {
    "claude-opus-4-7": "2026-04-16",
    "claude-sonnet-4-6": "2026-02-17",
    "claude-haiku-4-5": "2025-10-15",
    "azure/gpt-5.4": "2026-03-05",
    "azure/gpt-5.5": "2026-04-23",
    "together/Qwen/Qwen3.5-397B-A17B": "2026-02-16",
    "together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput": "2025-07-21",
    "together/deepseek-ai/DeepSeek-V3.1": "2025-08-21",
    "together/deepseek-ai/DeepSeek-R1": "2025-01-20",
    "together/openai/gpt-oss-120b": "2025-08-05",
    "together/openai/gpt-oss-20b": "2025-08-05",
}


# Preferred ordering for "headline" models; everything else falls in alphabetically.
_MODEL_PRIORITY = (
    "opus-4-7",
    "sonnet-4-6",
    "haiku-4-5",
    "gpt-5-5",
    "gpt-5-4",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "deepseek-v3",
    "deepseek-r1",
    "qwen3-5",       # Qwen 3.5  (Qwen3.5-397B-A17B → "qwen3-5-...")
    "qwen3-235b",    # Qwen 3.2 Instruct
)


def _model_sort_key(model: str):
    # Match against the full provider-prefixed name (normalized), so renames in
    # `_short()` (e.g. "Qwen 3.5" → with a space) don't break substring matching.
    norm = model.lower().replace(".", "-")
    for i, tag in enumerate(_MODEL_PRIORITY):
        if tag in norm:
            return (i, norm)
    return (len(_MODEL_PRIORITY), norm)


def _trial_values(by_trial, model, world):
    values = by_trial.get((model, world), [])
    errs = [
        float(e) for e, _ in values if isinstance(e, (int, float)) and math.isfinite(e)
    ]
    scores = [
        float(s) for _, s in values if isinstance(s, (int, float)) and math.isfinite(s)
    ]
    return errs, scores


def _bootstrap_ci(values):
    """Return (mean, err_lo, err_hi) — distances to the 2.5/97.5 percentiles."""
    if not values:
        return None
    if len(values) == 1:
        return (values[0], 0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(_BOOTSTRAP_RESAMPLES, arr.size))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    m = float(arr.mean())
    return (m, m - lo, hi - m)


def _bootstrap_ci_geom(values):
    """Bootstrap CI of the geometric mean. Drops values < _GEOM_MIN."""
    arr = np.asarray([v for v in values if v >= _GEOM_MIN], dtype=float)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return (float(arr[0]), 0.0, 0.0)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(_BOOTSTRAP_RESAMPLES, arr.size))
    boot = np.exp(np.log(arr[idx]).mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    m = float(np.exp(np.log(arr).mean()))
    return (m, m - lo, hi - m)


def _trial_passes(err, score) -> bool:
    """A single trial passes iff mean_pos_error < threshold AND explanation_score >= threshold."""
    if not isinstance(err, (int, float)) or not math.isfinite(err):
        return False
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        return False
    return float(err) < _PASS_ERR_THRESHOLD and float(score) >= _PASS_SCORE_THRESHOLD


def _passed_per_model(by_trial, models, worlds) -> dict[str, int]:
    """Sum of trial passes across every (world, seed) for each model."""
    counts = {m: 0 for m in models}
    for (m, _w), values in by_trial.items():
        if m not in counts:
            continue
        for err, score in values:
            if _trial_passes(err, score):
                counts[m] += 1
    return counts


def _expected_worlds_passed_at_k(
    by_seed,
    models,
    worlds,
    k: int,
    pool_size: int = _SEED_POOL_SIZE,
    n_samples: int = _EXPECTED_PASSED_SAMPLES,
) -> dict[str, tuple[float, float]]:
    """Return {model: (mean, sem)} of worlds_passed@k under random seed sampling.

    For each Monte Carlo draw, sample `k` distinct seed positions from
    {0..pool_size-1} (without replacement), then count how many worlds had at
    least one passing seed within that draw. Mean and standard error of the
    mean are computed over `n_samples` draws (RNG seed `_BOOTSTRAP_SEED`).

    Missing seeds count as fails for their slot — identical treatment to the
    deterministic _worlds_passed_at_k metric.
    """
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = np.stack(
        [rng.choice(pool_size, size=k, replace=False) for _ in range(n_samples)]
    )
    out: dict[str, tuple[float, float]] = {}
    for m in models:
        per_sample_counts = np.zeros(n_samples, dtype=float)
        for w in worlds:
            seed_results = by_seed.get((m, w), {})
            passes = np.array([
                _trial_passes(*seed_results.get(s, (None, None)))
                for s in range(pool_size)
            ])
            if not passes.any():
                continue
            per_sample_counts += passes[samples].any(axis=1).astype(float)
        mean = float(per_sample_counts.mean())
        sem = (
            float(per_sample_counts.std(ddof=1) / math.sqrt(n_samples))
            if n_samples > 1
            else 0.0
        )
        out[m] = (mean, sem)
    return out


def _expected_pass_rate_per_world_at_k(
    by_seed,
    models,
    worlds,
    k: int,
    pool_size: int = _SEED_POOL_SIZE,
    n_samples: int = _EXPECTED_PASSED_SAMPLES,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Return {(model, world): (mean, sem)} of pass-rate@k per world.

    For each Monte Carlo draw, sample `k` distinct seed positions from
    {0..pool_size-1} (without replacement); a (model, world) cell counts as
    passed for that draw iff at least one sampled seed produced a trial-pass.
    Mean and SEM are computed over `n_samples` draws (RNG seed
    `_BOOTSTRAP_SEED`). Missing seeds count as fails for their slot.
    """
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = np.stack(
        [rng.choice(pool_size, size=k, replace=False) for _ in range(n_samples)]
    )
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for m in models:
        for w in worlds:
            seed_results = by_seed.get((m, w), {})
            passes = np.array([
                _trial_passes(*seed_results.get(s, (None, None)))
                for s in range(pool_size)
            ])
            if not passes.any():
                out[(m, w)] = (0.0, 0.0)
                continue
            per_sample = passes[samples].any(axis=1).astype(float)
            mean = float(per_sample.mean())
            sem = (
                float(per_sample.std(ddof=1) / math.sqrt(n_samples))
                if n_samples > 1
                else 0.0
            )
            out[(m, w)] = (mean, sem)
    return out


def _worlds_passed_at_k(by_seed, models, worlds, k: int) -> dict[str, int]:
    """Count of worlds (out of len(worlds)) passed at threshold k.

    A world is passed at k iff at least one of seed indices 0..k-1 produced a
    trial-pass for that (model, world). Missing seeds in the 0..k-1 window
    count as fails for their slot but do not preclude other seeds in the same
    window from passing the world.
    """
    counts = {m: 0 for m in models}
    for m in models:
        for w in worlds:
            seed_results = by_seed.get((m, w), {})
            for seed in range(k):
                err, score = seed_results.get(seed, (None, None))
                if _trial_passes(err, score):
                    counts[m] += 1
                    break
    return counts


def _trials_per_model(by_trial, models, worlds) -> dict[str, int]:
    """Number of (world, seed) trials run for each model — denominator for passes."""
    return {
        m: sum(len(by_trial.get((m, w), [])) for w in worlds)
        for m in models
    }


def _make_plots(by_trial, by_seed, analysis_dir: Path, title: str) -> None:
    """Write summary.{png,pdf} (grouped bars) and runs.{png,pdf} (strip plot)."""
    if not by_trial:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots", file=sys.stderr)
        return

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })

    by_trial = {k: v for k, v in by_trial.items() if not _is_plot_excluded(k[0])}
    if not by_trial:
        return

    models = sorted({m for m, _ in by_trial.keys()}, key=_model_sort_key)
    worlds = sorted({w for _, w in by_trial.keys()})
    _make_bar_plot(by_trial, models, worlds, analysis_dir, title, plt)
    _make_strip_plot(by_trial, models, worlds, analysis_dir, title, plt)
    _make_per_model_plot(by_trial, models, analysis_dir, title, plt)
    _make_passed_plot(by_trial, by_seed, models, worlds, analysis_dir, title, plt)
    _make_pareto_plot(by_trial, models, worlds, analysis_dir, title, plt)
    _make_world_difficulty_plot(by_trial, worlds, analysis_dir, title, plt)
    _make_per_world_passed_plot(by_trial, models, worlds, analysis_dir, title, plt)
    _make_expected_pass_rate_per_world_plot(
        by_seed, by_trial, models, worlds, analysis_dir, title, plt, k=3
    )
    _make_worlds_passed_at_k_plot(by_seed, models, worlds, analysis_dir, title, plt)
    _make_worlds_passed_at_k_scatter(by_seed, models, worlds, analysis_dir, title, plt)
    _make_worlds_expected_passed_at_k_scatter(
        by_seed, models, worlds, analysis_dir, title, plt
    )
    _make_pareto_expected_combo_plot(
        by_trial, by_seed, models, worlds, analysis_dir, title, plt
    )
    _make_expected_passed_vs_release_date_plot(
        by_seed, models, worlds, analysis_dir, title, plt, k=3
    )
    _make_pareto_expected_release_combo_plot(
        by_trial, by_seed, models, worlds, analysis_dir, title, plt, k=3
    )


def _make_per_model_plot(by_trial, models, analysis_dir, title, plt) -> None:
    """One bar per model in two side-by-side panels: expl. score and MPE."""
    pooled = _per_model_pooled(by_trial)
    n_models = len(models)
    if n_models == 0:
        return
    fig_w = max(8.0, 1.2 * n_models * 2)
    fig, (ax_score, ax_err) = plt.subplots(1, 2, figsize=(fig_w, 5))
    x = np.arange(n_models)
    model_colors = _model_color_map(models, plt)
    colors = [model_colors[m] for m in models]

    score_means, score_lo, score_hi = [], [], []
    err_means, err_lo, err_hi = [], [], []
    for model in models:
        errs, scores = pooled.get(model, ([], []))
        s = _bootstrap_ci(scores)
        e = _bootstrap_ci_geom(errs)
        score_means.append(s[0] if s else np.nan)
        score_lo.append(s[1] if s else 0.0)
        score_hi.append(s[2] if s else 0.0)
        err_means.append(e[0] if e else np.nan)
        err_lo.append(e[1] if e else 0.0)
        err_hi.append(e[2] if e else 0.0)

    ax_score.bar(x, score_means, yerr=[score_lo, score_hi], color=colors, capsize=3)
    ax_err.bar(x, err_means, yerr=[err_lo, err_hi], color=colors, capsize=3)

    labels = [_short(m) for m in models]
    for ax, ax_title, ylabel in [
        (
            ax_score,
            "Explanation score (pooled across worlds)",
            "score [0, 1] ↑",
        ),
        (
            ax_err,
            "Mean position error (geom. mean, pooled across worlds)",
            "geom. mean error ↓",
        ),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=15)
        ax.set_ylabel(ylabel)
        ax.set_title(ax_title)
    ax_score.set_ylim(0, 1.05)
    ax_err.set_yscale("log")

    fig.suptitle(f"Per-model rollup: {title}", y=1.02)
    fig.tight_layout()
    fig.savefig(analysis_dir / "summary_per_model.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "summary_per_model.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_worlds_passed_at_k_plot(
    by_seed, models, worlds, analysis_dir, title, plt
) -> None:
    """5-row plot: worlds_passed@k for k=1..5, one bar per model per row."""
    n_models = len(models)
    n_worlds = len(worlds)
    if n_models == 0 or n_worlds == 0:
        return

    worlds_at_k = {
        k: _worlds_passed_at_k(by_seed, models, worlds, k) for k in _PASS_K_VALUES
    }
    fig_w = max(8.0, 1.3 * n_models)
    n_rows = len(_PASS_K_VALUES)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(fig_w, 2.3 * n_rows + 1.5), sharex=True
    )
    if n_rows == 1:
        axes = [axes]
    x = np.arange(n_models)
    model_colors = _model_color_map(models, plt)
    colors = [model_colors[m] for m in models]
    ylabel_kw = {"fontsize": 16}

    for ax, k in zip(axes, _PASS_K_VALUES):
        counts = [worlds_at_k[k].get(m, 0) for m in models]
        ax.bar(x, counts, color=colors)
        ax.set_ylim(0, n_worlds + 0.5)
        ax.set_ylabel(f"@k={k}", **ylabel_kw)
        ax.tick_params(axis="y", labelsize=12)

    axes[0].set_title(
        f"Worlds passed at k (out of {n_worlds})", fontsize=16
    )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [_short(m) for m in models], rotation=25, ha="right", fontsize=14
    )

    fig.tight_layout()
    fig.savefig(analysis_dir / "worlds_passed_at_k.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "worlds_passed_at_k.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_worlds_passed_at_k_scatter(
    by_seed, models, worlds, analysis_dir, title, plt
) -> None:
    """Scatter+line plot: worlds_passed@k vs k, one curve per model."""
    n_models = len(models)
    n_worlds = len(worlds)
    if n_models == 0 or n_worlds == 0:
        return
    from matplotlib.lines import Line2D

    worlds_at_k = {
        k: _worlds_passed_at_k(by_seed, models, worlds, k) for k in _PASS_K_VALUES
    }
    model_colors = _model_color_map(models, plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = list(_PASS_K_VALUES)
    for model in models:
        ys = [worlds_at_k[k].get(model, 0) for k in xs]
        ax.plot(
            xs, ys,
            marker="o", color=model_colors[model],
            linewidth=1.8, markersize=10,
            markeredgecolor="white", markeredgewidth=0.7,
        )

    ax.set_xticks(xs)
    ax.set_xlabel("k", fontsize=18)
    ax.set_ylabel(f"worlds passed @k (out of {n_worlds}) ↑", fontsize=18)
    ax.set_ylim(-0.5, n_worlds + 0.5)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=10, linestyle="", label=_short(m),
        )
        for m in models
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(handles), 5),
        frameon=False,
        borderaxespad=0,
    )

    fig.tight_layout()
    fig.savefig(
        analysis_dir / "worlds_passed_at_k_scatter.png",
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        analysis_dir / "worlds_passed_at_k_scatter.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _make_worlds_expected_passed_at_k_scatter(
    by_seed, models, worlds, analysis_dir, title, plt
) -> None:
    """Scatter+line plot: expected_passed@k vs k with ±SEM error bars."""
    n_models = len(models)
    n_worlds = len(worlds)
    if n_models == 0 or n_worlds == 0:
        return
    from matplotlib.lines import Line2D

    expected_at_k = {
        k: _expected_worlds_passed_at_k(by_seed, models, worlds, k)
        for k in _PASS_K_VALUES
    }
    model_colors = _model_color_map(models, plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = list(_PASS_K_VALUES)
    for model in models:
        means = [expected_at_k[k].get(model, (0.0, 0.0))[0] for k in xs]
        sems = [expected_at_k[k].get(model, (0.0, 0.0))[1] for k in xs]
        ax.errorbar(
            xs, means, yerr=sems,
            marker="o", color=model_colors[model],
            ecolor=model_colors[model],
            linewidth=1.8, markersize=10,
            markeredgecolor="white", markeredgewidth=0.7,
            capsize=3, elinewidth=1.0,
        )

    ax.set_xticks(xs)
    ax.set_xlabel("k", fontsize=18)
    ax.set_ylabel(f"expected passed @k (out of {n_worlds}) ↑", fontsize=18)
    ax.set_ylim(-0.5, n_worlds + 0.5)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=10, linestyle="", label=_short(m),
        )
        for m in models
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(handles), 5),
        frameon=False,
        borderaxespad=0,
    )

    fig.tight_layout()
    fig.savefig(
        analysis_dir / "worlds_expected_passed_at_k_scatter.png",
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        analysis_dir / "worlds_expected_passed_at_k_scatter.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _make_passed_plot(
    by_trial, by_seed, models, worlds, analysis_dir, title, plt
) -> None:
    """3-row plot: E[@k=3], explanation score, MPE — model on x-axis."""
    n_models = len(models)
    n_worlds = len(worlds)
    if n_models == 0:
        return

    pooled = _per_model_pooled(by_trial)

    fig_w = max(8.0, 1.3 * n_models)
    fig, (ax_pass, ax_score, ax_err) = plt.subplots(
        3, 1, figsize=(fig_w, 11), sharex=True
    )
    x = np.arange(n_models)
    model_colors = _model_color_map(models, plt)
    colors = [model_colors[m] for m in models]
    ylabel_kw = {"fontsize": 18}

    # Row 1: expected worlds passed at k=3 with SEM error bars.
    expected_at_3 = _expected_worlds_passed_at_k(by_seed, models, worlds, 3)
    e3_means = [expected_at_3.get(m, (0.0, 0.0))[0] for m in models]
    e3_sems = [expected_at_3.get(m, (0.0, 0.0))[1] for m in models]
    ax_pass.bar(x, e3_means, yerr=e3_sems, color=colors, capsize=3)
    ax_pass.set_ylabel("E[@k=3] ↑", **ylabel_kw)
    ax_pass.set_ylim(0, n_worlds + 0.5)

    # Rows 2 & 3: pooled across all worlds for each model.
    score_means, score_lo, score_hi = [], [], []
    err_means, err_lo, err_hi = [], [], []
    for model in models:
        errs, scores = pooled.get(model, ([], []))
        s = _bootstrap_ci(scores)
        e = _bootstrap_ci_geom(errs)
        score_means.append(s[0] if s else np.nan)
        score_lo.append(s[1] if s else 0.0)
        score_hi.append(s[2] if s else 0.0)
        err_means.append(e[0] if e else np.nan)
        err_lo.append(e[1] if e else 0.0)
        err_hi.append(e[2] if e else 0.0)

    ax_score.bar(
        x, score_means, yerr=[score_lo, score_hi], color=colors, capsize=3
    )
    ax_score.set_ylabel("Explanation score (0-1) ↑", **ylabel_kw)
    ax_score.set_ylim(0, 1.05)

    ax_err.bar(x, err_means, yerr=[err_lo, err_hi], color=colors, capsize=3)
    ax_err.set_ylabel("MPE (geom. mean) ↓", **ylabel_kw)
    ax_err.set_yscale("log")

    ax_err.set_xticks(x)
    ax_err.set_xticklabels(
        [_short(m) for m in models], rotation=25, ha="right", fontsize=15
    )

    fig.tight_layout()
    fig.savefig(analysis_dir / "summary_passed.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "summary_passed.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_bar_plot(by_trial, models, worlds, analysis_dir, title, plt) -> None:
    n_models = len(models)
    n_worlds = len(worlds)
    # Stacked layout: explanation score on top, position error on bottom,
    # sharing one set of world tick labels across both panels.
    fig_w = max(8.0, 0.8 * n_worlds * max(1, n_models))
    fig, (ax_score, ax_err) = plt.subplots(2, 1, figsize=(fig_w, 9), sharex=True)
    width = 0.8 / max(1, n_models)
    x = np.arange(n_worlds)
    model_colors = _model_color_map(models, plt)

    for i, model in enumerate(models):
        sh, sl, su = [], [], []
        eh, el, eu = [], [], []
        for w in worlds:
            errs, scores = _trial_values(by_trial, model, w)
            s = _bootstrap_ci(scores)
            e = _bootstrap_ci_geom(errs)
            sh.append(s[0] if s else np.nan)
            sl.append(s[1] if s else 0.0)
            su.append(s[2] if s else 0.0)
            eh.append(e[0] if e else np.nan)
            el.append(e[1] if e else 0.0)
            eu.append(e[2] if e else 0.0)
        offset = (i - (n_models - 1) / 2) * width
        color = model_colors[model]
        ax_score.bar(
            x + offset,
            sh,
            width=width,
            yerr=[sl, su],
            label=_short(model),
            color=color,
            capsize=3,
        )
        ax_err.bar(
            x + offset,
            eh,
            width=width,
            yerr=[el, eu],
            color=color,
            capsize=3,
        )

    for ax, ax_title, ylabel in [
        (ax_score, "Explanation score (higher = better)", "score [0, 1] ↑"),
        (ax_err, "Mean position error (geom. mean, lower = better)", "geom. mean error ↓"),
    ]:
        ax.set_xticks(x)
        ax.set_ylabel(ylabel)
        ax.set_title(ax_title)
    # With sharex, only the bottom panel needs the world labels.
    ax_err.set_xticklabels([_world_label(w) for w in worlds], rotation=20, ha="right")
    ax_score.set_ylim(0, 1.05)
    ax_err.set_yscale("log")

    fig.suptitle(f"Benchmark: {title}", y=1.00)
    handles, labels = ax_score.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, max(1, n_models)),
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    fig.savefig(analysis_dir / "summary.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "summary.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_strip_plot(by_trial, models, worlds, analysis_dir, title, plt) -> None:
    n_models = len(models)
    n_worlds = len(worlds)
    fig_w = max(8.0, 1.3 * n_worlds * max(1, n_models))
    fig, (ax_score, ax_err) = plt.subplots(1, 2, figsize=(fig_w, 5))
    spread = 0.7 / max(1, n_models)
    x = np.arange(n_worlds)
    model_colors = _model_color_map(models, plt)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * spread
        color = model_colors[model]
        first_label_used = False
        for j, w in enumerate(worlds):
            errs, scores = _trial_values(by_trial, model, w)
            if scores:
                jitter = rng.normal(0, 0.03, size=len(scores))
                ax_score.scatter(
                    np.full(len(scores), x[j] + offset) + jitter,
                    scores,
                    color=color,
                    alpha=0.75,
                    s=30,
                    edgecolors="white",
                    linewidth=0.5,
                    label=_short(model) if not first_label_used else None,
                )
                first_label_used = True
            if errs:
                jitter = rng.normal(0, 0.03, size=len(errs))
                ax_err.scatter(
                    np.full(len(errs), x[j] + offset) + jitter,
                    errs,
                    color=color,
                    alpha=0.75,
                    s=30,
                    edgecolors="white",
                    linewidth=0.5,
                )

    for ax, ax_title, ylabel in [
        (ax_score, "Explanation score per run", "score [0, 1] ↑"),
        (ax_err, "Mean position error per run", "error ↓"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([_world_label(w) for w in worlds], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ax_title)
    ax_score.set_ylim(-0.05, 1.05)
    ax_err.set_yscale("log")

    fig.suptitle(f"Benchmark (per-run): {title}", y=1.02)
    handles, labels = ax_score.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, max(1, n_models)),
            bbox_to_anchor=(0.5, -0.05),
        )
    fig.tight_layout()
    fig.savefig(analysis_dir / "runs.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "runs.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_pareto_plot(by_trial, models, worlds, analysis_dir, title, plt) -> None:
    """One point per model: pooled (geom-mean error, mean score) across all worlds and seeds.

    Error bars are 95% bootstrap CIs.
    """
    if not by_trial or not models:
        return
    from matplotlib.lines import Line2D

    pooled = _per_model_pooled(by_trial)
    model_colors = _model_color_map(models, plt)

    fig, ax = plt.subplots(figsize=(9, 6))

    plotted_models = []
    for model in models:
        errs, scores = pooled.get(model, ([], []))
        e = _bootstrap_ci_geom(errs)
        s = _bootstrap_ci(scores)
        if e is None or s is None:
            continue
        x_mean, x_lo, x_hi = e
        y_mean, y_lo, y_hi = s
        ax.errorbar(
            x_mean, y_mean,
            xerr=[[x_lo], [x_hi]],
            yerr=[[y_lo], [y_hi]],
            fmt="o",
            color=model_colors[model],
            ecolor=model_colors[model],
            elinewidth=1.2,
            capsize=3,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
        plotted_models.append(model)

    if not plotted_models:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_xlabel("MPE ↓", fontsize=18)
    ax.set_ylabel("Evaluation score (0-1) ↑", fontsize=18)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_ylim(-0.05, 1.05)

    model_handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=10, linestyle="", label=_short(m),
        )
        for m in plotted_models
    ]
    ax.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(plotted_models), 5),
        frameon=False,
        borderaxespad=0,
    )

    fig.tight_layout()
    fig.savefig(analysis_dir / "pareto.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_pareto_expected_combo_plot(
    by_trial, by_seed, models, worlds, analysis_dir, title, plt
) -> None:
    """Two-panel plot: pareto (left) + expected_passed@k scatter (right).

    Single legend over the top maps model → color (shared across both panels).
    """
    if not by_trial or not models:
        return
    from matplotlib.lines import Line2D

    pooled = _per_model_pooled(by_trial)
    model_colors = _model_color_map(models, plt)
    n_worlds = len(worlds)

    fig, (ax_pareto, ax_ek) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: pareto (geom-mean MPE, mean explanation score) per model.
    plotted_models = []
    for model in models:
        errs, scores = pooled.get(model, ([], []))
        e = _bootstrap_ci_geom(errs)
        s = _bootstrap_ci(scores)
        if e is None or s is None:
            continue
        x_mean, x_lo, x_hi = e
        y_mean, y_lo, y_hi = s
        ax_pareto.errorbar(
            x_mean, y_mean,
            xerr=[[x_lo], [x_hi]],
            yerr=[[y_lo], [y_hi]],
            fmt="o",
            color=model_colors[model],
            ecolor=model_colors[model],
            elinewidth=1.2,
            capsize=3,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
        plotted_models.append(model)

    if not plotted_models:
        plt.close(fig)
        return

    ax_pareto.set_xscale("log")
    ax_pareto.set_xlabel("MPE ↓", fontsize=18)
    ax_pareto.set_ylabel("Evaluation score (0-1) ↑", fontsize=18)
    ax_pareto.tick_params(axis="both", labelsize=13)
    ax_pareto.set_ylim(-0.05, 1.05)

    # Right: expected worlds passed @k vs k, ±SEM error bars.
    expected_at_k = {
        k: _expected_worlds_passed_at_k(by_seed, models, worlds, k)
        for k in _PASS_K_VALUES
    }
    xs = list(_PASS_K_VALUES)
    for model in models:
        means = [expected_at_k[k].get(model, (0.0, 0.0))[0] for k in xs]
        sems = [expected_at_k[k].get(model, (0.0, 0.0))[1] for k in xs]
        ax_ek.errorbar(
            xs, means, yerr=sems,
            marker="o", color=model_colors[model],
            ecolor=model_colors[model],
            linewidth=1.8, markersize=10,
            markeredgecolor="white", markeredgewidth=0.7,
            capsize=3, elinewidth=1.0,
        )

    ax_ek.set_xticks(xs)
    ax_ek.set_xlabel("k", fontsize=18)
    ax_ek.set_ylabel(f"expected passed @k (out of {n_worlds}) ↑", fontsize=18)
    ax_ek.set_ylim(-0.5, n_worlds + 0.5)
    ax_ek.tick_params(axis="both", labelsize=13)
    ax_ek.grid(axis="y", linestyle=":", alpha=0.4)

    handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=10, linestyle="", label=_short(m),
        )
        for m in plotted_models
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        borderaxespad=0,
        fontsize=16,
    )

    fig.tight_layout()
    fig.savefig(
        analysis_dir / "pareto_and_expected_passed_at_k.png",
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        analysis_dir / "pareto_and_expected_passed_at_k.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _make_expected_passed_vs_release_date_plot(
    by_seed, models, worlds, analysis_dir, title, plt, k: int = 3
) -> None:
    """Scatter: expected worlds passed @k vs model release date.

    One point per model with ±std error bars (std across MC draws, reflecting
    draw-to-draw spread). Models without a known release date in
    `_MODEL_RELEASE_DATES` are skipped. Models are identified via a top
    legend that reuses the canonical color per model.
    """
    if not by_seed or not models or not worlds:
        return
    import datetime as _dt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    n_worlds = len(worlds)
    model_colors = _model_color_map(models, plt)

    # Compute (mean, std) of worlds_passed@k across MC draws — std reflects the
    # draw-to-draw spread (SEM at n_samples=1000 is sub-pixel and not useful here).
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = np.stack(
        [rng.choice(_SEED_POOL_SIZE, size=k, replace=False)
         for _ in range(_EXPECTED_PASSED_SAMPLES)]
    )
    mean_std_by_model: dict[str, tuple[float, float]] = {}
    for m in models:
        per_sample_counts = np.zeros(_EXPECTED_PASSED_SAMPLES, dtype=float)
        for w in worlds:
            seed_results = by_seed.get((m, w), {})
            passes = np.array([
                _trial_passes(*seed_results.get(s, (None, None)))
                for s in range(_SEED_POOL_SIZE)
            ])
            if not passes.any():
                continue
            per_sample_counts += passes[samples].any(axis=1).astype(float)
        mean = float(per_sample_counts.mean())
        std = (
            float(per_sample_counts.std(ddof=1))
            if _EXPECTED_PASSED_SAMPLES > 1 else 0.0
        )
        mean_std_by_model[m] = (mean, std)

    points = []
    for m in models:
        date_str = _MODEL_RELEASE_DATES.get(m)
        if date_str is None:
            continue
        try:
            d = _dt.date.fromisoformat(date_str)
        except ValueError:
            continue
        mean, std = mean_std_by_model.get(m, (0.0, 0.0))
        points.append((d, mean, std, m))
    if not points:
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted_models = []
    for d, mean, std, m in points:
        ax.errorbar(
            d, mean, yerr=std,
            marker="o", color=model_colors[m], ecolor=model_colors[m],
            markersize=11, capsize=4, elinewidth=1.4,
            markeredgecolor="white", markeredgewidth=0.7,
            linestyle="",
        )
        plotted_models.append(m)

    ax.set_xlabel("Model release date", fontsize=18)
    ax.set_ylabel(f"expected passed @k={k} (out of {n_worlds}) ↑", fontsize=18)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(-0.5, n_worlds + 0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=11, linestyle="", label=_short(m),
        )
        for m in sorted(plotted_models, key=_model_sort_key)
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(handles), 5),
        frameon=False,
        borderaxespad=0,
    )

    fig.tight_layout()
    fig.savefig(
        analysis_dir / f"expected_passed_at_k{k}_vs_release_date.png",
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        analysis_dir / f"expected_passed_at_k{k}_vs_release_date.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _make_pareto_expected_release_combo_plot(
    by_trial, by_seed, models, worlds, analysis_dir, title, plt, k: int = 3
) -> None:
    """Three-panel combo: pareto | expected_passed@k vs k | expected@k vs release date.

    Single legend above the whole figure maps model → color (shared across all
    three panels). Right panel uses ±std error bars from MC draws (not SEM).
    """
    if not by_trial or not models:
        return
    import datetime as _dt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    pooled = _per_model_pooled(by_trial)
    model_colors = _model_color_map(models, plt)
    n_worlds = len(worlds)

    fig, (ax_pareto, ax_ek, ax_date) = plt.subplots(1, 3, figsize=(22, 6.5))

    # Panel 1: pareto (geom-mean MPE, mean explanation score) per model.
    plotted_models: list[str] = []
    for model in models:
        errs, scores = pooled.get(model, ([], []))
        e = _bootstrap_ci_geom(errs)
        s = _bootstrap_ci(scores)
        if e is None or s is None:
            continue
        x_mean, x_lo, x_hi = e
        y_mean, y_lo, y_hi = s
        ax_pareto.errorbar(
            x_mean, y_mean,
            xerr=[[x_lo], [x_hi]],
            yerr=[[y_lo], [y_hi]],
            fmt="o",
            color=model_colors[model],
            ecolor=model_colors[model],
            elinewidth=1.2,
            capsize=3,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
        plotted_models.append(model)

    if not plotted_models:
        plt.close(fig)
        return

    ax_pareto.set_xscale("log")
    ax_pareto.set_xlabel("MPE ↓", fontsize=20)
    ax_pareto.set_ylabel("Evaluation score (0-1) ↑", fontsize=20)
    ax_pareto.tick_params(axis="both", labelsize=13)
    ax_pareto.set_ylim(-0.05, 1.05)

    # Panel 2: expected worlds passed @k vs k, ±SEM error bars.
    expected_at_k = {
        kk: _expected_worlds_passed_at_k(by_seed, models, worlds, kk)
        for kk in _PASS_K_VALUES
    }
    xs = list(_PASS_K_VALUES)
    for model in models:
        means = [expected_at_k[kk].get(model, (0.0, 0.0))[0] for kk in xs]
        sems = [expected_at_k[kk].get(model, (0.0, 0.0))[1] for kk in xs]
        ax_ek.errorbar(
            xs, means, yerr=sems,
            marker="o", color=model_colors[model],
            ecolor=model_colors[model],
            linewidth=1.8, markersize=10,
            markeredgecolor="white", markeredgewidth=0.7,
            capsize=3, elinewidth=1.0,
        )

    ax_ek.set_xticks(xs)
    ax_ek.set_xlabel("k", fontsize=20)
    ax_ek.set_ylabel(f"Expected passed @k (out of {n_worlds}) ↑", fontsize=20)
    ax_ek.set_ylim(-0.5, n_worlds + 0.5)
    ax_ek.tick_params(axis="both", labelsize=13)
    ax_ek.grid(axis="y", linestyle=":", alpha=0.4)

    # Panel 3: expected worlds passed @k vs release date, ±std error bars.
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = np.stack(
        [rng.choice(_SEED_POOL_SIZE, size=k, replace=False)
         for _ in range(_EXPECTED_PASSED_SAMPLES)]
    )
    mean_std_by_model: dict[str, tuple[float, float]] = {}
    for m in models:
        per_sample_counts = np.zeros(_EXPECTED_PASSED_SAMPLES, dtype=float)
        for w in worlds:
            seed_results = by_seed.get((m, w), {})
            passes = np.array([
                _trial_passes(*seed_results.get(s, (None, None)))
                for s in range(_SEED_POOL_SIZE)
            ])
            if not passes.any():
                continue
            per_sample_counts += passes[samples].any(axis=1).astype(float)
        mean = float(per_sample_counts.mean())
        std = (
            float(per_sample_counts.std(ddof=1))
            if _EXPECTED_PASSED_SAMPLES > 1 else 0.0
        )
        mean_std_by_model[m] = (mean, std)

    for m in models:
        date_str = _MODEL_RELEASE_DATES.get(m)
        if date_str is None:
            continue
        try:
            d = _dt.date.fromisoformat(date_str)
        except ValueError:
            continue
        mean, std = mean_std_by_model.get(m, (0.0, 0.0))
        ax_date.errorbar(
            d, mean, yerr=std,
            marker="o", color=model_colors[m], ecolor=model_colors[m],
            markersize=11, capsize=4, elinewidth=1.4,
            markeredgecolor="white", markeredgewidth=0.7,
            linestyle="",
        )

    ax_date.set_xlabel("Model release date", fontsize=20)
    ax_date.set_ylabel(f"Expected passed @k={k} (out of {n_worlds}) ↑", fontsize=20)
    ax_date.tick_params(axis="both", labelsize=13)
    ax_date.set_ylim(-0.5, n_worlds + 0.5)
    ax_date.grid(axis="y", linestyle=":", alpha=0.4)
    ax_date.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_date.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax_date.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=11, linestyle="", label=_short(m),
        )
        for m in sorted(plotted_models, key=_model_sort_key)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(handles), 6),
        frameon=False,
        borderaxespad=0,
        fontsize=19,
    )

    fig.tight_layout()
    fig.savefig(
        analysis_dir / "pareto_expected_release_combo.png",
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        analysis_dir / "pareto_expected_release_combo.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _make_noise_ablation_plot(by_noise, analysis_dir, title) -> None:
    """For each (model, world), score & MPE vs σ — one line per (model, world).

    Skipped unless ≥2 distinct noise levels were found across the run.
    """
    if not by_noise:
        return
    noise_levels = sorted({sigma for _m, _w, sigma in by_noise.keys()})
    if len(noise_levels) < 2:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })

    models = sorted({m for m, _w, _s in by_noise.keys()}, key=_model_sort_key)
    worlds = sorted({w for _m, w, _s in by_noise.keys()})
    model_colors = _model_color_map(models, plt)
    for m in list(model_colors):
        norm = m.lower().replace(".", "-")
        if "opus-4-7" in norm:
            model_colors[m] = "dimgray"
        elif "gpt-5-5" in norm:
            model_colors[m] = "darkorange"

    fig, (ax_pass, ax_score, ax_err) = plt.subplots(1, 3, figsize=(17, 5))

    for model in models:
        for world in worlds:
            xs = []
            pass_counts = []
            score_means, score_lo, score_hi = [], [], []
            err_means, err_lo, err_hi = [], [], []
            for sigma in noise_levels:
                values = by_noise.get((model, world, sigma), [])
                scores = [
                    float(s) for _e, s in values
                    if isinstance(s, (int, float)) and math.isfinite(s)
                ]
                errs = [
                    float(e) for e, _s in values
                    if isinstance(e, (int, float)) and math.isfinite(e)
                ]
                s_ci = _bootstrap_ci(scores)
                e_ci = _bootstrap_ci_geom(errs)
                if s_ci is None and e_ci is None:
                    continue
                xs.append(sigma)
                pass_counts.append(
                    sum(1 for err, score in values if _trial_passes(err, score))
                )
                if s_ci is not None:
                    score_means.append(s_ci[0])
                    score_lo.append(s_ci[1])
                    score_hi.append(s_ci[2])
                else:
                    score_means.append(np.nan)
                    score_lo.append(0.0)
                    score_hi.append(0.0)
                if e_ci is not None:
                    err_means.append(e_ci[0])
                    err_lo.append(e_ci[1])
                    err_hi.append(e_ci[2])
                else:
                    err_means.append(np.nan)
                    err_lo.append(0.0)
                    err_hi.append(0.0)
            if not xs:
                continue
            label = (
                _short(model) if len(worlds) == 1
                else f"{_short(model)} / {_world_label(world)}"
            )
            color = model_colors[model]
            ax_pass.plot(
                xs, pass_counts,
                marker="o", color=color,
                linewidth=1.8, markersize=8,
                markeredgecolor="white", markeredgewidth=0.6,
                label=label,
            )
            ax_score.errorbar(
                xs, score_means, yerr=[score_lo, score_hi],
                marker="o", color=color, ecolor=color,
                capsize=3, linewidth=1.6, markersize=7,
                markeredgecolor="white", markeredgewidth=0.6,
            )
            ax_err.errorbar(
                xs, err_means, yerr=[err_lo, err_hi],
                marker="o", color=color, ecolor=color,
                capsize=3, linewidth=1.6, markersize=7,
                markeredgecolor="white", markeredgewidth=0.6,
            )

    ax_pass.set_xlabel(r"observation noise $\sigma$", fontsize=20)
    ax_pass.set_ylabel("worlds passed ↑", fontsize=20)
    ax_pass.tick_params(axis="both", labelsize=14)

    ax_score.set_xlabel(r"observation noise $\sigma$", fontsize=20)
    ax_score.set_ylabel("Evaluation score (0-1) ↑", fontsize=20)
    ax_score.set_ylim(-0.05, 1.05)
    ax_score.tick_params(axis="both", labelsize=14)

    ax_err.set_xlabel(r"observation noise $\sigma$", fontsize=20)
    ax_err.set_ylabel("MPE (geom. mean) ↓", fontsize=20)
    ax_err.set_yscale("log")
    ax_err.tick_params(axis="both", labelsize=14)

    handles, labels = ax_pass.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize=18,
        )

    fig.tight_layout()
    fig.savefig(analysis_dir / "noise_ablation.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "noise_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def _world_score_pools(by_trial, worlds):
    """Pool errors and scores across models per world."""
    errs_per_world: dict[str, list[float]] = {w: [] for w in worlds}
    scores_per_world: dict[str, list[float]] = {w: [] for w in worlds}
    for (_model, world), values in by_trial.items():
        if world not in errs_per_world:
            continue
        for err, score in values:
            if isinstance(err, (int, float)) and math.isfinite(err) and err >= _GEOM_MIN:
                errs_per_world[world].append(float(err))
            if isinstance(score, (int, float)) and math.isfinite(score):
                scores_per_world[world].append(float(score))
    return errs_per_world, scores_per_world


def _world_order_by_score(by_trial, worlds) -> list[str]:
    """Worlds ordered by descending median explanation score (easy → hard)."""
    _errs, scores_per_world = _world_score_pools(by_trial, worlds)

    def _median_score(w: str) -> float:
        vals = scores_per_world[w]
        return float(np.median(vals)) if vals else float("-inf")

    return sorted(worlds, key=_median_score, reverse=True)


def _make_per_world_passed_plot(
    by_trial, models, worlds, analysis_dir, title, plt
) -> None:
    """Grouped bars: per-world pass rate (%) with one bar per model."""
    if not by_trial or not models or not worlds:
        return
    from matplotlib.lines import Line2D

    ordered = _world_order_by_score(by_trial, worlds)
    n_models = len(models)
    n_worlds = len(ordered)
    model_colors = _model_color_map(models, plt)

    fig_w = max(10.0, 0.9 * n_worlds * max(1, n_models) * 0.35 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    width = 0.8 / max(1, n_models)
    x = np.arange(n_worlds)

    for i, model in enumerate(models):
        pcts = []
        for w in ordered:
            values = by_trial.get((model, w), [])
            n = len(values)
            if n == 0:
                pcts.append(np.nan)
                continue
            passed = sum(1 for err, score in values if _trial_passes(err, score))
            pcts.append(100.0 * passed / n)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(
            x + offset,
            pcts,
            width=width,
            color=model_colors[model],
            label=_short(model),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_world_label(w) for w in ordered], rotation=20, ha="right", fontsize=14
    )
    ax.set_ylabel("worlds passed (%) ↑", fontsize=16)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 105)

    handles = [
        Line2D(
            [0], [0],
            marker="s", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=10, linestyle="", label=_short(m),
        )
        for m in models
    ]
    ax.legend(
        handles=handles, title="model",
        loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
    )

    fig.tight_layout()
    fig.savefig(analysis_dir / "per_world_passed.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "per_world_passed.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_expected_pass_rate_per_world_plot(
    by_seed, by_trial, models, worlds, analysis_dir, title, plt, k: int = 3
) -> None:
    """Grouped bars: per-world expected pass rate @k, one bar per model.

    Each bar's height is the (model, world) expected probability of passing
    the world when k seeds are sampled uniformly without replacement from a
    `_SEED_POOL_SIZE`-seed pool. Worlds sorted by difficulty (median
    explanation score, easy → hard). Error bars are ±SEM from the MC draws.
    """
    if not by_seed or not models or not worlds:
        return
    from matplotlib.lines import Line2D

    ordered = _world_order_by_score(by_trial, worlds)
    n_models = len(models)
    n_worlds = len(ordered)
    model_colors = _model_color_map(models, plt)
    rates = _expected_pass_rate_per_world_at_k(by_seed, models, ordered, k)

    fig_w = max(10.0, 0.9 * n_worlds * max(1, n_models) * 0.35 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    width = 0.8 / max(1, n_models)
    x = np.arange(n_worlds)

    for i, model in enumerate(models):
        means = [rates.get((model, w), (0.0, 0.0))[0] for w in ordered]
        sems = [rates.get((model, w), (0.0, 0.0))[1] for w in ordered]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=sems,
            color=model_colors[model],
            label=_short(model),
            capsize=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_world_label(w) for w in ordered], rotation=20, ha="right", fontsize=14
    )
    ax.set_ylabel(f"expected pass rate @k={k} ↑", fontsize=16)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 1.05)

    handles = [
        Line2D(
            [0], [0],
            marker="s", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=10, linestyle="", label=_short(m),
        )
        for m in models
    ]
    ax.legend(
        handles=handles, title="model",
        loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
    )

    fig.tight_layout()
    fig.savefig(
        analysis_dir / f"expected_pass_rate_per_world_at_k{k}.png",
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        analysis_dir / f"expected_pass_rate_per_world_at_k{k}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _make_world_difficulty_plot(by_trial, worlds, analysis_dir, title, plt) -> None:
    """Pool trials across models per world; violin of error and score, sorted by median score."""
    if not by_trial or not worlds:
        return

    errs_per_world, scores_per_world = _world_score_pools(by_trial, worlds)
    ordered = _world_order_by_score(by_trial, worlds)
    included = [w for w in ordered if errs_per_world[w] or scores_per_world[w]]
    if not included:
        return

    err_data = [np.log10(np.asarray(errs_per_world[w])) for w in included if errs_per_world[w]]
    err_labels = [w for w in included if errs_per_world[w]]
    score_data = [scores_per_world[w] for w in included if scores_per_world[w]]
    score_labels = [w for w in included if scores_per_world[w]]

    fig_w = max(12.0, 0.9 * len(included) * 2)
    fig, (ax_err, ax_score) = plt.subplots(1, 2, figsize=(fig_w, 6))

    if err_data:
        ax_err.violinplot(err_data, showmedians=True, showextrema=False)
        ax_err.set_xticks(range(1, len(err_labels) + 1))
        ax_err.set_xticklabels(
            [_world_label(w) for w in err_labels], rotation=20, ha="right"
        )
        ymin, ymax = ax_err.get_ylim()
        lo_t = int(math.floor(ymin))
        hi_t = int(math.ceil(ymax))
        ticks = list(range(lo_t, hi_t + 1))
        ax_err.set_yticks(ticks)
        ax_err.set_yticklabels([f"$10^{{{t}}}$" for t in ticks])
    ax_err.set_ylabel("mean_pos_error (log scale) ↓")
    ax_err.set_title("Per-world error (pooled across models)")

    if score_data:
        ax_score.violinplot(score_data, showmedians=True, showextrema=False)
        ax_score.set_xticks(range(1, len(score_labels) + 1))
        ax_score.set_xticklabels(
            [_world_label(w) for w in score_labels], rotation=20, ha="right"
        )
    ax_score.set_ylabel("explanation score ↑")
    ax_score.set_title("Per-world score (pooled across models)")
    ax_score.set_ylim(-0.05, 1.05)

    fig.suptitle(
        f"World difficulty ranking (sorted by median explanation score): {title}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(analysis_dir / "world_difficulty.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "world_difficulty.pdf", bbox_inches="tight")
    plt.close(fig)

    if score_data:
        fig2, ax2 = plt.subplots(figsize=(max(8.0, 0.9 * len(score_labels)), 6))
        parts = ax2.violinplot(score_data, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("black")
            body.set_edgecolor("black")
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("black")
        ax2.set_xticks(range(1, len(score_labels) + 1))
        ax2.set_xticklabels(
            [_world_label(w) for w in score_labels],
            rotation=20,
            ha="right",
            fontsize=16,
        )
        ax2.set_ylabel("explanation score ↑", fontsize=18)
        ax2.tick_params(axis="y", labelsize=14)
        ax2.set_ylim(-0.05, 1.05)
        fig2.tight_layout()
        fig2.savefig(
            analysis_dir / "world_difficulty_score.png", dpi=150, bbox_inches="tight"
        )
        fig2.savefig(analysis_dir / "world_difficulty_score.pdf", bbox_inches="tight")
        plt.close(fig2)


def _fmt_mean_bootstrap(values: list[float]) -> str:
    """Mean and 95% bootstrap CI of the mean (`_BOOTSTRAP_RESAMPLES` resamples)."""
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(_BOOTSTRAP_RESAMPLES, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return f"{float(arr.mean()):.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_geom_mean_bootstrap(values: list[float]) -> str:
    """Geom. mean and 95% bootstrap CI of the geom. mean. Drops values < _GEOM_MIN."""
    filtered = [float(v) for v in values if v >= _GEOM_MIN]
    if not filtered:
        return "n/a"
    if len(filtered) == 1:
        return f"{filtered[0]:.3f}"
    arr = np.asarray(filtered, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(_BOOTSTRAP_RESAMPLES, arr.size))
    boot = np.exp(np.log(arr[idx]).mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    m = float(np.exp(np.log(arr).mean()))
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        help="Path to a results directory (e.g. results/yml_bench/production_r2)",
    )
    parser.add_argument(
        "--out",
        default="analysis",
        help="Subdirectory name (under results_dir) for analysis output (default: analysis)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"error: {results_dir} is not a directory", file=sys.stderr)
        return 1

    analysis_dir = results_dir / args.out
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_path = aggregate(results_dir, analysis_dir)
    print(f"summary written to {summary_path}")
    print()
    print(summary_path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
