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
import sys
from pathlib import Path

import numpy as np

_BOOTSTRAP_RESAMPLES = 5000
_BOOTSTRAP_SEED = 0  # fixed so summary.txt is reproducible across re-aggregation

# Errors aggregate as geometric means; values below this are dropped (log undefined).
_GEOM_MIN = 1e-14
# A (model, world) pair "passes" when these are both met across its seeds.
_PASS_ERR_THRESHOLD = 0.5     # geom. mean of mean_pos_error must be < this
_PASS_SCORE_THRESHOLD = 0.5   # arithmetic mean of explanation score must be >= this


def aggregate(results_dir: Path, analysis_dir: Path) -> Path:
    """Walk results_dir for per-trial JSONs and write summary.txt into analysis_dir."""
    by_trial: dict[tuple[str, str], list[tuple[float | None, float | None]]] = {}

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
    _make_plots(by_trial, analysis_dir, title)
    _write_per_model_summary(by_trial, analysis_dir, title)
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


def _write_per_model_summary(by_trial, analysis_dir: Path, title: str) -> Path:
    pooled = _per_model_pooled(by_trial)
    models = sorted({m for m, _ in by_trial.keys()}, key=_model_sort_key)
    worlds = sorted({w for _, w in by_trial.keys()})
    passed = _passed_per_model(by_trial, models, worlds)
    n_trials = _trials_per_model(by_trial, models, worlds)
    lines = []
    lines.append(f"Per-model summary  ({title})")
    lines.append("=" * 124)
    header = (
        f"{'model':<50} {'n_trials':>9} {'passed':>10} "
        f"{'expl_score [95% CI]':>22} {'geom_pos_err [95% CI]':>26}"
    )
    lines.append(header)
    lines.append("-" * 124)
    for model in sorted(pooled, key=_model_sort_key):
        errs, scores = pooled[model]
        n = max(len(errs), len(scores))
        passed_str = f"{passed.get(model, 0)}/{n_trials.get(model, n)}"
        lines.append(
            f"{model:<50} {n:>9} {passed_str:>10} "
            f"{_fmt_mean_bootstrap(scores):>22} {_fmt_geom_mean_bootstrap(errs):>26}"
        )
    lines.append("-" * 124)
    lines.append("Pooled across all worlds and seeds (every trial counts equally).")
    lines.append(
        f"passed: number of trials (summed over worlds & seeds) with mean_pos_error < "
        f"{_PASS_ERR_THRESHOLD} AND explanation_score >= {_PASS_SCORE_THRESHOLD}."
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


def _trials_per_model(by_trial, models, worlds) -> dict[str, int]:
    """Number of (world, seed) trials run for each model — denominator for passes."""
    return {
        m: sum(len(by_trial.get((m, w), [])) for w in worlds)
        for m in models
    }


def _make_plots(by_trial, analysis_dir: Path, title: str) -> None:
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
    _make_passed_plot(by_trial, models, worlds, analysis_dir, title, plt)
    _make_pareto_plot(by_trial, models, worlds, analysis_dir, title, plt)
    _make_world_difficulty_plot(by_trial, worlds, analysis_dir, title, plt)


def _make_per_model_plot(by_trial, models, analysis_dir, title, plt) -> None:
    """One bar per model in two side-by-side panels: expl. score and MSE."""
    pooled = _per_model_pooled(by_trial)
    n_models = len(models)
    if n_models == 0:
        return
    fig_w = max(8.0, 1.2 * n_models * 2)
    fig, (ax_score, ax_err) = plt.subplots(1, 2, figsize=(fig_w, 5))
    x = np.arange(n_models)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_models)]

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
            "score [0, 1]",
        ),
        (
            ax_err,
            "Mean position error (geom. mean, pooled across worlds)",
            "geom. mean error",
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


def _make_passed_plot(by_trial, models, worlds, analysis_dir, title, plt) -> None:
    """3-row plot: worlds passed, explanation score, MSE — model on x-axis."""
    n_models = len(models)
    n_worlds = len(worlds)
    if n_models == 0:
        return

    pooled = _per_model_pooled(by_trial)
    passed = _passed_per_model(by_trial, models, worlds)

    fig_w = max(8.0, 1.3 * n_models)
    fig, (ax_pass, ax_score, ax_err) = plt.subplots(
        3, 1, figsize=(fig_w, 11), sharex=True
    )
    x = np.arange(n_models)
    cmap = plt.get_cmap("Dark2")
    colors = [cmap(i % cmap.N) for i in range(n_models)]
    ylabel_kw = {"fontsize": 18}

    # Row 1: trials passed (summed over all worlds and seeds).
    pass_counts = [passed.get(m, 0) for m in models]
    ax_pass.bar(x, pass_counts, color=colors)
    ax_pass.set_ylabel("worlds passed", **ylabel_kw)

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
    ax_score.set_ylabel("Explanation score (0-1)", **ylabel_kw)
    ax_score.set_ylim(0, 1.05)

    ax_err.bar(x, err_means, yerr=[err_lo, err_hi], color=colors, capsize=3)
    ax_err.set_ylabel("MSE (geom. mean)", **ylabel_kw)
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
    cmap = plt.get_cmap("tab10")

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
        color = cmap(i % 10)
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
        (ax_score, "Explanation score (higher = better)", "score [0, 1]"),
        (ax_err, "Mean position error (geom. mean, lower = better)", "geom. mean error"),
    ]:
        ax.set_xticks(x)
        ax.set_ylabel(ylabel)
        ax.set_title(ax_title)
    # With sharex, only the bottom panel needs the world labels.
    ax_err.set_xticklabels(worlds, rotation=20, ha="right")
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
    cmap = plt.get_cmap("tab10")
    rng = np.random.default_rng(_BOOTSTRAP_SEED)

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * spread
        color = cmap(i % 10)
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
        (ax_score, "Explanation score per run", "score [0, 1]"),
        (ax_err, "Mean position error per run", "error"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(worlds, rotation=20, ha="right")
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
    cmap = plt.get_cmap("tab10")
    model_colors = {m: cmap(i % 10) for i, m in enumerate(models)}

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
    ax.set_xlabel("mean_pos_error (geom. mean across worlds, lower = better)")
    ax.set_ylabel("explanation score (mean across worlds, higher = better)")
    ax.set_title(f"Score vs error per model: {title}")
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
        handles=model_handles, title="model",
        loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
    )

    fig.tight_layout()
    fig.savefig(analysis_dir / "pareto.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_world_difficulty_plot(by_trial, worlds, analysis_dir, title, plt) -> None:
    """Pool trials across models per world; violin of error and score, sorted by median error."""
    if not by_trial or not worlds:
        return

    errs_per_world: dict[str, list[float]] = {w: [] for w in worlds}
    scores_per_world: dict[str, list[float]] = {w: [] for w in worlds}
    for (_model, world), values in by_trial.items():
        for err, score in values:
            if isinstance(err, (int, float)) and math.isfinite(err) and err >= _GEOM_MIN:
                errs_per_world[world].append(float(err))
            if isinstance(score, (int, float)) and math.isfinite(score):
                scores_per_world[world].append(float(score))

    # Single ordering shared across both panels: ascending median error (easy → hard).
    # Worlds with no error data sink to the bottom.
    def _median_err(w: str) -> float:
        vals = errs_per_world[w]
        return float(np.median(vals)) if vals else float("inf")

    ordered = sorted(worlds, key=_median_err)
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
        ax_err.set_xticklabels(err_labels, rotation=20, ha="right")
        ymin, ymax = ax_err.get_ylim()
        lo_t = int(math.floor(ymin))
        hi_t = int(math.ceil(ymax))
        ticks = list(range(lo_t, hi_t + 1))
        ax_err.set_yticks(ticks)
        ax_err.set_yticklabels([f"$10^{{{t}}}$" for t in ticks])
    ax_err.set_ylabel("mean_pos_error (log scale)")
    ax_err.set_title("Per-world error (pooled across models)")

    if score_data:
        ax_score.violinplot(score_data, showmedians=True, showextrema=False)
        ax_score.set_xticks(range(1, len(score_labels) + 1))
        ax_score.set_xticklabels(score_labels, rotation=20, ha="right")
    ax_score.set_ylabel("explanation score")
    ax_score.set_title("Per-world score (pooled across models)")
    ax_score.set_ylim(-0.05, 1.05)

    fig.suptitle(
        f"World difficulty ranking (sorted by median error): {title}", y=1.02
    )
    fig.tight_layout()
    fig.savefig(analysis_dir / "world_difficulty.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "world_difficulty.pdf", bbox_inches="tight")
    plt.close(fig)


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
