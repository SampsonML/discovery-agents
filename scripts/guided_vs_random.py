#!/usr/bin/env python
"""
Guided-vs-random comparison plots from two parallel results directories.

Walks the per-trial JSONs in `<guided_dir>` and `<random_dir>` (same layout
as `scripts/yml_benchmark.py` produces) and writes comparison plots and a
summary table into `<guided_dir>/analysis/` (or `--out`).

Scatter plots distinguish condition by marker: circle = guided, x = random.
Bar plots place guided and random side-by-side; color is keyed to the model.

Usage:
    python scripts/guided_vs_random.py results/yml_bench/production_r2 \\
                                       results/yml_bench/production_r2_random
    python scripts/guided_vs_random.py guided_dir random_dir --out my_compare
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_SEED_RE = re.compile(r"_seed(\d+)\.json$")

from production_analysis import (
    _BOOTSTRAP_SEED,
    _GEOM_MIN,
    _PASS_ERR_THRESHOLD,
    _PASS_SCORE_THRESHOLD,
    _bootstrap_ci,
    _bootstrap_ci_geom,
    _fmt_geom_mean_bootstrap,
    _fmt_mean_bootstrap,
    _is_plot_excluded,
    _model_color_map,
    _model_sort_key,
    _per_model_pooled,
    _short,
    _trial_passes,
    _trial_values,
    _world_label,
    _world_order_by_score,
)


_GUIDED = "guided"
_RANDOM = "random"
_CONDITIONS = (_GUIDED, _RANDOM)
_MARKERS = {_GUIDED: "o", _RANDOM: "X"}


def _walk(results_dir: Path):
    """Walk per-trial JSONs and return {(model, world): [(err, score), ...]}."""
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
    return by_trial


def _walk_seeded(results_dir: Path):
    """Like `_walk`, but also returns a seed-indexed view.

    Returns (by_trial, by_seed) where
      by_trial: {(model, world): [(err, score), ...]}
      by_seed:  {(model, world): {seed_idx: (err, score)}}
    Seed indices come from the filename suffix `_seed<N>.json`.
    """
    by_trial: dict[tuple[str, str], list[tuple[float | None, float | None]]] = {}
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
        m_seed = _SEED_RE.search(json_path.name)
        if m_seed is not None:
            by_seed.setdefault((model, world), {})[int(m_seed.group(1))] = (
                mpe, score
            )
    return by_trial, by_seed


def _passed_count(by_trial) -> dict[str, int]:
    counts: dict[str, int] = {}
    for (m, _w), values in by_trial.items():
        c = counts.setdefault(m, 0)
        for err, score in values:
            if _trial_passes(err, score):
                counts[m] = c + 1
                c = counts[m]
    return counts


def aggregate(guided_dir: Path, random_dir: Path, analysis_dir: Path) -> Path:
    by_cond = {_GUIDED: _walk(guided_dir), _RANDOM: _walk(random_dir)}

    title = f"{guided_dir.name} vs {random_dir.name}"
    summary_path = analysis_dir / "summary.txt"
    summary_path.write_text(_format_summary(by_cond, title))
    _make_plots(by_cond, analysis_dir, title)
    return summary_path


def _format_summary(by_cond, title: str) -> str:
    lines = []
    lines.append(f"Guided vs random comparison  ({title})")
    lines.append("=" * 132)
    header = (
        f"{'model':<50} {'world':<15} {'cond':<7} {'n':>3} "
        f"{'expl_score [95% CI]':>22} {'geom_pos_err [95% CI]':>22}"
    )
    lines.append(header)
    lines.append("-" * 132)

    keys = sorted(
        {key for cond in _CONDITIONS for key in by_cond[cond].keys()},
        key=lambda mw: (_model_sort_key(mw[0]), mw[1]),
    )
    for model, world in keys:
        for cond in _CONDITIONS:
            values = by_cond[cond].get((model, world), [])
            if not values:
                continue
            scores = [
                float(s) for _, s in values
                if isinstance(s, (int, float)) and math.isfinite(s)
            ]
            errs = [
                float(e) for e, _ in values
                if isinstance(e, (int, float)) and math.isfinite(e)
            ]
            n = len(values)
            score_str = _fmt_mean_bootstrap(scores)
            err_str = _fmt_geom_mean_bootstrap(errs)
            lines.append(
                f"{model:<50} {world:<15} {cond:<7} {n:>3} "
                f"{score_str:>22} {err_str:>22}"
            )

    lines.append("-" * 132)
    lines.append(
        "expl_score: mean explanation judge score; "
        f"geom_pos_err drops values < {_GEOM_MIN:.0e}; "
        f"format = point [2.5%, 97.5%], bootstrap seed={_BOOTSTRAP_SEED}."
    )
    lines.append(
        f"passed: trial passes iff mean_pos_error < {_PASS_ERR_THRESHOLD} "
        f"AND explanation_score >= {_PASS_SCORE_THRESHOLD}."
    )
    return "\n".join(lines) + "\n"


def _make_plots(by_cond, analysis_dir: Path, title: str) -> None:
    if not any(by_cond[c] for c in _CONDITIONS):
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

    by_cond_filtered = {
        c: {k: v for k, v in by_cond[c].items() if not _is_plot_excluded(k[0])}
        for c in _CONDITIONS
    }
    if not any(by_cond_filtered[c] for c in _CONDITIONS):
        return

    models = sorted(
        {m for c in _CONDITIONS for (m, _w) in by_cond_filtered[c].keys()},
        key=_model_sort_key,
    )
    worlds = sorted(
        {w for c in _CONDITIONS for (_m, w) in by_cond_filtered[c].keys()}
    )

    _make_strip_plot(by_cond_filtered, models, worlds, analysis_dir, title, plt)
    _make_pareto_plot(by_cond_filtered, models, analysis_dir, title, plt)
    _make_per_model_plot(by_cond_filtered, models, analysis_dir, title, plt)
    _make_passed_plot(by_cond_filtered, models, analysis_dir, title, plt)
    _make_per_world_passed_plot(
        by_cond_filtered, models, worlds, analysis_dir, title, plt
    )


def _make_strip_plot(by_cond, models, worlds, analysis_dir, title, plt) -> None:
    """Per-trial scatter, marker = condition, color = model, x-axis = worlds."""
    n_models = len(models)
    n_worlds = len(worlds)
    if n_models == 0 or n_worlds == 0:
        return
    from matplotlib.lines import Line2D

    fig_w = max(10.0, 1.3 * n_worlds * max(1, n_models))
    fig, (ax_score, ax_err) = plt.subplots(1, 2, figsize=(fig_w, 5))
    spread = 0.7 / max(1, n_models)
    x = np.arange(n_worlds)
    model_colors = _model_color_map(models, plt)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * spread
        color = model_colors[model]
        for cond in _CONDITIONS:
            marker = _MARKERS[cond]
            for j, w in enumerate(worlds):
                errs, scores = _trial_values(by_cond[cond], model, w)
                if scores:
                    jitter = rng.normal(0, 0.03, size=len(scores))
                    ax_score.scatter(
                        np.full(len(scores), x[j] + offset) + jitter,
                        scores,
                        color=color, marker=marker,
                        alpha=0.8, s=36,
                        edgecolors="white", linewidth=0.5,
                    )
                if errs:
                    jitter = rng.normal(0, 0.03, size=len(errs))
                    ax_err.scatter(
                        np.full(len(errs), x[j] + offset) + jitter,
                        errs,
                        color=color, marker=marker,
                        alpha=0.8, s=36,
                        edgecolors="white", linewidth=0.5,
                    )

    for ax, ylabel in [
        (ax_score, "Evaluation score (0-1) ↑"),
        (ax_err, "MPE ↓"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_world_label(w) for w in worlds], rotation=20, ha="right"
        )
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
    ax_score.set_ylim(-0.05, 1.05)
    ax_err.set_yscale("log")

    model_handles = [
        Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=10, linestyle="", label=_short(m),
        )
        for m in models
    ]
    cond_handles = [
        Line2D(
            [0], [0], marker=_MARKERS[c], color="black",
            markersize=9, linestyle="", label=c,
        )
        for c in _CONDITIONS
    ]
    fig.legend(
        handles=model_handles + cond_handles,
        loc="lower center", ncol=min(len(model_handles) + 2, 6),
        bbox_to_anchor=(0.5, -0.02), frameon=False,
    )

    fig.tight_layout()
    fig.savefig(analysis_dir / "runs.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "runs.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_pareto_plot(by_cond, models, analysis_dir, title, plt) -> None:
    """One point per (model, condition): pooled (geom-mean MPE, mean score)."""
    if not models:
        return
    from matplotlib.lines import Line2D

    pooled_per_cond = {c: _per_model_pooled(by_cond[c]) for c in _CONDITIONS}
    model_colors = _model_color_map(models, plt)

    fig, ax = plt.subplots(figsize=(9, 6))

    plotted_any = False
    for model in models:
        for cond in _CONDITIONS:
            errs, scores = pooled_per_cond[cond].get(model, ([], []))
            e = _bootstrap_ci_geom(errs)
            s = _bootstrap_ci(scores)
            if e is None or s is None:
                continue
            ax.errorbar(
                e[0], s[0],
                xerr=[[e[1]], [e[2]]],
                yerr=[[s[1]], [s[2]]],
                fmt=_MARKERS[cond],
                color=model_colors[model],
                ecolor=model_colors[model],
                elinewidth=1.2, capsize=3, markersize=11,
                markeredgecolor="white", markeredgewidth=0.7,
            )
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_xlabel("MPE ↓", fontsize=18)
    ax.set_ylabel("Evaluation score (0-1) ↑", fontsize=18)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_ylim(-0.05, 1.05)

    model_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=10, linestyle="", label=_short(m),
        )
        for m in models
    ]
    cond_handles = [
        Line2D(
            [0], [0], marker=_MARKERS[c], color="black",
            markersize=10, linestyle="", label=c,
        )
        for c in _CONDITIONS
    ]
    ax.legend(
        handles=model_handles + cond_handles,
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(model_handles) + 2, 6),
        frameon=False, borderaxespad=0,
    )

    fig.tight_layout()
    fig.savefig(analysis_dir / "pareto.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_per_model_plot(by_cond, models, analysis_dir, title, plt) -> None:
    """Per-model bars: guided & random side-by-side, score (left) and MPE (right)."""
    if not models:
        return
    from matplotlib.patches import Patch

    pooled_per_cond = {c: _per_model_pooled(by_cond[c]) for c in _CONDITIONS}
    n_models = len(models)
    fig_w = max(10.0, 1.4 * n_models * 2)
    fig, (ax_score, ax_err) = plt.subplots(1, 2, figsize=(fig_w, 5.5))
    x = np.arange(n_models)
    model_colors = _model_color_map(models, plt)
    width = 0.38
    hatches = {_GUIDED: "", _RANDOM: "//"}

    for k, cond in enumerate(_CONDITIONS):
        offset = (k - 0.5) * width
        score_means, score_lo, score_hi = [], [], []
        err_means, err_lo, err_hi = [], [], []
        for model in models:
            errs, scores = pooled_per_cond[cond].get(model, ([], []))
            s = _bootstrap_ci(scores)
            e = _bootstrap_ci_geom(errs)
            score_means.append(s[0] if s else np.nan)
            score_lo.append(s[1] if s else 0.0)
            score_hi.append(s[2] if s else 0.0)
            err_means.append(e[0] if e else np.nan)
            err_lo.append(e[1] if e else 0.0)
            err_hi.append(e[2] if e else 0.0)
        colors = [model_colors[m] for m in models]
        ax_score.bar(
            x + offset, score_means, width=width,
            yerr=[score_lo, score_hi], color=colors, capsize=3,
            edgecolor="black", linewidth=0.6, hatch=hatches[cond],
        )
        ax_err.bar(
            x + offset, err_means, width=width,
            yerr=[err_lo, err_hi], color=colors, capsize=3,
            edgecolor="black", linewidth=0.6, hatch=hatches[cond],
        )

    for ax, ylabel in [
        (ax_score, "Evaluation score (0-1) ↑"),
        (ax_err, "MPE (geom. mean) ↓"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_short(m) for m in models], rotation=25, ha="right", fontsize=13
        )
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis="y", labelsize=12)
    ax_score.set_ylim(0, 1.05)
    ax_err.set_yscale("log")

    cond_handles = [
        Patch(
            facecolor="lightgray", edgecolor="black",
            hatch=hatches[c], label=c,
        )
        for c in _CONDITIONS
    ]
    fig.legend(
        handles=cond_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
    )

    fig.tight_layout()
    fig.savefig(analysis_dir / "summary_per_model.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "summary_per_model.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_passed_plot(by_cond, models, analysis_dir, title, plt) -> None:
    """Trials passed per model: guided vs random as paired bars."""
    if not models:
        return
    from matplotlib.patches import Patch

    passed_per_cond = {c: _passed_count(by_cond[c]) for c in _CONDITIONS}
    n_models = len(models)
    fig_w = max(8.0, 1.3 * n_models)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    x = np.arange(n_models)
    model_colors = _model_color_map(models, plt)
    width = 0.38
    hatches = {_GUIDED: "", _RANDOM: "//"}

    for k, cond in enumerate(_CONDITIONS):
        offset = (k - 0.5) * width
        counts = [passed_per_cond[cond].get(m, 0) for m in models]
        ax.bar(
            x + offset, counts, width=width,
            color=[model_colors[m] for m in models],
            edgecolor="black", linewidth=0.6, hatch=hatches[cond],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_short(m) for m in models], rotation=25, ha="right", fontsize=13
    )
    ax.set_ylabel("worlds passed ↑", fontsize=18)
    ax.tick_params(axis="y", labelsize=12)

    cond_handles = [
        Patch(
            facecolor="lightgray", edgecolor="black",
            hatch=hatches[c], label=c,
        )
        for c in _CONDITIONS
    ]
    ax.legend(
        handles=cond_handles, loc="upper right", frameon=False,
    )

    fig.tight_layout()
    fig.savefig(analysis_dir / "summary_passed.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "summary_passed.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_per_world_passed_plot(
    by_cond, models, worlds, analysis_dir, title, plt
) -> None:
    """Per-world pass rate (%): guided vs random in side-by-side panels.

    World order matches `production_analysis.py`: descending median explanation
    score on the guided run (falls back to random if guided is empty).
    """
    if not models or not worlds:
        return
    from matplotlib.lines import Line2D

    seed_map = by_cond[_GUIDED] if by_cond[_GUIDED] else by_cond[_RANDOM]
    ordered = _world_order_by_score(seed_map, worlds)
    n_models = len(models)
    n_worlds = len(ordered)
    model_colors = _model_color_map(models, plt)

    fig_w = max(12.0, 0.9 * n_worlds * max(1, n_models) * 0.35 + 4)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 6), sharey=True)
    width = 0.8 / max(1, n_models)
    xs = np.arange(n_worlds)

    for ax, cond in zip(axes, _CONDITIONS):
        for i, model in enumerate(models):
            pcts = []
            for w in ordered:
                values = by_cond[cond].get((model, w), [])
                n = len(values)
                if n == 0:
                    pcts.append(np.nan)
                    continue
                p = sum(1 for err, score in values if _trial_passes(err, score))
                pcts.append(100.0 * p / n)
            offset = (i - (n_models - 1) / 2) * width
            ax.bar(
                xs + offset, pcts, width=width,
                color=model_colors[model], label=_short(model),
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [_world_label(w) for w in ordered], rotation=20, ha="right", fontsize=12
        )
        ax.set_title(cond, fontsize=15)
        ax.set_ylim(0, 105)
        ax.tick_params(axis="y", labelsize=12)
    axes[0].set_ylabel("worlds passed (%) ↑", fontsize=16)

    handles = [
        Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=10, linestyle="", label=_short(m),
        )
        for m in models
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.05), ncol=min(len(models), 5),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(analysis_dir / "per_world_passed.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "per_world_passed.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("guided_dir", help="Results directory for guided trials")
    parser.add_argument("random_dir", help="Results directory for random trials")
    parser.add_argument(
        "--out",
        default="analysis_guided_vs_random",
        help="Subdir under guided_dir for output (default: analysis_guided_vs_random)",
    )
    args = parser.parse_args()

    guided_dir = Path(args.guided_dir)
    random_dir = Path(args.random_dir)
    for d, label in [(guided_dir, "guided_dir"), (random_dir, "random_dir")]:
        if not d.is_dir():
            print(f"error: {label} {d} is not a directory", file=sys.stderr)
            return 1

    analysis_dir = guided_dir / args.out
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_path = aggregate(guided_dir, random_dir, analysis_dir)
    print(f"summary written to {summary_path}")
    print()
    print(summary_path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
