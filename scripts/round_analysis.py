#!/usr/bin/env python
"""
Plot worlds passed, evaluation score, and MPE as a function of round budget
for opus-4-7 and gpt-5.5, comparing guided vs random experiments.

Takes eight results directories in this exact order:

    2-rounds-guided  2-rounds-random
    4-rounds-guided  4-rounds-random
    8-rounds-guided  8-rounds-random
    16-rounds-guided 16-rounds-random

Each (model, condition) is pooled across all worlds and seeds inside its
directory. Color = model, marker = guided (circle) vs random (X). Output
goes to `analysis_rounds/round_analysis.{png,pdf}` in the repo root.

Usage:
    python scripts/round_analysis.py r2g r2r r4g r4r r8g r8r r16g r16r
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from guided_vs_random import _walk, _walk_seeded
from production_analysis import (
    _BOOTSTRAP_SEED,
    _EXPECTED_PASSED_SAMPLES,
    _SEED_POOL_SIZE,
    _bootstrap_ci,
    _bootstrap_ci_geom,
    _model_color_map,
    _model_sort_key,
    _short,
    _trial_passes,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

_ROUND_BUDGETS = (2, 4, 8, 16)
_GUIDED = "guided"
_RANDOM = "random"
_CONDITIONS = (_GUIDED, _RANDOM)
_MARKERS = {_GUIDED: "o", _RANDOM: "X"}
_LINESTYLES = {_GUIDED: "-", _RANDOM: "--"}

# Y-axis cap for worlds-passed@k plots; matches the canonical 11-world grid.
_TOTAL_WORLDS = 11
# k values for the per-k variants of the headline plot.
_PASS_K_VARIANTS = (1, 3, 5)

# Substring matchers (against lowercased, dot→dash-normalized model strings)
# for the two models we want to keep.
_MODEL_TAGS = {
    "claude-opus-4-7": "opus-4-7",
    "azure/gpt-5.5": "gpt-5-5",
}


def _normalize(model: str) -> str:
    return model.lower().replace(".", "-")


def _matches_target(model: str) -> str | None:
    """Return the canonical target model string this run belongs to, else None."""
    norm = _normalize(model)
    for canonical, tag in _MODEL_TAGS.items():
        if tag in norm:
            return canonical
    return None


def _pool_for_model(by_trial, target_model: str):
    """Pool (errs, scores) across every (target_model, world) entry."""
    errs: list[float] = []
    scores: list[float] = []
    for (model, _world), values in by_trial.items():
        if _matches_target(model) != target_model:
            continue
        for err, score in values:
            if isinstance(err, (int, float)) and math.isfinite(err):
                errs.append(float(err))
            if isinstance(score, (int, float)) and math.isfinite(score):
                scores.append(float(score))
    return errs, scores


def _pass_count_for_model(by_trial, target_model: str) -> int:
    n = 0
    for (model, _world), values in by_trial.items():
        if _matches_target(model) != target_model:
            continue
        for err, score in values:
            if _trial_passes(err, score):
                n += 1
    return n


def _worlds_passed_at_k_for_model(
    by_trial, target_model: str, k: int
) -> int:
    """Count worlds where ≥k seeds for `target_model` achieved a trial-pass."""
    seed_passes_per_world: dict[str, int] = {}
    for (model, world), values in by_trial.items():
        if _matches_target(model) != target_model:
            continue
        passed = sum(1 for err, score in values if _trial_passes(err, score))
        seed_passes_per_world[world] = (
            seed_passes_per_world.get(world, 0) + passed
        )
    return sum(1 for pc in seed_passes_per_world.values() if pc >= k)


def _expected_worlds_passed_at_k_for_model(
    by_seed, target_model: str, k: int
) -> tuple[float, float]:
    """(mean, sem) of worlds_passed@k for `target_model` under random seed
    sampling: each MC draw picks k distinct seed positions from
    {0..pool-1} (without replacement); world counts as passed iff at least
    one drawn position passes. Missing seeds count as fails for their slot.
    """
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = np.stack(
        [
            rng.choice(_SEED_POOL_SIZE, size=k, replace=False)
            for _ in range(_EXPECTED_PASSED_SAMPLES)
        ]
    )
    seeds_per_world: dict[
        str, dict[int, tuple[float | None, float | None]]
    ] = {}
    for (model, world), seed_map in by_seed.items():
        if _matches_target(model) != target_model:
            continue
        for s, val in seed_map.items():
            seeds_per_world.setdefault(world, {})[s] = val

    per_sample_counts = np.zeros(_EXPECTED_PASSED_SAMPLES, dtype=float)
    for seed_map in seeds_per_world.values():
        passes = np.array([
            _trial_passes(*seed_map.get(s, (None, None)))
            for s in range(_SEED_POOL_SIZE)
        ])
        if not passes.any():
            continue
        per_sample_counts += passes[samples].any(axis=1).astype(float)
    mean = float(per_sample_counts.mean())
    sem = (
        float(per_sample_counts.std(ddof=1) / math.sqrt(_EXPECTED_PASSED_SAMPLES))
        if _EXPECTED_PASSED_SAMPLES > 1
        else 0.0
    )
    return mean, sem


def make_plot(
    dirs: dict[tuple[int, str], Path],
    out_dir: Path,
    pass_k: int | None = None,
    expected_pass_k: int | None = None,
) -> Path:
    if pass_k is not None and expected_pass_k is not None:
        raise ValueError("pass_k and expected_pass_k are mutually exclusive")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; cannot make plot", file=sys.stderr)
        return Path()
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })

    # Walk each directory once. Seeded view is needed only for expected@k;
    # we always grab both to keep the code path uniform.
    by_run: dict[tuple[int, str], dict] = {}
    by_seed_run: dict[tuple[int, str], dict] = {}
    for (rounds, cond), path in dirs.items():
        by_run[(rounds, cond)], by_seed_run[(rounds, cond)] = _walk_seeded(path)

    if expected_pass_k is not None:
        def _row1(rounds, cond, model: str, _k=expected_pass_k):
            return _expected_worlds_passed_at_k_for_model(
                by_seed_run.get((rounds, cond), {}), model, _k
            )
        row1_label = f"expected passed @k={expected_pass_k} ↑"
        out_stem = f"round_analysis_expected_passed_at_{expected_pass_k}"
    elif pass_k is not None:
        def _row1(rounds, cond, model: str, _k=pass_k):
            return float(
                _worlds_passed_at_k_for_model(
                    by_run.get((rounds, cond), {}), model, _k
                )
            ), 0.0
        row1_label = f"worlds passed @k={pass_k} ↑"
        out_stem = f"round_analysis_passed_at_{pass_k}"
    else:
        def _row1(rounds, cond, model: str):
            return float(
                _pass_count_for_model(by_run.get((rounds, cond), {}), model)
            ), 0.0
        row1_label = "worlds passed ↑"
        out_stem = "round_analysis"

    target_models = list(_MODEL_TAGS.keys())
    model_colors = _model_color_map(target_models, plt)
    model_colors["claude-opus-4-7"] = "dimgray"
    model_colors["azure/gpt-5.5"] = "darkorange"

    fig, (ax_pass, ax_score, ax_err) = plt.subplots(
        3, 1, figsize=(8, 11), sharex=True
    )

    for model in target_models:
        for cond in _CONDITIONS:
            xs, pass_y, pass_yerr = [], [], []
            score_y, score_lo, score_hi = [], [], []
            err_y, err_lo, err_hi = [], [], []
            for rounds in _ROUND_BUDGETS:
                by_trial = by_run.get((rounds, cond), {})
                errs, scores = _pool_for_model(by_trial, model)
                if not errs and not scores:
                    continue
                xs.append(rounds)
                row1_mean, row1_sem = _row1(rounds, cond, model)
                pass_y.append(row1_mean)
                pass_yerr.append(row1_sem)

                s_ci = _bootstrap_ci(scores)
                score_y.append(s_ci[0] if s_ci else np.nan)
                score_lo.append(s_ci[1] if s_ci else 0.0)
                score_hi.append(s_ci[2] if s_ci else 0.0)

                e_ci = _bootstrap_ci_geom(errs)
                err_y.append(e_ci[0] if e_ci else np.nan)
                err_lo.append(e_ci[1] if e_ci else 0.0)
                err_hi.append(e_ci[2] if e_ci else 0.0)

            if not xs:
                continue
            color = model_colors[model]
            marker = _MARKERS[cond]
            ls = _LINESTYLES[cond]
            ax_pass.errorbar(
                xs, pass_y, yerr=pass_yerr,
                color=color, ecolor=color,
                marker=marker, linestyle=ls,
                linewidth=1.8, markersize=9,
                markeredgecolor="white", markeredgewidth=0.6,
                capsize=3, elinewidth=1.0,
            )
            ax_score.errorbar(
                xs, score_y, yerr=[score_lo, score_hi],
                color=color, ecolor=color, marker=marker, linestyle=ls,
                linewidth=1.8, markersize=9, capsize=3,
                markeredgecolor="white", markeredgewidth=0.6,
            )
            ax_err.errorbar(
                xs, err_y, yerr=[err_lo, err_hi],
                color=color, ecolor=color, marker=marker, linestyle=ls,
                linewidth=1.8, markersize=9, capsize=3,
                markeredgecolor="white", markeredgewidth=0.6,
            )

    label_kw = {"fontsize": 17}
    xlabel_kw = {"fontsize": 19}
    ax_pass.set_ylabel(row1_label, **label_kw)
    if pass_k is not None or expected_pass_k is not None:
        ax_pass.set_ylim(0, _TOTAL_WORLDS)
    ax_score.set_ylabel("Evaluation score (0-1) ↑", **label_kw)
    ax_score.set_ylim(0, 1.05)
    ax_err.set_ylabel("MPE (geom. mean) ↓", **label_kw)
    ax_err.set_yscale("log")
    ax_err.set_xlabel("rounds", **xlabel_kw)

    for ax in (ax_pass, ax_score, ax_err):
        ax.set_xscale("log", base=2)
        ax.set_xticks(_ROUND_BUDGETS)
        ax.set_xticklabels([str(r) for r in _ROUND_BUDGETS])
        ax.tick_params(axis="both", labelsize=12)

    model_handles = [
        Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=11, linestyle="", label=_short(m),
        )
        for m in sorted(target_models, key=_model_sort_key)
    ]
    cond_handles = [
        Line2D(
            [0], [0], marker=_MARKERS[c], color="black",
            linestyle=_LINESTYLES[c], markersize=9, linewidth=1.6, label=c,
        )
        for c in _CONDITIONS
    ]
    fig.legend(
        handles=model_handles + cond_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(model_handles) + len(cond_handles),
        frameon=False,
        fontsize=16,
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{out_stem}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dirs", nargs=8,
        metavar="DIR",
        help="Eight result directories in order: r2g r2r r4g r4r r8g r8r r16g r16r",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "analysis_rounds"),
        help="Output directory (default: <repo>/analysis_rounds)",
    )
    args = parser.parse_args()

    paths = [Path(d) for d in args.dirs]
    for p in paths:
        if not p.is_dir():
            print(f"error: {p} is not a directory", file=sys.stderr)
            return 1

    dirs: dict[tuple[int, str], Path] = {}
    for path, rounds, cond in zip(
        paths,
        [r for r in _ROUND_BUDGETS for _ in _CONDITIONS],
        list(_CONDITIONS) * len(_ROUND_BUDGETS),
    ):
        dirs[(rounds, cond)] = path

    out_dir = Path(args.out)
    out_path = make_plot(dirs, out_dir)
    if out_path:
        print(f"wrote {out_path}")
    for k in _PASS_K_VARIANTS:
        out_path_k = make_plot(dirs, out_dir, pass_k=k)
        if out_path_k:
            print(f"wrote {out_path_k}")
    for k in _PASS_K_VARIANTS:
        out_path_e = make_plot(dirs, out_dir, expected_pass_k=k)
        if out_path_e:
            print(f"wrote {out_path_e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
