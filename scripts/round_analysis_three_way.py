#!/usr/bin/env python
"""
Three-way round-budget analysis: guided vs random vs no_mse.

Same shape as round_analysis.py but adds a third condition (no_mse, triangle
marker, dotted linestyle). All result directories are hard-coded — no
positional arguments. Pulls helpers (`_pool_for_model`, `_pass_count_for_model`,
`_worlds_passed_at_k_for_model`, `_expected_worlds_passed_at_k_for_model`)
from `round_analysis.py` so the metric definitions stay in lockstep.

Outputs in --out (default <repo>/analysis_rounds), each in vertical (3×1),
horizontal (1×3) scatter, and horizontal bar layouts. Suffixes: `_horizontal`
for horizontal scatter, `_horizontal_bar` for horizontal grouped bars (color =
model, hatch = condition, x = round budget):
    round_analysis_three_way.{png,pdf}
    round_analysis_three_way_passed_at_{1,3,5}.{png,pdf}
    round_analysis_three_way_expected_passed_at_{1,3,5}.{png,pdf}

Usage:
    python scripts/round_analysis_three_way.py
    python scripts/round_analysis_three_way.py --out /path/to/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from guided_vs_random import _walk_seeded
from production_analysis import (
    _bootstrap_ci,
    _bootstrap_ci_geom,
    _model_color_map,
    _model_sort_key,
    _short,
)
from round_analysis import (
    _MODEL_TAGS,
    _expected_worlds_passed_at_k_for_model,
    _pass_count_for_model,
    _pool_for_model,
    _worlds_passed_at_k_for_model,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "yml_bench"

_ROUND_BUDGETS = (2, 4, 8, 16)
_GUIDED = "guided"
_RANDOM = "random"
_NO_MSE = "no_mse"
_CONDITIONS = (_GUIDED, _RANDOM, _NO_MSE)
_MARKERS = {_GUIDED: "o", _RANDOM: "X", _NO_MSE: "^"}
_LINESTYLES = {_GUIDED: "-", _RANDOM: "--", _NO_MSE: ":"}
# Bar-plot hatch per condition (color still encodes model).
_HATCHES = {_GUIDED: "", _RANDOM: "///", _NO_MSE: "xxx"}

_TOTAL_WORLDS = 11
_PASS_K_VARIANTS = (1, 3, 5)

# Hard-coded benchmark directories: (rounds, condition) → path under results/yml_bench/.
_DIRS: dict[tuple[int, str], Path] = {
    (2,  _GUIDED):  RESULTS_ROOT / "production_r2",
    (2,  _RANDOM):  RESULTS_ROOT / "production_r2_random",
    (2,  _NO_MSE):  RESULTS_ROOT / "production_no_mse_r2",
    (4,  _GUIDED):  RESULTS_ROOT / "production_r4",
    (4,  _RANDOM):  RESULTS_ROOT / "production_r4_random",
    (4,  _NO_MSE):  RESULTS_ROOT / "production_no_mse_r4",
    (8,  _GUIDED):  RESULTS_ROOT / "production_run_baseline",
    (8,  _RANDOM):  RESULTS_ROOT / "production_run_random",
    (8,  _NO_MSE):  RESULTS_ROOT / "production_no_mse",
    (16, _GUIDED):  RESULTS_ROOT / "production_r16",
    (16, _RANDOM):  RESULTS_ROOT / "production_r16_random",
    (16, _NO_MSE):  RESULTS_ROOT / "production_no_mse_r16",
}


def make_plot(
    dirs: dict[tuple[int, str], Path],
    out_dir: Path,
    pass_k: int | None = None,
    expected_pass_k: int | None = None,
    horizontal: bool = False,
    bar_plot: bool = False,
) -> Path:
    if pass_k is not None and expected_pass_k is not None:
        raise ValueError("pass_k and expected_pass_k are mutually exclusive")
    if bar_plot:
        # Bar variant currently only renders the horizontal (1×3) layout.
        horizontal = True
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; cannot make plot", file=sys.stderr)
        return Path()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

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
        out_stem = f"round_analysis_three_way_expected_passed_at_{expected_pass_k}"
    elif pass_k is not None:
        def _row1(rounds, cond, model: str, _k=pass_k):
            return float(
                _worlds_passed_at_k_for_model(
                    by_run.get((rounds, cond), {}), model, _k
                )
            ), 0.0
        row1_label = f"worlds passed @k={pass_k} ↑"
        out_stem = f"round_analysis_three_way_passed_at_{pass_k}"
    else:
        def _row1(rounds, cond, model: str):
            return float(
                _pass_count_for_model(by_run.get((rounds, cond), {}), model)
            ), 0.0
        row1_label = "worlds passed ↑"
        out_stem = "round_analysis_three_way"

    if horizontal:
        out_stem = f"{out_stem}_horizontal"
    if bar_plot:
        out_stem = f"{out_stem}_bar"

    target_models = list(_MODEL_TAGS.keys())
    model_colors = _model_color_map(target_models, plt)
    model_colors["claude-opus-4-7"] = "dimgray"
    model_colors["azure/gpt-5.5"] = "darkorange"

    if horizontal:
        fig, (ax_pass, ax_score, ax_err) = plt.subplots(
            1, 3, figsize=(28, 6.5)
        )
    else:
        fig, (ax_pass, ax_score, ax_err) = plt.subplots(
            3, 1, figsize=(8, 11), sharex=True
        )

    if bar_plot:
        n_rounds = len(_ROUND_BUDGETS)
        n_groups = len(target_models) * len(_CONDITIONS)
        bar_width = 0.85 / n_groups
        x_pos = np.arange(n_rounds)
        for mi, model in enumerate(target_models):
            for ci, cond in enumerate(_CONDITIONS):
                pass_y = [np.nan] * n_rounds
                pass_yerr = [0.0] * n_rounds
                score_y = [np.nan] * n_rounds
                score_lo = [0.0] * n_rounds
                score_hi = [0.0] * n_rounds
                err_y = [np.nan] * n_rounds
                err_lo = [0.0] * n_rounds
                err_hi = [0.0] * n_rounds
                for ri, rounds in enumerate(_ROUND_BUDGETS):
                    by_trial = by_run.get((rounds, cond), {})
                    errs, scores = _pool_for_model(by_trial, model)
                    if not errs and not scores:
                        continue
                    row1_mean, row1_sem = _row1(rounds, cond, model)
                    pass_y[ri] = row1_mean
                    pass_yerr[ri] = row1_sem
                    s_ci = _bootstrap_ci(scores)
                    if s_ci is not None:
                        score_y[ri] = s_ci[0]
                        score_lo[ri] = s_ci[1]
                        score_hi[ri] = s_ci[2]
                    e_ci = _bootstrap_ci_geom(errs)
                    if e_ci is not None:
                        err_y[ri] = e_ci[0]
                        err_lo[ri] = e_ci[1]
                        err_hi[ri] = e_ci[2]

                group_idx = mi * len(_CONDITIONS) + ci
                offset = (group_idx - (n_groups - 1) / 2) * bar_width
                color = model_colors[model]
                hatch = _HATCHES[cond]
                bar_kw = dict(
                    width=bar_width, color=color, hatch=hatch,
                    edgecolor="black", linewidth=0.6, capsize=2,
                )
                ax_pass.bar(x_pos + offset, pass_y, yerr=pass_yerr, **bar_kw)
                ax_score.bar(
                    x_pos + offset, score_y,
                    yerr=[score_lo, score_hi], **bar_kw,
                )
                ax_err.bar(
                    x_pos + offset, err_y,
                    yerr=[err_lo, err_hi], **bar_kw,
                )
    else:
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
                    capsize=3, elinewidth=1.0, alpha=0.75,
                )
                ax_score.errorbar(
                    xs, score_y, yerr=[score_lo, score_hi],
                    color=color, ecolor=color, marker=marker, linestyle=ls,
                    linewidth=1.8, markersize=9, capsize=3,
                    markeredgecolor="white", markeredgewidth=0.6, alpha=0.75,
                )
                ax_err.errorbar(
                    xs, err_y, yerr=[err_lo, err_hi],
                    color=color, ecolor=color, marker=marker, linestyle=ls,
                    linewidth=1.8, markersize=9, capsize=3,
                    markeredgecolor="white", markeredgewidth=0.6, alpha=0.75,
                )

    label_kw = {"fontsize": 26 if horizontal else 17}
    xlabel_kw = {"fontsize": 28 if horizontal else 19}
    ax_pass.set_ylabel(row1_label, **label_kw)
    if pass_k is not None or expected_pass_k is not None:
        ax_pass.set_ylim(-0.5, _TOTAL_WORLDS)
    else:
        ax_pass.set_ylim(bottom=-0.5)
    ax_score.set_ylabel("Evaluation score (0-1) ↑", **label_kw)
    ax_score.set_ylim(0, 1.05)
    ax_err.set_ylabel("MPE (geom. mean) ↓", **label_kw)
    ax_err.set_yscale("log")
    ax_err.set_xlabel("rounds", **xlabel_kw)
    if horizontal:
        ax_pass.set_xlabel("rounds", **xlabel_kw)
        ax_score.set_xlabel("rounds", **xlabel_kw)

    tick_size = 16 if horizontal else 12
    if bar_plot:
        x_pos = np.arange(len(_ROUND_BUDGETS))
        for ax in (ax_pass, ax_score, ax_err):
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(r) for r in _ROUND_BUDGETS])
            ax.tick_params(axis="both", labelsize=tick_size)
    else:
        for ax in (ax_pass, ax_score, ax_err):
            ax.set_xscale("log", base=2)
            ax.set_xticks(_ROUND_BUDGETS)
            ax.set_xticklabels([str(r) for r in _ROUND_BUDGETS])
            ax.tick_params(axis="both", labelsize=tick_size)

    model_handles = [
        Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=11, linestyle="", label=_short(m),
        )
        for m in sorted(target_models, key=_model_sort_key)
    ]
    if bar_plot:
        cond_handles = [
            Patch(
                facecolor="lightgray", edgecolor="black",
                hatch=_HATCHES[c], linewidth=0.6, label=c,
            )
            for c in _CONDITIONS
        ]
    else:
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
        ncol=6 if horizontal else 3,
        frameon=False,
        fontsize=25 if horizontal else 16,
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
        "--out",
        default=str(REPO_ROOT / "analysis_rounds"),
        help="Output directory (default: <repo>/analysis_rounds)",
    )
    args = parser.parse_args()

    for path in _DIRS.values():
        if not path.is_dir():
            print(f"error: missing benchmark directory {path}", file=sys.stderr)
            return 1

    out_dir = Path(args.out)
    for horiz in (False, True):
        out_path = make_plot(_DIRS, out_dir, horizontal=horiz)
        if out_path:
            print(f"wrote {out_path}")
        for k in _PASS_K_VARIANTS:
            p = make_plot(_DIRS, out_dir, pass_k=k, horizontal=horiz)
            if p:
                print(f"wrote {p}")
        for k in _PASS_K_VARIANTS:
            p = make_plot(_DIRS, out_dir, expected_pass_k=k, horizontal=horiz)
            if p:
                print(f"wrote {p}")

    # Bar-plot variants (always horizontal).
    out_path = make_plot(_DIRS, out_dir, bar_plot=True)
    if out_path:
        print(f"wrote {out_path}")
    for k in _PASS_K_VARIANTS:
        p = make_plot(_DIRS, out_dir, pass_k=k, bar_plot=True)
        if p:
            print(f"wrote {p}")
    for k in _PASS_K_VARIANTS:
        p = make_plot(_DIRS, out_dir, expected_pass_k=k, bar_plot=True)
        if p:
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
