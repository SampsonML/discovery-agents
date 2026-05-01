#!/usr/bin/env python
"""
Overlay plot: random-experiments benchmark vs. agent-designed rounds benchmark.

Takes two summary.jsonl files — one from scripts/round_benchmark.sh output and
one from scripts/random_benchmark.sh output — and produces a single grid plot
with one row per world × two columns (explanation score ↑, MSE ↓). Each panel
shows two lines:

  - agent-designed   (existing per-world pastel color, circle marker)
  - random experiments (dark gray, square marker)

Points present in one summary but not the other are silently skipped on the
missing line; the panel still renders whichever line has data.

Aggregation helpers (`load`, `agg`, `geomean_agg`) are imported from
scripts/analyze_rounds.py so the two plots stay in numerical lock-step.
"""

import argparse
import os
import sys
from collections import defaultdict

# Make sibling `analyze_rounds.py` importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_rounds import load, agg, geomean_agg


def _group_by_world_rounds(rows):
    groups = defaultdict(list)
    for r in rows:
        if r.get("world") is None or r.get("max_rounds") is None:
            continue
        key = (r["world"], int(r["max_rounds"]))
        groups[key].append(r)
    return groups


def _draw_lines(
    ax_exp,
    ax_mse,
    rnds_a,
    em_a,
    es_a,
    gm_a,
    lo_a,
    hi_a,
    rnds_r,
    em_r,
    es_r,
    gm_r,
    lo_r,
    hi_r,
    agent_color,
    random_color,
):
    import numpy as np

    has_any = False
    if len(rnds_r):
        ax_exp.errorbar(
            rnds_r,
            em_r,
            yerr=es_r,
            marker="s",
            linestyle="--",
            color=random_color,
            capsize=3,
            linewidth=1.8,
            alpha=0.8,
            zorder=2,
        )
        ax_mse.errorbar(
            rnds_r,
            gm_r,
            yerr=np.vstack([lo_r, hi_r]),
            marker="s",
            linestyle="--",
            color=random_color,
            capsize=3,
            linewidth=1.8,
            alpha=0.8,
            zorder=2,
        )
        has_any = True
    if len(rnds_a):
        ax_exp.errorbar(
            rnds_a,
            em_a,
            yerr=es_a,
            marker="o",
            linestyle="-",
            color=agent_color,
            capsize=3,
            linewidth=1.8,
            alpha=0.95,
            zorder=3,
        )
        ax_mse.errorbar(
            rnds_a,
            gm_a,
            yerr=np.vstack([lo_a, hi_a]),
            marker="o",
            linestyle="-",
            color=agent_color,
            capsize=3,
            linewidth=1.8,
            alpha=0.95,
            zorder=3,
        )
        has_any = True
    return has_any


def _series_for_world(groups, rounds_set, world):
    import numpy as np

    rounds_used, exp_means, exp_stds = [], [], []
    mse_gms, mse_los, mse_his = [], [], []
    for rnd in rounds_set:
        grp = groups.get((world, rnd), [])
        if not grp:
            continue
        em, es, _ = agg([r["explanation_score"] for r in grp])
        gm, err, _ = geomean_agg([r["mean_pos_error"] for r in grp])
        if em is None and gm is None:
            continue
        rounds_used.append(rnd)
        exp_means.append(em if em is not None else np.nan)
        exp_stds.append(es if es is not None else 0.0)
        mse_gms.append(gm if gm is not None else np.nan)
        lo, hi = err if err is not None else (0.0, 0.0)
        mse_los.append(lo)
        mse_his.append(hi)
    return (
        np.asarray(rounds_used, dtype=float),
        np.asarray(exp_means, dtype=float),
        np.asarray(exp_stds, dtype=float),
        np.asarray(mse_gms, dtype=float),
        np.asarray(mse_los, dtype=float),
        np.asarray(mse_his, dtype=float),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rounds-summary",
        required=True,
        help="summary.jsonl produced by scripts/round_benchmark.sh",
    )
    p.add_argument(
        "--random-summary",
        required=True,
        help="summary.jsonl produced by scripts/random_benchmark.sh",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to dir of --rounds-summary.",
    )
    p.add_argument(
        "--fname",
        default="rounds_vs_metrics_overlay.png",
        help="Output filename for the overlay plot.",
    )
    args = p.parse_args()

    rounds_rows = load(args.rounds_summary)
    random_rows = load(args.random_summary)
    print(
        f"[overlay] loaded {len(rounds_rows)} rounds rows, "
        f"{len(random_rows)} random rows"
    )
    if not rounds_rows and not random_rows:
        print("[overlay] both summaries empty; nothing to plot")
        return

    out_dir = (
        args.out_dir or os.path.dirname(os.path.abspath(args.rounds_summary)) or "."
    )
    os.makedirs(out_dir, exist_ok=True)

    # World + round axes — mirror analyze_rounds.py's canonical ordering,
    # then union in anything unexpected that appears in either summary.
    _BENCHMARK_WORLDS = [
        "gravity",
        "yukawa",
        "fractional",
        "dark_matter",
        "three_species",
    ]
    world_order = list(_BENCHMARK_WORLDS)
    seen = set(world_order)
    for r in rounds_rows + random_rows:
        w = r.get("world")
        if w and w not in seen:
            seen.add(w)
            world_order.append(w)

    _BENCHMARK_ROUNDS = [1, 2, 4, 8, 16, 32]
    observed_rounds = {
        int(r["max_rounds"])
        for r in (rounds_rows + random_rows)
        if r.get("max_rounds") is not None
    }
    rounds_set = sorted(set(_BENCHMARK_ROUNDS) | observed_rounds)

    rounds_groups = _group_by_world_rounds(rounds_rows)
    random_groups = _group_by_world_rounds(random_rows)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available; cannot plot.")
        return

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    palette = [
        "#8FB5C7",
        "#A8D5BA",
        "#F4B6C2",
        "#F5CBA7",
        "#AECDE0",
        "#E8D5A0",
        "#D4A5A5",
        "#C5E1C5",
    ]
    RANDOM_COLOR = "#444444"
    world_display = {"dark_matter": "dark matter", "three_species": "multi species"}

    n_worlds = len(world_order)
    fig, axes = plt.subplots(
        n_worlds,
        2,
        figsize=(11, 2.6 * n_worlds + 1.6),
        sharex=True,
        squeeze=False,
    )

    for i, world in enumerate(world_order):
        ax_exp = axes[i, 0]
        ax_mse = axes[i, 1]
        agent_color = palette[i % len(palette)]
        display = world_display.get(world, world.replace("_", " "))

        rnds_a, em_a, es_a, gm_a, lo_a, hi_a = _series_for_world(
            rounds_groups, rounds_set, world
        )
        rnds_r, em_r, es_r, gm_r, lo_r, hi_r = _series_for_world(
            random_groups, rounds_set, world
        )

        # Draw random first (lower z-order) so the agent line sits on top.
        has_any = _draw_lines(
            ax_exp,
            ax_mse,
            rnds_a,
            em_a,
            es_a,
            gm_a,
            lo_a,
            hi_a,
            rnds_r,
            em_r,
            es_r,
            gm_r,
            lo_r,
            hi_r,
            agent_color,
            RANDOM_COLOR,
        )
        if not has_any:
            for ax in (ax_exp, ax_mse):
                ax.text(
                    0.5,
                    0.5,
                    "(no data)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=10,
                )

        ax_exp.set_title(display, fontsize=16, loc="center", pad=6)
        ax_mse.set_title(display, fontsize=16, loc="center", pad=6)

        ax_exp.set_ylabel("explanation score (↑)", fontsize=13)
        ax_exp.set_ylim(0.0, 1.05)
        ax_mse.set_ylabel("MSE (↓)", fontsize=13)
        ax_mse.set_yscale("log")

        for ax in (ax_exp, ax_mse):
            ax.grid(True, alpha=0.3)
            ax.tick_params(
                direction="in", which="both", top=True, right=True, labelsize=11
            )

    for ax in axes[-1, :]:
        ax.set_xlabel("max rounds", fontsize=13)
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(rounds_set)
        ax.set_xticklabels([str(r) for r in rounds_set])

    # Figure-level legend — per-panel legends would repeat the same 2-entry
    # key on every subplot. Use a neutral swatch for the agent entry since
    # its real color varies by world; the shape (circle vs square) plus the
    # label carry the distinction.
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#888888",
            marker="o",
            linestyle="-",
            linewidth=1.8,
            label="agent-designed (color)",
        ),
        Line2D(
            [0],
            [0],
            color=RANDOM_COLOR,
            marker="s",
            linestyle="-",
            linewidth=1.8,
            label="random experiments",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.0),
        ncol=2,
        frameon=False,
        fontsize=12,
    )

    fig.tight_layout(rect=[0.08, 0.02, 1, 0.965])
    fig.subplots_adjust(left=0.14)

    # Physics-difficulty arrow on the far left (mirrors analyze_rounds.py).
    from matplotlib.patches import FancyArrowPatch

    top_bbox = axes[0, 0].get_position()
    bottom_bbox = axes[-1, 0].get_position()
    arrow_x = max(0.02, top_bbox.x0 - 0.10)
    y_top = top_bbox.y1
    y_bottom = bottom_bbox.y0
    arrow = FancyArrowPatch(
        (arrow_x, y_top),
        (arrow_x, y_bottom),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=25,
        color="black",
        linewidth=2,
    )
    fig.patches.append(arrow)
    fig.text(
        arrow_x,
        (y_top + y_bottom) / 2,
        "physics difficulty",
        ha="center",
        va="center",
        rotation=90,
        color="black",
        fontsize=16,
        style="italic",
        bbox=dict(facecolor=fig.get_facecolor(), edgecolor="none", pad=4),
    )

    out_path = os.path.join(out_dir, args.fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    # ------------------------------------------------------------------
    # Second figure: same rows, but each line-plot column is followed by
    # a "difference" column showing (agent − random) per round. Positive
    # Δexplanation means the agent scored higher; negative ΔMSE means the
    # agent's error was lower.
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(
        n_worlds,
        4,
        figsize=(20, 2.6 * n_worlds + 1.6),
        sharex=True,
        squeeze=False,
    )

    for i, world in enumerate(world_order):
        ax_exp = axes2[i, 0]
        ax_exp_d = axes2[i, 1]
        ax_mse = axes2[i, 2]
        ax_mse_d = axes2[i, 3]
        agent_color = palette[i % len(palette)]
        display = world_display.get(world, world.replace("_", " "))

        rnds_a, em_a, es_a, gm_a, lo_a, hi_a = _series_for_world(
            rounds_groups, rounds_set, world
        )
        rnds_r, em_r, es_r, gm_r, lo_r, hi_r = _series_for_world(
            random_groups, rounds_set, world
        )

        has_any = _draw_lines(
            ax_exp,
            ax_mse,
            rnds_a,
            em_a,
            es_a,
            gm_a,
            lo_a,
            hi_a,
            rnds_r,
            em_r,
            es_r,
            gm_r,
            lo_r,
            hi_r,
            agent_color,
            RANDOM_COLOR,
        )
        if not has_any:
            for ax in (ax_exp, ax_mse):
                ax.text(
                    0.5,
                    0.5,
                    "(no data)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=10,
                )

        a_em = dict(zip(rnds_a.astype(int).tolist(), em_a.tolist()))
        r_em = dict(zip(rnds_r.astype(int).tolist(), em_r.tolist()))
        a_gm = dict(zip(rnds_a.astype(int).tolist(), gm_a.tolist()))
        r_gm = dict(zip(rnds_r.astype(int).tolist(), gm_r.tolist()))
        common = sorted(set(a_em) & set(r_em))
        if common:
            c_arr = np.array(common, dtype=float)
            em_diff = np.array([a_em[r] - r_em[r] for r in common])
            gm_diff = np.array([a_gm[r] - r_gm[r] for r in common])
            ax_exp_d.plot(
                c_arr,
                em_diff,
                marker="o",
                linestyle="-",
                color=agent_color,
                linewidth=1.8,
                zorder=3,
            )
            ax_mse_d.plot(
                c_arr,
                gm_diff,
                marker="o",
                linestyle="-",
                color=agent_color,
                linewidth=1.8,
                zorder=3,
            )
        else:
            for ax in (ax_exp_d, ax_mse_d):
                ax.text(
                    0.5,
                    0.5,
                    "(no overlap)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=10,
                )

        for ax in (ax_exp_d, ax_mse_d):
            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

        ax_exp.set_title(display, fontsize=16, loc="center", pad=6)
        ax_exp_d.set_title(f"{display} (Δ)", fontsize=16, loc="center", pad=6)
        ax_mse.set_title(display, fontsize=16, loc="center", pad=6)
        ax_mse_d.set_title(f"{display} (Δ)", fontsize=16, loc="center", pad=6)

        ax_exp.set_ylabel("explanation score (↑)", fontsize=13)
        ax_exp.set_ylim(0.0, 1.05)
        ax_exp_d.set_ylabel("Δ explanation (agent − random, ↑)", fontsize=13)
        ax_mse.set_ylabel("MSE (↓)", fontsize=13)
        ax_mse.set_yscale("log")
        ax_mse_d.set_ylabel("Δ MSE (agent − random, ↓)", fontsize=13)

        for ax in (ax_exp, ax_mse, ax_exp_d, ax_mse_d):
            ax.grid(True, alpha=0.3)
            ax.tick_params(
                direction="in", which="both", top=True, right=True, labelsize=11
            )

    for ax in axes2[-1, :]:
        ax.set_xlabel("max rounds", fontsize=13)
    for ax in axes2.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(rounds_set)
        ax.set_xticklabels([str(r) for r in rounds_set])

    fig2.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.0),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig2.tight_layout(rect=[0.05, 0.02, 1, 0.965])
    fig2.subplots_adjust(left=0.08)

    top_bbox2 = axes2[0, 0].get_position()
    bottom_bbox2 = axes2[-1, 0].get_position()
    arrow_x2 = max(0.01, top_bbox2.x0 - 0.06)
    arrow2 = FancyArrowPatch(
        (arrow_x2, top_bbox2.y1),
        (arrow_x2, bottom_bbox2.y0),
        transform=fig2.transFigure,
        arrowstyle="-|>",
        mutation_scale=25,
        color="black",
        linewidth=2,
    )
    fig2.patches.append(arrow2)
    fig2.text(
        arrow_x2,
        (top_bbox2.y1 + bottom_bbox2.y0) / 2,
        "physics difficulty",
        ha="center",
        va="center",
        rotation=90,
        color="black",
        fontsize=16,
        style="italic",
        bbox=dict(facecolor=fig2.get_facecolor(), edgecolor="none", pad=4),
    )

    base, ext = os.path.splitext(args.fname)
    diff_fname = f"{base}_with_diff{ext}"
    out_path2 = os.path.join(out_dir, diff_fname)
    fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {out_path2}")


if __name__ == "__main__":
    main()
