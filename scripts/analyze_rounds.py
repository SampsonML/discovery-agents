#!/usr/bin/env python
"""
Aggregate the rounds-sweep summary.jsonl produced by round_benchmark.sh into:
  - rounds_agg.csv              one row per (world, max_rounds)
  - rounds_table.md             same, as a markdown table
  - rounds_vs_metrics.png       explanation score (↑) and MSE (↓) vs rounds, per world + aggregate

Assumes a single model / critic setting (claude-opus-4-6, critic off) per the
round_benchmark.sh configuration, but the plotting code filters on whatever
distinct (model, use_critic) pairs are present and picks the first.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean, stdev


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _clean(values):
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        out.append(f)
    return out


def agg(values):
    vals = _clean(values)
    if not vals:
        return None, None, 0
    m = mean(vals)
    s = stdev(vals) if len(vals) > 1 else 0.0
    return m, s, len(vals)


def geomean_agg(values):
    """Geometric mean with asymmetric (multiplicative) error offsets."""
    vals = [v for v in _clean(values) if v > 0]
    if not vals:
        return None, None, 0
    logs = [math.log(v) for v in vals]
    lm = mean(logs)
    gm = math.exp(lm)
    if len(logs) > 1:
        ls = stdev(logs)
        gsd = math.exp(ls)
        err = (gm - gm / gsd, gm * gsd - gm)
    else:
        err = (0.0, 0.0)
    return gm, err, len(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--fname",
        default="rounds_vs_metrics.png",
        help="Output filename for the main plot.",
    )
    args = p.parse_args()

    rows = load(args.summary)
    if not rows:
        print(f"[analyze_rounds] no rows in {args.summary}")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    # Always render all five benchmark worlds (difficulty order), even if
    # some have no completed runs yet — a missing row will show "(no data)".
    # Any extra worlds that appear in the summary are appended after.
    _BENCHMARK_WORLDS = [
        "gravity",
        "yukawa",
        "fractional",
        "dark_matter",
        "three_species",
    ]
    world_order = list(_BENCHMARK_WORLDS)
    seen = set(world_order)
    for r in rows:
        w = r.get("world")
        if w and w not in seen:
            seen.add(w)
            world_order.append(w)

    # Always plot the canonical round budgets from round_benchmark.sh
    # (even if some haven't run yet); union in any extras actually observed.
    _BENCHMARK_ROUNDS = [1, 2, 4, 8, 16, 32]
    observed_rounds = {
        int(r["max_rounds"]) for r in rows if r.get("max_rounds") is not None
    }
    rounds_set = sorted(set(_BENCHMARK_ROUNDS) | observed_rounds)

    # Group by (world, max_rounds) — the two axes we sweep.
    groups = defaultdict(list)
    for r in rows:
        if r.get("world") is None or r.get("max_rounds") is None:
            continue
        key = (r["world"], int(r["max_rounds"]))
        groups[key].append(r)

    # ---------- CSV + Markdown table ----------
    csv_path = os.path.join(args.out_dir, "rounds_agg.csv")
    md_path = os.path.join(args.out_dir, "rounds_table.md")

    md_lines = [
        "| world | rounds | n | MSE (geomean, ×/gsd) | explanation (mean±std) | pass rate |",
        "|---|---|---|---|---|---|",
    ]
    csv_rows = [
        [
            "world",
            "max_rounds",
            "n",
            "mean_pos_error_geomean",
            "mean_pos_error_gsd_lo",
            "mean_pos_error_gsd_hi",
            "mean_pos_error_arith_mean",
            "mean_pos_error_arith_std",
            "explanation_score_mean",
            "explanation_score_std",
            "pass_rate",
        ]
    ]

    def _sort_key(k):
        world, rnd = k
        try:
            w_idx = world_order.index(world)
        except ValueError:
            w_idx = len(world_order)
        return (w_idx, rnd)

    for key in sorted(groups.keys(), key=_sort_key):
        world, rnd = key
        grp = groups[key]
        mse_gm, mse_err, _ = geomean_agg([r["mean_pos_error"] for r in grp])
        mse_m, mse_s, _ = agg([r["mean_pos_error"] for r in grp])
        exp_m, exp_s, _ = agg([r["explanation_score"] for r in grp])
        passed = [1.0 if r.get("passed") else 0.0 for r in grp]
        pass_rate = mean(passed) if passed else 0.0

        if mse_gm is not None and mse_err is not None:
            lo, hi = mse_err
            mse_str = f"{mse_gm:.3g} (−{lo:.2g}/+{hi:.2g})"
        else:
            mse_str = "—"
        exp_str = f"{exp_m:.2f}±{exp_s:.2f}" if exp_m is not None else "—"

        md_lines.append(
            f"| {world} | {rnd} | {len(grp)} | {mse_str} | {exp_str} | {pass_rate:.2f} |"
        )
        lo, hi = mse_err if mse_err is not None else (None, None)
        csv_rows.append(
            [
                world,
                rnd,
                len(grp),
                mse_gm,
                lo,
                hi,
                mse_m,
                mse_s,
                exp_m,
                exp_s,
                pass_rate,
            ]
        )

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")

    # ---------- Plot ----------
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    # Hand-picked pastel palette (matches aggregate_bench.py).
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

    world_display = {"dark_matter": "dark matter", "three_species": "multi species"}

    def _series_for_world(world):
        """Return (rounds_used, exp_mean, exp_std, mse_gm, mse_lo, mse_hi)."""
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

    # One row per world, two columns (explanation score, MSE). Worlds are
    # listed top-to-bottom in the order they appear in the summary, which
    # matches the "physics difficulty" ladder used throughout the repo.
    n_worlds = len(world_order)
    fig, axes = plt.subplots(
        n_worlds,
        2,
        figsize=(11, 2.6 * n_worlds + 1.2),
        sharex=True,
        squeeze=False,
    )

    for i, world in enumerate(world_order):
        ax_exp = axes[i, 0]
        ax_mse = axes[i, 1]
        color = palette[i % len(palette)]
        display = world_display.get(world, world.replace("_", " "))

        rnds, em, es, gm, lo, hi = _series_for_world(world)
        if len(rnds):
            ax_exp.errorbar(
                rnds,
                em,
                yerr=es,
                marker="o",
                linestyle="-",
                color=color,
                capsize=3,
                linewidth=1.8,
                alpha=0.95,
            )
            ax_mse.errorbar(
                rnds,
                gm,
                yerr=np.vstack([lo, hi]),
                marker="o",
                linestyle="-",
                color=color,
                capsize=3,
                linewidth=1.8,
                alpha=0.95,
            )
        else:
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

        # World label sits centered above both column subplots.
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

    # Shared x-axis styling: log2 ticks at the swept round budgets, labels
    # only on the bottom row.
    for ax in axes[-1, :]:
        ax.set_xlabel("max rounds", fontsize=13)
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(rounds_set)
        ax.set_xticklabels([str(r) for r in rounds_set])

    # Reserve room on the left for the physics-difficulty arrow.
    fig.tight_layout(rect=[0.08, 0.02, 1, 1.0])
    fig.subplots_adjust(left=0.14)

    # Vertical "physics difficulty" arrow on the far left, spanning all
    # world rows top→bottom. Mirrors aggregate_bench.py's horizontal
    # arrow treatment but rotated 90° since rows (not columns) are now
    # the difficulty axis.
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

    out_path = os.path.join(args.out_dir, args.fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
