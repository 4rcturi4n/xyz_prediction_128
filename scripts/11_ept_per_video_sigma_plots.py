# scripts/17_ept_per_video_sigma_plots.py
#
# Post-processing only, no retraining. Replaces the fixed 0.05 um EPT
# threshold with each video's OWN measurement noise sigma_i (std of its 21
# repeat ImageJ measurements: x_error / y_error / z_error in the dataset CSV).
#
#   EPT_i = earliest frame t, BEFORE the last frame, such that
#           |pred_i(t') - true_i| < sigma_i for every t' >= t
#           (x/y models: x and y must both stay within their own sigma)
#
# A video that only gets inside its sigma at the final frame (100%), or never,
# is NOT stabilised: it is counted separately and left out of the EPT
# histogram and the mean/median EPT, instead of being filed as EPT = 100%.
#
# Reads:  results/<run>/fold_N/val_predictions_per_timestep.csv
# Writes (existing files are left untouched):
#   results/<run>/fold_N/error_vs_progress_sigma.png   error / own sigma vs video progress
#   results/<run>/fold_N/ept_distribution_sigma.png    EPT histogram, stabilised videos only
#   results/<run>/ept_summary_sigma.csv                one row per validation video, all folds
#   results/<run>/ept_summary_sigma.json               % stabilised, mean/median EPT, per fold
#
# Usage:
#   python scripts/17_ept_per_video_sigma_plots.py                     # every run
#   python scripts/17_ept_per_video_sigma_plots.py mamba_xy_video_only

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data", "processed")


def load_sigma():
    xy = pd.read_csv(os.path.join(DATA, "video_xy_dataset.csv")).drop_duplicates("video_id").set_index("video_id")
    z = pd.read_csv(os.path.join(DATA, "video_z_dataset.csv")).drop_duplicates("video_id").set_index("video_id")
    return {"x": xy["x_error"].to_dict(), "y": xy["y_error"].to_dict(), "z": z["z_error"].to_dict()}


def ept_before_last(within):
    """Earliest t < last frame from which `within` stays True to the end, else None."""
    ok_from = np.logical_and.accumulate(within[::-1])[::-1]  # ok_from[t]: within for all t' >= t
    hits = np.flatnonzero(ok_from[:-1])
    return int(hits[0]) if len(hits) else None


def per_video(df, axes, sigma):
    rows = []
    for vid, sub in df.groupby("video_id", sort=False):
        sub = sub.sort_values("frame_idx")
        pct = sub["frame_pct"].values
        row = {"video_id": vid}
        within_all = np.ones(len(sub), dtype=bool)
        for a in axes:
            s = sigma[a][vid]
            err = sub[f"abs_err_{a}_phys"].values
            within = err < s
            within_all &= within
            t = ept_before_last(within)
            row.update({f"sigma_{a}": s, f"final_err_{a}": err[-1], f"final_within_{a}": bool(within[-1]),
                        f"ept_{a}_pct": float(pct[t]) if t is not None else np.nan})
        t = ept_before_last(within_all)
        row["stabilised"] = t is not None
        row["ept_pct"] = float(pct[t]) if t is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_error_vs_progress(df, axes, sigma, fold, out_path):
    fig, axs = plt.subplots(len(axes), 1, figsize=(8, 4.5 * len(axes)), sharex=True, squeeze=False)
    for ax, a in zip(axs[:, 0], axes):
        d = df.assign(rel=df[f"abs_err_{a}_phys"] / df["video_id"].map(sigma[a]))
        g = d.groupby("frame_pct")["rel"]
        pct = g.mean().index.values
        ax.plot(pct, g.mean().values, color="steelblue", lw=2, marker="o", ms=3, label="mean |error| / own σ")
        ax.plot(pct, g.median().values, color="steelblue", lw=1.5, ls="--", label="median |error| / own σ")
        ax.axhline(1.0, color="gray", ls=":", label="= own measurement σ")
        ax.set_ylabel(f"{a} error / σ_video")
        ax.set_title(f"Fold {fold} — {a} error relative to each video's own σ vs. video progress")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(pct, (g.apply(lambda r: (r < 1).mean()) * 100).values, color="tomato", lw=2,
                 label="% videos within own σ")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("% videos within own σ", color="tomato")
        lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")
    axs[-1, 0].set_xlabel("Video progress (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ept_distribution(vdf, axes, fold, out_path):
    stab = vdf[vdf["stabilised"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(stab):
        ax.hist(stab["ept_pct"], bins=np.linspace(0, 100, 11), color="steelblue", edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("EPT — video progress when prediction stabilised (%)")
    ax.set_ylabel("Number of videos")
    which = " and ".join(axes)
    ax.set_title(f"Fold {fold} — EPT, threshold = each video's own σ ({which})\n"
                 f"stabilised before the last frame: {len(stab)}/{len(vdf)} videos "
                 f"(not stabilised: {len(vdf) - len(stab)})", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize(vdf):
    stab = vdf[vdf["stabilised"]]
    return {"n_videos": int(len(vdf)), "n_stabilised": int(len(stab)),
            "pct_stabilised": float(100 * len(stab) / len(vdf)) if len(vdf) else float("nan"),
            "mean_ept_pct_stabilised": float(stab["ept_pct"].mean()) if len(stab) else None,
            "median_ept_pct_stabilised": float(stab["ept_pct"].median()) if len(stab) else None}


def process_run(run_dir, sigma):
    fold_dirs = sorted(glob.glob(os.path.join(run_dir, "fold_*")))
    all_rows, per_fold = [], {}
    for fd in fold_dirs:
        path = os.path.join(fd, "val_predictions_per_timestep.csv")
        if not os.path.exists(path):
            continue
        fold = int(os.path.basename(fd).split("_")[1])
        df = pd.read_csv(path)
        axes = ["x", "y"] if "abs_err_x_phys" in df else ["z"]
        vdf = per_video(df, axes, sigma).assign(fold=fold)
        plot_error_vs_progress(df, axes, sigma, fold, os.path.join(fd, "error_vs_progress_sigma.png"))
        plot_ept_distribution(vdf, axes, fold, os.path.join(fd, "ept_distribution_sigma.png"))
        all_rows.append(vdf)
        per_fold[fold] = summarize(vdf)
    if not all_rows:
        return None
    vdf = pd.concat(all_rows, ignore_index=True)
    vdf.to_csv(os.path.join(run_dir, "ept_summary_sigma.csv"), index=False)
    summary = {"threshold": "each video's own measurement sigma",
               "stabilised_rule": "within sigma from some frame before the last one to the end",
               "all_folds": summarize(vdf), "folds": per_fold}
    with open(os.path.join(run_dir, "ept_summary_sigma.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary["all_folds"]


def main():
    sigma = load_sigma()
    names = sys.argv[1:]
    run_dirs = ([os.path.join(RESULTS, n) for n in names] if names else sorted(
        {os.path.dirname(os.path.dirname(p)) for p in
         glob.glob(os.path.join(RESULTS, "*", "fold_*", "val_predictions_per_timestep.csv"))}))
    for d in run_dirs:
        s = process_run(d, sigma)
        if s:
            med = s["median_ept_pct_stabilised"]
            print(f"{os.path.basename(d):55s} stabilised {s['n_stabilised']:3d}/{s['n_videos']:3d} "
                  f"({s['pct_stabilised']:5.1f}%)  median EPT {'-' if med is None else f'{med:5.1f}%'}")


if __name__ == "__main__":
    main()
