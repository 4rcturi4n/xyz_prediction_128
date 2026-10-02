# scripts/16_plot_learning_curves.py
#
# Post-processing only, no retraining. One figure per trained model: the five
# folds side by side, train_loss (left axis) and validation MAE in um (right
# axis) per epoch, dotted line at the epoch early stopping kept.
#
# x/y models show val MAE for x and y as separate lines (never averaged in the
# plot, and never computed anywhere); the "best" epoch for x/y models is
# marked using val MAE x alone (a visual aid only, matching the fallback
# already built into extra_plots.plot_learning_curve), not a combined metric.
#
# Reads:  results/<run>/history_all_folds.csv
# Writes: results/<run>/learning_curves.png
#
# Usage:
#   python scripts/16_plot_learning_curves.py                 # every run with a history
#   python scripts/16_plot_learning_curves.py mamba_z_video_only mamba_xy_fused_handcrafted_dense

import os
import sys
import glob

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "results")


def val_series(history):
    """[(column, label)] to plot, and the early-stopping criterion column."""
    if "final_mae_x_phys" in history:
        return [("final_mae_x_phys", "val MAE x"), ("final_mae_y_phys", "val MAE y")], "final_mae_x_phys"
    return [("final_mae_z_phys", "val MAE z")], "val_mae"


def plot_run(run_dir):
    history = pd.read_csv(os.path.join(run_dir, "history_all_folds.csv"))
    series, criterion = val_series(history)
    folds = sorted(history["fold"].unique())

    fig, axes = plt.subplots(1, len(folds), figsize=(4 * len(folds), 3.4))
    for i, (ax, fold) in enumerate(zip(axes, folds)):
        h = history[history["fold"] == fold]
        best = int(h.loc[h[criterion].idxmin(), "epoch"])

        ax.plot(h["epoch"], h["train_loss"], color="C0", label="train loss")
        ax2 = ax.twinx()
        for k, (col, label) in enumerate(series):
            ax2.plot(h["epoch"], h[col], color=f"C{k + 1}", label=label)
        ax.axvline(best, color="gray", linestyle=":", alpha=0.8)

        ax.set_title(f"fold {fold} (best ep {best})", fontsize=10)
        ax.set_xlabel("epoch")
        if i == 0:
            ax.set_ylabel("train_loss", color="C0")
            lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
            ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")
        if i == len(folds) - 1:
            ax2.set_ylabel("val MAE (µm)", color="C1")

    fig.suptitle(os.path.basename(run_dir), fontsize=11)
    fig.tight_layout()
    out = os.path.join(run_dir, "learning_curves.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    names = sys.argv[1:]
    run_dirs = ([os.path.join(RESULTS, n) for n in names] if names else
                sorted(os.path.dirname(p) for p in glob.glob(os.path.join(RESULTS, "*", "history_all_folds.csv"))))
    for d in run_dirs:
        print(plot_run(d))


if __name__ == "__main__":
    main()
