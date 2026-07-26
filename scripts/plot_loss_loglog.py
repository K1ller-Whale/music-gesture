#!/usr/bin/env python3
"""
plot_loss_loglog.py -- Parse a Sound-of-Motions training log (the format
train.py prints per step) and plot log(loss) vs. log(step) on a log-log
scale, e.g. to check whether the loss curve follows a power-law decay
(Kaplan et al. 2020 / Hestness et al. 2017 scaling-law style diagnostic)
rather than a true plateau.

Expected log line formats (train.py's stdout, e.g. train_log_phase0.txt):
    epoch <E> step <S>/<TOTAL> loss <L>
    epoch <E> done avg_loss <L>
Any other lines (blank lines, other prints, eval_som.py output mixed into
the same file, etc.) are ignored.

Global step numbering:
    Both epoch and step are 0-indexed (matches train.py's checkpoint "epoch"
    field and the "step 0/<TOTAL>" line printed at the start of every
    epoch). This script computes one monotonically increasing
        global_step = epoch * steps_per_epoch + step
    using steps_per_epoch as the most common "<TOTAL>" value seen in the
    file (and warns if it's inconsistent, e.g. the log spans a config
    change or a curriculum stage boundary with a different dataset size).

    log(0) is undefined, so the x-axis actually plots log(global_step + 1);
    this only shifts step numbering by one and does not change the shape of
    the curve.

Usage:
    python scripts/plot_loss_loglog.py train_log_phase0.txt
    python scripts/plot_loss_loglog.py train_log_phase0.txt --out loss_loglog.png
    python scripts/plot_loss_loglog.py train_log_phase0.txt --mark-epochs --fit
    python scripts/plot_loss_loglog.py train_log_phase0.txt --ln   # natural log instead of log10
"""
from __future__ import annotations

import argparse
import math
import re
import sys

STEP_RE = re.compile(r"epoch\s+(\d+)\s+step\s+(\d+)/(\d+)\s+loss\s+([\-0-9.eE]+)")
EPOCH_DONE_RE = re.compile(r"epoch\s+(\d+)\s+done\s+avg_loss\s+([\-0-9.eE]+)")


def parse_log(path):
    """Return (step_rows, epoch_done, steps_per_epoch).

    step_rows: list of (epoch, step, total, loss) from per-step lines, in
        file order.
    epoch_done: list of (epoch, avg_loss) from "epoch N done avg_loss L"
        lines, in file order.
    steps_per_epoch: the most common "<TOTAL>" seen across step_rows.
    """
    step_rows = []
    epoch_done = []
    totals_seen = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                epoch, step, total, loss = m.groups()
                epoch, step, total, loss = int(epoch), int(step), int(total), float(loss)
                step_rows.append((epoch, step, total, loss))
                totals_seen[total] = totals_seen.get(total, 0) + 1
                continue
            m = EPOCH_DONE_RE.search(line)
            if m:
                epoch, avg_loss = m.groups()
                epoch_done.append((int(epoch), float(avg_loss)))

    if not step_rows:
        raise SystemExit(
            f"no 'epoch <E> step <S>/<TOTAL> loss <L>' lines found in {path} -- "
            f"check the file is train.py's raw stdout/log, not eval_som.py output")

    if len(totals_seen) > 1:
        print(f"[warn] inconsistent steps-per-epoch total(s) across the log: {totals_seen} "
              f"-- using the most common one for global-step numbering "
              f"(a curriculum stage boundary or config change mid-log can cause this)",
              file=sys.stderr)
    steps_per_epoch = max(totals_seen, key=totals_seen.get)

    return step_rows, epoch_done, steps_per_epoch


def to_global_points(step_rows, epoch_done, steps_per_epoch):
    points = [(epoch * steps_per_epoch + step, loss) for epoch, step, _total, loss in step_rows]
    points.sort(key=lambda p: p[0])
    epoch_points = [(epoch * steps_per_epoch + steps_per_epoch - 1, avg_loss)
                    for epoch, avg_loss in epoch_done]
    epoch_points.sort(key=lambda p: p[0])
    return points, epoch_points


def least_squares_slope(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var = sum((x - mean_x) ** 2 for x in xs)
    slope = cov / var if var else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log_file", help="Path to the training log, e.g. train_log_phase0.txt")
    p.add_argument("--out", default=None,
                    help="Save the plot to this path (e.g. loss_loglog.png) instead of showing it "
                         "interactively.")
    p.add_argument("--ln", action="store_true", help="Use natural log instead of log10.")
    p.add_argument("--mark-epochs", action="store_true",
                    help="Overlay each epoch's avg_loss as a red marker line and draw light "
                         "vertical gridlines at epoch boundaries.")
    p.add_argument("--fit", action="store_true",
                    help="Fit a straight line (least squares) to log(step) vs log(loss) and "
                         "report its slope -- the exponent of an assumed power-law decay "
                         "loss ~ step^slope.")
    args = p.parse_args()

    step_rows, epoch_done, steps_per_epoch = parse_log(args.log_file)
    points, epoch_points = to_global_points(step_rows, epoch_done, steps_per_epoch)
    print(f"parsed {len(points)} per-step point(s), {len(epoch_points)} epoch-avg point(s), "
          f"steps_per_epoch={steps_per_epoch}")

    log = math.log if args.ln else math.log10
    log_label = "ln" if args.ln else "log10"

    xs, ys = [], []
    skipped = 0
    for step, loss in points:
        if loss <= 0:
            skipped += 1
            continue  # log undefined for non-positive loss; shouldn't happen for this loss_mode
        xs.append(log(step + 1))  # +1 avoids log(0) at the very first global step
        ys.append(log(loss))
    if skipped:
        print(f"[warn] skipped {skipped} point(s) with non-positive loss (log undefined)", file=sys.stderr)

    try:
        import matplotlib
        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required to plot -- pip install matplotlib")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xs, ys, ".", markersize=2, alpha=0.5, label="per-step loss")

    if args.mark_epochs and epoch_points:
        ex, ey = [], []
        for s, l in epoch_points:
            if l <= 0:
                continue
            ex.append(log(s + 1))
            ey.append(log(l))
        ax.plot(ex, ey, "o-", color="red", markersize=4, linewidth=1, label="epoch avg_loss")
        for s, _ in epoch_points:
            ax.axvline(log(s + 1), color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    if args.fit:
        if len(xs) < 2:
            print("[warn] not enough points to fit a line", file=sys.stderr)
        else:
            slope, intercept = least_squares_slope(xs, ys)
            fit_ys = [slope * x + intercept for x in xs]
            ax.plot(xs, fit_ys, "-", color="black", linewidth=1.5,
                    label=f"least-squares fit (slope={slope:.4f})")
            print(f"least-squares fit: {log_label}(loss) = {slope:.4f} * {log_label}(step+1) + {intercept:.4f}")
            print(f"implied power-law exponent: loss ~ step^{slope:.4f}")

    ax.set_xlabel(f"{log_label}(step + 1)")
    ax.set_ylabel(f"{log_label}(loss)")
    ax.set_title(f"Training loss (log-log): {args.log_file}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
