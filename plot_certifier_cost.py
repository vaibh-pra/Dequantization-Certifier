"""Figure for W4 (R3.4): measured certifier cost scaling.
Reads results/certifier_cost.csv, draws a 3-panel figure:
  (a) DLA closure time vs qubit count, one series per ceiling
  (b) fit time vs basis size B (log-log), measured points + B^2 guide
  (c) peak fit memory vs B
Output: results/certifier_cost.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "certifier_cost.csv")
OUT = os.path.join(HERE, "..", "results", "certifier_cost.png")

rows = list(csv.DictReader(open(CSV)))
dla = [r for r in rows if r["part"] == "dla"]
fit = [r for r in rows if r["part"] == "surrogate"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "lines.linewidth": 1.8, "lines.markersize": 6,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})
BLUE, RED, MUTED = "#2a78d6", "#e34948", "#666666"

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))

# (a) DLA closure time vs n, per cap
ax = axes[0]
for cap, marker, color in [(16, "s", MUTED), (64, "o", BLUE), (256, "^", RED)]:
    pts = sorted((int(r["n"]), float(r["time_s"]), r["dim"])
                 for r in dla if int(r["cap"]) == cap)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker + "-", color=color,
            label=f"ceiling {cap}")
    for nq, t, dim in pts:
        ax.annotate(f"$g$={dim}", (nq, t), textcoords="offset points",
                    xytext=(0, 6), fontsize=7.5, color=color, ha="center")
ax.set_yscale("log")
ax.set_xlabel("qubit count $n$ (ZZ feature map)")
ax.set_ylabel("Lie-closure wall time (s)")
ax.set_title("(a) DLA closure", fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")

# (b) fit time vs B, log-log
ax = axes[1]
pts = sorted((int(r["B"]), float(r["fit_time_s"])) for r in fit)
ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-", color=BLUE,
          label="measured")
guide = np.array([10, 10 ** 4.5])
ax.loglog(guide, (guide / pts[0][0]) ** 2 * pts[0][1], ":", color=MUTED,
          label=r"$\propto B^{2}$ guide")
for B, t in pts:
    ax.annotate(f"{B:,}", (B, t), textcoords="offset points", xytext=(4, 4),
                fontsize=7.5, color=MUTED)
ax.set_xlabel(r"basis size $B=(2K+1)^{C}$")
ax.set_ylabel("enumerate + least-squares fit time (s)")
ax.set_title("Surrogate fit", fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")

# (c) peak memory vs B
ax = axes[2]
pts = sorted((int(r["B"]), float(r["peak_mb"])) for r in fit)
ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-", color=RED)
for B, m in pts:
    ax.annotate(f"{B:,}", (B, m), textcoords="offset points", xytext=(4, 4),
                fontsize=7.5, color=MUTED)
ax.set_xlabel(r"basis size $B=(2K+1)^{C}$")
ax.set_ylabel("peak fit memory (MB)")
ax.set_title("Fit memory", fontsize=11)

fig.tight_layout()
fig.savefig(OUT := os.path.join(HERE, "..", "results", "certifier_cost.png"))
print("wrote", OUT)
