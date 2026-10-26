"""[sprint tooling] Report figures from analysis_branches outputs. Run after analyze_branches.py."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = Path(__file__).resolve().parent / "results" / "analysis_branches"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
GREY, BLUE = "#9a9a9a", "#2f6fb3"

# Fig 2: cartel formation by demand visibility
F = pd.read_csv(A / "formation.csv")
fig, ax = plt.subplots(figsize=(6.2, 2.9))
labels, vals, cols, ns = [], [], [], []
for _, r in F.sort_values(["model", "demand_visible", "trunk"]).iterrows():
    labels.append(f"{'gpt-oss' if 'gpt' in r.model else 'Qwen3'} | {'bids visible' if r.demand_visible else 'bids hidden'} | {'chat' if r.trunk == 'chat' else 'no chat'}")
    vals.append(100 * r.rate); cols.append(BLUE if r.demand_visible else GREY); ns.append(f"{int(r.cartels)}/{int(r.markets)}")
y = np.arange(len(vals))
ax.barh(y, vals, color=cols)
for i, (v, n) in enumerate(zip(vals, ns)): ax.text(v + 1.5, i, n, va="center", fontsize=8)
ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 115); ax.set_xlabel("Markets reaching cartel pricing by hour 20 (%)")
fig.suptitle("Seeing current bids, not chat, drives cartel formation", x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(A / "fig_formation.png", dpi=220); fig.savefig(A / "fig_formation.pdf")

# Fig 3: effect of each restriction on established cartels (gpt-oss, bids visible)
S = pd.read_csv(A / "summary.csv")
S = S[(S.model == "gpt-oss-120b") & (S.demand_visible) & (S.branch != "B0") & (S.horizon == 12)].copy()
S["name"] = S.label + np.where(S.trunk == "chat", "  [chat cartels]", "  [tacit cartels]")
S = S.sort_values("premium_removed")
fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(S) + 1.0))
y = np.arange(len(S))
ax.errorbar(100 * S.premium_removed, y, xerr=[100 * (S.premium_removed - S.pr_lo), 100 * (S.pr_hi - S.premium_removed)], fmt="o",
            color=BLUE, ecolor=GREY, capsize=2, ms=4)
ax.axvline(0, color="black", lw=.6); ax.axvline(100, color=GREY, lw=.6, ls=":")
ax.text(99, len(S) - 0.6, "cartel fully broken", ha="right", fontsize=7, color=GREY)
ax.set_yticks(y, S.name); ax.set_xlim(-10, 105)
ax.set_xlabel("Markup removed vs. same cartel left alone (%)")
fig.suptitle("Only a committed low-price entrant dents an established cartel", x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(A / "fig_restrictions.png", dpi=220); fig.savefig(A / "fig_restrictions.pdf")
print("ok")
