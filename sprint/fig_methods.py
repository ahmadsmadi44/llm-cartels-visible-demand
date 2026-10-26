"""[sprint tooling] Methods chart (markup removed per method) and cost-effectiveness chart, gpt-oss, 12h, bids visible."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = Path(__file__).resolve().parent / "results" / "analysis_branches"
S = pd.read_csv(A / "summary.csv")
S = S[(S.model == "gpt-oss-120b") & (S.demand_visible) & (S.horizon == 12) & (S.branch != "B0")]
names = {"B1": "Ban chat", "B2": "50-char message cap", "B3": "Oversight + monitor", "B4": "Higher temperature",
         "B5": "Adaptive maverick bot", "B6": "Fresh AI seller swap", "B7": "Open a chat channel", "B8": "Hide current bids",
         "B10": "Fixed seller at $85", "E90": "Fixed seller at $90", "E95": "Fixed seller at $95", "E97": "Fixed seller at $97",
         "B11": "Strategic buyers", "B12": "Hide rivals' asks", "B13": "Antitrust notice", "B14": "'Compete aggressively' prompt",
         "B15": "$1 lowest-ask rebate"}
rows = []
for b, n in names.items():
    g = S[S.branch == b]
    if g.empty: continue
    get = lambda tk, col: g[g.trunk == tk][col].mean() if (g.trunk == tk).any() else np.nan
    rows.append({"code": b, "name": n, "chat": 100 * get("chat", "premium_removed"), "tacit": 100 * get("nochat", "premium_removed"),
                 "cost": np.nanmean([get("chat", "deploy_cost_per_hour"), get("nochat", "deploy_cost_per_hour")]),
                 "welfare": np.nanmean([get("chat", "welfare_change_per_hour"), get("nochat", "welfare_change_per_hour")]),
                 "n_chat": get("chat", "n_cartels"), "n_tacit": get("nochat", "n_cartels")})
D = pd.DataFrame(rows); D["key"] = D[["chat", "tacit"]].mean(axis=1, skipna=True); D = D.sort_values("key")
D.round(2).to_csv(A / "methods_table.csv", index=False); print(D.round(2).to_string(index=False))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(D) + 1.4)); y = np.arange(len(D)); h = 0.38
ax.barh(y + h / 2, D.tacit.fillna(0), h, color="#2f6fb3", label="Tacit cartels (no chat)")
ax.barh(y - h / 2, D.chat.fillna(0), h, color="#9fc0e3", label="Chat cartels")
for i, r in enumerate(D.itertuples()):
    for v, off in ((r.tacit, h / 2), (r.chat, -h / 2)):
        ax.text((0 if np.isnan(v) else max(v, 0)) + 0.6, i + off, "n/a" if np.isnan(v) else f"{v:.1f}%", va="center", fontsize=7, color="#444")
    ax.text(47, i, f"${r.cost:.2f}/h", va="center", fontsize=7.5, color="#222", ha="right")
ax.text(47, len(D) - 0.3, "cost to deploy", ha="right", fontsize=7.5, color="#666")
ax.set_yticks(y, D.name); ax.axvline(0, color="black", lw=.6); ax.set_xlim(-2, 47)
ax.set_xlabel("Share of the cartel's price markup removed after 12 hours (%)")
ax.legend(loc="lower right", bbox_to_anchor=(0.86, 0), frameon=False)
fig.suptitle("Strategic buyers and a fixed low-price seller are the only methods that dent a cartel", x=0.01, ha="left", fontsize=10.5)
fig.text(0.01, 0.005, "gpt-oss-120b sellers; vs. the same cartel left alone; 8 to 16 cartels per method, 2 replicates each; cost = $ per market-hour paid by whoever deploys it",
         fontsize=6.5, color="#666")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(A / "fig_methods.png", dpi=220)
