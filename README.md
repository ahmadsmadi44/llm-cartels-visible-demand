# Cartels Built on Visible Demand

Code and results for **"Cartels Built on Visible Demand: Single Restrictions Rarely Break Established LLM Seller Cartels"**
by Ahmad Al Smadi (Lumora) and Mustafa Naqvie (University of Toronto).
AI Collusion Research Sprint (Apart Research, with Poseidon Research and AE Studio), Oct 23 to 25, 2026. Track 1, question 1.5:
which single restriction removes the most collusion at the least cost?

## Findings in one paragraph

Four LLM sellers (gpt-oss-120b or Qwen3-30B-A3B-Instruct-2507) and three rule-based buyers trade in the continuous double
auction of Agrawal et al. (2025). When sellers can see buyers' current bids, every market reaches cartel pricing
(65/65 gpt-oss, 47/47 Qwen); with bids hidden, 3/43 and 0/20 do, and seller chat makes no difference. We save the full
state of each cartel and fork it into copies that differ in one restriction (175 trunks, 983 branches). Chat bans,
message caps, an LLM monitor, temperature, antitrust notices, "compete aggressively" prompts, rebates and fresh
replacement sellers remove almost none of the markup. For gpt-oss, bid-shading buyers remove 14 to 17% at no deployment
cost and a seller committed to an $85 ask removes up to 30% (about $6 per market-hour). Qwen cartels resist everything.

## What is in this repository

| Path | Contents |
|---|---|
| `sprint/` | All experiment code written during the sprint: trunk-and-fork runner (`branch.py`, `run_branches.sh`), analysis (`analyze_branches.py`), figures, talk measure (`talk.py`), Modal vLLM servers, offline mock endpoint. `sprint/README.md` has run instructions. |
| `results/` | Every number in the report: per-trunk (`trunks.csv`) and per-branch (`branches.csv`) results, summaries with bootstrap CIs and Wilcoxon tests (`summary.csv`, `summary.md`, `methods_table.csv`), formation counts, hourly trajectories, talk measure and its validation labels. |
| `figures/` | Figures produced by the analysis scripts. |
| `sprint-changes-to-starter-repo.patch` | Exact diff of our changes to the starter repository's `llm-agent-collusion/` harness. |
| `code-snapshot.zip` | Full runnable snapshot: the starter repository's `llm-agent-collusion/` and `util/` with our changes applied, plus `sprint/`. |

Raw per-market logs (about 900 MB) are not included; they are available from the authors on request.

## Pre-existing work vs. sprint work

**Pre-existing (not ours):** everything at commit `d585a7b` of the sprint starter repository
(`poseidon-research/collusion_hackathon`), including the `llm-agent-collusion/` double-auction harness (Agrawal et al. 2025,
adapted by Poseidon Research) and `util/`.

**Written during the sprint** (marked `[sprint]` in the code; see the patch file):
- Order-book snapshot per hour, fixing a same-hour information leak (upstream issue #4), with a regression test.
- Optional `buyers_first` timing (buyers post before sellers price).
- Maverick (ZIP-style), fixed-price seller and strategic (bid-shading) buyer agents; hidden-rival-asks, lowest-ask
  rebate and prompt-notice options.
- The trunk-and-fork runner with snapshot and resume, all analysis and figure scripts, and the Modal vLLM servers.

## Reproduce

See `sprint/README.md`. In short: deploy a vLLM server on Modal (`sprint/modal_gptoss.py`), put its URL and key in
`.secrets.env`, run `sprint/run_branches.sh`, then `python sprint/analyze_branches.py ...` and `python sprint/figures.py`.

## Talk-measure validation

`sprint/talk.py` flags a message as explicit price coordination when it contains a coordination phrase and a dollar
figure. On 17 six-hour excerpts (`sprint/labelling_key.json`), the rule agreed with blind labels from an independent LLM
judge (Claude, shown only the messages) on 17/17 for coordination on a named price and 16/17 for upward coordination
(`results/labels_llm_judge.json`).

## AI assistance

Claude (Anthropic) wrote and modified the experiment code, ran the experiments, computed the statistics and figures,
and drafted the report text; the authors chose the question, reviewed the design and results, and edited the report.
