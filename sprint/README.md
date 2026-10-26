# Prevention beats cure: single restrictions vs. established LLM seller cartels
AI Collusion Research Sprint (Apart Research, Oct 23-25 2026), question 1.5.

## What is pre-existing and what is new
Pre-existing (not ours): everything in this repository at upstream commit `d585a7b` of
poseidon-research/collusion_hackathon, including the `llm-agent-collusion/` double-auction harness (Agrawal et al. 2025,
adapted by Poseidon Research) and `util/`.

Written during the sprint (all marked `[sprint]` in the code):
- `market/market.py`: order-book snapshot per hour, fixing the same-round leak (upstream issue #4); regression test
  `llm-agent-collusion/tests/test_market_snapshot.py`. Optional `buyers_first` timing (rule-based buyers post before sellers).
- `market/agents.py`: `MaverickSeller` (adaptive-margin, ZIP-style, never chats) and `FixedPriceSeller` (aggressive entrant).
- `simulation.py`, `cda_types.py`, `util/misc.py`, `src/resources/model_wrappers.py`: maverick flags, buyer margin,
  token accounting, JSON-mode / reasoning-effort / max-token / retry knobs for self-hosted vLLM endpoints.
- `sprint/branch.py`: trunk-and-fork experiment (grow a market, gate on cartel pricing, fork the exact state into
  one-restriction branches). `sprint/run_branches.sh` launches it.
- `sprint/analyze_branches.py`, `sprint/figures.py`: every number and figure in the report.
- `sprint/run_grid.py`, `sprint/analyze.py`: the earlier condition-grid pilots.
- `sprint/modal_vllm.py`, `sprint/modal_gptoss*.py`: vLLM servers on Modal (Qwen3-30B-A3B-Instruct-2507-FP8, gpt-oss-120b).
- `sprint/mock_llm.py`: offline mock endpoint for pipeline tests.

## Reproduce
1. `modal secret create collusion-vllm-key VLLM_API_KEY=<random>` then `modal deploy sprint/modal_gptoss.py`.
2. Put `GPTOSS_URL`, `MODAL_SERVE_KEY` in `.secrets.env` (see `run_branches.sh`).
3. Main experiment: `BF=1 BRANCHES="B0 B1 B2 B3 B4 B5 B6 B8" ./sprint/run_branches.sh main_bf chat 8 600 601 ...`
   and `BF=1 BRANCHES="B0 B4 B5 B6 B7 B8" ./sprint/run_branches.sh main_bf nochat 8 700 701 ...`
4. `python sprint/analyze_branches.py breakcartel main_bf qwen_bf qwen_hidden long_bf entrant_bf` then `python sprint/figures.py`.
