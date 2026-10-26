#!/bin/bash
# [sprint tooling] launch break-the-cartel experiments: one process per trunk seed, P in parallel
# usage: sprint/run_branches.sh <grid> <trunk chat|nochat> <P> <seeds...>
GRID=$1; TRUNK=$2; P=$3; shift 3
source .secrets.env
export LLM_BASE_URL=${LLM_BASE_URL:-$GPTOSS_URL/v1} LLM_API_KEY=$MODAL_SERVE_KEY LLM_MAX_TOKENS=3000 LLM_MAX_RETRIES=5 LLM_BACKOFF_SCALE=3 LLM_REASONING_EFFORT=${LLM_REASONING_EFFORT:-low}
mkdir -p sprint/results/$GRID/logs
printf "%s\n" "$@" | xargs -P $P -I{} bash -c "cd llm-agent-collusion && uv run python ../sprint/branch.py --seed {} --trunk $TRUNK --K ${K:-20} --M ${M:-12} --reps 2 --workers ${W:-4} --threshold ${TH:-0.5} --branches ${BRANCHES:-B0 B1 B2 B3 B4 B5 B6} --model ${MODEL:-gpt-oss-120b} ${BF:+--buyers_first} --out ../sprint/results/$GRID > ../sprint/results/$GRID/logs/$TRUNK-{}.log 2>&1"
