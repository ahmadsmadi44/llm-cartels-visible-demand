#!/bin/bash
source .secrets.env; export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt
for i in $(seq 1 200); do
  out=$(timeout 300 .modalenv/bin/modal deploy sprint/modal_vllm.py 2>&1)
  if echo "$out" | grep -qi "payment method"; then echo "$(date +%T) waiting for card" > deploy_status.txt; sleep 30; else echo "$out" > deploy_status.txt; echo DONE >> deploy_status.txt; break; fi
done
