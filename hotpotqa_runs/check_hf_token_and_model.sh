#!/usr/bin/env bash
# Simple smoke-test for Hugging Face token and model accessibility
# Usage:
#   export HF_API_TOKEN=hf_xxx
#   export HF_MODEL_ID=meta-llama/Llama-3.1-8b-instruct
#   ./check_hf_token_and_model.sh

set -euo pipefail
MODEL_ID=${HF_MODEL_ID:-}
if [[ -z "$HF_API_TOKEN" ]]; then
  echo "HF_API_TOKEN is not set. Export it and try again."
  exit 2
fi
if [[ -z "$MODEL_ID" ]]; then
  echo "HF_MODEL_ID is not set. Set it to a model id (e.g. google/flan-t5-small)"
  exit 2
fi

echo "== whoami =="
curl -s -H "Authorization: Bearer $HF_API_TOKEN" https://huggingface.co/api/whoami-v2 || true

echo -e "\n== model metadata =="
curl -i -s -H "Authorization: Bearer $HF_API_TOKEN" https://huggingface.co/api/models/$MODEL_ID || true

echo -e "\n== try router inference (raw) =="
curl -i -s -X POST "https://router.huggingface.co/models/$MODEL_ID" \
  -H "Authorization: Bearer $HF_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs":"Summarize: The quick brown fox","parameters":{"max_new_tokens":16}}' || true

echo -e "\n== try legacy api-inference (raw) =="
curl -i -s -X POST "https://api-inference.huggingface.co/models/$MODEL_ID" \
  -H "Authorization: Bearer $HF_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs":"Summarize: The quick brown fox","parameters":{"max_new_tokens":16}}' || true
