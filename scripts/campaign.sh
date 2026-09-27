#!/usr/bin/env bash
# Whole-model controlled replay. Refuses existing outputs through the serving CLI.
set -euo pipefail
CAMPAIGN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN_PYTHON="${KERNELSCOPE_PYTHON:-$CAMPAIGN_ROOT/.venv/bin/python}"
CAMPAIGN_RESULTS="${KERNELSCOPE_RESULTS:-$CAMPAIGN_ROOT/../kernelscope/results}"
CAMPAIGN_OUT="${1:-$CAMPAIGN_RESULTS/serve_4090/graduation_$(date -u +%Y%m%dT%H%M%SZ)}"
CAMPAIGN_REPEATS="${REPEATS:-5}"
cd "$CAMPAIGN_ROOT"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
"$CAMPAIGN_PYTHON" -m kernelscope.cli serve doctor
CAMPAIGN_EXIT=0
for scenario in uniform ragged arrivals; do
  if [[ -f "$CAMPAIGN_OUT/$scenario/manifest.json" ]]; then
    if "$CAMPAIGN_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("status")=="complete" else 1)' "$CAMPAIGN_OUT/$scenario/manifest.json"; then
      printf 'Keeping completed experiment: %s\n' "$CAMPAIGN_OUT/$scenario"
      if ! "$CAMPAIGN_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("tokens_equivalent") is True else 1)' "$CAMPAIGN_OUT/$scenario/manifest.json"; then
        CAMPAIGN_EXIT=1
      fi
      continue
    fi
    printf 'Incomplete experiment already exists; choose a fresh campaign path: %s\n' "$CAMPAIGN_OUT/$scenario" >&2
    exit 1
  fi
  if "$CAMPAIGN_PYTHON" -m kernelscope.cli serve run \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --scenario "scenarios/graduation_$scenario.yaml" \
    --policy heuristic --policy fixed:8 --policy table:demo_data/dispatch_paged_cold.csv --policy model \
    --machine machines/rtx4090.json --params models/rtx4090.json \
    --kv-gib "${KV_GIB:-10}" --repeats "$CAMPAIGN_REPEATS" \
    --warmup-runs 1 --warmup-steps 2 --out "$CAMPAIGN_OUT/$scenario"; then
    :
  else
    # A completed negative equivalence result must not erase evidence or stop
    # independent scenarios. Operational failures still stop the campaign.
    if [[ -f "$CAMPAIGN_OUT/$scenario/manifest.json" ]] && \
       "$CAMPAIGN_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("status")=="complete" else 1)' "$CAMPAIGN_OUT/$scenario/manifest.json"; then
      CAMPAIGN_EXIT=1
      printf 'Recorded an equivalence failure; continuing independent scenarios.\n' >&2
    else
      exit 1
    fi
  fi
done
printf 'Recorded campaign: %s\n' "$CAMPAIGN_OUT"
exit "$CAMPAIGN_EXIT"
