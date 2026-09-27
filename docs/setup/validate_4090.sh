#!/usr/bin/env bash
# Full validation, after setup_accelsim_4090.sh {toolchain,build,gate,config}.
# Each invocation uses new artifact directories so repeated runs cannot mix.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/skkai/miniforge3/envs/gradkernel/bin/python
RUN=${SIM_VALIDATION_RUN:-$(date +%Y%m%d-%H%M%S)}
WORK=/home/skkai/accelsim/kernelscope_sim
SIM_RESULTS="$REPO/results/sim_4090/$RUN"
HW_RESULTS=${HW_RESULTS:-$REPO/results/hw_4090_simtrack/$RUN}
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0
# Fail before tracing if the installed attention library cannot load.
"$PY" -c 'import torch, flash_attn; print(torch.__version__, flash_attn.__version__)'
check_idle() {
    "$PY" - <<'PY'
import csv, io, subprocess
def query(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True)
raw = query('--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits')
rows = list(csv.reader(io.StringIO(raw)))
util = int(next(row[1] for row in rows if int(row[0]) == 0))
apps = query('--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits')
# A persistent desktop rerun viewer is visible as C+G even while idle.
foreign = [r for r in csv.reader(io.StringIO(apps)) if len(r) > 1 and 'rerun_cli/rerun' not in r[1]]
if util >= 5 or foreign:
    raise SystemExit(f'BLOCKED: require idle GPU, util={util}%, compute apps={foreign}')
print(f'GPU idle: {util}%; desktop processes remain present')
PY
}
check_results() {
    "$PY" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
path, track = Path(sys.argv[1]) / 'summaries.jsonl', sys.argv[2]
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
rows = [row for row in rows if 'status' in row]  # hygiene warnings are separate JSON records
assert rows, f'No summaries in {path}'
for row in rows:
    assert row['status'] == 'ok', row
    if track == 'sim':
        assert row.get('sim') and all(v == 'ok' for v in row['sim'].values()), row
    else:
        assert row.get('check_ok') is True, row
print(f'{track}: verified {len(rows)} completed cells')
PY
}
sim() {
    "$PY" -m kernelscope.cli simsweep "$@" --results "$SIM_RESULTS" \
        --accelsim-root /home/skkai/accelsim/accel-sim-framework \
        --work-dir "$WORK" --arch SM89_RTX4090 --device-index 0 --python "$PY" --sim-jobs 4
    check_results "$SIM_RESULTS" sim
}
hw() {
    check_idle
    "$PY" -m kernelscope.cli sweep "$@" --results "$HW_RESULTS" --python "$PY"
    check_results "$HW_RESULTS" hw
}
check_idle
SIM_RESULTS="$SIM_RESULTS/smoke" sim --grid grids/sim_smoke.yaml --plugins fa2,flashdecoding --variants base
# All seven variants on exactly the three requested attention cells.
for cell in B1_Lq1_Lkv1024 B1_Lq1_Lkv8192 B16_Lq1_Lkv1024; do
    key="decode_${cell}_Hq32_Hkv8_d128_float16_causal"
    hw --workload "$key" --plugins fa2,flashdecoding
    sim --workload "$key" --plugins fa2,flashdecoding \
        --variants base,l2_x2,l2_half,bw_x2,bw_half,sm_x2,sm_half
done
key=decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float32_causal
hw --workload "$key" --plugins naive_exec
sim --workload "$key" --plugins naive_exec --variants base
"$PY" -m kernelscope.cli report --results "$HW_RESULTS" "$SIM_RESULTS" \
    --clock-mhz 2520 --out "$SIM_RESULTS/validation.csv"
echo "Hardware: $HW_RESULTS"
echo "Simulation and validation table: $SIM_RESULTS"
