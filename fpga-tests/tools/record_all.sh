#!/usr/bin/env bash
# record_all.sh: full traces for every suite (run from hardware-projects/, after
# sourcing an env with OSS CAD Suite first on PATH and RISC-V GCC after it).
cd "$(dirname "$0")/.."
for s in systolic_array accel_axil riscv_rvtests riscv gpu_v1 gpu_v2 ew_v1 ew_v2; do
    echo "=== $s $(date +%T)"
    if [ "$s" = riscv ]; then export RAND_PROGRAMS=${RAND_PROGRAMS:-1000}; fi
    python tools/record.py "$s" > "traces_$s.log" 2>&1
    echo "exit $? $(date +%T)"; grep -E "passed|failed|\[record\].*cycles" "traces_$s.log"
done
echo ALL-DONE
