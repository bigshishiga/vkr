#!/bin/bash
set -e

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  echo "Error: CUDA_VISIBLE_DEVICES is not set. Please set this environment variable to specify which GPUs to use." >&2
  exit 1
fi

python3 src/run_eval.py --data-dir "drag_data/dragbench-dr" --save-dir "guidance-baseline-layer=0" --method guidance --guidance-layers 0 --energy-alpha 0 --energy-beta 1
python3 src/run_eval.py --data-dir "drag_data/dragbench-sr" --save-dir "guidance-baseline-layer=0" --method guidance --guidance-layers 0 --energy-alpha 0 --energy-beta 1
