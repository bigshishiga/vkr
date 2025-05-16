#!/bin/bash
set -e

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  echo "Error: CUDA_VISIBLE_DEVICES is not set. Please set this environment variable to specify which GPUs to use." >&2
  exit 1
fi

python3 src/run_eval.py --data-dir "drag_data/dragbench-dr" --save-dir "regiondrag-baseline" --method regiondrag --start-t 0.5 --ip-adapter --mask-blur-radius 0 --max-pairs 0 --energy-function null --skip-evaluation
python3 src/run_eval.py --data-dir "drag_data/dragbench-sr" --save-dir "regiondrag-baseline" --method regiondrag --start-t 0.5 --ip-adapter --mask-blur-radius 0 --max-pairs 0 --energy-function null --skip-evaluation
