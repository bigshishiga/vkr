import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='baseline')

args, unknown = parser.parse_known_args()
add_args = " ".join(unknown)

print("Passed to run_eval.py:", unknown)

exp_name = args.exp_name
script_name = f"run-{exp_name}.sh"

script = \
f"""#!/bin/bash
set -e

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  echo "Error: CUDA_VISIBLE_DEVICES is not set. Please set this environment variable to specify which GPUs to use." >&2
  exit 1
fi

python3 regiondrag/run_eval.py --data-dir "drag_data/dragbench-dr" --save-dir "{exp_name}" --method guidance {add_args} --energy-alpha 0 --energy-beta 1
python3 regiondrag/run_eval.py --data-dir "drag_data/dragbench-sr" --save-dir "{exp_name}" --method guidance {add_args} --energy-alpha 0 --energy-beta 1
"""

with open(script_name, "w") as f:
    f.write(script)
