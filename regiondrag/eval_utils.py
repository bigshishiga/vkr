import subprocess
import logging
from pythonjsonlogger import jsonlogger
import os
import argparse
import shutil

def get_git_commit_hash():
    try:
        # Run git command to get the current commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        # Return the hash (removing any trailing whitespace)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "Unable to get git commit hash"

def has_uncommitted_changes():
    try:
        # Run git status and check if there are changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        # If the output is not empty, there are uncommitted changes
        return len(result.stdout.strip()) > 0
    except subprocess.CalledProcessError:
        # Handle errors (e.g., not a git repository)
        print("Error checking git status")
        return False

def setup_logging(filename):
    logger = logging.getLogger()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logHandler = logging.FileHandler(filename)

    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def process_log_record(self, log_record):
            # Round all floats to 2 decimal places
            for key, value in log_record.items():
                if isinstance(value, float):
                    log_record[key] = round(value, 3)
            return super().process_log_record(log_record)

    formatter = CustomJsonFormatter()
    logHandler.setFormatter(formatter)

    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)

def parse_all_args(argv=None):
    args = {}
    i = 0

    while i < len(argv):
        if argv[i].startswith('--'):
            key = argv[i][2:]
            if i + 1 < len(argv) and not argv[i+1].startswith('--'):
                args[key] = argv[i+1]
                i += 2
            else:
                args[key] = True
                i += 1
        elif argv[i].startswith('-'):
            key = argv[i][1:]
            if i + 1 < len(argv) and not argv[i+1].startswith('-'):
                args[key] = argv[i+1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            if 'positional' not in args:
                args['positional'] = []
            args['positional'].append(argv[i])
            i += 1
    
    return args

def parse_list(arg):
    """Parse a comma-separated string into a list of integers."""
    if arg:
        return [int(item) for item in arg.split(',')]
    return []


def get_args():
    parser = argparse.ArgumentParser(description='Run the drag operation.')

    parser.add_argument(
        '--sd-version',
        type=str,
        default='v1-5',
        help="Version of the Stable Diffusion model to use. Options: 'v1-5', 'v1-5-inpainting', 'v2-1', 'xl'"
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help="Path to the evaluation data directory. Should start with 'drag_data/'"
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help="Path to the save directory (i.e. name of the run). If not provided, results will not be saved."
    )

    parser.add_argument(
        '--method',
        type=str,
        default='regiondrag',
        choices=['regiondrag', 'guidance', 'instantdrag', 'copy', 'id'],
        help="Method to use for the drag operation. Options: 'regiondrag', 'guidance', 'instantdrag', 'copy', 'id'"
    )

    parser.add_argument(
        '--sampler',
        choices=["ddim", "ddpm"],
        default="ddpm",
        help="Sampler to use for the drag operation. Options: 'ddim', 'ddpm'"
    )

    parser.add_argument(
        '--start-t',
        type=float,
        default=0.5,
        help="Timestep to which the inversion is applied (e.g. 0.5 for 50% of the way through the diffusion process)"
    )

    parser.add_argument(
        '--end-t',
        type=float,
        default=0.2,
        help="Timestep at which the editing is terminated (e.g. 0.2 for 20% of the way through the diffusion process)"
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=20,
        help="DDIM inversion steps"
    )

    parser.add_argument(
        '--noise-scale',
        type=float,
        default=1.0,
        help="Weight of the noise applied in the inpainting region"
    )

    parser.add_argument(
        '--guidance-mask-radius',
        type=int,
        default=None,
        help="Radius of the guidance mask. If None, guidance masking is disabled"
    )

    parser.add_argument(
        '--guidance-weight',
        type=float,
        default=1.0,
        help="If method is 'guidance', this is the weight of the energy function in denoising step"
    )

    parser.add_argument(
        '--guidance-layers',
        type=parse_list,
        default="1,2",
        help="If method is 'guidance', this is the comma-separated list of layers at which the guidance is applied (e.g., '1,2')"
    )

    parser.add_argument(
        '--energy-function',
        type=str,
        default=None,
        help="Energy function to use for the guidance. Options: 'dragon'"
    )

    parser.add_argument(
        '--sim-function',
        type=str,
        default=None,
        help="Similarity function to use for the guidance. Options: 'cosine_local'"
    )

    parser.add_argument(
        '--eps-clipping',
        type=float,
        default=None,
        help="If passed, clip the guidance norm to the denoise norm multiplied by this value"
    )

    parser.add_argument(
        '--disable-kv-copy',
        action='store_true',
        help="If passed, do not perform the key-value copy-paste operation in self-attention layers"
    )

    parser.add_argument(
        '--ip-adapter',
        action='store_true',
        help="If passed, use IP-Adapter"
    )

    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help="If passed, do not perform the evaluation (saves time if you only want the output images)"
    )

    parser.add_argument(
        '--override',
        action='store_true',
        help="If passed, override saving directory"
    )

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args, unknown = parser.parse_known_args()
    unknown = parse_all_args(unknown)

    args.bench_name = args.data_dir.removeprefix('drag_data/').removesuffix('/') if args.data_dir != 'drag_data' else ""
    args.save_dir = os.path.join('saved', args.save_dir, args.bench_name) if args.save_dir else None
    args.commit_hash = get_git_commit_hash()

    energy_args= {
        k.removeprefix("energy-"): v
        for k, v in unknown.items()
        if k.startswith("energy-")
    }
    sim_args = {
        k.removeprefix("sim-"): v
        for k, v in unknown.items()
        if k.startswith("sim-")
    }

    # Validate arguments
    assert args.data_dir.startswith('drag_data/'), "Data should lie in 'drag_data/'"
    assert args.method in ['regiondrag', 'guidance', 'instantdrag', 'copy', 'id'], "Invalid method"
    if args.save_dir and os.path.exists(args.save_dir):
        if args.override:
            shutil.rmtree(args.save_dir)
        else:
            raise FileExistsError(f"Save directory {args.save_dir} already exists")

    # Log stuff
    print(f"Using args: {args}", end="\n\n")
    print(f"Args for the energy function: {energy_args}", end="\n\n")
    print(f"Args for the similarity function: {sim_args}", end="\n\n")
    print("*" * 100)

    return args, energy_args, sim_args
