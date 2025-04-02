import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import gradio as gr
from PIL import Image

from region_utils.drag import drag, get_drag_data, get_meta_data, drag_copy_paste, drag_id
from region_utils.evaluator import DragEvaluator


def get_args():
    parser = argparse.ArgumentParser(description='Run the drag operation.')

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
        help="If method is 'regiondrag', this is the weight of the noise applied in the inpainting region"
    )

    parser.add_argument(
        '--disable-kv-copy',
        action='store_true',
        help="If passed, do not perform the key-value copy-paste operation in self-attention layers"
    )

    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help="If passed, do not perform the evaluation (saves time if you only want the output images)"
    )

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    args.bench_name = args.data_dir.removeprefix('drag_data/').removesuffix('/') if args.data_dir != 'drag_data' else ""
    args.save_dir = os.path.join('saved', args.save_dir, args.bench_name) if args.save_dir else None

    # Validate arguments
    assert args.data_dir.startswith('drag_data/'), "Data should lie in 'drag_data/'"
    assert args.method in ['regiondrag', 'guidance', 'instantdrag', 'copy', 'id'], "Invalid method"
    if args.save_dir and os.path.exists(args.save_dir):
        raise FileExistsError(f"Save directory {args.save_dir} already exists")

    # Log stuff
    print(f"Using args: {args}")
    print()
    print('Current commit:')
    os.system("git log -1")
    print("*" * 100)

    return args

args = get_args()

evaluator = DragEvaluator()
all_distances = []; all_lpips = []; all_names = []

data_dirs = [dirpath for dirpath, dirnames, _ in os.walk(args.data_dir) if not dirnames]

for data_path in tqdm(data_dirs):
    # Region-based Inputs for Editing
    drag_data = get_drag_data(data_path)
    ori_image = drag_data['ori_image'].copy()

    if args.method in ('regiondrag', 'instantdrag', 'guidance'):
        if args.method == 'regiondrag':
            method = 'Encode then CP'
        elif args.method == 'instantdrag':
            method = 'InstantDrag'
        elif args.method == 'guidance':
            method = 'guidance'

        out_image = drag(drag_data, args.steps, args.start_t, args.end_t, args.noise_scale, args.seed,
                        progress=gr.Progress(), device=args.device, disable_kv_copy=args.disable_kv_copy,
                        method='Encode then CP' if args.method == 'regiondrag' else 'InstantDrag')
    elif args.method == 'copy':
        out_image = drag_copy_paste(drag_data, device=args.device) 
    elif args.method == 'id':
        out_image = drag_id(drag_data, device=args.device)

    if args.save_dir is not None:
        save_name = os.path.join(args.save_dir, data_path.removeprefix(f'drag_data/{args.bench_name}/') + '.png')
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        Image.fromarray(out_image).save(save_name)

    if not args.skip_evaluation:
        # Point-based Inputs for Evaluation
        meta_data_path = os.path.join(data_path, 'meta_data.pkl')
        prompt, _, source, target = get_meta_data(meta_data_path)    

        all_distances.append(evaluator.compute_distance(ori_image, out_image, source, target, method='sd', prompt=prompt))
        all_lpips.append(evaluator.compute_lpips(ori_image, out_image))
        all_names.append(data_path)


if not args.skip_evaluation:
    mean_dist = torch.tensor(all_distances).mean().item() * 100
    mean_lpips = torch.tensor(all_lpips).mean().item() * 100
    print(f'MD: {mean_dist:.4f}\nLPIPS: {mean_lpips:.4f}\n')

    if args.save_dir is not None:
        md = np.array(all_distances)
        lpips = np.array(all_lpips)

        filepath = os.path.join(args.save_dir, 'metrics.npz')
        np.savez(
            filepath, md=md, lpips=lpips, names=all_names,
            mean_md=np.array(mean_dist), mean_lpips=np.array(mean_lpips)
        )
