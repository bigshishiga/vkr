import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import gradio as gr
from PIL import Image

from region_utils.drag import drag, get_drag_data, get_meta_data, drag_copy_paste, drag_id
from region_utils.evaluator import DragEvaluator

# Setting up the argument parser
parser = argparse.ArgumentParser(description='Run the drag operation.')
parser.add_argument('--data-dir', type=str) # 'drag_data/dragbench-dr' OR 'drag_data/dragbench-sr'
parser.add_argument('--save-dir', type=str, default='saved')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--method', type=str, default='regiondrag') # 'regiondrag' OR 'copy' OR 'id'
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--start-t', type=float, default=0.5)
parser.add_argument('--end-t', type=float, default=0.2)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--noise-scale', type=float, default=1.0)
args = parser.parse_args()

evaluator = DragEvaluator()
all_distances = []; all_lpips = []; all_names = []

save_dir = None if args.save_dir is None else args.save_dir.removesuffix("/")
data_dir = args.data_dir.removesuffix("/")
data_dirs = [dirpath for dirpath, dirnames, _ in os.walk(data_dir) if not dirnames]
device = args.device
method = args.method

bench = data_dir.split("/")[-1]

start_t = args.start_t
end_t = args.end_t
steps = args.steps
noise_scale = args.noise_scale
seed = args.seed

for data_path in tqdm(data_dirs):
    # Region-based Inputs for Editing
    drag_data = get_drag_data(data_path)
    ori_image = drag_data['ori_image']

    if method == 'regiondrag':
        out_image = drag(drag_data, steps, start_t, end_t, noise_scale, seed, progress=gr.Progress(), device=device)
    elif method == 'copy':
        out_image = drag_copy_paste(drag_data, device=device) 
    elif method == 'id':
        out_image = drag_id(drag_data, device=device)

    if save_dir is not None:
        save_name = f"{save_dir}/{data_path.removeprefix('drag_data/')}.png"
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        Image.fromarray(out_image).save(save_name)

    # Point-based Inputs for Evaluation
    meta_data_path = os.path.join(data_path, 'meta_data.pkl')
    prompt, _, source, target = get_meta_data(meta_data_path)    

    all_distances.append(evaluator.compute_distance(ori_image, out_image, source, target, method='sd', prompt=prompt))
    all_lpips.append(evaluator.compute_lpips(ori_image, out_image))
    all_names.append(data_path)


if all_distances:
    mean_dist = torch.tensor(all_distances).mean().item() * 100
    mean_lpips = torch.tensor(all_lpips).mean().item() * 100
    print(f'MD: {mean_dist:.4f}\nLPIPS: {mean_lpips:.4f}\n')

    if save_dir is not None:
        md = np.array(all_distances)
        lpips = np.array(all_lpips)

        filepath = f"{save_dir}/{bench}/metrics.npz"
        np.savez(
            filepath, md=md, lpips=lpips, names=all_names,
            mean_md=np.array(mean_dist), mean_lpips=np.array(mean_lpips)
        )
