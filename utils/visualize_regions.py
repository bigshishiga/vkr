import os
from pathlib import Path
import argparse
import torch
from tqdm import tqdm
import gradio as gr

from PIL import Image
from diffusers.utils import make_image_grid
from torchvision.utils import draw_segmentation_masks
from torch.nn.functional import interpolate
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

import sys
sys.path.append('.')

from regiondrag.region_utils.drag import drag, get_drag_data, get_meta_data
from regiondrag.region_utils.evaluator import DragEvaluator

# Setting up the argument parser
parser = argparse.ArgumentParser(description='Run the drag operation.')
parser.add_argument('--data-dir', type=str, default='drag_data')
parser.add_argument('--output-dir', type=str, default='masks')
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir

def main(data_dir, output_dir):
    data_dirs = [dirpath for dirpath, dirnames, _ in os.walk(data_dir) if not dirnames]

    for data_path in tqdm(data_dirs):
        drag_data = get_drag_data(data_path)

        assert len(drag_data['source'].shape) == 2 and len(drag_data['target'].shape) == 2
        assert drag_data['source'].shape == drag_data['target'].shape
        
        def to_mask(arr: np.ndarray, shape: tuple):
            mask = np.zeros(shape, dtype=np.dtype('bool'))
            mask[arr[:, 1], arr[:, 0]] = 1
            return torch.from_numpy(mask).float()

        image = drag_data['ori_image']
        latent_shape = (image.shape[0] // 8, image.shape[1] // 8)
        source_mask = interpolate(
            to_mask(drag_data['source'] // 8, latent_shape).unsqueeze(0).unsqueeze(0),
            size=(image.shape[0], image.shape[1]),
            mode='nearest',
        ).squeeze(0).squeeze(0).bool()
        target_mask = interpolate(
            to_mask(drag_data['target'] // 8, latent_shape).unsqueeze(0).unsqueeze(0),
            size=(image.shape[0], image.shape[1]),
            mode='nearest',
        ).squeeze(0).squeeze(0).bool()

        def fix_holes(mask, it=4):
            dilated_mask = binary_dilation(mask, iterations=it)
            eroded_mask = binary_erosion(dilated_mask, iterations=it)
            return torch.from_numpy(eroded_mask)

        masked_src_image = draw_segmentation_masks(
            torch.Tensor(image).to(torch.uint8).permute(2, 0, 1),
            fix_holes(source_mask),
            # source_mask,
            alpha=0.6,
            colors=(255, 0, 0)
        ).permute(1, 2, 0).numpy()

        masked_target_image = draw_segmentation_masks(
            torch.Tensor(image).to(torch.uint8).permute(2, 0, 1),
            fix_holes(target_mask),
            # target_mask,
            colors=(0, 0, 255)
        ).permute(1, 2, 0).numpy()

        image, masked_src_image, masked_target_image = (
            Image.fromarray(image), Image.fromarray(masked_src_image), Image.fromarray(masked_target_image)
        )

        save_path = f"{output_dir}/{'/'.join(data_path.split('/')[1:])}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        make_image_grid([image, masked_src_image, masked_target_image], 1, 3).save(
            save_path        
        )

if __name__ == '__main__':
    main(os.path.join(data_dir, "dragbench-sr"), output_dir)
    main(os.path.join(data_dir, "dragbench-dr"), output_dir)
