import os
import argparse
import torch
from tqdm import tqdm
import gradio as gr

from region_utils.drag import drag, get_drag_data, get_meta_data
from region_utils.evaluator import DragEvaluator

# Setting up the argument parser
parser = argparse.ArgumentParser(description='Run the drag operation.')
parser.add_argument('--data_dir', type=str, default='drag_data/dragbench-dr/') # OR 'drag_data/dragbench-sr/'
args = parser.parse_args()

evaluator = DragEvaluator()
all_distances = []; all_lpips = []

data_dir = args.data_dir
data_dirs = [dirpath for dirpath, dirnames, _ in os.walk(data_dir) if not dirnames]

start_t = 0.5
end_t = 0.2
steps = 20
noise_scale = 1.
seed = 42

for data_path in tqdm(data_dirs):
    # Region-based Inputs for Editing
    drag_data = get_drag_data(data_path)
    # ori_image = drag_data['ori_image']
    # out_image = drag(drag_data, steps, start_t, end_t, noise_scale, seed, progress=gr.Progress())

    # # Point-based Inputs for Evaluation
    # meta_data_path = os.path.join(data_path, 'meta_data.pkl')
    # prompt, _, source, target = get_meta_data(meta_data_path)    

    # all_distances.append(evaluator.compute_distance(ori_image, out_image, source, target, method='sd', prompt=prompt))
    # all_lpips.append(evaluator.compute_lpips(ori_image, out_image))

    assert len(drag_data['source'].shape) == 2 and len(drag_data['target'].shape) == 2
    assert drag_data['source'].shape == drag_data['target'].shape

    from PIL import Image
    from diffusers.utils import make_image_grid
    from torchvision.utils import draw_segmentation_masks
    import numpy as np

    def expand(arr: np.ndarray, shape: tuple):
        assert shape[0] == shape[1]
        lst = [arr + np.array([i, j]) for i in range(8) for j in range(8)]
        result = np.stack(lst, axis=0).reshape(-1, 2)
        return np.minimum(result, shape[0] - 1)

    def to_mask(arr: np.ndarray, shape: tuple):
        mask = np.zeros(shape, dtype=np.dtype('bool'))
        mask[arr[:, 1], arr[:, 0]] = 1
        return mask

    image = drag_data['ori_image']
    source_mask = to_mask(expand(drag_data['source'], image.shape[:2]), image.shape[:2])
    target_mask = to_mask(expand(drag_data['target'], image.shape[:2]), image.shape[:2])

    masked_src_image = draw_segmentation_masks(
        torch.Tensor(image).to(torch.uint8).permute(2, 0, 1),
        torch.BoolTensor(source_mask),
        alpha=0.6,
        colors=(255, 0, 0)
    ).permute(1, 2, 0).numpy()

    masked_target_image = draw_segmentation_masks(
        torch.Tensor(image).to(torch.uint8).permute(2, 0, 1),
        torch.BoolTensor(target_mask),
        colors=(0, 0, 255)
    ).permute(1, 2, 0).numpy()

    image, masked_src_image, masked_target_image = (
        Image.fromarray(image), Image.fromarray(masked_src_image), Image.fromarray(masked_target_image)
    )
    make_image_grid([image, masked_src_image, masked_target_image], 1, 3).save(f'results/{data_path.split("/")[-1]}.png')

if all_distances:
    mean_dist = torch.tensor(all_distances).mean().item()
    mean_lpips = torch.tensor(all_lpips).mean().item()
    print(f'MD: {mean_dist:.4f}\nLPIPS: {mean_lpips:.4f}\n')