import os
import torch
from diffusers.utils import make_image_grid
import numpy as np
from tqdm import tqdm
import gradio as gr
from PIL import Image
import logging

from region_utils.drag import drag, get_drag_data, get_meta_data, drag_copy_paste, drag_id
from region_utils.evaluator import DragEvaluator
from region_utils.energy import get_energy_function
from eval_utils import setup_logging, get_args, has_uncommitted_changes


def main():
    if has_uncommitted_changes() and not os.environ.get("DEBUG"):
        raise ValueError("Commit your changes before running the evaluation.")

    args, energy_args = get_args()
    setup_logging(os.path.join(args.save_dir, "log.log"))

    energy_function = get_energy_function(args.energy_function, **energy_args)

    logger = logging.getLogger()
    logger.info("params", extra=args.__dict__)

    evaluator = DragEvaluator()
    all_distances = []; all_lpips = []; all_names = []

    data_dirs = [dirpath for dirpath, dirnames, _ in os.walk(args.data_dir) if not dirnames]

    for data_path in tqdm(data_dirs):
        # Region-based Inputs for Editing
        drag_data = get_drag_data(data_path)
        ori_image = drag_data['ori_image'].copy()

        logger.info("image", extra={"path": data_path})

        if args.method in ('regiondrag', 'instantdrag', 'guidance'):
            out_image, forward_process, backward_process = drag(drag_data, args.steps, args.start_t, args.end_t, args.noise_scale, args.seed,
                            progress=gr.Progress(), device=args.device,
                            disable_kv_copy=args.disable_kv_copy,
                            disable_ip_adapter=not args.ip_adapter,
                            guidance_weight=args.guidance_weight, guidance_layers=args.guidance_layers,
                            method=args.method, sde=(args.sampler == "ddpm"),
                            energy_function=energy_function
                        )
        elif args.method == 'copy':
            out_image, forward_process, backward_process = drag_copy_paste(drag_data, device=args.device), None, None
        elif args.method == 'id':
            out_image, forward_process, backward_process = drag_id(drag_data, device=args.device), None, None

        if args.save_dir is not None:
            file_name = data_path.removeprefix(f'drag_data/{args.bench_name}/') + '.png'

            save_name = os.path.join(args.save_dir, file_name)
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            Image.fromarray(out_image).save(save_name)

            if forward_process:
                save_name = os.path.join(args.save_dir, "process", file_name)
                os.makedirs(os.path.dirname(save_name), exist_ok=True)

                grid = make_image_grid(
                    [Image.fromarray(image) for image in forward_process] + [Image.fromarray(image) for image in backward_process],
                    2, len(forward_process)
                )
                grid.save(save_name)

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

if __name__ == '__main__':
    main()
