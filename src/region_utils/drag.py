import typing
import math
import os
import numpy as np
import pickle
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F

import sys
sys.path.append('.')

from src.region_utils.cycle_sde import Sampler, GuidanceSampler, get_img_latent, get_text_embed, load_model, set_seed

import logging
logger = logging.getLogger(__name__)

# --- To include: InstantDrag (https://github.com/SNU-VGILab/InstantDrag) --- #
sys.path.append('instantdrag/')
if not os.path.exists('instantdrag/utils/__init__.py'):
    open('instantdrag/utils/__init__.py', 'a').close()

from huggingface_hub import snapshot_download
from instantdrag.demo.demo_utils import InstantDragPipeline
os.makedirs("./checkpoints", exist_ok=True)
snapshot_download("alex4727/InstantDrag", local_dir="./checkpoints")

def scale_schedule(begin, end, n, length, type='linear'):
    if type == 'constant':
        return end
    elif type == 'linear':
        return begin + (end - begin) * n / length
    elif type == 'cos':
        factor = (1 - math.cos(n * math.pi / length)) / 2
        return (1 - factor) * begin + factor * end
    else:
        raise NotImplementedError(type)
    
def get_meta_data(meta_data_path):
    with open(meta_data_path, 'rb') as file:
        meta_data = pickle.load(file)
        prompt = meta_data['prompt']
        mask = meta_data['mask']
        points = meta_data['points']
        source = points[0:-1:2]
        target = points[1::2]
    return prompt, mask, source, target

def get_drag_data(data_path):
    ori_image_path = os.path.join(data_path, 'original_image.png')
    meta_data_path = os.path.join(data_path, 'meta_data_region.pkl')

    original_image = np.array(Image.open(ori_image_path))
    prompt, mask, source, target = get_meta_data(meta_data_path)

    return {
        'ori_image' : original_image, 'preview' : original_image, 'prompt' : prompt, 
        'mask' : mask, 'source' : np.array(source), 'target' : np.array(target)
    }

def reverse_and_repeat_every_n_elements(lst, n, repeat=1):
    """
    Reverse every n elements in a given list, then repeat the reversed segments
    the specified number of times.
    Example:
    >>> reverse_and_repeat_every_n_elements([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 2)
    [3, 2, 1, 3, 2, 1, 6, 5, 4, 6, 5, 4, 9, 8, 7, 9, 8, 7]
    """
    if not lst or n < 1:
        return lst
    return [element for i in range(0, len(lst), n) for _ in range(repeat) for element in reversed(lst[i:i+n])]

def get_border_points(points):
    x_max, y_max = np.amax(points, axis=0) 
    mask = np.zeros((y_max+1, x_max+1), np.uint8)
    mask[points[:, 1], points[:, 0]] = 1
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_points = np.concatenate([contour[:, 0, :] for contour in contours], axis=0)
    return border_points

def postprocess(vae, latent, ori_image, mask, blur_radius=0):
    dtype = latent.dtype
    upcast_dtype = torch.float32 if 'xl-base-1.0' in vae.config._name_or_path and dtype == torch.float16 else dtype
    H, W = ori_image.shape[:2]

    if dtype == torch.float16:
        vae = vae.to(upcast_dtype)
        for module in [vae.post_quant_conv, vae.decoder.conv_in, vae.decoder.mid_block]:
            module = module.to(dtype)
    
    image = vae.decode(latent / 0.18215).sample / 2 + 0.5
    image = (image.clamp(0, 1).permute(0, 2, 3, 1)[0].cpu().numpy() * 255).astype(np.uint8)
    image = cv2.resize(image, (W, H))

    if not np.all(mask == 1):
        if blur_radius > 0:
            mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (blur_radius, blur_radius), 0)[:, :, None]
            image = mask_blurred * image + (1 - mask_blurred) * ori_image
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = np.where(mask[:, :, None], image, ori_image)
    
    return image

def copy_and_paste(source_latents, target_latents, source, target):
    target_latents[0, :, target[:, 1], target[:, 0]] = source_latents[0, :, source[:, 1], source[:, 0]]
    return target_latents

def blur_source(latents, noise_scale, source):
    img_scale = (1 - noise_scale ** 2) ** (0.5) if noise_scale < 1 else 0
    latents[0, :, source[:, 1], source[:, 0]] = latents[0, :, source[:, 1], source[:, 0]] * img_scale + \
        torch.randn_like(latents[0, :, source[:, 1], source[:, 0]]) * noise_scale
    return latents

def ip_encode_image(feature_extractor, image_encoder, image):
    if image_encoder is None:
        return None
    dtype = next(image_encoder.parameters()).dtype
    device = next(image_encoder.parameters()).device

    image = feature_extractor(image, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
    image_enc_hidden_states = image_encoder(image, output_hidden_states=True).hidden_states[-2]      
    uncond_image_enc_hidden_states = image_encoder(
        torch.zeros_like(image), output_hidden_states=True
    ).hidden_states[-2]
    image_embeds = torch.stack([uncond_image_enc_hidden_states, image_enc_hidden_states], dim=0)

    return [image_embeds]    
        
def forward(scheduler, sampler, steps, start_t, latent, text_embeddings, added_cond_kwargs, progress=tqdm, sde=True):
    forward_func = sampler.forward_sde if sde else sampler.forward_ode
    hook_latents = [latent,]; noises = []; cfg_scales = []
    start_t = int(start_t * steps)

    for index, t in enumerate(progress(scheduler.timesteps[(steps - start_t):].flip(dims=[0])), start=1):
        cfg_scale = scale_schedule(1, 1, index, steps, type='linear')
        latent, noise = forward_func(t, latent, cfg_scale, text_embeddings, added_cond_kwargs=added_cond_kwargs)
        hook_latents.append(latent); noises.append(noise); cfg_scales.append(cfg_scale)

    return hook_latents, noises, cfg_scales

def backward_step(sampler, t, latent, inv_latent, text_embeddings, cfg_scale, noise, source, target, mask, do_edit, sde, added_cond_kwargs):
    latent = copy_and_paste(inv_latent, latent, source, target) if do_edit else latent
    latent = torch.where(mask == 1, latent, inv_latent)
    latent = sampler.sample(t, latent, cfg_scale, text_embeddings, sde=sde, noise=noise, added_cond_kwargs=added_cond_kwargs)
    
    return latent

def backward_guidance_step(sampler, t, latent, inv_latent, inv_feature_maps, text_embeddings, cfg_scale, noise, source, target, mask, do_edit, sde, added_cond_kwargs):
    if do_edit:
        latent = sampler.sample(t, latent, cfg_scale, text_embeddings, sde=sde, noise=noise, added_cond_kwargs=added_cond_kwargs, inv_feature_maps=inv_feature_maps, source=source, target=target)
    else:
        latent = sampler.sample(t, latent, cfg_scale, text_embeddings, sde=sde, noise=noise, added_cond_kwargs=added_cond_kwargs)
    latent = torch.where(mask == 1, latent, inv_latent)

    return latent

def backward(scheduler, sampler, steps, start_t, end_t, noise_scale, inv_latents, inv_features, noises, cfg_scales, mask, text_embeddings, added_cond_kwargs, blur, source, target, progress=tqdm, latent=None, sde=True, mode=None):
    start_t = int(start_t * steps)
    end_t = int(end_t * steps)
    step_len = 1000 // steps

    latent = inv_latents[-1].clone() if latent is None else latent
    latent = blur_source(latent, noise_scale, blur)

    latents = []
    for t in progress(scheduler.timesteps[(steps-start_t- 1):-1]):
        do_edit = round((t / step_len).item()) > end_t + 1
        inv_latent = inv_latents.pop()
        cfg_scale = cfg_scales.pop()
        noise = noises.pop()

        if mode == 'regiondrag':
            latent = backward_step(sampler, t, latent, inv_latent, text_embeddings, cfg_scale, noise, source, target, mask, do_edit, sde, added_cond_kwargs)
        elif mode == 'guidance':
            inv_feature_maps = [inv_features.pop() for _ in range(len(sampler.guidance_layers))]
            latent = backward_guidance_step(sampler, t, latent, inv_latent, inv_feature_maps, text_embeddings, cfg_scale, noise, source, target, mask, do_edit, sde, added_cond_kwargs)
        else:
            raise ValueError("Invalid mode")
        latents.append(latent.clone())

    return latent, latents


@torch.no_grad()
def drag_copy_paste(drag_data, sd_version=None, device=None):
    torch_dtype = torch.float16 if 'cuda' in device else torch.float32

    global vae, tokenizer, text_encoder, unet, scheduler, feature_extractor, image_encoder, tokenizer_2, text_encoder_2
    if 'vae' not in globals():
        vae, tokenizer, text_encoder, unet, scheduler, feature_extractor, image_encoder, tokenizer_2, text_encoder_2 = load_model(sd_version, torch_device=device, torch_dtype=torch_dtype)

    ori_image, preview, prompt, mask, source, target = drag_data.values()
    source //= 8
    target //= 8

    source_latent = get_img_latent(ori_image, vae, torch_device=device, dtype=torch_dtype)
    target_latent = source_latent.clone()
    target_latent = copy_and_paste(source_latent, target_latent, source, target)
    target_image = postprocess(vae, target_latent, ori_image, mask)

    return target_image

@torch.no_grad()
def drag_id(drag_data, sd_version=None, device=None):
    torch_dtype = torch.float16 if 'cuda' in device else torch.float32

    global vae, tokenizer, text_encoder, unet, scheduler, feature_extractor, image_encoder, tokenizer_2, text_encoder_2
    if 'vae' not in globals():
        vae, tokenizer, text_encoder, unet, scheduler, feature_extractor, image_encoder, tokenizer_2, text_encoder_2 = load_model(sd_version, torch_device=device, torch_dtype=torch_dtype)

    ori_image, preview, prompt, mask, source, target = drag_data.values()
    source //= 8
    target //= 8

    source_latent = get_img_latent(ori_image, vae, torch_device=device, dtype=torch_dtype)
    target_image = postprocess(vae, source_latent, ori_image, mask)

    return target_image

def drag(
        drag_data,
        steps,
        start_t,
        end_t,
        noise_scale,
        seed,
        progress=tqdm,
        sde=True,
        method=None,
        device=None,
        disable_kv_copy=False,
        disable_ip_adapter=False,
        guidance_layers: list[int] = [1, 2],
        guidance_weight: float = 3000.0,
        energy_function = None,
        similarity_function = None,
        eps_clipping_coeff: float = 0,
        guidance_mask_radius: int = -1,
        sd_version = None,
        mask_blur_radius: int = 0,
        max_pairs: int = -1,
    ):
    assert (
        all(guidance_layer in (0, 1, 2, 3) for guidance_layer in guidance_layers) and
        sorted(guidance_layers) == guidance_layers
    ), f"Invalid guidance layers: {guidance_layers}"

    set_seed(seed)
    ori_image = drag_data['ori_image']
    prompt = drag_data['prompt']
    mask = drag_data['mask']
    source = drag_data['source']
    target = drag_data['target']

    if max_pairs != -1:
        assert source.shape[0] == target.shape[0]
        assert source.shape[1] == target.shape[1] == 2
        cur_samples = source.shape[0]
        new_samples = min(max_pairs, cur_samples)
        if cur_samples > 1:
            idxs = torch.randperm(cur_samples)[:new_samples]
            source = source[idxs]
            target = target[idxs]
        logger.info(f"subsamples", extra={"before": cur_samples, "after": new_samples})

    torch_dtype = torch.float16 if 'cuda' in device else torch.float32
    
    if method in ('regiondrag', 'guidance'):
        global vae, tokenizer, text_encoder, unet, scheduler, feature_extractor, image_encoder, tokenizer_2, text_encoder_2
        if 'vae' not in globals():
            vae, tokenizer, text_encoder, unet, scheduler, feature_extractor, image_encoder, tokenizer_2, text_encoder_2 = load_model(sd_version, torch_device=device, torch_dtype=torch_dtype, ip_adapter=not disable_ip_adapter)

        def copy_key_hook(module, input, output):
            keys.append(output)
        def copy_value_hook(module, input, output):
            values.append(output)
        def paste_key_hook(module, input, output):
            output[:] = keys.pop()
        def paste_value_hook(module, input, output):
            output[:] = values.pop()
        def save_features_hook(module, input, output):
            features.append(output)

        def register(do='copy'):
            key_handlers = []; value_handlers = []; feature_handlers = []
            do_copy = do == 'copy'
            do_guide = method == 'guidance'

            if not disable_kv_copy:
                key_hook, value_hook = (copy_key_hook, copy_value_hook) if do_copy else (paste_key_hook, paste_value_hook)
                for block in (*sampler.unet.down_blocks, sampler.unet.mid_block, *sampler.unet.up_blocks):
                    if not hasattr(block, 'attentions'):
                        continue
                    for attention in block.attentions:
                        for tb in attention.transformer_blocks:
                            key_handlers.append(tb.attn1.to_k.register_forward_hook(key_hook))
                            value_handlers.append(tb.attn1.to_v.register_forward_hook(value_hook))
            
            if do_guide and do_copy:
                for layer in guidance_layers:
                    block = sampler.unet.up_blocks[layer]
                    feature_handlers.append(block.register_forward_hook(save_features_hook))

            return key_handlers, value_handlers, feature_handlers

        def unregister(*handlers):
            for handler in handlers:
                handler.remove()
            if device == 'cuda':
                torch.cuda.empty_cache()

        source = torch.from_numpy(source).to(device) if isinstance(source, np.ndarray) else source.to(device)
        target = torch.from_numpy(target).to(device) if isinstance(target, np.ndarray) else target.to(device)
        source = source // 8; target = target // 8 # from img scale to latent scale

        blur_pts = source; copy_pts = source
        paste_pts = target
        
        if 'out_latent' not in drag_data:
            latent = get_img_latent(ori_image, vae, torch_device=device, dtype=torch_dtype)
        else:
            latent = drag_data['out_latent']
        
        if method == 'guidance':
            sampler = GuidanceSampler(
                unet=unet, scheduler=scheduler, num_steps=steps,
                guidance_weight=guidance_weight, guidance_layers=guidance_layers,
                energy_function=energy_function, similarity_function=similarity_function,
                eps_clipping_coeff=eps_clipping_coeff, guidance_mask_radius=guidance_mask_radius
            )
        else:
            sampler = Sampler(unet=unet, scheduler=scheduler, num_steps=steps)

        with torch.no_grad():
            neg_pooled_prompt_embeds, neg_prompt_embeds = get_text_embed("", tokenizer, text_encoder, tokenizer_2, text_encoder_2, torch_device=device)
            neg_prompt_embeds = neg_prompt_embeds if sd_version == 'xl' else neg_pooled_prompt_embeds
            pooled_prompt_embeds, prompt_embeds = get_text_embed(prompt, tokenizer, text_encoder, tokenizer_2, text_encoder_2, torch_device=device)
            prompt_embeds = prompt_embeds if sd_version == 'xl' else pooled_prompt_embeds
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

            image_embeds = ip_encode_image(feature_extractor, image_encoder, ori_image)
            
            H, W = ori_image.shape[:2]
            add_time_ids = torch.tensor([[H, W, 0, 0, H, W]]).to(prompt_embeds).repeat(2, 1)
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids} if sd_version == 'xl' else {}
            if image_embeds is not None:
                added_cond_kwargs["image_embeds"] = image_embeds 

            mask_pt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
            mask_pt = F.interpolate(mask_pt, size=latent.shape[2:]).expand_as(latent)

            keys = []; values = []; features = []
            key_handlers, value_handlers, feature_handlers = register(do='copy')
            hook_latents, noises, cfg_scales = forward(scheduler, sampler, steps, start_t, latent, prompt_embeds, added_cond_kwargs, progress=progress, sde=sde)
            start_latent = None
            unregister(*key_handlers, *value_handlers, *feature_handlers)
            forward_process = [postprocess(vae, latent, ori_image, mask, mask_blur_radius) for latent in hook_latents]

            keys = reverse_and_repeat_every_n_elements(keys, n=len(key_handlers))
            values = reverse_and_repeat_every_n_elements(values, n=len(value_handlers))
            features = reverse_and_repeat_every_n_elements(features, n=len(feature_handlers))

            key_handlers, value_handlers, feature_handlers = register(do='paste')
            latent, latents = backward(scheduler, sampler, steps, start_t, end_t, noise_scale, hook_latents, features, noises, cfg_scales, mask_pt, prompt_embeds, added_cond_kwargs, blur_pts, copy_pts, paste_pts, latent=start_latent, progress=progress, sde=sde, mode=method)
            unregister(*key_handlers, *value_handlers, *feature_handlers)

            drag_data['out_latent'] = latent
            drag_data['latents'] = latents
            image = postprocess(vae, latent, ori_image, mask, mask_blur_radius)
            backward_process = [forward_process[-1]] + [postprocess(vae, latent, ori_image, mask, mask_blur_radius) for latent in latents]
            backward_process.reverse()

    elif method == 'instantdrag':
        global instant_pipe
        if 'instant_pipe' not in globals():
            instant_pipe = InstantDragPipeline(seed, device, (torch.float16 if 'cuda' in device else torch.float32))
        flowgen_ckpt =  next((m for m in sorted(os.listdir("checkpoints/")) if "flowgen" in m), None)
        flowdiffusion_ckpt = next(f for f in sorted(os.listdir("checkpoints/")) if "flowdiffusion" in f)

        print('Unused parameters in utils.drag function: input, start_t, end_t, noise_scale, progress')
        selected_points = [point.tolist() for pair in zip(source, target) for point in pair]
        ori_H, ori_W = ori_image.shape[:2]; new_H, new_W = 512, 512
        selected_points = torch.tensor(selected_points) / torch.tensor([ori_W, ori_H]) * torch.tensor([new_W, new_H])
        image_guidance, flow_guidance, flowgen_output_scale = 1.5, 1.5, -1.0
        image = instant_pipe.run(cv2.resize(ori_image, (new_W, new_H)), selected_points.tolist(), flowgen_ckpt, flowdiffusion_ckpt, image_guidance,
                         flow_guidance, flowgen_output_scale, steps, save_results=False)
        image = cv2.resize(image, (ori_W, ori_H))
        forward_process = None
        backward_process = None

    else:
        raise ValueError('Select method from regiondrag, instantdrag, guidance')

    if device == 'cuda':
        torch.cuda.empty_cache()
    return image, forward_process, backward_process