# This file is developed based on the work of [ML-GSAI/SDE-Drag],
# which can be found at https://github.com/ML-GSAI/SDE-Drag

import torch
import random
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import logging

logger = logging.getLogger(__name__)


def load_model(version="v1-5", torch_device='cuda', torch_dtype=torch.float16, ip_adapter=True, verbose=True):
    pipe_paths = {
        'v1-5':            "runwayml/stable-diffusion-v1-5", 
        'v1-5-inpainting': "stable-diffusion-v1-5/stable-diffusion-inpainting",
        'v2-1':            "stabilityai/stable-diffusion-2-1",
        'xl':              "stabilityai/stable-diffusion-xl-base-1.0"
    }
    pipe_path = pipe_paths.get(version, pipe_paths['v1-5'])
    pipe = StableDiffusionPipeline if version in ['v1-5', 'v1-5-inpainting', 'v2-1'] else StableDiffusionXLPipeline
    
    if verbose:
        print(f'Loading model from {pipe_path}.')
    pipe = pipe.from_pretrained(pipe_path, torch_dtype=torch_dtype).to(torch_device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # IP-Adapter
    if ip_adapter:
        print("Loading IP-Adapter")
        if version in ['v1-5', 'v2-1']:
            subfolder, weight_name, ip_adapter_scale = "models", "ip-adapter-plus_sd15.bin", 0.5
        else: 
            subfolder, weight_name, ip_adapter_scale = "sdxl_models", "ip-adapter_sdxl.bin", 0.6
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder=subfolder, weight_name=weight_name)
        pipe.set_ip_adapter_scale(ip_adapter_scale)

    tokenizer_2 = pipe.tokenizer_2 if version == 'xl' else None
    text_encoder_2 = pipe.text_encoder_2 if version == 'xl' else None
    return pipe.vae, pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.scheduler, pipe.feature_extractor, pipe.image_encoder, tokenizer_2, text_encoder_2

@torch.no_grad()
def get_text_embed(prompt: list, tokenizer, text_encoder, tokenizer_2=None, text_encoder_2=None, torch_device='cuda'):
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    prompt_embeds = text_encoder(text_input.input_ids.to(torch_device), output_hidden_states=True)
    pooled_prompt_embeds, prompt_embeds = prompt_embeds[0], prompt_embeds.hidden_states[-2]

    if tokenizer_2 is not None and text_encoder_2 is not None:
        text_input = tokenizer_2(prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt")
        prompt_embeds_2 = text_encoder_2(text_input.input_ids.to(torch_device), output_hidden_states=True)
        pooled_prompt_embeds, prompt_embeds_2 = prompt_embeds_2[0], prompt_embeds_2.hidden_states[-2]
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

    return pooled_prompt_embeds, prompt_embeds

@torch.no_grad()
def get_img_latent(image, vae, torch_device=None, dtype=torch.float16, size=None):
    # upcast vae for sdxl, attention blocks can be in torch.float16
    upcast_dtype = torch.float32 if 'xl-base-1.0' in vae.config._name_or_path and dtype == torch.float16 else dtype
    if dtype == torch.float16:
        vae = vae.to(upcast_dtype)
        for module in [vae.post_quant_conv, vae.decoder.conv_in, vae.decoder.mid_block]:
            module = module.to(dtype)

    image = Image.open(image).convert('RGB') if isinstance(image, str) else Image.fromarray(image)
    image = image.resize(size) if size else image
    image = transforms.ToTensor()(image).unsqueeze(0).to(torch_device, upcast_dtype)
    latents = vae.encode(image * 2 - 1).latent_dist.sample() * 0.18215
    return latents.to(dtype)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Sampler():
    def __init__(self, unet, scheduler, num_steps=100):
        scheduler.set_timesteps(num_steps)
        self.num_inference_steps = num_steps
        self.num_train_timesteps = len(scheduler)

        self.alphas = scheduler.alphas
        self.alphas_cumprod = scheduler.alphas_cumprod

        self.final_alpha_cumprod = torch.tensor(1.0)
        self.initial_alpha_cumprod = torch.tensor(1.0)

        self.unet = unet

    def get_eps(self, img, timestep, guidance_scale, text_embeddings, lora_scale=None, added_cond_kwargs=None, **kwargs):
        guidance_scale = max(1, guidance_scale)
        
        text_embeddings = text_embeddings if guidance_scale > 1. else text_embeddings[-1:]
        latent_model_input = torch.cat([img] * 2) if guidance_scale > 1. else img
        cross_attention_kwargs = None if lora_scale is None else {"scale": lora_scale}

        if guidance_scale == 1. and added_cond_kwargs is not None:
            added_cond_kwargs = {k: v[-1:] for k, v in added_cond_kwargs.items()} 

        noise_pred = self.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs
        ).sample

        if guidance_scale > 1.:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        elif guidance_scale == 1.:
            noise_pred_text = noise_pred
            noise_pred_uncond = 0.
        else:
            raise NotImplementedError(guidance_scale)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    def sample(
        self,
        timestep,
        sample,
        guidance_scale,
        text_embeddings,
        sde=False,
        noise=None,
        eta=1.,
        lora_scale=None,
        added_cond_kwargs=None,
        **kwargs,
    ):
        eps = self.get_eps(sample, timestep, guidance_scale, text_embeddings, lora_scale, added_cond_kwargs, **kwargs)

        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        sigma_t = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** (0.5) * (1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5) if sde else 0

        pred_original_sample = (sample - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
        pred_sample_direction_coeff = (1 - alpha_prod_t_prev - sigma_t ** 2) ** (0.5)

        noise = torch.randn_like(sample, device=sample.device) if noise is None else noise
        img = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_coeff * eps + sigma_t * noise

        return img

    def forward_sde(self, timestep, sample, guidance_scale, text_embeddings, eta=1., lora_scale=None, added_cond_kwargs=None):
        prev_timestep = timestep + self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.initial_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t_prev = 1 - alpha_prod_t_prev

        x_prev = (alpha_prod_t_prev / alpha_prod_t) ** (0.5) * sample + (1 - alpha_prod_t_prev / alpha_prod_t) ** (0.5) * torch.randn_like(sample, device=sample.device)
        eps = self.get_eps(x_prev, prev_timestep, guidance_scale, text_embeddings, lora_scale, added_cond_kwargs)

        sigma_t_prev = eta * ((1 - alpha_prod_t) / (1 - alpha_prod_t_prev)) ** (0.5) * (1 - alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        pred_original_sample = (x_prev - beta_prod_t_prev ** (0.5) * eps) / alpha_prod_t_prev ** (0.5)
        pred_sample_direction_coeff = (1 - alpha_prod_t - sigma_t_prev ** 2) ** (0.5)

        noise = (sample - alpha_prod_t ** (0.5) * pred_original_sample - pred_sample_direction_coeff * eps) / sigma_t_prev

        return x_prev, noise
    
    def forward_ode(self, timestep, sample, guidance_scale, text_embeddings, lora_scale=None, added_cond_kwargs=None):
        prev_timestep = timestep + self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.initial_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t

        eps = self.get_eps(sample, timestep, guidance_scale, text_embeddings, lora_scale, added_cond_kwargs)
        pred_original_sample = (sample - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * eps

        img = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        
        noise = None
        return img, noise


class GuidanceSampler(Sampler):
    def __init__(
        self,
        *args,
        guidance_weight=3000.0,
        guidance_layers=[1, 2],
        energy_function=None,
        similarity_function=None,
        eps_clipping_coeff=0,
        guidance_mask_radius=-1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.guidance_weight = guidance_weight
        self.guidance_layers = guidance_layers
        self.energy_function = energy_function
        self.similarity_function = similarity_function
        self.eps_clipping_coeff = eps_clipping_coeff
        self.guidance_mask_radius = guidance_mask_radius

    def get_eps(self, img, timestep, guidance_scale, text_embeddings, lora_scale=None, added_cond_kwargs=None, **kwargs):
        inv_feature_maps = kwargs.get("inv_feature_maps", None)
        # When guidance is terminated
        if inv_feature_maps is None:
            return super().get_eps(img, timestep, guidance_scale, text_embeddings, lora_scale, added_cond_kwargs)

        source = kwargs["source"]
        target = kwargs["target"]

        feature_maps = []
        def forward_hook(module, input, output):
            nonlocal feature_maps
            feature_maps.append(output)

        handlers = []
        for layer in self.guidance_layers:
            handler = self.unet.up_blocks[layer].register_forward_hook(forward_hook)
            handlers.append(handler)

        with torch.enable_grad():
            img.requires_grad = True
            denoise_eps = super().get_eps(img, timestep, guidance_scale, text_embeddings, lora_scale, added_cond_kwargs, **kwargs)
            denoise_eps = denoise_eps.detach().requires_grad_(False)
            for handler in handlers:
                handler.remove()

            total_sim = 0
            stats = {}
            for layer, feature_map, inv_feature_map in zip(self.guidance_layers, feature_maps, inv_feature_maps):
                assert feature_map.shape == inv_feature_map.shape, (feature_map.shape, inv_feature_map.shape)

                # Feature map from lower layers. Needs to be upsampled.
                if feature_map.shape[-1] != img.shape[-1] or feature_map.shape[-2] != img.shape[-2]:
                    shape = (img.shape[-2], img.shape[-1])
                    feature_map = torch.nn.functional.interpolate(feature_map, size=shape, mode="bilinear", align_corners=False)
                    inv_feature_map = torch.nn.functional.interpolate(inv_feature_map, size=shape, mode="bilinear", align_corners=False)

                sim = self.similarity_function(
                    feature_map=feature_map,
                    inv_feature_map=inv_feature_map,
                    source=source,
                    target=target
                )
                total_sim = total_sim + sim

                stats[f"sim_{layer}"] = sim.item()
            total_sim = total_sim / len(self.guidance_layers)
            stats["total_sim"] = total_sim.item()

            energy = self.energy_function(total_sim)
            energy.backward()

            guidance_eps = img.grad
            img.grad = None
            img.requires_grad = False
        
        guidance_eps = guidance_eps * self.guidance_weight

        mask = np.zeros((img.shape[-2], img.shape[-1]))
        if self.guidance_mask_radius != -1:
            target_np = target.cpu().numpy()
            mask[target_np[:, 1], target_np[:, 0]] = 1
            mask = binary_dilation(mask, iterations=self.guidance_mask_radius)
            mask = torch.from_numpy(mask).to(guidance_eps.device)
        else:
            mask = torch.from_numpy(mask).fill_(1).to(guidance_eps.dtype).to(guidance_eps.device)
        
        stats["energy"] = energy.item()
        stats["guidance_norm_unclipped"] = guidance_eps.norm().item()
        stats["guidance_norm_masked_unclipped"] = (guidance_eps * mask).norm().item()

        if self.eps_clipping_coeff != 0:
            guidance_norm_masked, denoise_norm_masked = (guidance_eps * mask).norm(), (denoise_eps * mask).norm()
            coeff = (1 / self.eps_clipping_coeff) * guidance_norm_masked / denoise_norm_masked
            guidance_eps = guidance_eps / max(1, coeff)
        
        stats["guidance_norm"] = guidance_eps.norm().item()
        stats["denoise_norm"] = denoise_eps.norm().item()
        stats["guidance_norm_masked"] = (guidance_eps * mask).norm().item()
        stats["denoise_norm_masked"] = (denoise_eps * mask).norm().item()
        stats["timestep"] = timestep.item()

        logger.info("step", extra=stats)

        return denoise_eps + guidance_eps * mask
