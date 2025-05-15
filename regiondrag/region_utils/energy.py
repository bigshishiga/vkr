import torch

def feature_map_filter(feature_map, kernel_size=1):
    if kernel_size == 1:
        return feature_map

    # Pad H and W dimensions
    feature_map_padded = torch.nn.functional.pad(
        feature_map,
        (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
        mode="constant",
        value=0
    )

    # Kernel for the averaging filter
    channels = feature_map.shape[1]
    kernel = torch.ones(channels, 1, kernel_size, kernel_size) / kernel_size**2
    kernel = kernel.to(device=feature_map.device, dtype=feature_map.dtype)

    return torch.nn.functional.conv2d(feature_map_padded, kernel, groups=channels)

def get_dragon_energy_function(alpha, beta):
    def energy_function(x):
        return 1 / (float(alpha) + float(beta) * x)
    return energy_function

def get_negative_energy_function():
    def energy_function(x):
        return -x
    return energy_function

def get_cosine_local_similarity_function(kernel_size=1):
    kernel_size = int(kernel_size)

    def similarity_function(feature_map, inv_feature_map, source, target):
        feature_map_filtered = feature_map_filter(feature_map, kernel_size)
        inv_feature_map_filtered = feature_map_filter(inv_feature_map, kernel_size)

        sim = (
            torch.nn.functional.cosine_similarity(
                feature_map_filtered[0, :, target[:, 1], target[:, 0]],
                inv_feature_map_filtered[0, :, source[:, 1], source[:, 0]],
                dim=1
            )
            + 1.0
        ) / 2.0
        sim = sim.mean()
        return sim
    return similarity_function

def get_cosine_global_similarity_function():
    def similarity_function(feature_map, inv_feature_map, source, target):
        v1 = feature_map[0, :, target[:, 1], target[:, 0]].mean(dim=1)
        v2 = inv_feature_map[0, :, source[:, 1], source[:, 0]].mean(dim=1)
        sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0)
        sim = (sim + 1.0) / 2.0
        return sim
    return similarity_function

def get_cosine_mixed_similarity_function(local_w):
    local_w = float(local_w)
    assert local_w >= 0 and local_w <= 1

    local_similarity_function = get_cosine_local_similarity_function()
    global_similarity_function = get_cosine_global_similarity_function()

    def similarity_function(feature_map, inv_feature_map, source, target):
        local_sim = local_similarity_function(feature_map, inv_feature_map, source, target)
        global_sim = global_similarity_function(feature_map, inv_feature_map, source, target)
        return local_w * local_sim + (1 - local_w) * global_sim

    return similarity_function

def get_energy_function(name, **kwargs):
    if name == "dragon":
        return get_dragon_energy_function(**kwargs)
    elif name == "negative":
        return get_negative_energy_function(**kwargs)
    elif name is None or name == "null":
        return None
    else:
        raise ValueError(f"Energy function {name} not found")

def get_similarity_function(name, **kwargs):
    if name == "cosine_local":
        return get_cosine_local_similarity_function(**kwargs)
    elif name == "cosine_global":
        return get_cosine_global_similarity_function(**kwargs)
    elif name == "cosine_mixed":
        return get_cosine_mixed_similarity_function(**kwargs)
    elif name is None or name == "null":
        return None
    else:
        raise ValueError(f"Similarity function {name} not found")