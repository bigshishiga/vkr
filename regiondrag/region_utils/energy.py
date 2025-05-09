import torch

def get_dragon_energy_function(alpha, beta):
    def energy_function(x):
        return 1 / (float(alpha) + float(beta) * x)
    return energy_function

def get_negative_energy_function():
    def energy_function(x):
        return -x
    return energy_function

def get_cosine_local_similarity_function():
    def similarity_function(feature_map, inv_feature_map, source, target):
        sim = (
            torch.nn.functional.cosine_similarity(
                feature_map[0, :, target[:, 1], target[:, 0]],
                inv_feature_map[0, :, source[:, 1], source[:, 0]],
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
        return sim
    return similarity_function


def get_energy_function(name, **kwargs):
    if name == "dragon":
        return get_dragon_energy_function(**kwargs)
    elif name == "negative":
        return get_negative_energy_function(**kwargs)
    elif name is None:
        return None
    else:
        raise ValueError(f"Energy function {name} not found")

def get_similarity_function(name, **kwargs):
    if name == "cosine_local":
        return get_cosine_local_similarity_function(**kwargs)
    elif name == "cosine_global":
        return get_cosine_global_similarity_function(**kwargs)
    elif name is None:
        return None
    else:
        raise ValueError(f"Similarity function {name} not found")