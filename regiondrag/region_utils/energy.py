def get_dragon_energy_function(alpha, beta):
    def energy_function(x):
        return 1 / (float(alpha) + float(beta) * x)
    return energy_function


def get_energy_function(name, **kwargs):
    if name == "dragon":
        return get_dragon_energy_function(**kwargs)
    elif name is None:
        return None
    else:
        raise ValueError(f"Energy function {name} not found")
