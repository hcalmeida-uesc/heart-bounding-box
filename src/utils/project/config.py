"""Utilities para carregar/salvar e normalizar configurações JSON."""

import copy
import json
import os


def deep_update_dict(base, updates):
    """Atualiza um dicionário recursivamente (merge profundo)."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def normalize_runtime_config(config):
    """Converte tipos carregados de JSON para tipos esperados em runtime."""
    cfg = copy.deepcopy(config)

    if "DOWNSCALE_FACTORS" in cfg and isinstance(cfg["DOWNSCALE_FACTORS"], list):
        cfg["DOWNSCALE_FACTORS"] = tuple(cfg["DOWNSCALE_FACTORS"])

    return cfg


def serialize_config_for_json(config):
    """Converte config runtime em estrutura serializável para JSON."""
    cfg = copy.deepcopy(config)

    if "DOWNSCALE_FACTORS" in cfg and isinstance(cfg["DOWNSCALE_FACTORS"], tuple):
        cfg["DOWNSCALE_FACTORS"] = list(cfg["DOWNSCALE_FACTORS"])

    return cfg


def load_config_json(path, base_config):
    """Carrega configuração JSON e aplica sobre base_config."""
    with open(path, "r", encoding="utf-8") as f:
        file_cfg = json.load(f)

    merged = copy.deepcopy(base_config)
    deep_update_dict(merged, file_cfg)
    return normalize_runtime_config(merged)


def save_config_json(config, path):
    """Salva configuração (normalizada) em arquivo JSON."""
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_config_for_json(config), f, indent=2, ensure_ascii=False)
