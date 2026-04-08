"""Configuration utilities for loading, merging, and saving YAML configs."""

import os
import copy
import yaml


def load_config(config_path):
    """Load a YAML configuration file and return it as a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base, override):
    """Deep merge two configuration dictionaries.

    The override values take precedence. Nested dictionaries are merged
    recursively rather than being replaced entirely.

    Args:
        base: The base configuration dictionary.
        override: The override configuration dictionary.

    Returns:
        A new dictionary containing the merged configuration.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def save_config(config, output_path):
    """Save a configuration dictionary to a YAML file.

    Creates parent directories if they do not exist.

    Args:
        config: The configuration dictionary to save.
        output_path: The file path to write the YAML to.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
