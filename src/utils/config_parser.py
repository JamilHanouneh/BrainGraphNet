"""
Configuration file parser
"""

import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        config: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate config
    _validate_config(config)
    
    return config


def save_config(config, save_path):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def _validate_config(config):
    """Validate configuration dictionary"""
    required_keys = ['data', 'model', 'training', 'system']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # Validate data config
    if 'source' not in config['data']:
        raise ValueError("Data source not specified in config")
    
    # Validate model config
    if 'type' not in config['model']:
        raise ValueError("Model type not specified in config")
    
    # Validate training config
    if 'num_epochs' not in config['training']:
        raise ValueError("Number of epochs not specified in config")
    
    return True


def update_config(config, updates):
    """
    Update config with new values
    
    Args:
        config: Configuration dictionary
        updates: Dictionary of updates (nested keys supported)
    
    Returns:
        Updated config
    """
    import copy
    config = copy.deepcopy(config)
    
    for key, value in updates.items():
        if '.' in key:
            # Nested key (e.g., 'training.learning_rate')
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
        else:
            config[key] = value
    
    return config
