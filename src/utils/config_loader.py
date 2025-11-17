"""
Configuration Loader for Mamba-Killer ResNet-BK

Handles loading and merging YAML configuration files with inheritance support.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class ConfigLoader:
    """Load and merge YAML configuration files."""
    
    def __init__(self, config_dir: str = "./configs"):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
    
    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration file with inheritance support.
        
        Args:
            config_name: Name of config file (without .yaml extension)
        
        Returns:
            Merged configuration dictionary
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle inheritance
        if '_base_' in config:
            base_name = config.pop('_base_').replace('.yaml', '')
            base_config = self.load(base_name)
            config = self._merge_configs(base_config, config)
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge override config into base config.
        
        Args:
            base: Base configuration
            override: Override configuration
        
        Returns:
            Merged configuration
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save(self, config: Dict[str, Any], output_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration has required fields.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_sections = ['model', 'training', 'data']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate model config
        required_model_fields = ['vocab_size', 'd_model', 'n_layers', 'n_seq']
        for field in required_model_fields:
            if field not in config['model']:
                raise ValueError(f"Missing required model field: {field}")
        
        # Validate training config
        required_training_fields = ['learning_rate', 'batch_size', 'max_steps']
        for field in required_training_fields:
            if field not in config['training']:
                raise ValueError(f"Missing required training field: {field}")
        
        return True
    
    def list_configs(self) -> list:
        """List all available configuration files."""
        return [f.stem for f in self.config_dir.glob("*.yaml")]


def load_config(config_name: str, config_dir: str = "./configs") -> Dict[str, Any]:
    """
    Convenience function to load a configuration.
    
    Args:
        config_name: Name of config file (without .yaml extension)
        config_dir: Directory containing configuration files
    
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_dir)
    config = loader.load(config_name)
    loader.validate(config)
    return config


if __name__ == '__main__':
    # Test configuration loading
    loader = ConfigLoader()
    
    print("Available configurations:")
    for config_name in loader.list_configs():
        print(f"  - {config_name}")
    
    print("\nTesting configuration loading...")
    
    # Test base config
    base_config = loader.load('base_config')
    print(f"✓ Loaded base_config: {len(base_config)} sections")
    
    # Test inherited config
    long_context_config = loader.load('long_context_config')
    print(f"✓ Loaded long_context_config: {len(long_context_config)} sections")
    
    # Validate
    loader.validate(base_config)
    print("✓ Configuration validation passed")
