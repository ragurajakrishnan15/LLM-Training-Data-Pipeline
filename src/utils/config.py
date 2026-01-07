"""
Configuration management for the LLM Training Data Pipeline.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for the pipeline."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default config path
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "pipeline_config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Resolve relative paths
        self._resolve_paths()
    
    def _resolve_paths(self) -> None:
        """Convert relative paths to absolute paths."""
        project_root = Path(__file__).parent.parent.parent
        
        if 'paths' in self._config:
            for key, value in self._config['paths'].items():
                if isinstance(value, str) and not os.path.isabs(value):
                    self._config['paths'][key] = str(project_root / value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Example:
            config.get('ingestion.source')
            config.get('quality.min_words', 50)
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def paths(self) -> Dict[str, str]:
        """Get all path configurations."""
        return self._config.get('paths', {})
    
    @property
    def ingestion(self) -> Dict[str, Any]:
        """Get ingestion configuration."""
        return self._config.get('ingestion', {})
    
    @property
    def cleaning(self) -> Dict[str, Any]:
        """Get cleaning configuration."""
        return self._config.get('cleaning', {})
    
    @property
    def deduplication(self) -> Dict[str, Any]:
        """Get deduplication configuration."""
        return self._config.get('deduplication', {})
    
    @property
    def quality(self) -> Dict[str, Any]:
        """Get quality filtering configuration."""
        return self._config.get('quality', {})
    
    @property
    def tokenization(self) -> Dict[str, Any]:
        """Get tokenization configuration."""
        return self._config.get('tokenization', {})
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
