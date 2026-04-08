"""Tests for the configuration system."""

import pytest
from src.utils.config import load_config, merge_configs


def test_load_config():
    """Test loading the ultra-minimal config and verifying key values."""
    config = load_config('configs/exp_001_ultra_minimal.yaml')
    assert 'experiment' in config
    assert 'data' in config
    assert 'model' in config
    assert 'training' in config
    assert 'topo_loss' in config
    assert 'analysis' in config
    assert 'output' in config
    assert config['model']['depth'] == 4
    assert config['model']['hidden_dim'] == 128


def test_merge_configs():
    """Test deep merging of configuration dictionaries."""
    base = {'a': 1, 'b': {'c': 2, 'd': 3}}
    override = {'b': {'c': 10}, 'e': 5}
    merged = merge_configs(base, override)
    assert merged['a'] == 1
    assert merged['b']['c'] == 10
    assert merged['b']['d'] == 3
    assert merged['e'] == 5


def test_alpha_override():
    """Test that alpha can be overridden via merge."""
    config = load_config('configs/exp_001_ultra_minimal.yaml')
    assert config['experiment']['alpha'] == 0.0
    override = {'experiment': {'alpha': 0.5}}
    merged = merge_configs(config, override)
    assert merged['experiment']['alpha'] == 0.5
    # Original should be unchanged
    assert config['experiment']['alpha'] == 0.0
