#!/usr/bin/env python3
"""Debug script to check config field inheritance."""

from dataclasses import fields
from src.models.phase8.config import Phase8Config
from src.models.phase7.integrated_model import Phase7Config
from src.models.config import ResNetBKConfig

print("=" * 60)
print("ResNetBKConfig fields:")
print("=" * 60)
for f in fields(ResNetBKConfig):
    print(f"  {f.name}: default={f.default if f.default is not f.default_factory else 'factory'}")

print("\n" + "=" * 60)
print("Phase7Config fields:")
print("=" * 60)
for f in fields(Phase7Config):
    print(f"  {f.name}: default={f.default if f.default is not f.default_factory else 'factory'}")

print("\n" + "=" * 60)
print("Phase8Config fields:")
print("=" * 60)
for f in fields(Phase8Config):
    print(f"  {f.name}: default={f.default if f.default is not f.default_factory else 'factory'}")

print("\n" + "=" * 60)
print("Creating Phase8Config with vocab_size=50256, d_model=4096:")
print("=" * 60)
config = Phase8Config(vocab_size=50256, d_model=4096)
print(f"  vocab_size: {config.vocab_size}")
print(f"  d_model: {config.d_model}")
print(f"  n_layers: {config.n_layers}")

print("\n" + "=" * 60)
print("Getting all fields via fields():")
print("=" * 60)
config_dict = {}
for f in fields(config):
    config_dict[f.name] = getattr(config, f.name)
print(f"  vocab_size in dict: {config_dict.get('vocab_size')}")
print(f"  d_model in dict: {config_dict.get('d_model')}")
print(f"  n_layers in dict: {config_dict.get('n_layers')}")
