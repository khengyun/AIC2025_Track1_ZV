#!/bin/bash
# Patch inference configuration for Docker runtime
# Ensures dataset_root_dir points to the mounted volume inside container
# and sets appropriate inference parameters

set -e

CONFIG_FILE="${1:-V-DETR/configs/best.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Patching inference config: $CONFIG_FILE"

# Use Python to safely patch YAML while preserving structure
python3 << 'PYTHON_SCRIPT'
import yaml
import sys

config_file = sys.argv[1]

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Ensure inference settings
config['test_only'] = True
config['merge_cls'] = False
config['dataset_root_dir'] = '/workspace/AIC2025_Track1_ZV/dataset/pcd_dataset'

# Optional: Set output directory if not already set
if 'output_dir' not in config or config['output_dir'] is None:
    config['output_dir'] = '/workspace/AIC2025_Track1_ZV/outputs/detection'

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✓ Config patched successfully")
print(f"  - test_only: {config['test_only']}")
print(f"  - merge_cls: {config['merge_cls']}")
print(f"  - dataset_root_dir: {config['dataset_root_dir']}")
print(f"  - output_dir: {config.get('output_dir', 'N/A')}")
PYTHON_SCRIPT

echo "Done."
