# TopoDiffuser Configuration System

## Overview

The configuration system uses YAML files with inheritance support.

## Available Configs

| Config | Purpose | Description |
|--------|---------|-------------|
| `default.yaml` | Base configuration | Full paper settings, 5-channel input |
| `lidar_only.yaml` | LiDAR ablation | 3-channel input (Height, Intensity, Density) |
| `multimodal.yaml` | Full multimodal | 5-channel input + OSM maps |
| `debug.yaml` | Quick testing | Minimal data, small model, fast iteration |

## Usage

### Training with Config

```bash
# Use default config
python pipeline.py --mode train --config configs/default.yaml

# Use LiDAR-only ablation
python pipeline.py --mode train --config configs/lidar_only.yaml

# Debug mode (fast iteration)
python pipeline.py --mode debug --config configs/debug.yaml
```

### Overriding Config Values

```bash
# Override batch size and epochs
python pipeline.py --mode train --config configs/default.yaml \
    --override training.batch_size=128 training.epochs=100

# Change loss weight
python pipeline.py --mode train --config configs/default.yaml \
    --override training.loss.alpha_road=0.5

# Change model architecture
python pipeline.py --mode train --config configs/default.yaml \
    --override model.diffusion.denoising_network.architecture=cnn1d
```

## Config Structure

```yaml
project:
  name: "TopoDiffuser"
  
paths:
  data_root: "..."
  checkpoint_dir: "..."
  
data:
  train_sequences: ["00", "02", "05", "07"]
  val_sequences: ["08", "09", "10"]
  trajectory:
    num_future: 8
    waypoint_spacing: 2.0
  
model:
  encoder:
    input_channels: 3  # or 5 for multimodal
    conditioning_dim: 512
  diffusion:
    num_timesteps: 10
    
training:
  mode: "joint"  # "joint", "encoder_only", "diffusion_only"
  batch_size: 64
  epochs: 50
  loss:
    alpha_road: 0.1  # L_total = L_diff + alpha_road * L_road
```

## Key Parameters

### Training Mode

- `training.mode: joint` - Train encoder + diffusion together (recommended)
- `training.mode: encoder_only` - Train only encoder (for ablation)
- `training.mode: diffusion_only` - Train only diffusion (encoder frozen)

### Loss Weight (alpha_road)

Controls the auxiliary road segmentation loss:

- `0.0` - Diffusion only (no auxiliary task)
- `0.1` - Balanced (default, paper recommendation)
- `0.5` - Stronger road supervision
- `1.0` - Equal weighting

### Model Input Channels

- `3` - LiDAR only (Height, Intensity, Density)
- `5` - Full multimodal (LiDAR + Trajectory History + OSM Map)

## Creating Custom Configs

Create a new YAML file that inherits from a base:

```yaml
# my_experiment.yaml
base_config: "default.yaml"

project:
  name: "My-Experiment"

training:
  batch_size: 128
  loss:
    alpha_road: 0.2
```

## Config Validation

Configs are automatically validated on load. Checks include:
- Required sections present
- Paths exist
- Reasonable hyperparameter ranges
- Compatible settings
