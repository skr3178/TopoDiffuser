"""
TopoDiffuser Models Package.

This package contains the neural network models for the TopoDiffuser pipeline:
- Block 1: BEV Rasterization (bev_rasterization.py)
- Block 2: Multimodal Encoder (encoder.py)
- Block 3: Conditional Diffusion Policy (diffusion.py) - TODO
- Block 4: Loss Computation (losses.py) - TODO
- Block 5: Inference/Training (trainer.py) - TODO
"""

from .bev_rasterization import (
    BEVRasterizer,
    BEVRasterizationBlock,
    load_kitti_lidar,
    extract_trajectory_from_poses,
)

from .encoder import (
    CBRBlock,
    CBBlock,
    MultimodalEncoder,
    build_encoder,
)

__all__ = [
    # Block 1: BEV Rasterization
    'BEVRasterizer',
    'BEVRasterizationBlock',
    'load_kitti_lidar',
    'extract_trajectory_from_poses',
    # Block 2: Multimodal Encoder
    'CBRBlock',
    'CBBlock',
    'MultimodalEncoder',
    'build_encoder',
]
