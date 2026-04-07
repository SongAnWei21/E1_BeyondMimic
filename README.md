# E1_BeyondMimic: Whole-Body Imitation Learning for Humanoid Robots

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-0.5.0-green.svg)](https://isaac-sim.github.io/IsaacLab/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A reinforcement learning framework for whole-body imitation learning of humanoid robots, built on NVIDIA Isaac Lab. This project enables training and deployment of imitation policies for various humanoid robots including E1, Hi, Pi Plus, N1, and X1 series.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Supported Robots](#supported-robots)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Overview

E1_BeyondMimic is a research framework for whole-body motion imitation learning using reinforcement learning. It leverages NVIDIA Isaac Lab for simulation and provides tools for:
- Training imitation policies from motion capture data
- Deploying trained policies to real robots via ONNX/RKNN conversion
- Visualizing and replaying motion sequences
- Curriculum learning with adaptive force assistance

The framework implements advanced techniques such as adaptive sampling, force curriculum learning, and multi-robot support for robust imitation learning.

## Features

- **Multi-Robot Support**: Unified interface for E1 (19/23 DOF), Hi, Pi Plus, N1, and X1 humanoid robots
- **Motion Imitation**: Load and replay motion capture data from NPZ files
- **Adaptive Sampling**: Intelligent sampling of motion sequences based on failure regions
- **Force Curriculum**: Automatic adjustment of assistance forces during training
- **Export to Edge**: Convert trained policies to ONNX and RKNN formats for deployment on embedded devices (RK3588)
- **Weights & Biases Integration**: Logging and artifact management
- **Interactive Visualization**: Real-time visualization of robot states and forces in Isaac Sim

## Supported Robots

| Robot Model | DOF | Description |
|-------------|-----|-------------|
| E1 (19 DOF) | 19 | Bipedal humanoid with simplified upper body |
| E1 (23 DOF) | 23 | Full-body bipedal humanoid |
| Hi | 20 | High-torque humanoid robot |
| Pi Plus | 20 | Compact humanoid platform |
| N1 | 16 | Lightweight humanoid |
| X1 (23 DOF) | 23 | Advanced humanoid with enhanced mobility |

## Installation

### Prerequisites

- **NVIDIA Isaac Lab 0.5.0+** (requires NVIDIA Omniverse)
- **Python 3.8+**
- **CUDA 11.8+** (for GPU acceleration)
- **RKNN Toolkit 2** (optional, for deployment to Rockchip NPU)

### Step 1: Set up Isaac Lab

Follow the official [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/installation.html) to set up your environment.

### Step 2: Install Python Dependencies

```bash
pip install -e .
```

Additional dependencies are managed by Isaac Lab. The core package requirements are:
- `mujoco==3.3.3`
- `mujoco-python-viewer`
- `psutil`
- `joblib>=1.2.0`
- `pynput`

### Step 3: Set up Robot Assets

Ensure robot URDF/MJCF files are properly placed in the `e1_lab/assets/` directory. The project includes pre-configured models for all supported robots.

## Quick Start

### Training an Imitation Policy

1. Prepare motion data in NPZ format (joint positions, velocities, body poses)
2. Configure training parameters in `e1_lab/tasks/tracking/config/`
3. Launch training:

```bash
python -m e1_lab.tasks.tracking.train --robot e1 --motion_file path/to/motion.npz
```

### Replaying Motion Capture Data

Use the replay script to visualize motion sequences:

```bash
# For E1 robot
python scripts/replay_npz.py --robot e1 --motion_file path/to/motion.npz

# For Hi robot
python scripts/replay_npz.py --robot hi --motion_file path/to/motion.npz
```

### Converting Policy for Deployment

Convert trained ONNX models to RKNN format for edge deployment:

```bash
python scripts/convert2rknn.py
```

Modify the script to point to your trained ONNX model and target platform.

## Usage

### Motion Command System

The core of the framework is the `MotionCommand` class (`e1_lab/tasks/tracking/mdp/commands.py`), which:
- Loads motion sequences from NPZ files
- Provides adaptive sampling based on failure regions
- Implements force curriculum learning for difficult poses
- Computes imitation rewards and metrics

### Configuration

Training configurations are located in `e1_lab/tasks/tracking/config/`. Each robot has dedicated configuration files for:
- Environment settings
- Reward weights
- Curriculum parameters
- Observation/action spaces

### Custom Motion Data

To use custom motion capture data:
1. Convert your data to NPZ format with the following arrays:
   - `joint_pos`: [T, N_joints] - joint positions
   - `joint_vel`: [T, N_joints] - joint velocities
   - `body_pos_w`: [T, N_bodies, 3] - body positions in world frame
   - `body_quat_w`: [T, N_bodies, 4] - body orientations (quaternions)
   - `body_lin_vel_w`: [T, N_bodies, 3] - body linear velocities
   - `body_ang_vel_w`: [T, N_bodies, 3] - body angular velocities
   - `fps`: int - frames per second

2. Update the body name mapping in your configuration to match your robot's URDF

## Project Structure

```
E1_BeyondMimic/
├── e1_lab/                    # Core Python package
│   ├── assets/               # Robot assets (URDF, MJCF, meshes)
│   │   ├── e1/              # E1 robot models (19/23 DOF)
│   │   ├── e1_19dof/        # E1 19 DOF variant
│   │   └── e1_21dof/        # E1 21 DOF variant
│   ├── robots/              # Robot configuration classes
│   └── tasks/tracking/      # Imitation learning task
│       ├── config/          # Training configurations per robot
│       └── mdp/             # MDP components (commands, events)
├── scripts/                  # Utility scripts
│   ├── convert2rknn.py      # ONNX to RKNN conversion
│   ├── replay_npz.py        # Motion sequence visualization
│   ├── upload_npz.py        # W&B artifact upload
│   └── csv_cut.py           # Data processing utilities
├── setup.py                 # Package installation
└── README.md               # This file
```

## Scripts

- **`convert2rknn.py`**: Convert trained ONNX policies to RKNN format for Rockchip NPU deployment
- **`replay_npz.py`**: Interactive motion replay in Isaac Sim with support for all robots
- **`upload_npz.py`**: Upload motion data to Weights & Biases registry
- **`csv_cut.py`**: Process and filter CSV motion data
- **`rsl_rl/cli_args.py`**: RSL-RL training configuration utilities

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{e1_beyondmimic,
  title = {E1\_BeyondMimic: Whole-Body Imitation Learning for Humanoid Robots},
  author = {Song, Anwei},
  year = {2025},
  url = {https://github.com/username/E1_BeyondMimic},
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- Uses [RSL-RL](https://github.com/leggedrobotics/rsl_rl) for reinforcement learning
- Motion data processing inspired by [Lafan1](https://github.com/DeepMotionEditing/deep-motion-editing) dataset

