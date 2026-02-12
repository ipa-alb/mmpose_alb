# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMPose is an OpenMMLab toolbox for pose estimation built on PyTorch. It supports 2D/3D human body, hand, face, and animal pose estimation with 20+ algorithms and 40+ datasets. Version 1.3.2, requires Python 3.7+ and PyTorch 1.8+.

Core dependencies: **MMEngine** (0.6.0–1.0.0) and **MMCV** (2.0.0–3.0.0).

## Common Commands

### Installation
```bash
pip install -e .
```

### Training & Testing
```bash
# Single GPU training
python tools/train.py <config_file>

# Distributed training (N GPUs)
bash tools/dist_train.sh <config_file> <num_gpus>

# Testing
python tools/test.py <config_file> <checkpoint_file>
```

### Tests
```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_codecs/test_simcc_label.py

# Run a specific test
pytest tests/test_codecs/test_simcc_label.py::TestSimCCLabel::test_encode -v
```

### Linting & Formatting
```bash
# Run all pre-commit checks
pre-commit run --all-files

# Individual tools
flake8 mmpose/
isort mmpose/
yapf -i -r mmpose/
```

Pre-commit hooks enforce: flake8, isort (line_length=79), yapf (PEP8-based), trailing-whitespace, double-quote-string-fixer, fix-encoding-pragma (--remove), docformatter (wrap at 79), codespell, mdformat, and OpenMMLab copyright headers.

## Architecture

### Config-Driven Design

Everything is driven by Python config files. A typical config inherits from base configs and defines the full pipeline:

```
configs/<task>/<algorithm>/<backbone>_<dataset>_<input_size>.py
```

Tasks: `body_2d_keypoint`, `body_3d_keypoint`, `hand_2d_keypoint`, `face_2d_keypoint`, `wholebody_2d_keypoint`, `animal_2d_keypoint`, etc. Base configs in `configs/_base_/` define shared defaults (datasets, default_runtime).

### Registry System

All modules are registered via MMEngine's Registry pattern (`mmpose/registry.py`). Key registries: MODELS, DATASETS, TRANSFORMS, KEYPOINT_CODECS, METRICS, VISUALIZERS, HOOKS. Each is a child of the corresponding MMEngine root registry. Components are instantiated from config dicts using `type` keys:

```python
dict(type='TopdownPoseEstimator', backbone=dict(type='HRNet', ...))
```

### Model Architecture (mmpose/models/)

Three pose estimator types in `models/pose_estimators/`:
- **TopdownPoseEstimator** — detect person first, then estimate per-crop (most common)
- **BottomupPoseEstimator** — detect all keypoints then group by person
- **PoseLifter** — lifts 2D predictions to 3D

Each estimator composes: **backbone** → **neck** (optional) → **head**. Heads are organized by approach: `heatmap_heads/`, `regression_heads/`, `coord_cls_heads/` (SimCC), `transformer_heads/`.

### Keypoint Codecs (mmpose/codecs/)

Codecs handle encoding ground-truth keypoints into training targets and decoding model outputs back to keypoints. Each codec pairs with specific head types:
- `MSRAHeatmap`, `UDPHeatmap`, `MegviiHeatmap` — for heatmap heads
- `RegressionLabel`, `IntegralRegressionLabel` — for regression heads
- `SimCCLabel` — for coordinate classification (SimCC) heads
- `AssociativeEmbedding`, `SPR` — for bottom-up methods
- `VideoPoseLifting`, `ImagePoseLifting`, `MotionBERTLabel` — for 3D lifting

### Data Pipeline (mmpose/datasets/)

Datasets load annotations; transforms define the augmentation and preprocessing pipeline. The pipeline is specified in config as a list of transform dicts. Key transforms: `LoadImage`, `GetBBoxCenterScale`, `RandomFlip`, `RandomHalfBody`, `TopdownAffine`, `GenerateTarget` (invokes the codec).

### Evaluation (mmpose/evaluation/)

Metrics like `CocoMetric`, `PCKAccuracy`, `AUC`, `EPE` are registered and specified in config. Evaluation runs automatically after validation/test loops.

### Projects Directory (projects/)

Semi-independent community projects (RTMPose, RTMO, RTMPose3D, PoseAnything, etc.) with flexible structure. These can define their own models, codecs, and configs outside the main mmpose package.

## Code Style

- Single quotes for strings (enforced by pre-commit)
- No encoding pragma (removed by pre-commit)
- Imports sorted by isort with `known_first_party = mmpose`
- All new files under `mmpose/`, `tests/`, `demo/`, `tools/` need OpenMMLab copyright headers
- 80% minimum docstring coverage (enforced by interrogate in CI)
