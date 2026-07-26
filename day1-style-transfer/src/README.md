# Style Transfer Project

## Overview

This is a style transfer implementation using VGG19. It applies the artistic style of one image to the content of another image using feature extraction and loss minimization.

## Project Structure

```
day1-style-transfer/
├── src/                     # Main source code (this directory)
│   ├── __init__.py
│   ├── config.py           # Configuration & hyperparameters
│   ├── models.py           # VGG19 and loss layer definitions (EXERCISE FILE)
│   ├── image_utils.py      # Image loading and tensor conversion
│   ├── mlflow_utils.py     # MLflow logging utilities
│   ├── main.py             # Style transfer pipeline (EXERCISE FILE)
│   └── README.md           # This file
├── img/                     # Image directory structure
│   ├── content/            # Place content images here
│   ├── style/              # Place style images here
│   └── generated/          # Generated outputs saved here
├── pytorch.md              # PyTorch tutorial
├── mlflow.md               # MLflow tutorial
├── setup.md                # Installation and setup guide
├── pyproject.toml          # Project dependencies
└── run_mlflow.bat          # Script to start MLflow UI
```

## Module Descriptions

### `config.py`
Central configuration class for all hyperparameters and settings.

**Key parameters:**
- `IMG_SIZE_GPU` / `IMG_SIZE_CPU`: Image dimensions based on device
- `NUM_STEPS`: Optimization iterations (default: 200)
- `STYLE_WEIGHT` / `CONTENT_WEIGHT`: Loss weighting (style: 1e6, content: 1)
- `CONTENT_LAYERS`: VGG layers for content loss (["conv_4"])
- `STYLE_LAYERS`: VGG layers for style loss (["conv_1" through "conv_5"])
- `MEAN` / `STD`: ImageNet normalization constants
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `MLFLOW_EXPERIMENT_NAME`: Experiment name for tracking

### `models.py` (EXERCISE - Fill in TODOs)
Neural network components for style transfer. Contains several TODO sections:

**Functions to implement:**
- `get_vgg19_model()`: Load pre-trained VGG19 (**TODO**)
- `gram_matrix(x)`: Compute Gram matrix for style (**TODO**)

**Layers to implement:**
- `NormalizationLayer.forward()`: Apply ImageNet normalization (**TODO**)
- `ContentLoss.forward()`: Measure content preservation (**TODO**)
- `StyleLoss.__init__()`: Initialize with Gram matrix (**TODO**)
- `StyleLoss.forward()`: Measure style similarity (**TODO**)

### `image_utils.py`
Utilities for image I/O and tensor conversion.

**Functions:**
- `load_image(image_path, transform, device)`: Load and preprocess image
- `tensor_to_image(tensor)`: Convert tensor to PIL Image
- `validate_image_sizes(image1, image2)`: Verify matching dimensions

### `mlflow_utils.py`
MLflow integration for experiment tracking and model logging.

**Functions:**
- `log_model_info()`: Log model architecture details
- `log_trained_model()`: Save model to MLflow registry

### `main.py`
Main style transfer pipeline.

**Key functions to implement:**
- `setup()`: Initialize device, transforms, VGG model (**TODO**)
- `get_style_transfer_model_and_features()`: Build loss network (**TODO**)
- `run_style_transfer()`: Optimization loop with closure (**TODO**)
- `main()`: Load images and run pipeline (**TODO**)

## Exercise Instructions

This project is designed as an educational exercise. Follow these steps:

1. **Read the Tutorials (if you need a refresher on PyTorch and MLFlow)**
   - Start with [PyTorch Tutorial](../pytorch.md) to understand the basics
   - Read [MLflow Tutorial](../mlflow.md) for experiment tracking concepts

2. **Review the Code Structure**
   - `config.py` is complete - read it first to understand the architecture
   - `image_utils.py` and `mlflow_utils.py` are complete - reference them as needed

3. **Implement TODOs**
   - **`models.py`**: Implement the 6 marked TODO sections:
     - VGG19 model loading
     - Gram matrix computation
     - Normalization, content loss, and style loss layers
   - **`main.py`**: Implement the 8 marked TODO sections:
     - Device setup and transforms
     - Model and features building
     - Optimizer initialization
     - Optimization closure function
     - Image loading and pipeline orchestration

4. **Monitor with MLflow**
   - Run `run_mlflow.bat` to start the MLflow UI
   - View experiment tracking at `http://localhost:8888`

5. **Test Your Implementation**
   - Place test images in `img/content/` and `img/style/`
   - Run: `uv run src/main.py --content test.png --style style.png`
   - Check `img/generated/` for output



## Command Line Arguments

When you have completed the TODOs, run:

```bash
uv run src/main.py --content piata_sfatului.png --style starry_night.png
```

**Arguments:**
- `--content`: Content image filename (in `img/content/`)
- `--style`: Style image filename (in `img/style/`)
- `--generated`: Output image filename (default: "output.png")
- `--use_content_as_input`: Use content image as initialization (default: true)

**Example:**
```bash
uv run src/main.py --content beach.jpg --style van_gogh.jpg --generated beach_van_gogh.png
```

## How It Works

1. **Initialization** : Load pre-trained VGG19 and set up normalization
2. **Feature Extraction** : Extract features from content and style images
3. **Loss Network Construction** : Build network with content & style loss layers
4. **Optimization** : Minimize combined loss using LBFGS
5. **Logging** : Track metrics and model with MLflow
6. **Output** : Save generated image with artistic style applied

## Loss Functions

- **Content Loss**: Measures feature similarity between content and generated image
  - Formula: `MSE(content_features, generated_features) / 2`
  - Preserves the structure and content

- **Style Loss**: Measures style similarity using Gram matrices
  - Formula: `MSE(Gram(style), Gram(generated)) / (4 * C² * (H*W)²)`
  - Captures texture and artistic patterns

- **Total Loss**: Weighted combination
  - Formula: `content_weight * content_loss + style_weight * style_loss`
  - Higher `style_weight` : more emphasis on style
  - Higher `content_weight` : more emphasis on content preservation

## Hyperparameter Tuning Guide

- **`STYLE_WEIGHT`** (default: 1e6): Increase for more artistic style, decrease for more content preservation
- **`CONTENT_WEIGHT`** (default: 1): Increase to preserve content details better
- **`NUM_STEPS`** (default: 200): More steps = better quality but slower (100-500 recommended)
- **`IMG_SIZE`** (512 GPU / 256 CPU): Larger size = higher quality but more memory
- **`CONTENT_LAYERS`**: Deeper layers preserve more structure (typically ["conv_4"])
- **`STYLE_LAYERS`**: Multiple layers capture different scale styles

## Troubleshooting

**AttributeError: module has no attribute**
- Ensure all TODOs in `models.py` and `main.py` are implemented
- Check that variable names match between functions

**RuntimeError: CUDA out of memory**
- Reduce `IMG_SIZE_GPU` in config.py
- Reduce `NUM_STEPS`

**Images not found**
- Verify images are in correct subdirectories: `img/content/` and `img/style/`
- Check filename spelling in command line arguments

**No MLflow UI**
- Run `run_mlflow.bat` before executing the main script
- Check that tracking URI matches in `config.py`

## Technical Details

### Architecture
- **Model**: VGG19 (19 convolutional layers pre-trained on ImageNet)
- **Optimizer**: L-BFGS (second-order optimization, effective for this task)
- **Device**: GPU (CUDA) or CPU (auto-detected)

### Key Technologies
- **PyTorch**: Deep learning framework
- **TorchVision**: Pre-trained models and image transforms
- **MLflow**: Experiment tracking and model logging
- **Pillow (PIL)**: Image processing

### Performance Notes
- GPU mode uses 512×512 images for better quality
- CPU mode uses 256×256 images for memory efficiency
- LBFGS typically converges in 100-500 steps
- Each step involves 1 forward pass + 1 backward pass through VGG19
