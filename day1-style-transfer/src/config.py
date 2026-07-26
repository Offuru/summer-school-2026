from typing import List


class Config:
    """Configuration class for style transfer hyperparameters and settings."""

    # Image size
    IMG_SIZE_GPU: int = 512
    IMG_SIZE_CPU: int = 256

    # Hyperparameters
    NUM_STEPS: int = 200
    LOG_INTERVAL: int = 50
    STYLE_WEIGHT: float = 1e6
    CONTENT_WEIGHT: float = 1
    CONTENT_LAYERS: List[str] = ["conv_4"]
    STYLE_LAYERS: List[str] = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    # ImageNet1K normalization tensors
    MEAN: List[float] = [0.485, 0.456, 0.406]
    STD: List[float] = [0.229, 0.224, 0.225]

    # MLflow parameters
    MLFLOW_EXPERIMENT_NAME: str = "style_transfer"
    MLFLOW_TRACKING_URI: str = "http://localhost:8888"
