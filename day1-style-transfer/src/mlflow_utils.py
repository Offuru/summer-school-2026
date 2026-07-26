import mlflow
import torch
import torch.nn as nn


def log_model_info(model: nn.Module) -> None:
    """Log model architecture and parameter information to MLflow.

    Args:
        model: PyTorch model to log.
    """
    model_summary = str(model)
    mlflow.log_text(model_summary, "model_summary.txt")

    layer_info = []
    for name, module in model.named_modules():
        layer_info.append(f"{name}: {module.__class__.__name__}")
    mlflow.log_text("\n".join(layer_info), "layer_info.txt")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    mlflow.log_param("total_params", total_params)
    mlflow.log_param("trainable_params", trainable_params)


@torch.no_grad()
def log_trained_model(model: nn.Module, imsize: int, device: torch.device) -> None:
    """Log trained model to MLflow.

    Args:
        model: PyTorch model to log.
        imsize: Image size for reference.
        device: Device model is on (cpu or cuda).
    """
    mlflow.pytorch.log_model(
        model,
        "style_transfer_model",
        serialization_format="pickle",
    )
