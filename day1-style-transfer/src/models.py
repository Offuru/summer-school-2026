import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


def get_vgg19_model() -> nn.Module:
    """Load pre-trained VGG19 model for feature extraction.

    Returns:
        nn.Module: Pre-trained VGG19 model in eval mode.
    """
    # TODO: Load pre-trained VGG19 with default ImageNet weights and set to eval mode.
    # Use torchvision.models [https://docs.pytorch.org/vision/main/models.html]
    vgg = None  # Replace with the implementation above
    return vgg


def gram_matrix(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Compute Gram matrix for style representation.

    Args:
        x: Feature tensor of shape (batch, channels, height, width).

    Returns:
        Tuple containing:
            - Gram matrix (channels, channels)
            - Number of channels
            - Number of spatial features (height * width)
    """
    # TODO: Extract tensor dimensions, reshape features, and compute Gram matrix.
    # Steps:
    # 1. Retrieve the tensor's dimensions: batch, channels, height, width
    # 2. Squash the 3D tensor into a 2D tensor where the first dimension is the number of channels
    # 3. Compute Gram matrix using the squashed features
    _, c, h, w = ...
    features = ...
    G = ...
    return G, c, h * w


class NormalizationLayer(nn.Module):
    """Normalization layer for preprocessing input images."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Initialize normalization layer with mean and std.

        Args:
            mean: Normalization mean tensor (C,).
            std: Normalization std tensor (C,).
        """
        super(NormalizationLayer, self).__init__()
        self.register_buffer("mean", mean.view(-1, 1, 1))
        self.register_buffer("std", std.view(-1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        # TODO: Normalize input x using self.mean and self.std.
        return ...


class ContentLoss(nn.Module):
    """Content loss module for style transfer."""

    def __init__(self, target: torch.Tensor) -> None:
        """Initialize content loss with target features.

        Args:
            target: Target content features to match.
        """
        super(ContentLoss, self).__init__()
        self.register_buffer("target", target.detach())
        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute content loss.

        Args:
            x: Input features.

        Returns:
            Unchanged input tensor (loss stored in self.loss).
        """
        # TODO: Compute MSE loss between x and self.target, then divide by 2.
        # Use nn.functional [https://docs.pytorch.org/docs/2.13/nn.functional.html]
        self.loss = ...  # Replace with actual loss computation
        return x


class StyleLoss(nn.Module):
    """Style loss module for style transfer."""

    def __init__(self, target: torch.Tensor) -> None:
        """Initialize style loss with target gram matrix.

        Args:
            target: Target feature tensor for style extraction.
        """
        super(StyleLoss, self).__init__()
        # TODO: Compute Gram matrix of target features.
        target_gram = ...
        self.register_buffer("target", target_gram.detach())
        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute style loss using gram matrices.

        Args:
            x: Input features.

        Returns:
            Unchanged input tensor (loss stored in self.loss).
        """
        # TODO: Compute Gram matrix of input features.
        # Steps:
        # 1. Get the Gram matrix of x
        # 2. Compute MSE loss between the Gram matrix of x and self.target
        # 3. Normalize the loss
        G, channels, features = ...  # Replace with gram_matrix(x)
        self.loss = ...  # Replace with normalized MSE loss
        return x
