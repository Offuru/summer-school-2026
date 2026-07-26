import copy
import argparse
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision import transforms
from typing import Tuple, List, Optional

try:
    from .config import Config
    from . import image_utils
    from . import mlflow_utils
    from . import models
except ImportError:
    from config import Config
    import image_utils
    import mlflow_utils
    import models

device = None
image_size = None
image_transform = None
vgg = None


def setup() -> None:
    """Initialize device, image transform, VGG model, and image directories."""
    global device, image_size, image_transform, vgg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = Config.IMG_SIZE_GPU if device.type == "cuda" else Config.IMG_SIZE_CPU
    # TODO: Create image transformation pipeline.
    # Use transforms.Compose
    # The pipeline should include a resize to an appropriate size and conversion to tensor. Check torchvision.transforms [https://docs.pytorch.org/vision/0.8/transforms.html]
    image_transform = ...  # Replace with the transformations above
    # TODO: Load pre-trained VGG19 model features and move to device.
    vgg = ...  # Replace with the implementation above

    img_path = Path("img")

    if not (img_path / "content").exists():
        (img_path / "content").mkdir(parents=True, exist_ok=True)
    if not (img_path / "style").exists():
        (img_path / "style").mkdir(parents=True, exist_ok=True)
    if not (img_path / "generated").exists():
        (img_path / "generated").mkdir(parents=True, exist_ok=True)


def get_style_transfer_model_and_features(
    cnn: nn.Module,
    normalization_mean: List[float],
    normalization_std: List[float],
    style_img: torch.Tensor,
    content_img: torch.Tensor,
    content_layers: Optional[List[str]] = None,
    style_layers: Optional[List[str]] = None,
) -> Tuple[nn.Sequential, List[models.ContentLoss], List[models.StyleLoss]]:
    """Build style transfer model with loss layers.

    Args:
        cnn: Pre-trained VGG19 feature extractor.
        normalization_mean: ImageNet normalization mean.
        normalization_std: ImageNet normalization std.
        style_img: Style image tensor.
        content_img: Content image tensor.
        content_layers: Layers to compute content loss. Defaults to Config.CONTENT_LAYERS.
        style_layers: Layers to compute style loss. Defaults to Config.STYLE_LAYERS.

    Returns:
        Tuple of (model, content_losses, style_losses).
    """
    if content_layers is None:
        content_layers = Config.CONTENT_LAYERS
    if style_layers is None:
        style_layers = Config.STYLE_LAYERS

    cnn = copy.deepcopy(cnn)

    # TODO: Create tensors for normalization mean and std.
    # Convert the input lists to tensors and reshape to (C, 1, 1) for broadcasting.
    normalization_mean = ...  # Replace with actual mean
    normalization_std = ...  # Replace with actual std

    # TODO: Create a NormalizationLayer using the tensors above. Don't forget to move it to the correct device.
    normalization = ...  # Replace with the implementation above

    content_losses = []
    style_losses = []

    # TODO: Initialize the style transfer model as nn.Sequential [https://docs.pytorch.org/docs/2.13/generated/torch.nn.Sequential.html].
    # Start it with the normalization layer.
    style_transfer_model = ...

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer type: {layer}")

        style_transfer_model.add_module(name, layer)

        if name in content_layers:
            target = style_transfer_model(content_img).detach()
            content_loss = models.ContentLoss(target).to(device)
            style_transfer_model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = style_transfer_model(style_img).detach()
            style_loss = models.StyleLoss(target_feature).to(device)
            style_transfer_model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(style_transfer_model) - 1, -1, -1):
        if isinstance(style_transfer_model[i], (models.ContentLoss, models.StyleLoss)):
            break

    style_transfer_model = style_transfer_model[: i + 1]

    return style_transfer_model, content_losses, style_losses


def run_style_transfer(
    cnn: nn.Module,
    normalization_mean: List[float],
    normalization_std: List[float],
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    input_img: Optional[torch.Tensor] = None,
    num_steps: Optional[int] = None,
    style_weight: Optional[float] = None,
    content_weight: Optional[float] = None,
) -> torch.Tensor:
    """Run style transfer optimization.

    Args:
        cnn: Pre-trained VGG19 feature extractor.
        normalization_mean: ImageNet normalization mean.
        normalization_std: ImageNet normalization std.
        content_img: Content image tensor.
        style_img: Style image tensor.
        input_img: Initial input tensor. Defaults to random noise.
        num_steps: Number of optimization steps. Defaults to Config.NUM_STEPS.
        style_weight: Weight for style loss. Defaults to Config.STYLE_WEIGHT.
        content_weight: Weight for content loss. Defaults to Config.CONTENT_WEIGHT.

    Returns:
        Optimized image tensor.
    """
    num_steps = num_steps or Config.NUM_STEPS
    style_weight = style_weight or Config.STYLE_WEIGHT
    content_weight = content_weight or Config.CONTENT_WEIGHT
    if input_img is None:
        input_img = torch.randn_like(content_img).to(device)

    params = {
        "num_steps": num_steps,
        "style_weight": style_weight,
        "content_weight": content_weight,
        "image_size": image_size,
        "content_layers": Config.CONTENT_LAYERS,
        "style_layers": Config.STYLE_LAYERS,
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
    }
    mlflow.log_params(params)

    # TODO: Build the style transfer model and extract loss layers.
    style_transfer_model, content_losses, style_losses = (
        ...
    )  # Replace with model from function call

    input_img.requires_grad_(True)
    # TODO: Create an LBFGS optimizer for the input image.
    # Use: optim.LBFGS [https://docs.pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html]
    optimizer = ...  # Replace with LBFGS optimizer

    run = [0]

    while run[0] < num_steps:

        def closure():
            # TODO: Implement the optimization closure function.
            # Steps:
            # 1. Clamp input_img to [0, 1], make sure gradients aren't computed during this step.
            # 2. Reset optimizer gradients to zero.
            # 3. Run forward pass / inference
            # 4. Compute total content and style losses
            # 5. Compute total loss as weighted sum of content and style losses
            # 6. Run backpropagation on total loss
            content_loss = ...
            style_loss = ...
            total_loss = ...
            pass  # Replace with the implementation above

            run[0] += 1

            if run[0] % 50 == 0:
                print(
                    f"Run {run[0]}: Total loss - {total_loss.item()}, Content loss - {content_loss.item()}, Style loss - {style_loss.item()}"
                )

                # TODO: Log metrics to MLflow for monitoring training progress [https://mlflow.org/docs/latest/ml/tracking/tracking-api/].
                # Log total_loss, content_loss, and style_loss
                pass  # Replace with MLflow logging calls

            return total_loss

        # TODO: Execute the optimizer step with the closure function.
        pass  # Replace with the step

    with torch.no_grad():
        input_img.clamp_(0, 1)

    mlflow_utils.log_model_info(style_transfer_model)
    mlflow_utils.log_trained_model(style_transfer_model, image_size, device)

    return input_img


def main(args: argparse.Namespace) -> None:
    """Run style transfer pipeline.

    Args:
        args: Command-line arguments containing content, style, generated image names.
    """
    setup()

    mlflow.set_tracking_uri(uri=Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():

        content_path = f"img/content/{args.content}"
        style_path = f"img/style/{args.style}"

        # TODO: Load content and style images using image_utils.load_image().
        content_img = ...  # Replace with loaded content image
        style_img = ...  # Replace with loaded style image

        if content_img is None or style_img is None:
            raise RuntimeError("Failed to load content or style image")

        input_img = (
            content_img.clone().to(device) if args.use_content_as_input else None
        )

        normalization_mean = Config.MEAN
        normalization_std = Config.STD

        # TODO: Run the style transfer optimization.
        output = ...

        generated_image = image_utils.tensor_to_image(output)
        generated_path = f"img/generated/{args.generated}"
        generated_image.save(generated_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style Transfer using VGG19")

    parser.add_argument(
        "--content",
        type=str,
        required=True,
        help="Name of the content image, from img/content/",
    )
    parser.add_argument(
        "--style",
        type=str,
        required=True,
        help="Name of the style image, from img/style/",
    )
    parser.add_argument(
        "--generated",
        type=str,
        default="output.png",
        help="Name of the generated image to save",
    )
    parser.add_argument(
        "--use_content_as_input",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Use content image as input for style transfer, default is True",
    )

    args = parser.parse_args()

    main(args)
