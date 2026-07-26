from PIL import Image
from torchvision import transforms
import torch
from typing import Optional


def load_image(
    image_path: str, image_transform: transforms.Compose, device: torch.device
) -> Optional[torch.Tensor]:
    """Load and transform an image from file.

    Args:
        image_path: Path to the image file.
        image_transform: Torchvision transforms to apply.
        device: Device to load tensor to (cpu or cuda).

    Returns:
        Transformed image tensor of shape (1, 3, H, W), or None if loading failed.
    """
    try:
        image = Image.open(image_path)
        image = image_transform(image).unsqueeze(0).to(device)
        return image
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return None


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor to a PIL Image.

    Args:
        tensor: Tensor of shape (1, 3, H, W) or (3, H, W).

    Returns:
        PIL Image object.
    """
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


def validate_image_sizes(image1: Image.Image, image2: Image.Image) -> bool:
    """Validate that two images have the same size.

    Args:
        image1: First image.
        image2: Second image.

    Returns:
        True if sizes match.

    Raises:
        ValueError: If image sizes don't match.
    """
    if image1.size != image2.size:
        raise ValueError(f"Image sizes do not match: {image1.size} vs {image2.size}")
    return True
