from pathlib import Path
import tiktoken
import requests
import torch
from config import Config


def load_data():
    """Load the Tiny Shakespeare dataset.

    Returns:
        str: Raw text of the dataset.
    """
    # Load the dataset
    data_path = Path("data/tinyshakespeare.txt")
    if not data_path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading dataset from {url}...")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(requests.get(url).text)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def encode_text(text: str) -> tuple[torch.Tensor, tiktoken.Encoding]:
    """Encode text using BPE tokenizer.

    Args:
        text: Raw text string.

    Returns:
        Tuple containing:
            - encoded_text: Tensor of encoded token indices.
            - tokenizer: Tokenizer object for encoding/decoding text.
    """
    # Load BPE Tokenizer (GPT-2 standard)
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(
        Config.device
    )
    return encoded_text, tokenizer


def get_batch(data: torch.Tensor):
    """Generate a batch of data for training.

    Args:
        data: Tensor of encoded token indices.

    Returns:
        Tuple containing:
            - x: Input batch tensor.
            - y: Target batch tensor.
    """

    # TODO: Generate random batch indices
    # 1. Create random indices for batch start positions
    # 2. Use torch.randint to generate a tensor of random start indices in the range [0, len(data) - Config.block_size)
    # 3. Stack input sequences (x) and target sequences (y) of size (Config.block_size)
    # 4. Target is offset by 1 token from input
    # 5. Move both tensors to Config.device before returning
    x = ...
    y = ...

    return x.to(Config.device), y.to(Config.device)
