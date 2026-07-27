import torch


class Config:

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # mps for macOS GPU support

    # Transformer config
    batch_size: int = 32
    block_size: int = 256  # Maximum sequence length for model predictions, in tokens
    embedding_dim: int = 384  # Dimensionality of the token embeddings
    num_heads: int = (
        6  # Number of attention heads in the multi-head attention mechanism
    )
    num_layers: int = 3  # Number of transformer blocks (layers) in the model
    dropout: float = 0.1  # Dropout rate for regularization

    # Training config
    learning_rate: float = 3e-4
    epochs: int = 1000
    eval_interval: int = 10  # Interval for evaluating the model

    torch.manual_seed(42)  # Set random seed for reproducibility
