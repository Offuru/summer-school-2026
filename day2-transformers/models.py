import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"

        # TODO: Initialize attention components
        # 1. Store embedding_dim, num_heads, and compute head_dim (embedding_dim // num_heads)
        # 2. Create linear layers for QKV projection and output projection
        # 3. Create dropout layers for attention and residual connections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement multi-head attention forward pass
        # 1. Extract batch size, sequence length from input x
        # 2. Project input to Q, K, V using qkv_proj and split into heads
        # 3. Compute attention logits: (Q @ K.T) / sqrt(head_dim)
        # 4. Apply causal mask to prevent attending to future positions
        # 5. Apply softmax and dropout to get attention weights
        # 6. Apply attention weights to values and concatenate heads
        # 7. Project output and apply residual dropout
        pass


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        # TODO: Create feed-forward network
        # Use nn.Sequential with:
        # 1. Linear layer: embedding_dim -> ff_dim
        # 2. GELU activation
        # 3. Linear layer: ff_dim -> embedding_dim
        # 4. Dropout for regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Pass input through feed-forward network
        pass


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        # TODO: Assemble transformer block components
        # 1. Create MultiheadAttention module
        # 2. Create FeedForward module with ff_dim = 4 * embedding_dim
        # 3. Create two LayerNorm modules for pre-normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement transformer block forward pass with residual connections
        # 1. Apply layer norm to input
        # 2. Pass through self-attention and add residual connection
        # 3. Apply layer norm to result
        # 4. Pass through feed-forward and add residual connection
        # 5. Return final output
        pass


class PositionalEncoding(nn.Module):
    def __init__(self, n_embd: int, max_seq_len: int = 1024):
        super().__init__()

        # TODO: Compute sinusoidal positional encodings
        # 1. Create pe tensor of shape (max_seq_len, n_embd)
        # 2. Compute position indices and division term
        # 3. Apply sine to even dimensions and cosine to odd dimensions
        # 4. Add batch dimension and register as buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Add positional encodings to input
        # 1. Extract sequence length from input x
        # 2. Add self.pe to the input (first seq_len positions)
        # 3. Return the result with positional encodings added
        pass


class ShakespeareTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # TODO: Initialize transformer model components
        # 1. Store block_size for later use in generate()
        # 2. Create token embedding table (vocab_size -> embedding_dim)
        # 3. Create positional encoding with max_seq_len=block_size
        # 4. Create stack of transformer blocks using nn.Sequential
        # 5. Create final layer normalization
        # 6. Create output linear layer (embedding_dim -> vocab_size)
        # 7. Share weights between token embedding and output layer

    def forward(self, idx: torch.Tensor, targets=None):
        # TODO: Implement forward pass for transformer language model
        # 1. Extract batch size and sequence length from input idx
        # 2. Get token embeddings using embedding table
        # 3. Add positional encodings to embeddings
        # 4. Pass through all transformer blocks
        # 5. Apply final layer normalization
        # 6. Project to vocabulary logits using output layer
        # 7. If targets provided, compute cross-entropy loss:
        #    - Reshape logits and targets to (batch*seq_len, vocab_size) and (batch*seq_len,)
        #    - Use F.cross_entropy() for loss computation
        # 8. Return (logits, loss) tuple
        pass

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate new tokens autoregressively.

        Args:
            idx: Initial token indices of shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate

        Returns:
            Generated token indices of shape (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # TODO: Generate next token in autoregressive manner
            # 1. Crop input to last block_size tokens (context window)
            # 2. Get logits from forward pass
            # 3. Extract logits for last position (next token prediction)
            # 4. Convert logits to probabilities with softmax
            # 5. Sample next token from distribution using multinomial
            # 6. Concatenate to sequence and repeat
            pass

        return idx
