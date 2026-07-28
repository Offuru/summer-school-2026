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

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embedding_dim = x.size()

        qkv = self.qkv_proj(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        logits = q @ k.transpose(-2, -1) / np.sqrt(self.head_dim)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(x.device)

        logits = logits.masked_fill(~causal_mask, float("-inf"))

        attention_weights = F.softmax(logits, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        out = attention_weights @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, embedding_dim)
        out = self.residual_dropout(self.out_proj(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiheadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embedding_dim, 4 * embedding_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, n_embd: int, max_seq_len: int = 1024):
        super().__init__()

        pe = torch.zeros(max_seq_len, n_embd)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2).float() * (-np.log(10000.0) / n_embd)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len]


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

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_seq_len=block_size
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embedding_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.layer_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        self.token_embedding_table.weight = self.layer_head.weight

    def forward(self, idx: torch.Tensor, targets=None):
        batch, seq_len = idx.shape

        token_embeddings = self.token_embedding_table(idx)

        x = self.positional_encoding(token_embeddings)

        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.layer_head(x)

        loss = None

        if targets is not None:
            batch, seq_len, vocab_size = logits.shape
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
