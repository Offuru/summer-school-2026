# Transformer Implementation Details

## Table of Contents
1. [Multi-Head Attention](#multi-head-attention)
2. [Positional Encoding](#positional-encoding)
3. [Tensor Operations & Reshaping](#tensor-operations--reshaping)
4. [Feed-Forward Networks](#feed-forward-networks)
5. [Residual Connections](#residual-connections)
6. [Causal Masking](#causal-masking)

---

## Multi-Head Attention

### The Problem
Traditional attention weights every input position equally when computing the output for a position. Multi-head attention solves this by learning multiple different ways to attend to the input, each focusing on different aspects of the data.

### QKV Projection Strategy

The key insight is that we want to compute attention between Query (Q), Key (K), and Value (V) representations, but we want to do this for **multiple "heads" simultaneously** for efficiency.

**Composed Matrix Approach:**
- Instead of creating separate linear layers for Q, K, and V (3 separate operations), we create **one large linear layer** that outputs Q, K, V concatenated together
- This single layer transforms input of size `(batch, seq_len, embedding_dim)` to `(batch, seq_len, 3*embedding_dim)`
- You then split this output into three equal parts: Q, K, V, each of size `(batch, seq_len, embedding_dim)`
- The `chunk()` or indexing operations can split the output efficiently

**Why compose them?**
- Computational efficiency: one matrix multiplication instead of three
- PyTorch can optimize better with larger batches of operations
- Reduced memory overhead

### Head Splitting

After getting Q, K, V:

1. **Reshape for multi-head attention**: Transform from `(batch, seq_len, embedding_dim)` to `(batch, seq_len, num_heads, head_dim)`
   - `head_dim = embedding_dim // num_heads`
   - This separates the embedding dimension into different "heads"

2. **Transpose for batch operations**: Rearrange to `(batch, num_heads, seq_len, head_dim)`
   - This groups all sequences for each head together
   - Allows parallel computation of attention across all heads

**Example dimensions with embedding_dim=384, num_heads=6, head_dim=64:**
- Q/K/V after projection: (batch, seq_len, 384)
- After reshape: (batch, seq_len, 6, 64)
- After transpose: (batch, 6, seq_len, 64)

### Attention Computation

With reshaped tensors, compute attention scores:

```
scores = (Q @ K^T) / sqrt(head_dim)
```

**Multi-dimensional matrix multiplication:**
- Q shape: `(batch, num_heads, seq_len, head_dim)`
- K^T shape: `(batch, num_heads, head_dim, seq_len)` (last two dims transposed)
- Result shape: `(batch, num_heads, seq_len, seq_len)`

The `@` operator in PyTorch automatically broadcasts over the batch and head dimensions, computing attention for all batch samples and all heads in parallel.

**Why divide by sqrt(head_dim)?**
- Prevents attention scores from becoming too large
- Maintains stable gradients during backpropagation
- Standard normalization in transformer attention

### Output Projection

After attention weighting:

1. **Concatenate heads**: Transform from `(batch, num_heads, seq_len, head_dim)` back to `(batch, seq_len, embedding_dim)`
   - Use transpose to move num_heads back to embedding: `(batch, seq_len, num_heads, head_dim)`
   - Use reshape/view to flatten the last two dimensions: `(batch, seq_len, embedding_dim)`

2. **Apply output projection**: Linear layer from `embedding_dim` to `embedding_dim`
   - Learns how to combine information from different heads
   - Same shape transformation as input

---

## Positional Encoding

### The Problem
Transformers have no inherent sense of sequence order because they treat all positions equally through attention. Without position information, the model cannot distinguish between "The cat sat on the mat" and "The mat sat on the cat."

### Sinusoidal Encoding Strategy

Rather than learning position embeddings, we use **sinusoidal functions** with different frequencies:

**Encoding formula:**
- For even dimensions (0, 2, 4, ...): PE[pos, i] = sin(pos / 10000^(i/d))
- For odd dimensions (1, 3, 5, ...): PE[pos, i] = cos(pos / 10000^((i-1)/d))

Where:
- `pos` is the position in the sequence
- `i` is the dimension index
- `d` is the total embedding dimension

**Implementation approach:**

1. **Create position indices**: Tensor of shape `(max_seq_len,)` with values [0, 1, 2, ..., max_seq_len-1]

2. **Compute division term**: This controls frequency variation
   - Create a term that decreases exponentially: 10000^(2i/d)
   - Apply this to pairs of dimensions to get different frequency bands
   - Apply logarithm and exponential for numerical stability

3. **Apply sine and cosine**:
   - Split the PE tensor into even and odd dimensions
   - Apply sine to even dimensions (0, 2, 4, ...)
   - Apply cosine to odd dimensions (1, 3, 5, ...)

4. **Add batch dimension**: Use `unsqueeze()` to add a batch dimension for broadcasting

5. **Register as buffer**: Use `register_buffer()` so the PE tensor stays on the device with the model, but isn't updated during training

**Why sinusoidal encoding?**
- No learned parameters (unlike embedding tables)
- Works for any sequence length (can extrapolate beyond training length)
- Different frequencies capture both short and long-range relationships
- Mathematical properties allow efficient addition to embeddings

### Adding to Embeddings

During forward pass:
- Extract PE for the actual sequence length: `PE[:, :seq_len]`
- Add directly to token embeddings: `embeddings + positional_encoding`
- Broadcasting handles the batch dimension automatically

---

## Tensor Operations & Reshaping

### Understanding View vs Reshape

Both `view()` and `reshape()` change tensor dimensions without copying data (when possible).

**Key difference:**
- `view()`: Requires tensor to be contiguous in memory; faster but stricter
- `reshape()`: More flexible, may copy if needed for contiguity

**Common pattern:**
```
Before: (batch, seq_len, num_heads, head_dim)
After:  (batch, seq_len, embedding_dim)
```

This combines the last two dimensions: `num_heads * head_dim = embedding_dim`

### Transpose for Batched Operations

The `transpose()` operation swaps two dimensions:
- `transpose(-2, -1)`: Swaps last two dimensions (common for matrix transpose)
- `transpose(1, 2)`: Swaps dimensions 1 and 2 (common for moving head dimension)

**Why important for attention:**
- Initial Q, K, V: `(batch, seq_len, embedding_dim)`
- Need for attention: `(batch, num_heads, seq_len, head_dim)`
- This ordering puts the sequence and head dimensions where matrix ops need them

### Batched Matrix Multiplication

PyTorch's `@` operator automatically broadcasts over batch and other dimensions:

```
(batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
= (batch, num_heads, seq_len, seq_len)
```

This is equivalent to computing `num_heads * batch` separate matrix multiplications in parallel.

---

## Feed-Forward Networks

### Two-Layer MLP Structure

The feed-forward component uses an "expand-then-contract" pattern:

1. **Expansion layer**: Linear from `embedding_dim` to `ff_dim` (typically 4x)
   - For example: 384 → 1536 (4 * 384)
   - Increases representation capacity

2. **Activation function**: GELU (Gaussian Error Linear Unit)
   - Smooth, non-linear activation
   - More sophisticated than ReLU; works well with transformers
   - Element-wise operation: doesn't change tensor shape

3. **Contraction layer**: Linear from `ff_dim` back to `embedding_dim`
   - For example: 1536 → 384
   - Projects back to original dimension for residual connection

**Why 4x expansion?**
- Empirically works well for transformers
- Provides more expressive power through width
- Common convention in transformer implementations

### Sequential Composition

Use `nn.Sequential()` to stack these layers. This automatically handles:
- Forward pass through each layer in order
- Shape transformations: (batch, seq_len, embedding_dim) → (batch, seq_len, ff_dim) → (batch, seq_len, embedding_dim)

---

## Residual Connections

### The Concept

Residual connections (skip connections) add the input directly to the output:

```
output = input + sublayer(input)
```

Instead of:

```
output = sublayer(input)
```

### Benefits

1. **Gradient flow**: Gradients can flow directly through the residual path during backpropagation
2. **Identity preservation**: Easy for the network to learn identity mappings if the sublayer isn't helpful
3. **Deeper networks**: Enables training of much deeper networks without vanishing gradients

### Pre-Norm Architecture

In this implementation, we use **pre-normalization**:

```
output = input + sublayer(LayerNorm(input))
```

Instead of post-norm:

```
output = LayerNorm(input + sublayer(input))
```

**Pre-norm advantages:**
- Better gradient flow
- More stable training
- Works better for deeper models

### Implementation Pattern

For each transformer block:
1. Apply LayerNorm to input
2. Pass through attention sublayer
3. Add to original input (residual)
4. Apply LayerNorm to result
5. Pass through feed-forward sublayer
6. Add to the result from step 3 (residual)

---

## Causal Masking

### The Problem

In language modeling, we want to predict the next token based on previous tokens. However, during training, the attention mechanism could "cheat" by looking at future tokens.

Causal masking prevents this by masking out future positions.

### Mask Creation

Create a **lower triangular matrix** of shape `(seq_len, seq_len)`:

```
Example for seq_len=4:
[1 0 0 0]
[1 1 0 0]
[1 1 1 0]
[1 1 1 1]
```

This matrix represents which positions can attend to which:
- Row i can attend to columns 0 through i (past and current positions)
- Row i cannot attend to columns i+1 onward (future positions)

**PyTorch function**: `torch.tril()` creates lower triangular matrices

### Applying the Mask

1. **Create boolean mask**: Convert to boolean tensor (True where attention is allowed, False where not)

2. **Apply to attention scores**: 
   - Before softmax, set masked positions to `-infinity`
   - Use `masked_fill()` with the inverted mask (~mask)
   - Softmax of -infinity becomes 0, so those positions contribute nothing to attention

3. **Move to correct device**: Ensure mask is on the same device as attention scores

### Device Handling

Remember to move the mask to the input device using `.to(x.device)`. This ensures compatibility when using GPU.

---

## Training Loop Considerations

### Gradient Flow

With multiple layers and residual connections, ensure gradients propagate well:
- Dropout helps regularize and prevents co-adaptation
- Use smaller learning rates for stable training

### Batch Dimension

Always maintain the batch dimension throughout:
- Input: `(batch, seq_len)` — token indices
- After embedding: `(batch, seq_len, embedding_dim)`
- After each layer: Same shape maintained
- Output logits: `(batch, seq_len, vocab_size)`

### Loss Computation

Cross-entropy loss expects:
- Predictions: `(batch*seq_len, vocab_size)`
- Targets: `(batch*seq_len,)` — single token indices, not one-hot

Reshape tensors appropriately: flatten the batch and sequence dimensions together.

---

## Key Tensor Dimension Tracking

Here's a reference table for tensor shapes through the model:

| Stage | Shape | Notes |
|-------|-------|-------|
| Input tokens | `(batch, seq_len)` | Token indices |
| Token embedding | `(batch, seq_len, embedding_dim)` | 384 |
| + Positional encoding | `(batch, seq_len, embedding_dim)` | Added element-wise |
| QKV projection | `(batch, seq_len, 3*embedding_dim)` | Combined before splitting |
| Q, K, V (separated) | `(batch, seq_len, embedding_dim)` | Each individually |
| Reshaped for heads | `(batch, seq_len, num_heads, head_dim)` | 64 per head |
| After transpose | `(batch, num_heads, seq_len, head_dim)` | Ready for matmul |
| Attention scores | `(batch, num_heads, seq_len, seq_len)` | Score for each position pair |
| Attention weights | `(batch, num_heads, seq_len, seq_len)` | Normalized with softmax |
| After attention × Values | `(batch, num_heads, seq_len, head_dim)` | Context vector per head |
| Concatenated heads | `(batch, seq_len, embedding_dim)` | Back to original dim |
| After output projection | `(batch, seq_len, embedding_dim)` | Ready for residual add |
| After feed-forward | `(batch, seq_len, embedding_dim)` | Same shape |
| Final logits | `(batch, seq_len, vocab_size)` | Probability per token |

---

## Common Pitfalls

### Device Mismatch
- Tensors on different devices (CPU vs GPU) cause errors
- Especially important for mask and positional encoding

### Dimension Errors
- Verify shapes at each step
- Use `print(tensor.shape)` during debugging

### Gradient Issues
- `detach()` breaks gradient flow — use carefully
- Ensure learnable parameters have `requires_grad=True` (automatic for nn.Module)
- Positional encoding should NOT update (use `register_buffer`)

### Attention Score Overflow
- Forgetting to divide by sqrt(head_dim) can cause numerical instability
- Large scores lead to near-zero gradients after softmax (vanishing gradient)
- Always normalize before softmax

---

## Testing Your Implementation

1. **Forward pass with tiny inputs**: Test with batch_size=1, seq_len=2 to verify shapes
2. **Check output shape**: Should be (batch, seq_len, vocab_size)
3. **Generate samples**: Test `generate()` produces sensible sequences
