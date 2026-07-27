# Transformers
## Overview

This is a character-level language model implementation using a Transformer architecture trained on Shakespeare's works. It learns to generate text in the style of Shakespeare by predicting the next token in a sequence using multi-head attention and feed-forward networks.

## Project Structure

```
day2-transformers/
├── config.py           # Configuration constants and hyperparameters
├── data.py             # Data loading and batch generation utilities
├── models.py           # Transformer architecture components
├── train.py            # Training pipeline with MLflow-style logging
├── inference.py        # Model inference for text generation
├── README.md           # This file
└── data/
    └── tinyshakespeare.txt  # Training data (auto-downloaded)
```

## Module Descriptions

### `config.py`
Central configuration class for all hyperparameters and settings.

**Key parameters:**
- `device`: CUDA if available, else CPU (with mps support for macOS)
- `batch_size`: Batch size for training (default: 32)
- `block_size`: Maximum sequence length in tokens (default: 256)
- `embedding_dim`: Token embedding dimensionality (default: 384)
- `num_heads`: Number of attention heads (default: 6)
- `num_layers`: Number of transformer blocks (default: 3)
- `dropout`: Dropout rate for regularization (default: 0.1)
- `learning_rate`: Adam optimizer learning rate (default: 3e-4)
- `epochs`: Total training epochs (default: 1000)
- `eval_interval`: Frequency of validation (default: 10)

### `data.py`
Utilities for data loading, encoding, and batch generation.

**Functions:**
- `load_data()`: Download and load the Tiny Shakespeare dataset
- `encode_text()`: Tokenize text using GPT-2 BPE tokenizer
- `split_data()`: Split data into train/val sets
- `get_batch()`: Generate random batches for training

**TODO Assignments:**
- Implement batch sampling with random indices
- Stack input and target sequences
- Move tensors to correct device

### `models.py`
Transformer components for language modeling.

**Classes:**
- `MultiheadAttention`: Multi-head self-attention mechanism
  - TODO: Implement QKV projections, attention computation, causal masking, output projection
- `FeedForward`: Feed-forward network (MLP)
  - TODO: Implement linear layers with GELU activation
- `TransformerBlock`: Combined attention + feed-forward with residual connections
  - TODO: Assemble attention and feed-forward with layer norms
- `PositionalEncoding`: Positional encoding for sequence positions
- `ShakespeareTransformer`: Full transformer-based language model
  - TODO: Assemble embeddings, positional encoding, transformer blocks, and output head

### `train.py`
Training pipeline with progress tracking.

**Key functions:**
- `main()`: Orchestrate training loop
  - Load and encode data
  - Initialize model and optimizer
  - Training loop with validation and checkpointing
  - Generate sample text at intervals
  - Progress tracking with tqdm

### `inference.py`
Generate text using a trained model.

**Workflow:**
- Load trained model checkpoint
- Tokenize initial prompt
- Generate new tokens autoregressively
- Decode and display results

## Usage

### Training

```bash
uv run .\train.py
```

This will:
1. Download Tiny Shakespeare dataset (if not present)
2. Tokenize and prepare data
3. Train the model for specified epochs
4. Generate sample Shakespeare text every `eval_interval` epochs
5. Save checkpoints to `saves/` directory

### Inference

```bash
uv run .\inference.py
```

Edit the `prompt` variable in `inference.py` to generate different text:
```python
prompt = "ROMEO: "  # Model will continue from this prompt
```

## How It Works

### Training
1. **Data Loading**: Load Tiny Shakespeare and tokenize with GPT-2 BPE
2. **Batch Generation**: Randomly sample sequences of length `block_size`
3. **Forward Pass**: 
   - Embed tokens
   - Add positional encodings
   - Pass through transformer blocks
   - Project to vocabulary size
4. **Loss Computation**: Cross-entropy loss between predictions and targets
5. **Optimization**: Backpropagation with AdamW optimizer
6. **Checkpointing**: Save model weights at regular intervals

### Inference
1. **Initialization**: Tokenize the prompt
2. **Autoregressive Generation**: 
   - For each step, predict the next token
   - Sample from the probability distribution
   - Append to sequence
   - Repeat until max tokens generated
3. **Decoding**: Convert token IDs back to text

## Architecture Details

### Multi-Head Attention
- Projects input to Query, Key, Value
- Splits into multiple heads for parallel attention
- Applies causal masking to prevent attending to future tokens
- Concatenates heads and projects output

### Transformer Block
- Self-attention followed by feed-forward
- Layer normalization before each sub-layer (pre-norm)
- Residual connections around each sub-layer
- Dropout for regularization

### Full Model
- Token embeddings map vocab indices to embedding dimension
- Positional encodings add sequence position information
- Stack of transformer blocks for deep representation learning
- Output projection to vocabulary size for next-token prediction

## Configuration Notes

- Larger models require more VRAM; reduce `batch_size` or `block_size` if out of memory
- Increasing `num_layers` and `num_heads` increases model capacity but training time
- `dropout` helps prevent overfitting; increase if model memorizes training data
- Lower `learning_rate` may help convergence; higher for faster training
- Set `eval_interval` to 1 to validate at every epoch (slower but more feedback)

## Dependencies

- `torch`: PyTorch deep learning framework
- `tiktoken`: GPT-2 tokenizer
- `tqdm`: Progress bar for training
- `requests`: Download dataset
- `numpy`: Numerical computing
