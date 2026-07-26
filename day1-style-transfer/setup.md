# Setup Instructions

This guide will help you set up the style transfer project using `uv`.

## Installing `uv`

`uv` is a fast, reliable Python package manager written in Rust. Follow the instructions below for your operating system:

### Windows

#### Option 1: Using PowerShell (Recommended)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Option 2: Using pip
```bash
pip install uv
```

For more installation options and details, see [uv Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, add `uv` to your PATH by following the on-screen instructions.

### Verify Installation

```bash
uv --version
```

You should see the version number if installation was successful.

## Setting Up the Project

### For NVIDIA GPU Users (CUDA 11.8)

If you have an NVIDIA GPU, the project is pre-configured to use PyTorch with CUDA 11.8 support. Simply run:

```bash
uv sync
```

This will install all dependencies including CUDA-optimized PyTorch.

### For Non-NVIDIA GPU Users (CPU Only)

If you don't have an NVIDIA GPU, you need to remove the CUDA index from the `pyproject.toml` file:

#### Step 1: Edit `pyproject.toml`

Open the `pyproject.toml` file and remove or comment out the following section:

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu118", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu118", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
```

After removing this section, your `pyproject.toml` will use the default PyTorch CPU build.

#### Step 2: Install Dependencies

```bash
uv sync
```

#### Step 3: Verify CPU-Only Setup

To verify that you have the CPU version of PyTorch installed, run the following code in a file using `uv run <filename>`:

```python
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
```

You should see `CUDA available: False` in the output.

## Running the Project

Once you've completed the setup, you can run the style transfer application (check the README in `src`)

## Troubleshooting

### Dependencies Not Installing Correctly

1. **Clear the cache**: `uv cache clean`
2. **Try syncing again**: `uv sync --refresh`

### PyTorch Not Found

Ensure you've correctly modified the `pyproject.toml` for your setup (GPU or CPU) and run `uv sync` again.

### uv Command Not Found

Make sure `uv` is in your system PATH:

- **Windows**: Restart your terminal or PowerShell after installation
- **macOS/Linux**: Run `source $HOME/.local/bin/env` or restart your terminal

For additional help, visit the [uv Documentation](https://docs.astral.sh/uv/).
