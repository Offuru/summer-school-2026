# PyTorch Tutorial

This tutorial covers essential PyTorch concepts for machine learning and deep learning projects.

## Table of Contents

1. [Tensors](#tensors)
2. [Neural Networks](#neural-networks)
3. [Optimization](#optimization)
4. [Backpropagation](#backpropagation)
5. [Device Management](#device-management)

## Tensors

Tensors are the fundamental data structures in PyTorch. They are multi-dimensional arrays similar to NumPy arrays but with GPU support.

### Creating Tensors

```python
import torch

# Create tensors from Python lists
x = torch.tensor([1, 2, 3])
y = torch.tensor([[1, 2], [3, 4]])

# Create random tensors
z = torch.randn(3, 4)  # Normal distribution
ones = torch.ones(2, 3)  # All ones

# Create tensors on GPU
x_gpu = torch.tensor([1, 2, 3], device="cuda")
```

### Tensor Operations

```python
# Element-wise operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

c = a + b  # [5, 7, 9]
d = a * b  # [4, 10, 18]

# Matrix multiplication
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
C = A @ B  # or torch.matmul(A, B)

# Reshape
x = torch.randn(3, 4)
y = x.view(2, 6)  # Flatten and reshape
z = x.reshape(12)  # Flatten to 1D

# Transpose
x = torch.randn(2, 3, 4)
y = x.transpose(0, 1)  # Swap first two dimensions
z = x.t()  # For 2D tensors only
```

### Important Tensor Properties

```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

print(x.shape)    # torch.Size([2, 2])
print(x.dtype)    # torch.float32
print(x.device)   # cpu
print(x.requires_grad)  # False (by default)
```

## Neural Networks

PyTorch provides pre-built models through `torchvision.models`. The VGG19 model is used in this project for feature extraction.

### Loading Pre-trained Models

```python
from torchvision import models

# Load VGG19 with pre-trained ImageNet weights
vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

# Set to evaluation mode (disables dropout, batch norm updates)
vgg19.eval()

# Extract only the feature extraction layers (exclude classification head)
features = vgg19.features
```

### Understanding Pre-trained Models

Pre-trained models like VGG19 are neural networks trained on large datasets (e.g., ImageNet) and can be fine-tuned or used for feature extraction on new tasks.

**Architecture hierarchy:**
- **Early Layers**: Detect simple features (edges, textures, colors)
- **Middle Layers**: Detect more complex patterns and shapes
- **Deep Layers**: Detect high-level semantic features and objects

Pre-trained models are useful for:
- **Transfer Learning**: Fine-tune on new datasets with less data
- **Feature Extraction**: Use intermediate layer outputs as features
- **Computer Vision Tasks**: Object detection, image classification, etc.

### Building Custom Neural Network Modules

```python
import torch.nn as nn

class CustomModule(nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = CustomModule()
output = model(torch.randn(3, 10))  # Batch of 3 samples
```

### Sequential Models

`nn.Sequential` stacks modules together:

```python
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)
```

## Optimization

Optimization algorithms update model parameters to minimize a loss function. PyTorch provides several optimizers in `torch.optim`.

### Common Optimizers

```python
import torch.optim as optim

# SGD (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Adam (Adaptive Moment Estimation) - Often works better
optimizer = optim.Adam(model.parameters(), lr=0.001)

# LBFGS (Limited-memory BFGS) - Second-order optimization
optimizer = optim.LBFGS([input_tensor], lr=1.0)
```

### LBFGS Optimizer

LBFGS is a second-order optimization method that is particularly effective for certain problems because:

- It's a **second-order optimization method** (uses curvature information)
- Converges faster than first-order methods like SGD or Adam
- Requires a **closure function** that computes the loss

```python
optimizer = optim.LBFGS([input_img])

def closure():
    optimizer.zero_grad()
    loss = compute_loss(input_img)
    loss.backward()
    return loss

for step in range(num_steps):
    optimizer.step(closure)
```

## Backpropagation

Backpropagation computes gradients of the loss with respect to all parameters. PyTorch automatically computes gradients using the **autograd** system.

### Computing Gradients

```python
import torch.nn.functional as F

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
target = torch.tensor([2.0, 4.0, 6.0])

# Forward pass
output = x * 2
loss = F.mse_loss(output, target)

# Backward pass
loss.backward()  # Computes gradients

# Access gradients
print(x.grad)  # Gradient of loss with respect to x
```

### Important Gradient Operations

```python
# Zero gradients (required before each backward pass in training)
optimizer.zero_grad()

# Compute gradients
loss.backward()

# Prevent gradient computation (for inference or fixed operations)
with torch.no_grad():
    output = model(input_data)

# Detach tensor from computation graph
features = model(x).detach()  # Gradients won't flow back through this
```

### Clipping and Constraints

```python
# Clamp tensor values to a range [0, 1]
tensor.clamp_(0, 1)  # In-place operation (modifies tensor)

# Or create a new tensor
clamped = torch.clamp(tensor, 0, 1)
```

## Device Management

PyTorch can run computations on CPU or GPU. Consistent device placement is crucial.

### Device Operations

```python
import torch

# Check if GPU is available
cuda_available = torch.cuda.is_available()

# Create device object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move tensor to device
x = torch.randn(3, 4)
x = x.to(device)

# Create tensor directly on device
y = torch.randn(3, 4, device=device)

# Move model to device
model = model.to(device)
```

### Common Device Pattern

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare input
input_data = torch.randn(3, 256, 256)
input_data = input_data.to(device)

# Load model
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
model = model.to(device)
model.eval()

# Process
with torch.no_grad():
    output = model(input_data)
```

## Loss Functions

Loss functions measure how well the model performs. PyTorch provides common losses in `torch.nn.functional`.

### Common Loss Functions

```python
import torch.nn.functional as F

# Mean Squared Error - for regression tasks
predicted = torch.randn(2, 3)
target = torch.randn(2, 3)
mse_loss = F.mse_loss(predicted, target, reduction="mean")

# Cross Entropy Loss - for classification tasks
logits = torch.randn(2, 10)  # 2 samples, 10 classes
labels = torch.tensor([3, 7])
ce_loss = F.cross_entropy(logits, labels)

# L1 Loss - for robust regression
l1_loss = F.l1_loss(predicted, target)
```

### Loss Reduction Options

```python
# 'mean': Average loss across all elements
loss = F.mse_loss(predicted, target, reduction="mean")

# 'sum': Sum of all losses
loss = F.mse_loss(predicted, target, reduction="sum")

# 'none': Return loss for each element (no reduction)
loss = F.mse_loss(predicted, target, reduction="none")  # Shape: (batch_size,)
```

## Quick Reference

| Task | Code |
|------|------|
| Create tensor | `torch.tensor([1, 2, 3])` |
| Random tensor | `torch.randn(3, 4)` |
| Zeros/Ones | `torch.zeros(2, 3)`, `torch.ones(2, 3)` |
| Load pre-trained model | `models.resnet50(weights=models.ResNet50_Weights.DEFAULT)` |
| Move to GPU | `tensor.to("cuda")`, `model.to("cuda")` |
| Compute loss | `F.mse_loss(output, target)` |
| Backward pass | `loss.backward()` |
| Zero gradients | `optimizer.zero_grad()` |
| Optimization step | `optimizer.step()` |
| Disable gradients | `with torch.no_grad(): ...` |
| Reshape tensor | `tensor.view(3, 4)` or `tensor.reshape(3, 4)` |

## Further Reading

- [PyTorch Official Tutorial](https://pytorch.org/tutorials/)
- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
