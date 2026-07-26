# MLflow Tutorial for Experiment Tracking

This tutorial covers MLflow, an open-source platform for managing machine learning experiments, enabling reproducible and organized ML workflows.

## Table of Contents

1. [What is MLflow?](#what-is-mlflow)
2. [Core Concepts](#core-concepts)
3. [Logging Experiments](#logging-experiments)
4. [Tracking UI](#tracking-ui)
5. [Best Practices](#best-practices)

## What is MLflow?

MLflow is an open-source platform that helps manage the complete machine learning lifecycle:

- **Experiment Tracking**: Log parameters, metrics, and artifacts from ML runs
- **Model Registry**: Version and manage models in a central repository
- **Model Serving**: Deploy models for inference
- **Projects**: Package ML code for reproducibility

For effective experiment management, we use MLflow for **experiment tracking** to monitor training progress and compare different model configurations.

## Core Concepts

### Experiments

An **experiment** is a collection of related runs. Think of it as a project or group of trials.

```python
import mlflow

# Set the experiment
mlflow.set_experiment("model_training")

# Or create a new one (if it doesn't exist)
experiment = mlflow.get_experiment_by_name("model_training")
if experiment is None:
    experiment_id = mlflow.create_experiment("model_training")
```

### Runs

A **run** is a single execution of your model training or evaluation. Each run logs:

- **Parameters**: Hyperparameters (fixed values that don't change)
- **Metrics**: Performance measurements (can change over time)
- **Artifacts**: Files (models, images, plots, etc.)
- **Tags**: Metadata for organization

### Structure

```
Experiment: "image_classification"
├── Run 1: classification_v1
│   ├── Parameters: learning_rate=0.001, batch_size=32
│   ├── Metrics: accuracy, loss (updates at each epoch)
│   └── Artifacts: model.pth
├── Run 2: classification_v2
│   ├── Parameters: learning_rate=0.0001, batch_size=64
│   ├── Metrics: accuracy, loss (updates at each epoch)
│   └── Artifacts: model.pth
└── Run 3: classification_v3
    └── ...
```

## Logging Experiments

### Basic Run Structure

```python
import mlflow
import mlflow.pytorch

# Set tracking URI (where to store data)
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("model_training")

# Start a run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("num_epochs", 10)
    
    # Perform training
    for epoch in range(num_epochs):
        loss = train_step()
        
        # Log metric (changes over time)
        mlflow.log_metric("loss", loss, step=step)
    
    # Save artifacts
    output_image.save("output.png")
    mlflow.log_artifact("output.png")
```

### Logging Parameters

Parameters are hyperparameters that **don't change** during training:

```python
# Log individual parameters
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("optimizer", "Adam")
mlflow.log_param("batch_size", 32)

# Log multiple parameters at once
params = {
    "num_epochs": 100,
    "learning_rate": 0.001,
    "batch_size": 32,
    "weight_decay": 0.0001,
    "dropout_rate": 0.5
}
mlflow.log_params(params)
```

### Logging Metrics

Metrics are **measurements** that change during training:

```python
# Log a single metric value
mlflow.log_metric("total_loss", 0.5)

# Log metric at a specific step
mlflow.log_metric("total_loss", 0.5, step=0)
mlflow.log_metric("total_loss", 0.3, step=1)
mlflow.log_metric("total_loss", 0.1, step=2)

# Log multiple metrics
mlflow.log_metrics({
    "train_loss": 0.5,
    "train_accuracy": 0.85,
    "val_loss": 0.6
}, step=0)
```

### Logging Artifacts

Artifacts are **files** associated with a run:

```python
import mlflow

# Log a local file
mlflow.log_artifact("output_image.png")

# Log with a destination folder
mlflow.log_artifact("output_image.png", artifact_path="images")

# Log directory
mlflow.log_artifacts("./results/")

# Save and log a model
mlflow.pytorch.log_model(model, artifact_path="model")
```

### Example: Logging a Training Run

```python
import mlflow
import torch
import torch.optim as optim

mlflow.set_tracking_uri("http://localhost:8888")
mlflow.set_experiment("model_training")

with mlflow.start_run():
    # Log hyperparameters
    params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 10,
        "optimizer": "Adam"
    }
    mlflow.log_params(params)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass and compute loss
            loss = train_step(batch)
            
            # Log loss at each epoch
            mlflow.log_metric("train_loss", loss.item(), step=epoch)
        
        # Evaluate on validation set
        val_loss = evaluate(val_dataloader)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
    
    # Save model
    torch.save(model.state_dict(), "model.pth")
    mlflow.log_artifact("model.pth")
    mlflow.pytorch.log_model(model, artifact_path="model")
```

## Tracking UI

The MLflow Tracking UI provides a web interface to view all your experiments and runs.

### Accessing the UI

1. **Start the server** (if not already running):
   ```bash
   mlflow ui --port 8888
   ```

2. **Open in browser**: `http://localhost:8888`

### UI Features

- **Experiments List**: View all experiments with run counts
- **Run Comparison**: Compare metrics and parameters across runs
- **Metric Charts**: Visualize how metrics changed over steps
- **Artifact Browser**: View logged images, models, and other files
- **Run Details**: See all parameters, metrics, and tags for a specific run

### Understanding the Dashboard

**Main View:**
- Left sidebar: Experiment selector
- Run list: All runs in the selected experiment
- Columns: Parameters, metrics, and run duration

**Compare Runs:**
1. Select multiple runs
2. Click "Compare"
3. View side-by-side comparison of parameters and metrics
4. Visualize metric trends in charts

## Best Practices

### 1. Organize with Tags

Add metadata to runs for better organization:

```python
mlflow.set_tag("team", "research")
mlflow.set_tag("model_type", "resnet")
mlflow.set_tag("dataset", "imagenet")
mlflow.set_tag("status", "production")
```

### 2. Consistent Experiment Names

Use meaningful, consistent experiment names:

```python
# Good
mlflow.set_experiment("image_classification_resnet")
mlflow.set_experiment("object_detection_yolo")

# Bad
mlflow.set_experiment("exp1")
mlflow.set_experiment("test")
```

### 3. Log Early and Often

Log metrics regularly to track progress:

```python
for step in range(num_steps):
    loss = compute_loss()
    if step % 50 == 0:  # Log every 50 steps
        mlflow.log_metric("loss", loss, step=step)
```

### 4. Save Intermediate Results

Log artifacts throughout training, not just at the end:

```python
for epoch in range(num_epochs):
    train()
    
    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"checkpoint_{epoch}.pt")
        mlflow.log_artifact(f"checkpoint_{epoch}.pt")
```

### 5. Log Model Information

Help future you (or collaborators) understand the model:

```python
mlflow.log_param("model_architecture", "ResNet50")
mlflow.log_param("pretrained", True)
mlflow.log_param("num_parameters", count_parameters(model))

# Or log model info as an artifact
with open("model_info.txt", "w") as f:
    f.write(f"Model: ResNet50\n")
    f.write(f"Parameters: {count_parameters(model)}\n")
mlflow.log_artifact("model_info.txt")
```

### 6. Use Context Managers

Always use `mlflow.start_run()` as a context manager to ensure proper cleanup:

```python
# Good
with mlflow.start_run():
    mlflow.log_param("x", 1)
    # Run is automatically ended here

# Avoid
mlflow.start_run()
mlflow.log_param("x", 1)
mlflow.end_run()
```

### 7. Compare Experiments Systematically

Test one variable at a time:

```python
# Test different learning rates
for learning_rate in [0.0001, 0.001, 0.01, 0.1]:
    with mlflow.start_run():
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.set_tag("experiment", "learning_rate_ablation")
        # Run training...
```

## Common Issues

### Cannot Connect to Tracking Server

**Error**: `failed to connect to tracking server at http://localhost:5000`

**Solution**: Start the MLflow UI first
```bash
mlflow ui --port 5000
```

### No Run is Active

**Error**: `You must be in a run to log metrics`

**Solution**: Use `mlflow.start_run()` context manager
```python
with mlflow.start_run():
    mlflow.log_metric("loss", 0.5)
```

### Metrics Not Appearing in UI

**Possible causes:**
- Run hasn't been ended yet (use context manager)
- Tracking URI mismatch between script and UI
- Firewall blocking access

## Quick Reference

| Task | Code |
|------|------|
| Set tracking URI | `mlflow.set_tracking_uri("http://localhost:5000")` |
| Set experiment | `mlflow.set_experiment("experiment_name")` |
| Start run | `with mlflow.start_run():` |
| Log parameter | `mlflow.log_param("name", value)` |
| Log metric | `mlflow.log_metric("loss", 0.5, step=0)` |
| Log artifact | `mlflow.log_artifact("file.png")` |
| Add tag | `mlflow.set_tag("key", "value")` |
| Save model | `mlflow.pytorch.log_model(model, "model")` |
| Start UI | `mlflow ui --port 8888` |
| Get run info | `mlflow.active_run()` |

## Further Reading

- [MLflow Documentation](https://mlflow.org/docs)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Examples](https://github.com/mlflow/mlflow/tree/master/examples)
- [PyTorch Integration](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)
