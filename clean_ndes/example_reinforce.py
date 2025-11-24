"""
Example script for training on Reinforce dataset.
"""

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from clean_ndes import (
    create_reinforce_dataset,
    get_reinforce_config,
    AdherencePredictor,
    train_one_epoch,
    SEED,
    LEARNING_RATE,
    BATCH_SIZE,
)

# Set random seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load Reinforce dataset
filepath = "/home/andrei/Desktop/PhD/Carepath/tensor_Reinforce.npy"
print("Loading Reinforce dataset...")
dataset = create_reinforce_dataset(filepath)

# Get dataset configuration
# Load raw data to get shape info
data_raw = np.load(filepath)
config = get_reinforce_config(data_raw)

print(f"\nDataset configuration:")
print(f"  Patients: {config['n_patients']}")
print(f"  Days: {config['n_days']}")
print(f"  Dimensions: {config['n_dimensions']}")
print(f"  Controls: {config['n_controls']}")
print(f"  Classes: {config['n_classes']}")

# Split into train and validation (80/20)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_set, val_set = random_split(
    dataset, 
    [train_size, val_size], 
    generator=torch.Generator().manual_seed(SEED)
)

print(f"\nTrain set: {train_size} patients")
print(f"Validation set: {val_size} patients")

# Create data loaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Create model
model = AdherencePredictor(
    in_dim=config['input_dim'], 
    out_dim=config['output_dim'], 
    hidden_dim=64
)

print(f"\nModel created:")
print(f"  Input dim: {config['input_dim']} (target + {config['n_controls']} controls)")
print(f"  Output dim: {config['output_dim']} (5 classes)")
print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

# Train the model
print("\n" + "=" * 80)
print("Training NDE model on Reinforce dataset")
print("=" * 80)

train_losses, val_losses = train_one_epoch(
    model, 
    train_loader, 
    val_loader, 
    learning_rate=LEARNING_RATE,
    verbose=True
)

print(f"\nTraining complete!")
print(f"  Final train loss: {train_losses[-1]:.4f}")
print(f"  Final val loss: {val_losses[-1]:.4f}")
print(f"  Total batches: {len(train_losses)}")

