"""
Training logic for adherence prediction.
"""

import numpy as np
import torch
import torch.optim

from .model import AdherencePredictor
from .losses import get_batch_loss, get_batch_loss_validation


def train_one_epoch(model, training_loader, validation_loader, learning_rate=0.001, 
                   verbose=True):
    """
    Train model for one epoch and compute validation loss.
    
    Parameters
    ----------
    model : AdherencePredictor
        Model to train
    training_loader : DataLoader
        Training data loader
    validation_loader : DataLoader
        Validation data loader
    learning_rate : float
        Learning rate for optimizer
    verbose : bool
        Whether to print loss values
        
    Returns
    -------
    tuple
        (train_losses, val_losses) arrays of losses for each batch
    """
    train_losses = []
    val_losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Select a random validation batch for monitoring
    random_vbatch_idx = torch.randint(0, len(validation_loader), (1,))
    random_vbatch = None
    
    for j, vdata in enumerate(validation_loader):
        if j == random_vbatch_idx.item():
            random_vbatch = vdata
            T = np.shape(np.array(random_vbatch[0]))[1]
            ts = torch.tensor([i for i in range(1, T + 1)])
            break
    
    if random_vbatch is None:
        raise ValueError("Could not select validation batch")
    
    # Training loop
    for i, (inputs, labels) in enumerate(training_loader):
        # Validation step
        model.eval()
        with torch.no_grad():
            vinputs, vlabels = random_vbatch
            batch_size = vinputs.shape[0]
            
            outputs = torch.zeros(batch_size, T, model.out_dim)
            for sample_idx in range(batch_size):
                # y0: [target at T=0, controls at T=0]
                y0 = torch.cat([
                    vlabels[sample_idx, 0].unsqueeze(0), 
                    vinputs[sample_idx][0]
                ])
                aux_trajectory = model.solve_for_inference(
                    ts=ts, cs=vinputs[sample_idx], y0=y0
                )
                # Convert predicted classes to one-hot
                aux_outputs = torch.nn.functional.one_hot(
                    aux_trajectory[:, 0].long(), 
                    num_classes=model.out_dim
                ).float()
                outputs[sample_idx] = aux_outputs
            
            vloss = get_batch_loss_validation(outputs, vlabels)
            val_losses.append(vloss.item())
            if verbose:
                print(f"Batch {i}: val_loss = {vloss.item():.4f}")
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        batch_size = inputs.shape[0]
        T_train = labels.shape[1]
        ts_train = torch.tensor([j for j in range(1, T_train + 1)])
        
        outputs = torch.zeros(batch_size, T_train, model.out_dim)
        for sample_idx in range(batch_size):
            # y0: [target at T=0, controls at T=0]
            y0 = torch.cat([
                labels[sample_idx, 0].unsqueeze(0), 
                inputs[sample_idx][0]
            ])
            outputs[sample_idx] = model.solve_for_training(
                ts=ts_train, cs=inputs[sample_idx], y0=y0
            )
        
        loss = get_batch_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        if verbose:
            print(f"Batch {i}: train_loss = {loss.item():.4f}")
    
    return np.array(train_losses), np.array(val_losses)


def train_multiple_runs(num_runs, model_class, training_loader, validation_loader,
                       **model_kwargs):
    """
    Run training for multiple runs with different random seeds.
    
    Parameters
    ----------
    num_runs : int
        Number of runs
    model_class : class
        Model class to instantiate
    training_loader : DataLoader
        Training data loader
    validation_loader : DataLoader
        Validation data loader
    **model_kwargs : dict
        Additional arguments for model initialization
        
    Returns
    -------
    np.ndarray
        Shape (runs, n_batches, 2) containing train and val losses for each run
    """
    runs_loss_history = []
    
    for run in range(num_runs):
        torch.manual_seed(run)
        model = model_class(**model_kwargs)
        
        train_losses, val_losses = train_one_epoch(
            model, training_loader, validation_loader, verbose=True
        )
        
        # Store losses for this run
        run_history = np.stack([train_losses, val_losses], axis=1)
        runs_loss_history.append(run_history)
    
    return np.array(runs_loss_history)
