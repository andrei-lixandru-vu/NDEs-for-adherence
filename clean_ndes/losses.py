"""
Loss functions for training and validation.
"""

import torch


def get_batch_loss(outputs, labels):
    """
    Training loss: Cross Entropy Loss on logits.
    
    Parameters
    ----------
    outputs : torch.Tensor
        Shape (batch_size, T, n_classes) - logits from model
    labels : torch.Tensor
        Shape (batch_size, T) - true class indices
        
    Returns
    -------
    torch.Tensor
        Scalar loss value
    """
    batch_size, T, n_classes = outputs.shape
    outputs_flat = outputs.reshape(-1, n_classes)
    labels_flat = labels.reshape(-1).long()
    
    loss = torch.nn.CrossEntropyLoss()(outputs_flat, labels_flat)
    return loss


def get_batch_loss_validation(outputs, labels):
    """
    Validation loss: MSE Loss on probabilities.
    
    Computes MSE between predicted probabilities and one-hot labels.
    Removes first timestep from evaluation as it's given by the initial condition.
    
    Parameters
    ----------
    outputs : torch.Tensor
        Shape (batch_size, T, n_classes) - logits from model
    labels : torch.Tensor
        Shape (batch_size, T) - true class indices
        
    Returns
    -------
    torch.Tensor
        Scalar loss value
    """
    # Remove first timestep from evaluation
    outputs = outputs[:, 1:, :]
    labels = labels[:, 1:]
    
    batch_size, T, n_classes = outputs.shape
    outputs_flat = outputs.reshape(-1, n_classes)
    labels_flat = labels.reshape(-1)
    
    # Convert labels to one-hot encoding for MSE
    labels_onehot = torch.nn.functional.one_hot(
        labels_flat.long(), 
        num_classes=n_classes
    ).float()
    
    loss = torch.nn.MSELoss()(outputs_flat, labels_onehot)
    return loss
