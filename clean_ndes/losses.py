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
    Validation loss: MSE Loss on raw class indices.
    
    Computes MSE between predicted class indices and true class indices.
    Removes first timestep from evaluation as it's given by the initial condition.
    
    Parameters
    ----------
    outputs : torch.Tensor
        Shape (batch_size, T, n_classes) - one-hot encoded outputs from model
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
    
    # Get predicted class indices from one-hot outputs
    pred_class_indices = torch.argmax(outputs_flat, dim=1).float()
    
    # Convert labels to float for MSE computation
    labels_flat_float = labels_flat.float()
    
    loss = torch.nn.MSELoss()(pred_class_indices, labels_flat_float)
    return loss
