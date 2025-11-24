"""
Neural network model for adherence prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AdherencePredictor(nn.Module):
    """
    Neural network for predicting adherence given state and control inputs.
    """
    
    def __init__(self, in_dim, out_dim, hidden_dim=64,
                 optimizer_cls=None, optimizer_kwargs=None):
        """
        Initialize the model.
        
        Parameters
        ----------
        in_dim : int
            Input dimension (state + controls)
        out_dim : int
            Output dimension (number of adherence classes)
        hidden_dim : int
            Hidden layer dimension
        optimizer_cls : torch.optim.Optimizer, optional
            Optimizer class to use for training (default: Adam)
        optimizer_kwargs : dict, optional
            Keyword arguments passed to optimizer (default: {"lr": 1e-3})
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.optimizer_cls = optimizer_cls or optim.Adam
        self.optimizer_kwargs = {"lr": 1e-3}
        if optimizer_kwargs:
            self.optimizer_kwargs.update(optimizer_kwargs)

        # Build network layers
        layers = []
        layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=True))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=True))

        self.net = nn.Sequential(*layers)
        
        # Weight initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.optimizer = self.optimizer_cls(
            self.parameters(),
            **self.optimizer_kwargs
        )

    def forward(self, y):
        """
        Forward pass returning logits.
        
        Parameters
        ----------
        y : torch.Tensor
            Input state vector
            
        Returns
        -------
        torch.Tensor
            Logits for each adherence class
        """
        return self.net(y)

    def inference(self, y):
        """
        Inference mode returning predicted class.
        
        Parameters
        ----------
        y : torch.Tensor
            Input state vector
            
        Returns
        -------
        torch.Tensor
            Predicted class index
        """
        logits = self.net(y)
        probs = F.softmax(logits, dim=0)
        return torch.argmax(probs).unsqueeze(0)

    def solve_for_inference(self, ts, cs, y0):
        """
        Solve trajectory for inference.
        
        Parameters
        ----------
        ts : torch.Tensor
            Shape (T,) time steps
        cs : torch.Tensor
            Shape (T, D-1) control signals
        y0 : torch.Tensor
            Shape (D,) initial condition
        
        Returns
        -------
        torch.Tensor
            Shape (T+1, D) trajectory
        """
        ys = []
        last_y = y0.float()  # Ensure y0 is float for consistency
        
        for t, c in zip(ts, cs[1:]):  # Skip first control as it's in y0
            ys.append(last_y)
            target_next_day = self.inference(y=last_y)
            # Convert class index to float to match control dtype
            target_next_day_float = target_next_day.float()
            last_y = torch.cat([target_next_day_float, c])
        ys.append(last_y)
        
        return torch.stack(ys)

    def solve_for_training(self, ts, cs, y0):
        """
        Solve trajectory for training.
        
        Parameters
        ----------
        ts : torch.Tensor
            Shape (T,) time steps
        cs : torch.Tensor
            Shape (T, D-1) control signals
        y0 : torch.Tensor
            Shape (D,) initial condition (first element is target class index)
        
        Returns
        -------
        torch.Tensor
            Shape (T, out_dim) logits for each time step
        """
        ys_target_logits = torch.zeros(len(ts), self.out_dim)
        
        # Ensure y0 is float for consistency
        last_y = y0.float()
        
        # Logits for t=0 are one-hot encoded target from initial condition
        # y0[0] should be the class index (as float, convert to long for one-hot)
        ys_target_logits[0] = F.one_hot(last_y[0].long(), num_classes=self.out_dim).float()
        
        for idx, (t, c) in enumerate(zip(ts, cs[1:]), start=1):  # Skip first control
            target_next_day = self.inference(y=last_y)
            ys_target_logits[idx] = self.forward(y=last_y)
            # Convert class index to float to match control dtype
            target_next_day_float = target_next_day.float()
            last_y = torch.cat([target_next_day_float, c])
        
        return ys_target_logits
