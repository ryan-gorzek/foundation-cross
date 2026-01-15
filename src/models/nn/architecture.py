"""
Neural network classifier architectures for label transfer.
"""
import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    """Multi-layer perceptron classifier for gene expression data"""

    def __init__(self, n_inputs, n_outputs, hidden=(512, 512, 256), dropout=0.2):
        super.__init__()
        layers = []
        in_dim = n_inputs
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h), # input, output
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        layers += [nn.Linear(in_dim, n_outputs)]
    self.model = nn.Sequential(*layers)

    def forward(x):
        return self.model(x)
