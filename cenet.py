"""
Constrained Encoder Network
"""

# Author: Qi Xu <qxu6@uci.edu>

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim.lr_scheduler import ExponentialLR

# define the network structure

class ConstrEncoderNet(nn.Module):

    def __init__(self, input_size, layer=2, act="relu", width=20, width_embed=5):

        super().__init__()

        cov_dim = input_size

        self.layer = layer
        self.act = act
        self.width = width
        self.width_embed = width_embed

        # define the covariate encoder

        self.cov_input = nn.Linear(cov_dim, width)

        self.cov_hidden = nn.ModuleList()
        
        for i in range(layer):

            self.cov_hidden.append(nn.Linear(width, width))
            self.cov_hidden.append(nn.BatchNorm1d(num_features=width))

        self.cov_embed = nn.Linear(width, width_embed)

    def covariate_embedding(self, X):

        # covaraite encoding

        cov = self.cov_input(X)
        if self.act == "relu":
            cov = F.relu(cov)
        elif self.act == "linear":
            cov = cov

        for index, layer in enumerate(self.cov_hidden):
            if index % 2 == 0:
                cov = layer(cov)
                break
            elif index % 2 == 1:
                cov = layer(cov)
                if self.act == "relu":
                    cov = F.relu(cov)
                elif self.act == "linear":
                    cov = cov

        cov_embed = self.cov_embed(cov)

        return cov_embed

    def forward(self, X, betas):

        alpha = self.covariate_embedding(X)

        output = torch.tensordot(alpha, torch.transpose(betas, 0, 1), dims=1)

        return output


if __name__ == "__main__":

    cenet = ConstrEncoderNet(5, width_embed=5)

    X = torch.rand(100, 5)

    betas = torch.rand(4, 5)

    output = cenet(X, betas)

    print(output)