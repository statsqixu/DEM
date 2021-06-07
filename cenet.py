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

        output = torch.softmax(output, dim=1)

        return output

    def loss_fn(self, mask_panel, trt_panel, cost, budget, lambd):

        loss1 = - torch.sum(mask_panel * trt_panel)
        loss2 = lambd * torch.maximum((torch.sum(mask_panel * cost) - budget), torch.Tensor([0])) ** 2

        return loss1 + loss2

    def train(self, data, epochs=10000, learning_rate=1e-3, lambd=1e3, verbose=0):

        # define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        optimizer.zero_grad()
        scheduler = ExponentialLR(optimizer, gamma=0.999)
        # load data
        X, betas, trt_panel, cost, budget = data

        lambd_list = torch.linspace(start=1, end=lambd, steps=epochs // 10)
        e_idx = -1

        for epoch in range(epochs):

            # generate predict
            output = self(X, betas)

            if epoch % 10 == 0:
                e_idx = e_idx + 1

            # calculate loss
            loss = self.loss_fn(output, trt_panel, cost, budget, lambd_list[e_idx])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if verbose == 1:
                print("Epoch {0} - Training loss: {1} -".format(epoch, loss))
                

    def getmask(self, X, betas):

        output = self(X, betas)
        mask = torch.zeros(output.size())
        argm = torch.argmax(output, dim=1)
        mask[torch.arange(output.size()[0]), argm] = 1

        return mask.detach().numpy()
