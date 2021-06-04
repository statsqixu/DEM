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
        loss2 = lambd * torch.maximum(torch.sum(mask_panel * cost) - budget, torch.Tensor([0])) ** 2

        return loss1 + loss2

    def train(self, data, epochs=100, learning_rate=1e-3, lambd=100, verbose=0):

        # define optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr = learning_rate)
        optimizer.zero_grad()
        scheduler = ExponentialLR(optimizer, gamma=0.999)

        # load data
        X, betas, trt_panel, cost, budget = data

        for epoch in range(epochs):

            # generate predict
            output = self(X, betas)

            # calculate loss
            loss = self.loss_fn(output, trt_panel, cost, budget, lambd)

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


if __name__ == "__main__":

    cenet = ConstrEncoderNet(5, width_embed=5)

    X = torch.rand(100, 5)

    betas = torch.rand(4, 5)

    trt_panel = torch.rand(100, 4)

    cost_panel = torch.Tensor([[0, 0, 1, 1]])

    budget = torch.Tensor([20])

    data = (X, betas, trt_panel, cost_panel, budget)

    cenet.train(data, 100, 1e-2, verbose=1)
    
    mask = cenet.getmask(X, betas)

    print(mask)