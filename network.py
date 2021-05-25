"""
Embedding Inner-product Network
"""

# Author: Qi Xu <qxu6@uci.edu>


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim.lr_scheduler import ExponentialLR



# Define the network structure

class EINet(nn.Module):
    
    def __init__(self, input_size, layer_trt=2, layer_cov=2, act_trt="linear", act_cov="linear", 
                                   width_trt=20, width_cov=20, width_embed=5):

        super(EINet, self).__init__()

        trt_dim, cov_dim = input_size

        self.layer_trt, self.layer_cov = layer_trt, layer_cov
        self.act_trt, self.act_cov = act_trt, act_cov
        self.width_trt, self.width_cov, self.width_embed = width_trt, width_cov, width_embed

        # define treatment encoder

        self.trt_input = nn.Linear(trt_dim, width_trt)

        self.trt_hidden = nn.ModuleList()
        for i in range(layer_trt):
            self.trt_hidden.append(nn.Linear(width_trt, width_trt))
            self.trt_hidden.append(nn.BatchNorm1d(num_features=width_trt))

        self.trt_embed = nn.Linear(width_trt, width_embed)

        # define covariate encoder

        self.cov_input = nn.Linear(cov_dim, width_cov)

        self.cov_hidden = nn.ModuleList()
        for i in range(layer_cov):
            self.cov_hidden.append(nn.Linear(width_cov, width_cov))
            self.cov_hidden.append(nn.BatchNorm1d(num_features=width_cov))

        self.cov_embed = nn.Linear(width_cov, width_embed)
        
    def weighted_mse_loss(self, input, target, weight):
        
        return (weight * (input - target) ** 2).mean()

    def treatment_weights(self, A):

        A_unique, A_inverse, A_count = torch.unique(A, return_counts=True, return_inverse=True, dim=0)
        W_ = 1 / A_count
        W = W_[A_inverse]

        num_channels = len(A_unique)
        num_samples = len(A)
        W_mat = torch.zeros((num_channels, num_samples))

        for i in range(num_channels):
            W_mat[i, A_inverse == i] = W[A_inverse == i]

        return W_mat
        
        
    def treatment_embed(self, A):
        
        # treatment encoding

        trt = self.trt_input(A)
        if self.act_trt == "relu":
            trt = F.relu(trt)
        elif self.act_trt == "linear":
            trt = trt
        
        for index, layer in enumerate(self.trt_hidden):
            if index % 2 == 0:
                trt = layer(trt)
                break
            elif index % 2 == 1:
                trt = layer(trt)
                if self.act_trt == "relu":
                    trt = F.relu(trt)
                elif self.act_trt == "linear":
                    trt = trt

        trt = self.trt_embed(trt)

        # centralize embedding to satisfy the constraints
        weight = self.treatment_weights(A)
        trt_w = weight.matmul(trt)
        trt_mean = torch.mean(trt_w, dim=0)
        
        trt_embed = trt - trt_mean
        
        return trt_embed
    
    def covariate_embed(self, X):
        
        # covaraite encoding

        cov = self.cov_input(X)
        if self.act_cov == "relu":
            cov = F.relu(cov)
        elif self.act_cov == "linear":
            cov = cov

        for index, layer in enumerate(self.cov_hidden):
            if index % 2 == 0:
                cov = layer(cov)
                break
            elif index % 2 == 1:
                cov = layer(cov)
                if self.act_cov == "relu":
                    cov = F.relu(cov)
                elif self.act_cov == "linear":
                    cov = cov

        cov_embed = self.cov_embed(cov)

        return cov_embed

    def forward(self, X, A):

        trt_embed = self.treatment_embed(A)
        
        cov_embed = self.covariate_embed(X)
        
        output = torch.sum(torch.mul(trt_embed, cov_embed), dim=1)

        return output
    
    def training_step(self, batch):
        
        # load data
        R, X, A, weight = batch
        
        # generate prediction
        output = self(X, A)
        
        # calculate loss
        loss = self.weighted_mse_loss(output, R, weight)
        
        return loss
    
    def epoch_end(self, epoch, result):
        
        print("Epoch: {} - Training Loss: {:.4f}".format(epoch, result))


# Define the network trainer
class Trainer():
    
    def fit(self, epochs, learning_rate, model, train_loader, print_history, opt_func, weight_decay, device):
        
        history = []
        optimizer = opt_func(model.parameters(), learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()
        scheduler = ExponentialLR(optimizer, gamma=0.999)
        
        for epoch in range(epochs):
            # training
            for batch in train_loader:
                batch = [item.to(device) for item in batch]
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
            result = self._evaluate(model, train_loader, device)
            if print_history:
                model.epoch_end(epoch, result)
            history.append(result)
            
        return history
            
    def _evaluate(self, model, train_loader, device):
        
        outputs = []
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            outputs.append(model.training_step(batch))
        
        return torch.stack(outputs).mean()