"""
Duo Encoder Network
"""

# Author: Qi Xu <qxu6@uci.edu>


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim.lr_scheduler import ExponentialLR

# cancel out layer
class CancelOut(nn.Module):
    '''
    CancelOut Layer
    
    x - an input data (vector, matrix, tensor)
    '''
    def __init__(self,inp, *kargs, **kwargs):
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(inp,requires_grad = True)+4)
    def forward(self, x):
        return (x * torch.sigmoid(self.weights.float()))

# Define the network structure

class DuoEncoderNet(nn.Module):
    
    def __init__(self, input_size, layer_trt=2, layer_cov=2, layer_men=2, 
                        act_trt="linear", act_cov="linear", act_men="linear",
                        width_trt=20, width_cov=20, width_men=20, width_embed=5, 
                        cov_cancel=True, men_cancel=True,
                        family="gaussian"):

        super().__init__()

        trt_dim, cov_dim = input_size

        self.layer_trt, self.layer_cov, self.layer_men = layer_trt, layer_cov, layer_men
        self.act_trt, self.act_cov, self.act_men = act_trt, act_cov, act_men
        self.width_trt, self.width_cov, self.width_embed, self.width_men = width_trt, width_cov, width_embed, width_men
        self.cov_cancel, self.men_cancel = cov_cancel, men_cancel
        self.family = family

        # define treatment encoder

        self.trt_input = nn.Linear(trt_dim, width_trt)

        self.trt_hidden = nn.ModuleList()
        for i in range(layer_trt):
            self.trt_hidden.append(nn.Linear(width_trt, width_trt))
            self.trt_hidden.append(nn.BatchNorm1d(num_features=width_trt))

        self.trt_embed = nn.Linear(width_trt, width_embed)

        # define covariate encoder

        if cov_cancel:
            self.cov_co = CancelOut(cov_dim)
    
        self.cov_input = nn.Linear(cov_dim, width_cov)

        self.cov_hidden = nn.ModuleList()
        
        for _ in range(layer_cov):
            self.cov_hidden.append(nn.Linear(width_cov, width_cov))
            self.cov_hidden.append(nn.BatchNorm1d(num_features=width_cov))

        self.cov_embed = nn.Linear(width_cov, width_embed)

        # define mean effect network

        if men_cancel:
            self.men_co = CancelOut(cov_dim)

        self.men_input = nn.Linear(cov_dim, width_men)

        self.men_hidden = nn.ModuleList()

        for _ in range(layer_men):
            self.men_hidden.append(nn.Linear(width_men, width_men))
            self.men_hidden.append(nn.BatchNorm1d(num_features=width_men))

        self.men_output = nn.Linear(width_men, 1)

        
    def weighted_mse_loss(self, input, target, weight):
        
        return (weight * (input - target) ** 2).mean()

    def weighted_crossentropy_loss(self, input, target, weight):
        
        loss_fn = nn.BCEWithLogitsLoss(weight=weight, reduction="mean")

        return loss_fn(input, target)

    def treatment_weights(self, A):

        A_unique, A_inverse, A_count = torch.unique(A, return_counts=True, return_inverse=True, dim=0)
        W_ = 1 / A_count
        W = W_[A_inverse]

        n_combinations = A_unique.shape[0]
        n_samples = A.shape[0]

        device = W.get_device()
        if device == -1:
            device = "cpu"
        W_mat = torch.zeros((n_combinations, n_samples))
        W_mat = W_mat.to(device)

        for i in range(n_combinations):

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

        if self.cov_cancel:
            cov = self.cov_co(X)
            cov = self.cov_input(cov)
        else:
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

    def mean_effect_network(self, X):

        # mean effect network

        if self.men_cancel:
            men = self.men_co(X)
            men = self.men_input(men)
        else:
            men = self.men_input(X)

        if self.act_men == "relu":
            men = F.relu(men)
        elif self.act_men == "linear":
            men = men

        for index, layer in enumerate(self.men_hidden):
            if index % 2 == 0:
                men = layer(men)
                break
            elif index % 2 == 1:
                men = layer(men)
                if self.act_men == "relu":
                    men = F.relu(men)
                elif self.act_men == "linear":
                    men = men
        
        men_output = self.men_output(men)

        return men_output

    def forward(self, X, A):

        trt_embed = self.treatment_embed(A)
        
        cov_embed = self.covariate_embed(X)
        
        men_output = self.mean_effect_network(X)

        output = torch.sum(torch.mul(trt_embed, cov_embed) + men_output, dim=1)

        return output
    
    def training_step(self, batch):
        
        # load data
        Y, X, A, weight = batch
        
        # generate prediction
        output = self(X, A)

        # calculate loss
        if self.family == "gaussian":
            loss = self.weighted_mse_loss(output, Y, weight)
        elif self.family == "bernoulli":
            loss = self.weighted_crossentropy_loss(output, Y, weight)
        
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