# Source code for Double Encoder Model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from skfda.representation.basis import BSpline
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.container import ITRDataset
from src.util import *

# Define the Double Encoder Model

class DoubleEncoderModel(nn.Module):

    def __init__(self, input_dim, embed_dim = 5, trt_encoder = "nn",
                 trt_layer = 2, trt_width = 20, trt_act = "relu", trt_num = None,
                 cov_encoder = "nn", cov_layer = 2, cov_width = 20,
                 cov_act = "relu", cov_degree = 3, cov_bs_order=3,
                 cov_bs_bases=6):
        
        """
        Define the Double Encoder Model

        Parameters
        ----------
        input_dim : tuple
            The dimension of the input data, (trt_dim, cov_dim)

        trt_encoder : str, {"nn", "dict"}
            The encoder for treatment, either neural network or dictionary

        trt_layer : int
            The number of layers for the neural network encoder

        trt_width : int
            The width of the neural network encoder

        trt_act : str, {"relu", "linear"}
            The activation function for the neural network encoder

        trt_num: int, 
            The number of treatments for the dictionary encoder

        cov_encoder : str, {"nn", "bs", "poly"}
            The encoder for covariates, either neural network, B-spline or polynomial
        
        cov_layer : int
            The number of layers for the neural network encoder

        cov_width : int
            The width of the neural network encoder

        cov_act : str, {"relu", "linear"}
            The activation function for the neural network encoder

        cov_degree : int
            The degree of the polynomial encoder

        cov_bs_order : int
            The order of the B-spline encoder

        cov_bs_bases : int
            The number of bases for the B-spline encoder

        """
        
        super().__init__()

        trt_dim, cov_dim = input_dim

        self.trt_dim = trt_dim

        self.cov_dim = cov_dim

        self.embed_dim = embed_dim

        self.trt_encoder = trt_encoder

        if trt_encoder == "nn":

            self.trt_layer = trt_layer
            self.trt_width = trt_width
            self.trt_act = trt_act

        elif trt_encoder == "dict":

            self.trt_num = trt_num

        self.cov_encoder = cov_encoder

        if cov_encoder == "nn":

            self.cov_layer = cov_layer
            self.cov_width = cov_width
            self.cov_act = cov_act

        elif cov_encoder == "bs":

            self.cov_bs_order = cov_bs_order
            self.cov_bs_bases = cov_bs_bases

        elif cov_encoder == "poly":

            self.cov_degree = cov_degree

        # define treatment encoder

        # additive encoder

        self.trt_additive_encoder_model = nn.Linear(trt_dim, embed_dim, bias=False)

        # interactive encoder

        # interactive encoder cutoffs
        ## if the input treatment is a single treatment, it will be cutoff
        ## so no interaction will be added; if the input treatment is a combination
        ## of treatments, it will not be cutoff, so the interaction will be added

        self.trt_interactive_cutoff = nn.Linear(trt_dim, 1, bias=False)
        self.trt_interactive_cutoff.weight = nn.Parameter(torch.ones(trt_dim))

        if trt_encoder == "nn":

            self.trt_interactive_encoder_model = nn.ModuleList()
            self.trt_interactive_encoder_model.append(nn.Linear(trt_dim, trt_width))

            for _ in range(trt_layer):
                self.trt_interactive_encoder_model.append(nn.BatchNorm1d(num_features=trt_width))
                self.trt_interactive_encoder_model.append(nn.Linear(trt_width, trt_width))

            self.trt_interactive_encoder_model.append(nn.Linear(trt_width, embed_dim))

        elif trt_encoder == "dict":

            self.trt_interactive_encoder_model = nn.Linear(trt_num, embed_dim, bias=False, dtype=torch.float32)

        # define covariate encoder

        if cov_encoder == "nn":

            self.cov_encoder_model = nn.ModuleList()
            self.cov_encoder_model.append(nn.Linear(cov_dim, cov_width))

            for _ in range(cov_layer):
                self.cov_encoder_model.append(nn.BatchNorm1d(num_features=cov_width))
                self.cov_encoder_model.append(nn.Linear(cov_width, cov_width))

            self.cov_encoder_model.append(nn.Linear(cov_width, embed_dim))

        elif cov_encoder == "bs":

            self.cov_encoder_model = nn.Linear(cov_bs_bases * cov_dim, embed_dim, bias=False)

        elif cov_encoder == "poly":

            self.cov_encoder_model = nn.Linear(cov_degree * cov_dim, embed_dim, bias=False)

    def weighted_mse_loss(self, input, target, weight):

        return (weight * (input - target) ** 2).mean()
    
    def trt_additive_embed(self, A):

        return self.trt_additive_encoder_model(A)

    def trt_interactive_embed(self, A):

        if self.trt_encoder == "nn":
            
            trt_cutoff = torch.minimum(torch.maximum(self.trt_interactive_cutoff(A) - 1, torch.Tensor([0])), torch.Tensor([1]))

            trt_interactive = self.trt_interactive_encoder_model[0](A)

            if self.trt_act == "relu":

                trt_interactive = F.relu(trt_interactive)

            elif self.trt_act == "linear":

                trt_interactive = F.relu(trt_interactive)

            for i in range(1, 2 * self.trt_layer + 1):

                if i % 2 == 1:

                    trt_interactive = self.trt_interactive_encoder_model[i](trt_interactive)

                elif i % 2 == 0:

                    trt_interactive = self.trt_interactive_encoder_model[i](trt_interactive)

                    if self.trt_act == "relu":

                        trt_interactive = F.relu(trt_interactive)

                    elif self.trt_act == "linear":

                        trt_interactive = F.relu(trt_interactive)

            trt_interactive = self.trt_interactive_encoder_model[2 * self.trt_layer + 1](trt_interactive)

            trt_interactive = torch.mul(trt_interactive, trt_cutoff[:, None])

        elif self.trt_encoder == "dict":
            
            trt_cutoff = torch.minimum(torch.maximum(self.trt_interactive_cutoff(A) - 1, torch.Tensor([0])), torch.Tensor([1]))

            _, A_cate = torch.unique(A, return_inverse=True, dim=0)

            A_onehot = F.one_hot(A_cate, num_classes=self.trt_num).to(torch.float)

            trt_interactive = self.trt_interactive_encoder_model(A_onehot)

            trt_interactive = torch.mul(trt_interactive, trt_cutoff[:, None])

        return trt_interactive
    
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
    
    def trt_embed(self, A):

        trt_additive = self.trt_additive_embed(A)
        trt_interactive = self.trt_interactive_embed(A)

        trt = trt_additive + trt_interactive

        # centralize the treatment embedding

        weight = self.treatment_weights(A)
        trt_w = weight.matmul(trt)
        trt_mean = torch.mean(trt_w, dim = 0)

        trt = trt - trt_mean

        return trt
    
    def cov_embed(self, X):

        if self.cov_encoder == "nn":

            cov = self.cov_encoder_model[0](X)

            if self.cov_act == "relu":

                cov = F.relu(cov)

            elif self.cov_act == "linear":

                cov = F.relu(cov)

            for i in range(1, 2 * self.cov_layer + 1):

                if i % 2 == 1:

                    cov = self.cov_encoder_model[i](cov)

                elif i % 2 == 0:

                    cov = self.cov_encoder_model[i](cov)

                    if self.cov_act == "relu":

                        cov = F.relu(cov)

                    elif self.cov_act == "linear":

                        cov = F.relu(cov)

            cov = self.cov_encoder_model[2 * self.cov_layer + 1](cov)

        elif self.cov_encoder == "bs":

            bss = BSpline(order = self.cov_bs_order, n_basis=self.cov_bs_bases)
            
            X_expand = []

            for i in range(self.cov_dim):

                X_expand.append(torch.Tensor(np.squeeze(bss(X[:, i])).T))

            X_expand = torch.cat(X_expand, dim = 1)

            cov = self.cov_encoder_model(X_expand)

        elif self.cov_encoder == "poly":

            X_poly = torch.cat([X ** i for i in range(1, self.cov_degree + 1)], 1)

            cov = self.cov_encoder_model(X_poly)

        return cov
    
    def forward(self, X, A):

        trt = self.trt_embed(A)
        cov = self.cov_embed(X)

        output = torch.sum(torch.mul(trt, cov), dim = 1)

        return output
    
    def training_step(self, batch):

        # load data

        X, A, Y, weight = batch

        # generate treatment effects prediction

        output = self(X, A)

        loss = self.weighted_mse_loss(output, Y, weight)

        return loss
    
    def epoch_end(self, epoch, result):

        print("Epoch [{}], loss: {:.4f}".format(epoch, result))



# Define the model trainer

class Trainer():

    def fit(self, model, epochs, learning_rate, train_loader, 
            opt_func, weight_decay, interactive_weight_decay, device,
            print_history):
        
        history = []

        cov_optimizer = opt_func(model.cov_encoder_model.parameters(), learning_rate, weight_decay=weight_decay)
        trt_additive_optimizer = opt_func(model.trt_additive_encoder_model.parameters(), learning_rate, weight_decay=weight_decay)
        trt_interactive_optimizer = opt_func(model.trt_interactive_encoder_model.parameters(), learning_rate, weight_decay=interactive_weight_decay)

        cov_scheduler = ExponentialLR(cov_optimizer, gamma=0.95)
        trt_additive_scheduler = ExponentialLR(trt_additive_optimizer, gamma=0.95)
        trt_interactive_scheduler = ExponentialLR(trt_interactive_optimizer, gamma=0.95)

        cov_optimizer.zero_grad()
        trt_additive_optimizer.zero_grad()
        trt_interactive_optimizer.zero_grad()

        for epoch in range(epochs):

            for batch in train_loader:
                
                batch = [item.to(device) for item in batch]

                loss = model.training_step(batch)

                loss.backward()

                cov_optimizer.step()
                trt_additive_optimizer.step()
                trt_interactive_optimizer.step()

                cov_optimizer.zero_grad()
                trt_additive_optimizer.zero_grad()
                trt_interactive_optimizer.zero_grad()

            cov_scheduler.step()
            trt_additive_scheduler.step()
            trt_interactive_scheduler.step()

            result = self._evaluate(model, train_loader, device)

            if print_history:

                model.epoch_end(epoch, result)

            history.append(result)

        return history

    def _evaluate(self, model, train_loader, device):

        outputs = []

        for batch in train_loader:

            batch = [item.to(device) for item in batch]

            output = model.training_step(batch)

            outputs.append(output)

        return torch.stack(outputs).mean()


class ITR():

    """
    Individualized Treatment Rule (ITR) for Combination Treatments
    
    Parameters
    ----------
    SEE DETAILS FROM THE CLASS DoubleEncoderModel
    """

    def __init__(self, embed_dim = 5, trt_encoder = "nn",
                 trt_layer = 2, trt_width = 20, trt_act = "relu", trt_num = None,
                 cov_encoder = "nn", cov_layer = 2, cov_width = 20,
                 cov_act = "relu", cov_degree = 3, cov_bs_order = 3,
                 cov_bs_bases=6):
        
        self.embed_dim = embed_dim
        self.trt_encoder = trt_encoder
        self.trt_layer = trt_layer
        self.trt_width = trt_width
        self.trt_act = trt_act
        self.trt_num = trt_num
        self.cov_encoder = cov_encoder
        self.cov_layer = cov_layer
        self.cov_width = cov_width
        self.cov_act = cov_act
        self.cov_degree = cov_degree
        self.cov_bs_order = cov_bs_order
        self.cov_bs_bases = cov_bs_bases

    def fit(self, X, A, Y, mode="randomized", epochs=100, learning_rate=0.001,
            opt_func=Adam, weight_decay=0.001, interactive_weight_decay=0.1,
            batch_size=32, trt_free_model="linear", ps_model="multinomial", device="cpu", verbose=0):

        """
        Fit the ITR model

        Parameters
        ----------
        X: array-like, shape (n_samples, n_covariates)
            Covariates

        A: array-like, shape (n_samples, n_treatments)
            Treatment assignment matrix

        Y: array-like, shape (n_samples,)
            Outcome

        mode: str, optional (default="randomized")
            The mode of the ITR model. "randomized" for randomized ITR, "ps" for inverse propensity score weighted ITR.

        epochs: int, optional (default=100)
            Number of epochs

        learning_rate: float, optional (default=0.001)
            Learning rate

        opt_func: torch.optim, optional (default=torch.optim.Adam)
            Optimizer

        weight_decay: float, optional (default=0)
            Weight decay for covariates

        interactive_weight_decay: float, optional (default=0)
            Weight decay for interactive terms

        device: str, optional (default="cpu")
            Device to use for training

        verbose: int
            Whether to print the training history

        """

        _device = return_device(device)

        if verbose > 0:

            print("-------- The program is running on {0}----------".format(_device))

        if verbose == 1:

            print_history = True

        else:

            print_history = False

        self.device = _device

        self.mode = mode

        self.ps_model = ps_model
        self.trt_free_model = trt_free_model

        X = check_covariate(X)

        input_dim = (A.shape[1], X.shape[1])

        n_samples = X.shape[0]

        n_treatments = np.unique(A, axis=0).shape[0]
        self.trt_num = n_treatments

        self.model = DoubleEncoderModel(input_dim, self.embed_dim, self.trt_encoder,
                                        self.trt_layer, self.trt_width, self.trt_act,
                                        self.trt_num, self.cov_encoder, self.cov_layer,
                                        self.cov_width, self.cov_act, self.cov_degree,
                                        self.cov_bs_order, self.cov_bs_bases).to(_device)
        
        # compute residual as output

        Y = Y - estimate_treatment_free(X, Y, model=trt_free_model)

        if mode == "randomized":

            ips = np.ones(n_samples)

        elif mode == "ps":
            
            # compute propensity score

            ips = estimate_ips(X, A, stabilize=True, model=ps_model)

        # create the datset to fit the double encoder model

        X_tsr = torch.from_numpy(X).float()
        A_tsr = torch.from_numpy(A).float()
        Y_tsr = torch.from_numpy(Y).float()
        W_tsr = torch.from_numpy(ips).float()

        dataset = ITRDataset(X_tsr, A_tsr, Y_tsr, W_tsr)

        loader = DataLoader(dataset, batch_size=batch_size)

        trainer = Trainer()

        history = []
        history += trainer.fit(self.model, epochs, learning_rate, loader, 
                               opt_func, weight_decay, interactive_weight_decay,
                               device, print_history)
        
        return history

    def get_trt_panel(self, X, A):

        X_tsr = torch.from_numpy(X).float().to(self.device)
        A_tsr = torch.from_numpy(A).float().to(self.device)

        A_unique = torch.unique(A_tsr, dim=0)
        cov_embed = self.model.cov_embed(X_tsr)
        trt_embed = self.model.trt_embed(A_unique)
        
        trt_panel = torch.matmul(cov_embed, torch.transpose(trt_embed, dim0=0, dim1=1))

        return trt_panel
    
    def predict(self, X, A):

        trt_panel = self.get_trt_panel(X, A)

        X_tsr = torch.from_numpy(X).float().to(self.device)
        A_tsr = torch.from_numpy(A).float().to(self.device)

        A_unique = torch.unique(A_tsr, dim = 0)

        idx = torch.argmax(trt_panel, dim=1)

        D = A_unique[idx]

        return D.cpu().numpy()

    def evaluate(self, Y, A, D, optA=None, X=None):

        n_samples = Y.shape[0]

        if self.mode == "randomized":

            ips = np.ones(n_samples)

        elif self.mode == "ps":

            ips = estimate_ips(X, A, stabilize=True, model=self.ps_model)

        output = []

        D = torch.from_numpy(D).float()
        A = torch.from_numpy(A).float()

        if optA is not None:

            optA = torch.from_numpy(optA).float()

            acc = torch.mean(torch.all(D == optA, dim=1) * 1.0)
            output.append(acc.numpy())

        # calculate value function

        nom = torch.sum(torch.all(D == A, dim = 1) * Y * ips)
        den = torch.sum(torch.all(D == A, dim = 1) * ips)

        val = nom / den

        output.append(val.numpy())

        return output
    