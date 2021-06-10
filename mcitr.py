"""
Multi-channel Individualized Treatment Rule
"""

# Author: Qi Xu <qxu6@uci.edu>

import torch
from torch.utils.data import DataLoader

from torch.optim import Adam

import numpy as np 
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

from util import plot_train_history
from denet import DuoEncoderNet, Trainer
from container import ITRDataset
from mckp import _mckp


## some helper function for MCITR

def _propensity_score(X, A, save_model=True):

    """
    Compute propensity score for observational study
    using MLP classifier

    X: covariates, 2d array
    A: assigned treatment, in categorical coding: 0, 1, 2, ...
    """

    n_sample, n_feature = X.shape

    mlp = MLPClassifier()
    mlp.fit(X, A)

    prob = mlp.predict_proba(X) # probability
    prop = prob[np.arange(n_sample), A]

    if save_model is True:

        return prop, mlp

    else:

        return prop

def _residual(Y, X, weight=None):

    linreg = LinearRegression()
    linreg.fit(X, Y, sample_weight=weight)
    residual = Y - linreg.predict(X)

    return residual


def _check_covariate(X):

    """
    Check whether covariate is 2d array,
    if not, add an extra axis

    X: covariates: 1d/2d
    """

    if X.ndim == 1:

        X = X[:, np.newaxis]

    return X

def _check_treatment(A):

    """
    Check whether treatment is multi-channel,
    if not, raise exception

    A: treatment: binary coding
    """

    if A.ndim == 1:

        raise Exception("Input treatment only has one channel.")

    else:

        return A


def _categorical_treatment(A):

    """
    Create categorical representation of multi-channel treatment
    """

    _, A_cate = np.unique(A, return_inverse=True, axis=0)

    return A_cate


def _return_device(device):

    if device == "cpu":

        return "cpu"

    elif device == "gpu":

        return "cuda:0"

    elif device == "default":

        if torch.cuda.is_available():
            return "cuda:0"

        else:
            return "cpu"



class MCITR():

    """
    Multi-Channel Individualized Treatment Rule

    Estimate continuous representation in the latent space for treatment and subject 
    pre-treatment covariates through two separate multi-layer perceptron, and model 
    the residual as inner product between two latent embedding.

    This class also implement subject-wise rotation to find sub-optimal treatment 
    to satisfy the budget constraint.

    Parameters
    ----------
    act_trt: {'relu', 'linear'}, default='relu'
        activation function of treatment embedding network

    act_cov: {'relu', 'linear'}, default='relu'
        activation function of covariate embedding network

    depth_trt: int, default=2
        depth of treatment embedding network, excluding input layer

    depth_cov: int, default=2
        depth of covariate embedding network, excluding input layer

    width_trt: int, default=10
        width of treatment embedding network

    width_cov: int, default=10
        width of covariate embedding network

    width_embed: int, default=5
        width of last layers in both treatment and covariate embedding network

    Attributes
    ----------


    Methods
    ----------

    """

    def __init__(self, act_trt="relu", act_cov="relu", depth_trt=2, depth_cov=2, width_trt=20, width_cov=20, width_embed=5, cov_cancel=True):
        
        self.act_trt = act_trt
        self.act_cov = act_cov
        self.depth_trt = depth_trt
        self.depth_cov = depth_cov
        self.width_trt = width_trt
        self.width_cov = width_cov
        self.width_embed = width_embed
        self.cov_cancel = cov_cancel
        

    def fit(self, Y, X, A, scenario="ct", epochs=100, learning_rate=1e-3, verbose=0, opt_func=Adam, weight_decay=0.01, batch_size=32, device="default"):

        """
        Fit the model according to the given training data.

        Parameters:

        Y: array-like of shape (n_samples, )
            outcome vector relative to X

        X: array-like of shape (n_samples, n_features)
            pre-treatment covariate

        A: array-like of shape (n_samples, n_channels)
            multi-channel treatment

        scenario: {"ct", "os"}, default="ct"
            problem scenario, if scnario="ct", propensity score 
            will be treated as fixed and identical for each 
            treatment combination; if scnario="os", propensity 
            score will be estimated through a multilayer perceptron

        epochs: int, default=100
            training epochs for the network
        
        learning_rate: float, default=1e-3
            learning rate for the network, learning rate will be decreased exponentially by a factor 0.999

        opt_func: see torch.optim, default=Adam
            optimiation algorithm for the network

        verbose: {0, 1, 2, 3}, default=0
            output status:
                0: no output
                1: output training loss history
                2: plot training loss history
                3: both 1 and 2

        weight_decay: float, default=0.01
            L2 penalty

        batch_size: int, default=32
            training batch size

        device: {"default", "cpu", "gpu"}, default="default"
            "default": use gpu if detected, otherwise use cpu
            "cpu": specify to use cpu
            "gpu": specify to use gpu

        """

        _device = _return_device(device)

        if verbose > 0:
            print("--------- The program is running on {0}----------".format(_device))

        self.device = _device

        A = _check_treatment(A)

        X = _check_covariate(X)

        input_dim = (A.shape[1], X.shape[1])
        n_samples = X.shape[0]

        self.model = DuoEncoderNet(input_dim, self.depth_trt, 
                            self.depth_cov, self.act_trt,
                            self.act_cov, self.width_trt, 
                            self.width_cov, self.width_embed, self.cov_cancel)

        self.model = self.model.to(self.device)

        # compute propensity score

        if scenario == "ct":

            W = np.ones((n_samples,))

        elif scenario == "os":

            A_cate = _categorical_treatment(A)
            prop, prop_model = _propensity_score(X, A_cate)

            W = 1 / prop

            self.prop_model = prop_model # used to predict propensity score for new data

        R = _residual(Y, X, weight=W)

        # create dataset to fit torch model
        R_tsr = torch.from_numpy(R).float()
        X_tsr = torch.from_numpy(X).float()
        A_tsr = torch.from_numpy(A).float()
        W_tsr = torch.from_numpy(W).float()

        dataset = ITRDataset(R_tsr, X_tsr, A_tsr, W_tsr)

        loader = DataLoader(dataset, batch_size=batch_size)

        if verbose == 0:
            print_history = False
            plot_history = False
        elif verbose == 1:
            print_history = True
            plot_history = False
        elif verbose == 2:
            print_history = False
            plot_history = True
        elif verbose == 3:
            print_history = True
            plot_history = True

        trainer = Trainer()
        history = []
        history += trainer.fit(epochs=epochs, learning_rate=learning_rate, model=self.model, train_loader=loader, 
                                print_history=print_history, opt_func=opt_func, weight_decay=weight_decay, device=self.device)

        if plot_history:
            plot_train_history(history)

        return history


    def predict(self, X, A):

        """
        Predict optimal multi-channel individualized treatment rule

        The returned prediction is in binary coding

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            pre-treatment covariate

        A: array-like of shape (n_samples, n_channels)
            multi-channel treatment, only used to know which treatment
            are being used

        Returns
        ---------
        D: array-like of shape (n_samples, n_channels)
            predicted optimal multi-channel treatment
        """

        X_tsr = torch.from_numpy(X).float().to(self.device)
        A_tsr = torch.from_numpy(A).float().to(self.device)

        A_unique = torch.unique(A_tsr, dim=0)
        cov_embed = self.model.covariate_embed(X_tsr)
        trt_embed = self.model.treatment_embed(A_unique)
        
        trt_panel = torch.matmul(cov_embed, torch.transpose(trt_embed, dim0=0, dim1=1))
        
        idx = torch.argmax(trt_panel, dim=1)
        
        D = A_unique[idx]

        return D.cpu().numpy()    

    def evaluate(self, Y, A, D, X=None, optA=None, scenario="ct", accuracy=True, value=True):

        """
        Evaluate a given treatment rule

        Parameters
        ----------

        Y: array-like of shape (n_samples, )
            outcome vector relative to X

        A: array-like of shape (n_samples, n_channels)
            multi-channel treatment

        D: array-like of shape (n_samples, n_channels)
            multi-channel treatment need to be evaluated

        X: array-like of shape (n_samples, n_features)
            pre-treatment covariate, needed in the observational study scenario
        
        optA: array-like of shape (n_samples, n_channels)
            optimal treatment rule, unknown in the real data 

        accuracy: bool
            whether or not to return accuracy 

        value: bool
            whether or not to return value function

        Returns
        ----------
        output: list
            may include value function, or accuracy, or both

        """

        Y = torch.from_numpy(Y).float()
        A = torch.from_numpy(A).float()
        D = torch.from_numpy(D).float()
        
        n_samples = len(Y)

        if scenario == "ct":

            prop = np.ones((n_samples,))
            
        elif scenario == "os":

            if X is None:

                raise Exception("pre-treatment covariates are unknown, propensity scores are not estimable.")

            else: 

                A_cate = _categorical_treatment(A)
                prob = self.prop_model.predict_proba(X)
                prop = prob[np.arange(n_samples), A_cate]


        output = []
        if accuracy:

            if optA is None:
                raise Exception("optimal assignment is unknown.")

            else:
                optA = torch.from_numpy(optA).float()
                acc = torch.mean(torch.all(D == optA, dim=1) * 1.0)
                output.append(acc.numpy())

        if value:

            nom = torch.sum(torch.all(D == A, dim=1) * Y / prop) / n_samples
            den = torch.sum(torch.all(D == A, dim=1) * 1.0 / prop) / n_samples

            val = nom / den
            output.append(val.numpy())
        
        return output


    def realign_random(self, X, A, cost, budgets):

        """
        Random select samples to assign treatment to satisfy the budget constraint

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            pre-treatment covariate

        A: array-like of shape (n_samples, n_channels)
            multi-channel treatment

        cost: array-like of shape (n_channels, )
            cost for each treatment channel
        
        budgets: array-like of shape (n_channels,)
            budget for each channels

        """
        n_channels = A.shape[1]

        D = self.predict(X, A)

        limit = np.divide(budgets, cost)

        for c in range(n_channels):

            if np.sum(D[:, c]) < limit[c]:
                
                pass

            else: 

                col = D[:, c]
                idx = np.squeeze(np.argwhere(col == 1))

                idx_select = np.random.choice(idx, size=(int(limit[c]), ))

                D[:, c] = 0
                D[idx_select, c] = 1

        return D

    def realign_mckp(self, X, A, cost, budget):

        """
        solve the budget constraint problem with MCKP

        Parameters
        -----------
        X: array-like of shape (n_samples, n_features)
            pre-treatment covariate

        A: array-like of shape (n_samples, n_channels)
            multi-channel treatment

        cost: array-like of shape (n_combinations, )
            cost for each treatment combination

        budgets: float
            total budget over population

        """


        X_tsr = torch.from_numpy(X).float()
        if self.device == "gpu":
            X_tsr = X_tsr.to(self.device)

        A_unique = np.unique(A, axis=0)
        A_tsr = torch.from_numpy(A_unique).float()
        if self.device == "gpu":
            A_tsr = A_tsr.to(self.device)

        if self.device == "gpu":

            alphas = self.model.covariate_embed(X_tsr).detach().cpu().numpy() # covariate embedding
            betas = self.model.treatment_embed(A_tsr).detach().cpu().numpy() # treatment embedding

        else:

            alphas = self.model.covariate_embed(X_tsr).detach().numpy() # covariate embedding
            betas = self.model.treatment_embed(A_tsr).detach().numpy() # treatment embedding

        trt_panel = alphas.dot(betas.transpose())

        sol = _mckp(trt_panel, cost, budget)

        argm = np.argmax(sol, axis=1)

        D = A_unique[argm]

        return D

    






