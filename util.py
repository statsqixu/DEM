import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



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

# Define the dataset
class ITRDataset(Dataset):
    
    def __init__(self, R, X, A, W):
        
        self.output, self.covariate, self.treatment, self.weight = R, X, A, W
        
    def __len__(self):
        return len(self.output)
    
    def __getitem__(self, idx):
        return self.output[idx], self.covariate[idx], self.treatment[idx], self.weight[idx]

# some utility functions
## plot training history  
def plot_train_history(history):
    
    plt.plot(history, "-x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")



class MCITR():

    def __init__(self, layer_trt=2, layer_cov=2, act_trt="linear", act_cov="linear", width_trt=20, width_cov=20, width_embed=5, scenario="ct"):

        self.layer_trt, self.layer_cov = layer_trt, layer_cov
        self.act_trt, self.act_cov = act_trt, act_cov
        self.width_trt, self.width_cov, self.width_embed = width_trt, width_cov, width_embed
        self.scenario = scenario

    def model_def(self, input_dim):

        model = EINet(input_dim, self.layer_trt, self.layer_cov, self.act_trt, self.act_cov, self.width_trt, self.width_cov, self.width_embed)

        return model 

    def fit(self, Y, X, A, epochs=100, learning_rate=1e-3, verbose=0, opt_func=torch.optim.Adam, weight_decay=0.01, batch_size=32, device="default"):

        if device == "default":
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")
        elif device == "cpu":
            device = "cpu"
        elif device == "gpu":
            device = "cuda:0"

        self.device = device

        if A.ndim == 1:
            raise Exception("Only one channel treatment.")
        elif A.ndim > 1:
            if X.ndim == 1:
                input_dim = (A.shape[1], 1) 
                X_ = X[:, np.newaxis]
            elif X.ndim > 1:
                input_dim = (A.shape[1], X.shape[1])
                X_ = X

        sample_size = A.shape[0]

        self.model = self.model_def(input_dim)
        self.model = self.model.to(device)

        # compute propensity score

        if self.scenario == "ct":

            W = np.ones((sample_size,))
            self.prop = np.ones((sample_size, )) / (2 ** A.shape[1])

            W = 1 / self.prop

        elif self.scenario == "os":

            _, A_compact = np.unique(A, axis=0, return_inverse=True)
            self.mlp = MLPClassifier(learning_rate="invscaling", max_iter=500)
            self.mlp.fit(X_, A_compact)
            prob = self.mlp.predict_proba(X_)
            self.prop = prob[np.arange(sample_size), A_compact]

            W = 1 / self.prop

        # compute residuel 
        linreg = LinearRegression()
        linreg.fit(X_, Y, sample_weight=W)
        R = Y - linreg.predict(X_)

        # create dataset
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

        X_tsr = torch.from_numpy(X).float().to(self.device)
        A_tsr = torch.from_numpy(A).float().to(self.device)

        A_unique = torch.unique(A_tsr, dim=0)
        cov_embed = self.model.covariate_embed(X_tsr)
        trt_embed = self.model.treatment_embed(A_unique)
        
        trt_panel = torch.matmul(cov_embed, torch.transpose(trt_embed, 0, 1))
        
        idx = torch.argmax(trt_panel, dim=1)
        
        D = A_unique[idx]

        return D.cpu().numpy()    

    def evaluate(self, Y, A, X, D, optA, accuracy=True, value=True):

        Y = torch.from_numpy(Y).float()
        A = torch.from_numpy(A).float()
        D = torch.from_numpy(D).float()
        optA = torch.from_numpy(optA).float()

        sample_size = len(Y)

        if self.scenario == "ct":
            W = np.ones((sample_size,))
            prop = np.ones((sample_size, )) / (2 ** A.shape[1])
        elif self.scenario == "os":
            _, A_compact = np.unique(A, axis=0, return_inverse=True)
            prob = self.mlp.predict_proba(X)
            prop = prob[np.arange(sample_size), A_compact]


        output = []
        if accuracy:

            acc = torch.mean(torch.all(D == optA, dim=1) * 1.0)
            output.append(acc.numpy())

        if value:

            nom = torch.sum(torch.all(D == A, dim=1) * Y / prop) / sample_size
            den = torch.sum(torch.all(D == A, dim=1) * 1.0 / prop) / sample_size

            val = nom / den
            output.append(val.numpy())
            #output.append(nom.numpy())
            #output.append(den.numpy())
        return output








        










    