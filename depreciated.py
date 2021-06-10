# Rotation approach to constrained problem


def _create_loss_individual(alpha, betas, cost, budget, lambda_1, lambda_2):

    """
    create loss function for individual-level budget constrained problem

    Parameters
    ----------
    alpha: array-like of shape (n_samples, n_embedding)
        covariate embedding 

    betas: array-like of shape (n_combinations, n_embedding)
        unique treatment embedding

    cost: array-like of shape (n_combinations, )
        cost for each treatment combination

    budget: array-like of shape (n_samples, )
        cost budget for each subject

    lambda_1: float, default=0.1
        penalty_coefficient for identity constraint

    lambda_2: float, default=1000
        peanlty_coefficient for unsatisification of budget constraint
    """
    
    def loss(X):

        I = anp.array([alpha.dot(X.transpose()).dot(beta.transpose()) for beta in betas])
        
        loss1 = - asp.special.logsumexp(I)
        penalty1 =  - lambda_1 * alpha.dot(X.transpose()).dot(alpha.transpose()) / alpha.dot(alpha)
        penalty2 = lambda_2 * anp.maximum(cost.dot(anp.exp(I) / anp.sum(anp.exp(I))) - budget, 0)

        return loss1 + penalty1 + penalty2

    return loss


def _binary_panel(I):

    I = np.exp(I) / np.sum(np.exp(I))
    I_binary = I
    I_binary[np.argmax(I)] = 1
    I_binary[I_binary < 1] = 0

    return I_binary


def _create_loss_population(alpha, alphas_r, betas, cost, budget, lambda_1, lambda_2):
    
    """
    create loss function for population-level budget constrained problem

    Parameters
    ----------
    alpha: array-like of shape (1, n_embedding)
        covariate embedding 

    alphas_r: array-like of shape (n_samples, n_embedding)
        remaining rotated covaraite embedding in the population

    betas: array-like of shape (n_combinations, n_embedding)
        unique treatment embedding

    cost: array-like of shape (n_combinations, )
        cost for each treatment combination

    budget: array-like of shape (n_samples, )
        total cost budget within population

    lambda_1: float, default=0.1
        penalty_coefficient for identity constraint

    lambda_2: float, default=1000
        peanlty_coefficient for unsatisification of budget constraint
    """

    def loss(X):
    
        I = anp.array([alpha.dot(X.transpose()).dot(beta.transpose()) for beta in betas])

        Is = anp.array(alphas_r.dot(betas.transpose()))

        loss1 = - asp.special.logsumexp(I)
        loss2 = - anp.sum(asp.special.logsumexp(Is, axis=1))
        
        penalty1 = -lambda_1 * alpha.dot(X.transpose()).dot(alpha.transpose()) / alpha.dot(alpha)
        
        cost1 = cost.dot(anp.exp(I) / anp.sum(anp.exp(I)))
        cost2 = anp.sum((anp.exp(Is) / anp.sum(anp.exp(Is), axis=1)[:, anp.newaxis]).dot(cost))

        penalty2 = lambda_2 * anp.maximum((cost1 + cost2 - budget), 0)

        return loss1 + loss2 + penalty1 + penalty2

    return loss




def realign_rotate(self, X, A, cost, budgets, lambda_1=1, lambda_2=1, budget_level="individual", verbose=0):

    """
    Rotate covaraite embedding to satisfy the budget constraint

    Parameters
    -----------
    X: array-like of shape (n_samples, n_features)
        pre-treatment covariate

    A: array-like of shape (n_samples, n_channels)
        multi-channel treatment

    cost: array-like of shape (n_combinations, )
        cost for each treatment combination

    budgets: array-like of shape (n_samples, ) or float
        if budget_level="individual", budgets include budget for each subject,
        if budget_level="population", budgets is total budget over population

    lambda_1: float, default=0.1
        penalty_coefficient for identity constraint

    lambda_2: float, default=1000
        peanlty_coefficient for unsatisification of budget constraint

    budget_level: {"individual", "population"}, default="individual"
        cost budget type

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

    n_embedding = alphas.shape[1]

    if budget_level == "individual":

        treatment_realign = []
        constraint_compliance = []

        # iterater over all samples
        for idx, alpha in tqdm(enumerate(alphas), ncols=100):

            budget = budgets[idx]

            I = anp.array([alpha.dot(beta.transpose()) for beta in betas])
            I_binary = _binary_panel(I)
            satisfy = (cost.dot(I_binary) - budget < 0)

            # first check whether the constraint is satisfied under optimal treatment
            if satisfy:
                argm = np.argmax(I)
                treatment_realign.append(A_unique[argm])
                constraint_compliance.append(satisfy)

            # if not, find a rotation that maximize its value under the constraint
            else:
                time = 0
                while not satisfy: 
                    loss = _create_loss_individual(alpha, betas, cost, budget, lambda_1, lambda_2)
                    manifold = Rotations(n_embedding)
                    problem = Problem(manifold, loss, verbosity=0)
                    solver = SteepestDescent(maxiter=50)
                    sol = solver.solve(problem)

                    I = anp.array([alpha.dot(sol.transpose()).dot(beta.transpose()) for beta in betas])
                    I_binary = _binary_panel(I)
                    satisfy = (cost.dot(I_binary) - budget < 0)
                    
                    time = time + 1
                    if time > 5:
                        break

                argm = np.argmax(alpha.dot(sol.transpose()).dot(betas.transpose()))
                treatment_realign.append(A_unique[argm])
                constraint_compliance.append(satisfy)

            if verbose == 1:

                print("The subject {0} constraint compliance: {1}".format(idx, satisfy))

            
        return np.array(treatment_realign), np.array(constraint_compliance)

    elif budget_level == "population":

        n_samples = alphas.shape[0]

        subject_rotations = [np.eye(n_embedding)] * n_samples

        alphas_rotate = np.zeros((n_samples, n_embedding))

        for i in range(n_samples):

            alphas_rotate[i, :] = alphas[i, :].dot(subject_rotations[i])
        
        iterations = 20

        loss_fn_cur = np.inf

        for iter in range(iterations):

            loss_fn_prev = loss_fn_cur

            rng = np.random.default_rng()
            samples = np.arange(n_samples)
            rng.shuffle(samples)

            for i in samples:

                alpha = alphas[i]
                alphas_r = alphas_rotate[np.delete(np.arange(n_samples), i), :]

                loss = _create_loss_population(alpha, alphas_r, betas, cost, budgets, lambda_1, lambda_2 * 2 ** iter)
                manifold = Rotations(n_embedding)
                problem = Problem(manifold, loss, verbosity=0)
                solver = SteepestDescent(maxiter=10, logverbosity=2)
                sol, log = solver.solve(problem)

                loss_fn_cur = log["iterations"]["f(x)"][-1]
                
                subject_rotations[i] = sol
                alphas_rotate[i, :] = alphas[i].dot(sol.transpose())

            if verbose == 1:
                print("-- iteration: {0}, loss: {1} --".format(iter, loss_fn_cur))

            if np.abs(loss_fn_cur - loss_fn_prev) / np.abs(loss_fn_cur) < 1e-4:
                break

            if iter == iterations - 1:
                
                if verbose == 1:
                    print("Algorithm reaches maximum itreations, does not converge.")

            
        trt_panel = alphas_rotate.dot(betas.transpose())

        argm = np.argmax(trt_panel, axis=1)

        treatment_realign = A_unique[argm]

        constraint_compliance = (np.sum(cost[argm]) < budgets)

        return treatment_realign, constraint_compliance

# Masking approach for constrained problem


class ITRDataset2(Dataset):
    
    def __init__(self, X, betas, trt_panel, cost, budget):

        self.X, self.betas, self.trt_panel, self.cost, self.budget = X, betas, trt_panel, cost, budget

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.betas, self.trt_panel[idx], self.cost, self.budget 

def realign_mask(self, X, A, cost, budget, layer=2, width=10, epochs=5000, learning_rate=1e-2, lambd=10, verbose=0):
        
        """
        solve the budget constraint problem with masking

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

        n_samples, n_features = X_tsr.size()

        A_unique = np.unique(A, axis=0)
        A_tsr = torch.from_numpy(A_unique).float()
        if self.device == "gpu":
            A_tsr = A_tsr.to(self.device)

        if self.device == "gpu":

            alphas = self.model.covariate_embed(X_tsr).detach() # covariate embedding
            betas = self.model.treatment_embed(A_tsr).detach() # treatment embedding

        else:

            alphas = self.model.covariate_embed(X_tsr).detach() # covariate embedding
            betas = self.model.treatment_embed(A_tsr).detach() # treatment embedding

        trt_panel = torch.tensordot(alphas, torch.transpose(betas, 0, 1), dims=1)

        cost_tsr = torch.from_numpy(cost)
        budget_tsr = torch.Tensor([budget])

        dataset = ITRDataset2(X_tsr, betas, trt_panel, cost_tsr, budget_tsr)

        loader = DataLoader(dataset, batch_size=50, shuffle=True)

        model = ConstrEncoderNet(input_size=n_features, layer=layer, width=width, width_embed=self.width_embed)

        model.train(loader, n_samples=n_samples, epochs=epochs, learning_rate=learning_rate, lambd=lambd, verbose=verbose)

        mask = model.getmask(X_tsr, betas)

        argm = np.argmax(mask, axis=1)

        D = A_unique[argm]

        return D


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

    def train(self, data_loader, n_samples, epochs=1000, learning_rate=1e-3, lambd=10, verbose=0):

        # define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        optimizer.zero_grad()
        scheduler = ExponentialLR(optimizer, gamma=0.999)

        for epoch in range(epochs):

            for batch in data_loader:

                X, betas, trt_panel, cost, budget = batch

                betas = betas[0]
                cost = cost[0]
                budget = budget[0]

                budget = budget / (n_samples / 50)

                # generate predict
                output = self(X, betas)

                # calculate loss
                loss = self.loss_fn(output, trt_panel, cost, budget, lambd)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            results = self._evaluate(data_loader, n_samples, lambd)

            if verbose == 1:
                print("- Epoch: {0}, loss: {1}".format(epoch, torch.mean(results)))

    def _evaluate(self, data_loader, n_samples, lambd):

        outputs = []

        for batch in data_loader:

            X, betas, trt_panel, cost, budget = batch

            betas = betas[0]
            cost = cost[0]
            budget = budget[0]

            budget = budget / (n_samples / 50)

            # generate predict
            output = self(X, betas)

            # if epoch % 10 == 0:
            #     e_idx = e_idx + 1

            # calculate loss
            loss = self.loss_fn(output, trt_panel, cost, budget, lambd)

            outputs.append(loss)

            return torch.Tensor(outputs)


    def getmask(self, X, betas):

        output = self(X, betas)
        mask = torch.zeros(output.size())
        argm = torch.argmax(output, dim=1)
        mask[torch.arange(output.size()[0]), argm] = 1

        return mask.detach().numpy()


