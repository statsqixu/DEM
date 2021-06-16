from random import sample
import matplotlib.pyplot as plt
import numpy as np

# some utility functions
## plot training history  
def plot_train_history(history):
    
    plt.plot(history, "-x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")


def _low_dim_cov(sample_size):

    return np.random.uniform(low=-1, high=1, size=(sample_size, 5)) # low-dim

def _high_dim_cov(sample_size):

    return np.random.uniform(low=-1, high=1, size=(sample_size, 50)) # high-dim

def _2_channel_trt(sample_size):

    return np.random.choice([0, 1], size=(sample_size, 2))

def _3_channel_trt(sample_size):

    return np.random.choice([0, 1], size=(sample_size, 3))

def _2_channel_trt_embed(A):

    assert A.shape[1] == 2

    sample_size = A.shape[0]
    beta = np.zeros((sample_size, 3))

    beta[np.all(A == np.array([0, 0]), axis=1), :] = np.array([1, -1, 2])
    beta[np.all(A == np.array([0, 1]), axis=1), :] = np.array([-2, 2, 1])
    beta[np.all(A == np.array([1, 0]), axis=1), :] = np.array([0, 1, 1])
    beta[np.all(A == np.array([1, 1]), axis=1), :] = np.array([1, -2, -4])

    return beta

def _3_channel_trt_embed(A):

    assert A.shape[1] == 3

    sample_size = A.shape[0]
    beta = np.zeros((sample_size, 4))

    beta[np.all(A == np.array([0, 0, 0]), axis=1), :] = np.array([1, 0, -1, 2])
    beta[np.all(A == np.array([0, 0, 1]), axis=1), :] = np.array([0, -1, 1, -1])
    beta[np.all(A == np.array([0, 1, 0]), axis=1), :] = np.array([-2, 1, -2, 1])
    beta[np.all(A == np.array([0, 1, 1]), axis=1), :] = np.array([1, 1, -2, -1])
    beta[np.all(A == np.array([1, 0, 0]), axis=1), :] = np.array([0, 2, -1, 1])
    beta[np.all(A == np.array([1, 0, 1]), axis=1), :] = np.array([-1, -3, 3, -1])
    beta[np.all(A == np.array([1, 1, 0]), axis=1), :] = np.array([0, 2, 1, -2])
    beta[np.all(A == np.array([1, 1, 1]), axis=1), :] = np.array([1, -2, 1, 1])

    return beta


def _2_channel_cov_embed(X):

    sample_size = X.shape[0]
    alpha = np.zeros((sample_size, 3))
    alpha[:, 0] = (X[:, 0] ** 2 + X[:, 1] ** 3 - 1)
    alpha[:, 1] = np.sin(np.pi * (X[:, 2] * X[:, 4]))
    alpha[:, 2] = X[:, 4] ** 2 * (X[:, 1] >= 0)

    return alpha

def _3_channel_cov_embed(X):

    sample_size = X.shape[0]
    alpha = np.zeros((sample_size, 4))
    alpha[:, 0] = (X[:, 0] ** 2 + X[:, 1] ** 3 - 1)
    alpha[:, 1] = np.sin(np.pi * (X[:, 2] * X[:, 4]))
    alpha[:, 2] = X[:, 4] ** 2 * (X[:, 1] >= 0)
    alpha[:, 3] = np.exp(X[:, 1] * X[:, 2] - X[:, 3] ** 2) 

    return alpha



def getdata(sample_size, case=1, family="gaussian", seed=None):

    """
    Simulation data generation

    parameters
    ----------
    sample_size: int
        number of subjects to be generated
    
    case: {1, 2, ..., 8}
        simulation case number
        case 1: low-dim covariates, linear, 2-channels
        case 2: low-dim covariates, linear, 3-channels
        case 3: low-dim covariates, nonlinear, 2-channels
        case 4: low-dim covariates, nonlinear, 3-channels
        case 5: high-dim covariates, linear, 2-channels
        case 6: high-dim covariates, linear, 3-channels
        case 7: high-dim covariates, nonlinear, 2-channels
        case 8: high-dim covariates, nonlinear, 3-channels

    family: {"gaussian", "bernoulli"}
        outcome setting, "gaussian": continuous outcome, "bernoulli": binary outcome

    seed: int, default=None
        random generating seed
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # treatment latent embedding
    
    if case in [1, 2, 3, 4]:

        X = _low_dim_cov(sample_size)

    else: 

        X = _high_dim_cov(sample_size)

    if case in [1, 3, 5, 7]:

        A = _2_channel_trt(sample_size)
        beta = _2_channel_trt_embed(A)

    else:

        A = _3_channel_trt(sample_size)
        beta = _3_channel_trt_embed(A)

    if case in [1, 5]:

        alpha = X[:, 0: 3]

    elif case in [2, 6]:

        alpha = X[:, 0: 4]

    elif case in [3, 7]:

        alpha = _2_channel_cov_embed(X)

    else:

        alpha = _3_channel_cov_embed(X)


    Y = 1 + X[:, 0] + X[:, 1] + np.sum(np.multiply(alpha, beta), axis=1)

    if family == "gaussian":

        Y = Y + np.random.normal(size=(sample_size, ))

    elif family == "bernoulli":

        Y = 1 / (1 + np.exp(- Y))

        Y = np.random.binomial(1, Y, size=(sample_size, ))

    A_unique, A_idx = np.unique(A, return_index=True, axis=0)
    beta_unique = beta[A_idx, :]
    trt_panel = alpha.dot(beta_unique.transpose())

    _optA = np.argmax(trt_panel, axis=1)    
    optA = A_unique[_optA, :]
    
    return Y, X, A, optA


def getdata2(sample_size, case=1, family="gaussian", seed=None):

    """
    Simulation data generation (multi-armed setting)

    parameters
    ----------
    sample_size: int
        number of subjects to be generated
    
    case: {1, 2, 3}
        three different scenarios to be considered

    family: {"gaussian", "bernoulli"}
        outcome setting, "gaussian": continuous outcome, "bernoulli": binary outcome

    seed: int, default=None
        random generating seed
    """

    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.uniform(low=-1, high=1, size=(sample_size, 6)) # covariates

    A = np.random.choice([0, 1, 2, 3], size=(sample_size, )) # treatment

    if case == 1:

        delta = (1 + X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3]) * (A == 0) + \
                (1 + X[:, 0] - X[:, 1] - X[:, 2] + X[:, 3]) * (A == 1) + \
                (1 + X[:, 0] - X[:, 1] + X[:, 2] - X[:, 3]) * (A == 2) + \
                (1 - X[:, 0] - X[:, 1] + X[:, 2] + X[:, 3]) * (A == 3)

    elif case == 2:

        delta = (3 * (X[:, 0] <= 0.5) * ((X[:, 1] > -0.6) - 1)) * (A == 0) + \
                ((X[:, 2] <= 1) * (2 * (X[:, 3] <= -0.3) - 1)) * (A == 1) + \
                (4 * (X[:, 4] <= 0) - 2) * (A == 2) + \
                (4 * (X[:, 5] <= 0) - 2) * (A == 3)

    elif case == 3:

        delta = (0.2 + X[:, 0] ** 2 + X[:, 1] ** 2 - X[:, 2] ** 2 - X[:, 3] ** 2) * (A == 0) + \
                (0.2 + X[:, 1] ** 2 + X[:, 2] ** 2 - X[:, 1] ** 2 - X[:, 3] ** 2) * (A == 1) + \
                (0.2 + X[:, 0] ** 2 + X[:, 3] ** 2 - X[:, 1] ** 2 - X[:, 2] ** 2) * (A == 2) + \
                (0.2 + X[:, 1] ** 2 + X[:, 2] ** 2 - X[:, 0] ** 2 - X[:, 3] ** 2) * (A == 3)

    mu = 1 + X[:, 0] + X[:, 1]

    if family == "gaussian":

        Y = mu + delta + np.random.normal(size=(sample_size, ))

    elif family == "bernoulli":

        p = 1 / (1 + np.exp(- mu - delta))
        Y = np.random.binomial(1, p, size=(sample_size, ))

    A_ = np.zeros((sample_size, len(np.unique(A))))
    A_[np.arange(sample_size), A] = 1
    A = A_

    return Y, X, A