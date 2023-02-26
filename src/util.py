import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

def _categorical_coding(A):

    """
    Create categorical representation of combination treatment from compact coding
    """

    _, A_cate = np.unique(A, return_inverse=True, axis=0)

    return A_cate

def _compact_coding2(A):

    """
    Create compact coding with {-1, 1} from compact coding with {0, 1}
    """

    A_compact2 = 2 * A - 1

    return A_compact2

def check_treatment_coding(A):

    if A.ndim == 2:

        if np.min(A) == 0:

            return "compact1"
        
        elif np.min(A) == -1:

            return "compact2"
        
    else:

        return "categorical"


def check_covariate(X):

    """
    Check whether covariate is 2d array,
    if not, add an extra axis

    X: covariates: 1d/2d
    """
     
    if X.ndim == 1:

        X = X[:, np.newaxis]

    return X

def return_device(device):

    if device == "cpu":

        return "cpu"

    elif device == "gpu":

        return "cuda:0"

    elif device == "default":

        if torch.cuda.is_available():
            return "cuda:0"

        else:
            return "cpu"
    
def _ps_model(X, input_dim, output_dim):

    """
    Propensity score model for simulation settings shown in paper "Optimal Individualized Treatment Rule for Combination Treatment"
    """


    np.random.seed(0)

    beta = np.random.normal(size=(input_dim, ))
    alpha = 0.2 * np.arange(1, output_dim + 1)[:, np.newaxis]

    prob = X.dot(beta)[:, np.newaxis].dot(alpha.T)
    prob = np.exp(prob) / np.sum(np.exp(prob), axis=1)[:, np.newaxis]

    return prob

def _generate_treatment1(sample_size, mode="randomized", seed=None, X=None): 

    """
    Generate treatment for case 1 and 2
    """

    if seed is not None:
        np.random.seed(seed)

    A_candidates = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0],
                             [0, 1, 1], [1, 0, 0], [1, 1, 1]])

    if mode == "randomized":

        A = A_candidates[np.random.randint(low = 0, high = 6, size = (sample_size, )), :]

    elif mode == "ps":

        try:

            prob = _ps_model(X, 10, 6)

            A = np.zeros((sample_size, 3))

            for i in range(sample_size):

                A[i, :] = A_candidates[np.random.choice(np.arange(6), p = prob[i, :]), :]

        except:

            raise ValueError("ps mode requires X to be provided")

    return A


def _generate_treatment2(sample_size, mode="randomized", seed=None, X=None):

    """
    Generate treatment for case 3 and 4
    """

    if seed is not None:
        np.random.seed(seed)

    A_candidate = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 1], [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, 1],
                            [0, 1, 1, 0, 0], [0, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 1],
                            [1, 0, 0, 1, 0], [1, 0, 1, 0, 0], [1, 0, 1, 0, 1], [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0], [1, 1, 0, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])

    if mode == "randomized":

        A = A_candidate[np.random.randint(low = 0, high = 20, size = (sample_size, )), :]

    elif mode == "ps":

        try:

            prob = _ps_model(X, 10, 20)

            A = np.zeros((sample_size, 5))

            for i in range(sample_size):

                A[i, :] = A_candidate[np.random.choice(np.arange(20), p = prob[i, :]), :]

        except:

            raise ValueError("ps mode requires X to be provided")

    return A, A_candidate


def generate_data(sample_size, case = 1, 
                  mode = "randomized", outcome_shift = None, 
                  treatment_coding = "compact1", seed = None):

    """
    Simulation data generation

    Paper "Optimal Individualized Treatment Rule for Combination Treatment"

    parameters
    ----------
    sample_size: int
        number of subjects to be generated
    
    case: {1, 2, 3, 4}
        simulation case number
        case 1: 3-channels, 6 candidates, no interaction
        case 2: 3-channels, 6 candidates, interaction effects
        case 3: 5-channels, 20 candiates, no interaction effects
        case 4: 5-channels, 20 candidates, interaction effects
    
    mode: {"randomized", "ps"}
        treatment assignment mode
        random: randomized assignment
        ps: propensity score-based assignment, the propensity score model is based on function `ps_model`

    outcome_shift: float, default=0
        shift the outcome sucht that the minimum of the outcome is the specified constant

    treatment_coding: {"compact1", "compact2", "categorical"}, default="compact1"
        treatment coding method
        compact1: compact coding with {0, 1}
        compact2: compact coding with {-1, 1}
        categorical: categorical coding

    seed: int, default=None
        random generating seed
    """

    if seed is not None:
        np.random.seed(seed)

    # generate 10 dim covariates from uniform distribution on [-1, 1]

    X = 2 * np.random.rand(sample_size, 10) - 1

    # generate treatment-free effects

    m = 3 + X[:, 0] + 2 * X[:, 1]

    if case == 1: 

        # generate treatment effects for case 1

        A = _generate_treatment1(sample_size, mode, seed, X)

        A_cate = _categorical_coding(A)

        trt_panel = np.zeros((sample_size, 6))

        trt_panel[:, 0] = 0
        
        trt_panel[:, 1] = 2 * X[:, 0] + np.exp(X[:, 2] + X[:, 3]) 

        trt_panel[:, 2] = X[:, 1] * np.log(X[:, 4] ** 2) + X[:, 6]

        trt_panel[:, 3] = 2 * X[:, 0] + np.exp(X[:, 2] + X[:, 3]) + X[:, 1] * np.log(X[:, 4] ** 2) + X[:, 6]

        trt_panel[:, 4] = np.sin(X[:, 2]) + np.log(X[:, 3] ** 2) + np.log(X[:, 6] ** 2)

        trt_panel[:, 5] = 2 * X[:, 0] + np.exp(X[:, 2] + X[:, 3]) + X[:, 1] * np.log(X[:, 4] ** 2) + X[:, 6] + np.sin(X[:, 2]) + np.log(X[:, 3] ** 2) + np.log(X[:, 6] ** 2)

        trt_panel = trt_panel - np.mean(trt_panel, axis=1)[:, np.newaxis] # normalize the treatment effects

        # observed treatment effects

        trt = trt_panel[np.arange(sample_size), A_cate]

    elif case == 2:

        # generate treatment effects for case 2

        A = _generate_treatment1(sample_size, mode, seed, X)

        A_cate = _categorical_coding(A)

        trt_panel = np.zeros((sample_size, 6))

        trt_panel[:, 0] = 0
        
        trt_panel[:, 1] = 2 * X[:, 0] + np.exp(X[:, 2] + X[:, 3]) 

        trt_panel[:, 2] = X[:, 1] * np.log(X[:, 4] ** 2) + X[:, 6]

        trt_panel[:, 3] = 2 * X[:, 0] + np.exp(X[:, 2] + X[:, 3]) + X[:, 1] * np.log(X[:, 4] ** 2) + X[:, 6] + np.sin(5 * X[:, 0] ** 2) - 3 * (X[:, 1] - 0.5) ** 2

        trt_panel[:, 4] = np.sin(X[:, 2]) + np.log(X[:, 3] ** 2) + np.log(X[:, 6] ** 2)

        trt_panel[:, 5] = 2 * X[:, 0] + np.exp(X[:, 2] + X[:, 3]) + X[:, 1] * np.log(X[:, 4] ** 2) + X[:, 6] + np.sin(X[:, 2]) + np.log(X[:, 3] ** 2) + np.log(X[:, 6] ** 2) + 2 * np.sin((X[:, 1] - X[:, 3]) ** 2)

        trt_panel = trt_panel - np.mean(trt_panel, axis=1)[:, np.newaxis] # normalize the treatment effects

        # observed treatment effects

        trt = trt_panel[np.arange(sample_size), A_cate]

    elif case == 3:

        # generate treatment effects for case 3

        A, A_candidate = _generate_treatment2(sample_size, mode, seed, X)

        A_cate = _categorical_coding(A)

        trt_panel = np.zeros((sample_size, 20))

        trt_candidate = np.c_[(X[:, 0] - 0.25) ** 3, 
                               np.log(X[:, 2] ** 2) + 2 * np.log(X[:, 7] ** 2) * np.cos(2 * np.pi * X[:, 9]),
                               X[:, 1] * np.sin(X[:, 3]) - 1, 
                               (X[:, 0] + X[:, 4] - X[:, 7] ** 2) ** 3,
                               np.exp(X[:, 1] - X[:, 4])]

        trt_panel = trt_candidate.dot(A_candidate.transpose())

        trt_panel = trt_panel - np.mean(trt_panel, axis=1)[:, np.newaxis] # normalize the treatment effects

        # observed treatment effects

        trt = trt_panel[np.arange(sample_size), A_cate]

    elif case == 4:

        # generate treatment effects for case 4

        A, A_candidate = _generate_treatment2(sample_size, mode, seed, X)

        A_cate = _categorical_coding(A)

        trt_panel = np.zeros((sample_size, 20))

        trt_candidate = np.c_[(X[:, 0] - 0.25) ** 3, 
                               np.log(X[:, 2] ** 2) + 2 * np.log(X[:, 7] ** 2) * np.cos(2 * np.pi * X[:, 9]),
                               X[:, 1] * np.sin(X[:, 3]) - 1, 
                               (X[:, 0] + X[:, 4] - X[:, 7] ** 2) ** 3,
                               np.exp(X[:, 1] - X[:, 4])]

        trt_panel = trt_candidate.dot(A_candidate.transpose())

        trt_panel[:, 4] += np.exp(2 * X[:, 1])
        trt_panel[:, 14] -= 3/2 * np.cos(2 * np.pi * X[:, 0] + X[:, 7] ** 2)
        trt_panel[:, 6] += np.exp(2 * X[:, 3] + X[:, 8])
        trt_panel[:, 7] += -2 * np.log(X[:, 5] ** 2)
        trt_panel[:, 16] += -2 * np.log(X[:, 5] ** 2)
        trt_panel[:, 17] += X[:, 5] ** 2 + 1/2 * np.sin(2 * np.pi / X[:, 6])

        trt_panel = trt_panel - np.mean(trt_panel, axis=1)[:, np.newaxis]

        # observed treatment effects

        trt = trt_panel[np.arange(sample_size), A_cate]

    # generate the optimal treatment 

    A_unique = np.unique(A, axis=0)
    _optA = np.argmax(trt_panel, axis=1)
    optA = A_unique[_optA, :]

    # generate outcome

    Y = m + trt + np.random.normal(0, 1, sample_size)

    if outcome_shift is not None:

        Y_output = Y - np.min(Y) + outcome_shift

    else:

        Y_output = Y

    # generate the treatment coding

    if treatment_coding == "categorical":

        A_output = A_cate

    elif treatment_coding == "compact1":

        A_output = A

    elif treatment_coding == "compact2":

        A_output = _compact_coding2(A)

    # generate the optimal treatment coding

    if treatment_coding == "categorical":

        optA_output = _categorical_coding(optA)

    elif treatment_coding == "compact1":

        optA_output = optA

    elif treatment_coding == "compact2":

        optA_output = _compact_coding2(optA)

    return X, A_output, Y_output, optA_output


def verify_dist_optimal_treatment(optA):

    """Verify the distribution of the optimal treatment"""

    optA_unique, optA_count = np.unique(optA, axis=0, return_counts=True)

    optA_dist = optA_count / optA.shape[0]

    return optA_dist

def estimate_ips(X, A, model="multinomial", stabilize=True):

    """
    Estimate the propensity score

    Parameters
    ----------

    X: array-like, shape (n_samples, n_features)
        The covariates

    A: array-like, shape (n_samples, )
        The treatment assignment

    model: string, {"multinomial", "nn"} default="multinomial"
        The model to estimate the propensity score
    
    stabilize: bool, default=True
        Whether to stabilize the propensity score
    """

    trt_coding = check_treatment_coding(A)

    if trt_coding == "compact1":

        A = _categorical_coding(A)

    elif  trt_coding == "compact2":

        A = _categorical_coding((A + 1) / 2)

    n = A.shape[0]

    if model == "multinomial":

        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=123)

        clf.fit(X, A)

        ps = clf.predict_proba(X)

    elif model == "nn":

        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=123)

        clf.fit(X, A)

        ps = clf.predict_proba(X)

    if stabilize:

        ps = ps[np.arange(n), A]

        _, inverse, count = np.unique(A, return_inverse=True, return_counts=True)

        freq = count / n

        freq = freq[inverse]
        
        ps = ps / freq[inverse]

    return 1 / ps


def estimate_treatment_free(X, Y, model="linear"):

    """
    Estimate the treatment-free effects

    Parameters
    ----------

    X: array-like, shape (n_samples, n_features)
        The covariates
    
    Y: array-like, shape (n_samples, )

    model: string, {"linear", "nn"} default="linear"
        The model to estimate the treatment-free effects
    """

    if model == "linear":

        clf = LinearRegression()

        clf.fit(X, Y)

        Y_pred = clf.predict(X)

    elif model == "nn":

        clf = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=123)

        clf.fit(X, Y)

        Y_pred = clf.predict(X)

    return Y_pred

    