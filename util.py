import matplotlib.pyplot as plt
import numpy as np

# some utility functions
## plot training history  
def plot_train_history(history):
    
    plt.plot(history, "-x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

def getdata(sample_size, case=1, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    

    X = np.random.uniform(low=-1, high=1, size=(int(sample_size), 2))
    A = np.random.choice([0.0, 1.0], size=(int(sample_size), 2))
    
    alpha_ = np.zeros((sample_size, 3))
    
    mu = 10 + 5 * X[:, 0] - 2 * X[:, 1]
    
    alpha_ = np.zeros((sample_size, 2))

    for i in range(sample_size):

        if (A[i, :] == np.array([0, 0])).all():

            alpha_[i, :] = np.array([0, 0])

        elif (A[i, :] == np.array([1, 0])).all():

            alpha_[i, :] = np.array([np.sqrt(3)/2, -1/2])
        
        elif (A[i, :] == np.array([0, 1])).all():

            alpha_[i, :] = np.array([-np.sqrt(3)/2, -1/2])

        elif (A[i, :] == np.array([1, 1])).all():

            alpha_[i, :] = np.array([0, 1])
            
    alpha_uq = np.array([[0, 0], [np.sqrt(3)/2, -1/2], [-np.sqrt(3)/2, -1/2], [0, 1]])
            
    beta_0 = np.zeros((sample_size, 2))

    beta_0[:, 0] = X[:, 0] ** 2
    beta_0[:, 1] = X[:, 1] ** 2
    
    beta_1 = np.zeros((sample_size, 2))

    beta_1[:, 0] = X[:, 0] ** 2 + X[:, 1] ** 2 - 1
    beta_1[:, 1] = np.sin(- (X[:, 0] + X[:, 1]) * np.pi)
    
    
    if case == 1:
        Y = mu + np.sum(np.multiply(alpha_, X), axis=1)
        trt_panel = X.dot(alpha_uq.transpose())
    
    elif case == 2:
        Y = mu + np.sum(np.multiply(alpha_, beta_0), axis=1)
        trt_panel = beta_0.dot(alpha_uq.transpose())
    
    elif case == 3:
        Y = mu + np.sum(np.multiply(alpha_, beta_1), axis=1)
        trt_panel = beta_1.dot(alpha_uq.transpose())
        
            
    Y = Y + np.random.normal(size=(sample_size, )) 
        
    optA_ = np.argmax(trt_panel, axis=1)
    
    optA = np.zeros((sample_size, 2))
    
    optA[optA_ == 0] = np.array([0, 0])
    optA[optA_ == 1] = np.array([1, 0])
    optA[optA_ == 2] = np.array([0, 1])
    optA[optA_ == 3] = np.array([1, 1])
    

    return Y, X, A, optA

