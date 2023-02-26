"""
ITR Dataset
"""

# Author: Qi Xu <qxu6@uci.edu>

from torch.utils.data import Dataset


# Define the dataset
class ITRDataset(Dataset):
    
    def __init__(self, X, A, Y, W):
        
        self.covariate, self.treatment, self.outcome, self.weight = X, A, Y, W
        
    def __len__(self):
        return len(self.outcome)
    
    def __getitem__(self, idx):
        return self.covariate[idx], self.treatment[idx], self.outcome[idx], self.weight[idx]


