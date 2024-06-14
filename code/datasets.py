from torch.utils.data import Dataset
import torch

class TrajectoryDataset(Dataset):
    def __init__(self, data:torch.Tensor, labels:torch.Tensor):
        '''
            Dataset for the real train and test data.
            ## Inputs:
            - data: tensor with the trajectories of shape (N, L). N being the number of trajectories and L the length of each trajectory
            - labels: tensor of shape (N). N being the number of trajectories. It can be either one or zero, depending on if its is positive or negative.
        '''
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SyntheticDataset(Dataset):
    def __init__(self, data:torch.Tensor):
        '''
            Dataset for the synthetic data obtained from a generator model.
            ## Inputs:
            - data: tensor with the synthetic trajectories of shape (N, L). N being the number of trajectories and L the length of each trajectory
        '''
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]