import torch
from torch.utils.data import Dataset

from utils.exceptions import ModelNotFit, NotAvailableFeature

class PyNN():
    def __init__(self, distance:str = 'euc') -> None:
        ''' 
        Distances can be:
            - euc: Euclidean distance
            - sqd: Squared euclidean distance
            - max: Tchebychev distance (maximum metric)
            - cab: Taxicab distance or Manhattan distance
        '''
        self.constellation = None
        self.dim = None
        self.available_dist = ['euc', 'sqd', 'max', 'cab']
        self.ensure_distance(distance)
        self.distance = distance
    
    def fit(self, constellation:torch.Tensor)-> None:
        self.constellation = constellation
        self.dim = len(self.constellation.shape)-1
    
    def set_distance(self, distance: str)-> None:
        self.ensure_distance(distance)
        self.distance = distance
    
    def ensure_distance(self, distance):
        if distance not in self.available_dist:
            raise NotAvailableFeature(f'Feature not available {distance}, currently only available distances: {self.available_dist}')
    
    def compute_distance(self, X:torch.Tensor)-> torch.Tensor:
        '''
        Computing distances from X to each point in the constellation depending on the self.distance value the metric will differ:
            - euc: Euclidean distance between points
            - sqd: Squared euclidean distance
            - max: Tchebychev distance (maximum metric)
            - cab: Taxicab distance or Manhattan distance
        ## Inputs:
            - X: A tensor with a single trajectory.
        ## Outputs:
            - dist: A tensor with the distance between X and all the constellation.
        '''
        if self.distance == self.available_dist[0]:
            dist = self.constellation.add( - X).pow(2).sum(dim=self.dim).pow(.5) 
        elif self.distance == self.available_dist[1]:
            dist = self.constellation.add( - X).pow(2).sum(dim=self.dim)
        elif self.distance == self.available_dist[2]:
            dist = torch.abs(self.constellation.add( - X)).max(dim=self.dim).values
        elif self.distance == self.available_dist[3]:
            dist = torch.abs(self.constellation.add( - X)).sum(dim=self.dim)
        return dist
    
    def kneighbours(self, X:torch.Tensor, k:int=4)-> tuple:
        '''
        ## Inputs:
        - X: A single trajectory that we want to check if it is similar to the original data.
        - k: the number of k nearest neighbours we want to see in the output
        ## Outputs:
        - dist: Distances to the first k-nearest neighbours
        - knn_indices: Indexes of the k-nearest neighbours
        '''
        if self.dim != None and self.dim>0:
            X = X.expand(self.constellation.shape) # Repeat the same two point traj N times until shape matches with the synthetic trajectories

            # Computing distance
            dist = self.compute_distance(X)
            # Getting the k nearest points
            knn_indices = dist.topk(k, largest=False, sorted=False)[1] # Getting the k smallest distances

            return dist[knn_indices], knn_indices
        else:
            raise ModelNotFit('The model has not been fit with the constellation or the constellation is empty. Before using model.kneighbours use model.fit with a propper constellation')
        
    def predict(self,test:Dataset, k:int = 2):
        distances = torch.LongTensor([])
        idx = torch.IntTensor([])
        labels = torch.IntTensor([])
        for i, (data, label) in enumerate(test):
            dist, knn_indices = self.kneighbours(data, k=k)
            dist=dist.unsqueeze(0)
            knn_indices = knn_indices.unsqueeze(0)
            label = label.unsqueeze(0)
            
            distances = torch.cat([distances, dist], dim=0)
            idx = torch.cat([idx, knn_indices], dim=0)
            labels = torch.cat([labels, label])
             
        return distances, idx, labels