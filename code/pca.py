import torch

from utils.exceptions import ModelNotFit

class PCA():
    
    def __init__(self) -> None:
        ''' 
        Initializer of the PCA class.
        '''
        self.U = None
        self.S = None
        self.V = None
    
    def get_vectors(self)-> torch.Tensor:
        self._check_fit()
        return self.V

    def get_n_components(self, desired_variance:float=0.90)->int:
        '''
        Return the number of components needed to get a specific cumulative variance.
        ## Inputs:
        - desired_variance: Percentage of the cumulative variance that we want to have in the PCA
        
        ## Outputs:
        - n_components: Number of components that have to be used in order to obtain the selected cumulative variance 
        '''
        cum_var = self.calculate_cumulative_variance()
        n_components = (torch.sum( cum_var < desired_variance) + 1).item()
        return n_components
    
    def perform_svd(self, dataset:torch.Tensor)->None:
        ''' 
        Perform singular value decomposition on the desired dataset. This will store the U,S,V values into the class object.
        '''
        self.U, self.S, self.V = torch.linalg.svd(dataset)
    
    def _check_fit(self)->None:
        if self.V == None or self.S == None:
            raise ModelNotFit('The PCA class is empty. Please perform the singular value decomposition first using perform_svd()')
    
    def calculate_cumulative_variance(self):
        ''' 
        Calculate the cumulative variance of the eigenvectors
        '''
        self._check_fit()
        # Calculate the variance explained by each principal component
        var = (self.S ** 2) / torch.sum(self.S ** 2)

        # Calculate the cumulative variance explained
        cum_var = torch.cumsum(var, dim=0)
        
        return cum_var
    
    def apply(self, data:torch.Tensor, n_components:int)->torch.Tensor:
        self._check_fit()
        return torch.matmul(data, self.V[:, :n_components])
    