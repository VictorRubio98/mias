import torch

from mia.pca import PCA
from mia.utils.exceptions import ModelNotFit

def test_error_get_vectors():
    pca = PCA()
    check=0
    try:
        pca.get_vectors()
    except ModelNotFit:
        check=1
    
    assert check == 1
    
def test_error_cum_var():
    pca = PCA()
    check=0
    try:
        pca.calculate_cumulative_variance()
    except ModelNotFit:
        check=1
    assert check == 1
    
def test_error_apply():
    pca = PCA()
    try:
        data = torch.Tensor([[1,2],[3,4]])  
        print(type(data), flush=True)     
        pca.apply(data, 2)
    except ModelNotFit:
        check=1
    assert check == 1
    
    
def test_all():
    pca = PCA()    
    dataset = torch.Tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    pca.perform_svd(dataset=dataset)
    n_comp = pca.get_n_components(desired_variance=0.99)
    X=torch.Tensor([[1,2,3,4], [1,2,3,4]])
    x_transformed = pca.apply(X, n_comp)
    assert x_transformed.shape[1] == n_comp