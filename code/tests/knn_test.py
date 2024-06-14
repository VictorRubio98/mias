import torch
import math 


from mia.knn import PyNN

def test_euc():
    euc = PyNN()
    const = torch.tensor([[1,2],[3,4]])
    X = torch.tensor([0,0])
    euc.fit(const)
    dist, idx = euc.kneighbours(X, 1)
    assert dist == math.sqrt(5) and idx == 0

def test_sqd():
    sqd = PyNN('sqd')
    const = torch.tensor([[1,2],[3,4]])
    X = torch.tensor([0,0])
    sqd.fit(const)
    dist, idx = sqd.kneighbours(X, 1)
    assert dist == 5 and idx == 0    

def test_max():
    max = PyNN('max')
    const = torch.tensor([[1,2],[3,4]])
    X = torch.tensor([0,0])
    max.fit(const)
    dist, idx = max.kneighbours(X, 1)
    assert dist == 2 and idx == 0    
    

def test_cab():
    cab = PyNN('cab')
    const = torch.tensor([[1,2],[3,4]])
    X = torch.tensor([0,0])
    cab.fit(const)
    dist, idx = cab.kneighbours(X, 1)
    assert dist == 3 and idx == 0    