import os
import argparse

import torch
import sklearn.metrics as metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import random_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from datetime import datetime

from datasets import SyntheticDataset, TrajectoryDataset
from knn import PyNN
from pca import PCA
from evaluations import calculateAdv

# Parser to define variables in the terminal
parser = argparse.ArgumentParser()
parser.add_argument('-v','--var', default=0.99, type=float, help = 'Desired cummulative variance of the eigenvectors')
parser.add_argument('-d','--dist', default='euc',choices=['euc', 'max', 'cab', 'sqd'], type=str, help = 'Distance used in the KNN')
parser.add_argument('-k',default=2, type=int, help='Number of neighbors')
parser.add_argument('-e', '--epsilon', default='baseline', type=str, help='Epsilon (privacy budget) of the desired dataset. If baseline means no differential privacy guarantees.')
parser.add_argument('--dataset', default='geolife', help='Name of the tataset to be used in the attack.')
parser.add_argument('-a', '--attack-model', dest='attack_model', default='none', choices=['none', 'rbfSVM', 'linearSVM', 'rf'], type=str, 
                    help='Attck model to use: - none means using the knn without any other model - rbfSVM: SVM with gaussian Kernel.' + 
                    ' - linearSVM: SVM with linear kernel. - rf: Random Forest')

opt = parser.parse_args()


# General Variables
epsilon = opt.epsilon
data_path = f'./data/{opt.dataset}'
file_path_test = f'{data_path}/test.data'
file_path_validation = f'{data_path}/val.data'
file_path_synthetic = f'{data_path}/{epsilon}/gene.data'
file_path_train = f'{data_path}/real.data'

# PCA Variables
desired_var = opt.var

#KNN Variables
dist_knn = opt.dist #max, euc, cab, sqd
K = opt.k


# Load Data
t = datetime.now()
current_time = t.strftime('%H:%M:%S')
print(f'[{current_time}]: Loading data...')
data_test = pd.read_csv(file_path_test, sep=" ", header=None)
# data_test = torch.unique(torch.Tensor(data_test.values), dim=0)
data_test = torch.Tensor(data_test.values)


data_validation = pd.read_csv(file_path_validation, sep=" ", header=None)
# data_validation = torch.unique(torch.Tensor(data_validation.values), dim=0)
data_validation = torch.Tensor(data_validation.values)

data_negative  = torch.cat([data_test, data_validation])

data_synthetic = pd.read_csv(file_path_synthetic, sep=" ", header=None)
data_synthetic = torch.unique(torch.Tensor(data_synthetic.values), dim=0)

data_train = pd.read_csv(file_path_train, sep=" ", header=None)
data_positive = torch.unique(torch.Tensor(data_train.values), dim=0)

negative = TrajectoryDataset(data_negative, torch.zeros(len(data_negative)))
synthetic = SyntheticDataset(data_synthetic)
positive = TrajectoryDataset(data_positive, torch.ones(len(data_positive)))

if opt.dataset == 'geolife':
    #Has to be specific for each dataset
    synthetic_1, synthetic_2 = random_split(synthetic, [math.floor(0.5*len(synthetic)), math.ceil(0.5*len(synthetic))])
    negative_train, negative_test = random_split(negative, [math.ceil(0.5*len(negative)), math.floor(0.5*len(negative))])
    positive_test, _ = random_split(positive, [math.ceil(0.5*len(negative)), len(positive)-math.ceil(0.5*len(negative))])

    training_dataset =TrajectoryDataset(torch.cat([synthetic_2.dataset[synthetic_2.indices], negative_train.dataset[negative_train.indices][0]]), torch.cat([torch.ones(len(synthetic_2)), torch.zeros(len(negative_train))]))
    #half negative half positive
    testing_dataset = TrajectoryDataset(torch.cat([negative_test.dataset[negative_test.indices][0], positive.data]), torch.cat([torch.ones(len(positive)), torch.zeros(len(negative_test))]))
    all_data = testing_dataset.data
    
# Perform PCA
pca = PCA()
pca.perform_svd(training_dataset.data)

n_comp = pca.get_n_components(desired_var)

t = datetime.now()
current_time = t.strftime('%H:%M:%S')
print(f'[{current_time}]: Number of components for a variance of {desired_var} is: {n_comp}', flush = True)

# Apply pca
training_pca = pca.apply(training_dataset.data, n_comp)
testing_pca = pca.apply(testing_dataset.data, n_comp)
constellation_pca = pca.apply(synthetic_1.dataset[synthetic_1.indices], n_comp)

training_dataset = TrajectoryDataset(training_pca, training_dataset.labels)
testing_dataset = TrajectoryDataset(testing_pca, testing_dataset.labels)

if opt.attack_model == 'none':
    # Perform KNN:
    knn = PyNN(dist_knn)
    knn.fit(constellation_pca)

    train_dist, train_idx, _ = knn.predict(training_dataset, K)
    train_prob  = torch.mean(1- (train_dist-train_dist.min()) / (train_dist.max()-train_dist.min()), dim=1)
    
    fpr, tpr, threshold = metrics.roc_curve(training_dataset.labels, train_prob)
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(threshold, index = i)})
    th = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    ths = th.thresholds.values[0]
    
    test_dist, test_idx, _ = knn.predict(testing_dataset, K)
    test_prob  = torch.mean(1- (test_dist-test_dist.min()) / (test_dist.max()-test_dist.min()), dim=1)
    
    y_pred = torch.Tensor([1 if p >=ths else 0 for p in test_prob])
else:
    #Split synthetic dataset training KNN and testing the whole model    
    #Fit KNN
    knn = PyNN(dist_knn)
    knn.fit(constellation_pca)
    
    # Predict knn for synthetic data and real data (We know this is a 0)
    train_dist, train_idx, train_labels = knn.predict(training_dataset, K)
    if 'SVM' in opt.attack_model:
        kernel = opt.attack_model.split('SVM')[0]
        model = svm.SVC(kernel=kernel)
    if 'rf' in opt.attack_model:
        model = RandomForestClassifier(n_estimators=100,criterion='entropy')
    #Train model

    model.fit(train_dist.tolist(), train_labels.tolist())
    
    #Test model
    test_dist, test_idx, test_labels = knn.predict(testing_dataset, K)
    y_pred = model.predict(test_dist.tolist())
    y_pred = torch.Tensor(y_pred)
    
y_test = testing_dataset.labels
    
# Plot resuts
results = torch.cat([all_data, torch.cat([y_pred.unsqueeze(-1), y_test.unsqueeze(-1)], dim=-1)], dim=-1)

y_pred= y_pred.tolist()
y_test = y_test.tolist()

aff = calculateAdv(results)
images_pth = os.listdir(f'{data_path}/{epsilon}/predictions/')
file_path = f'{aff:.4f}_{opt.attack_model}_k{K}_comp{n_comp}_{dist_knn}'
i = 0
length = len(file_path) #The length of characters that the image shall have without the {i}.png
images_pth = [pth[0:length] for pth in images_pth]
for pth in images_pth:
    if pth == file_path:
        i += 1

t = datetime.now()
current_time = t.strftime('%H:%M:%S')
print(f'[{current_time}]: Saving file as: {data_path}/{epsilon}/predictions/{file_path}_{i}.pt')
torch.save(results, f'{data_path}/{epsilon}/predictions/{file_path}_{i}.pt')
