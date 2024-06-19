import os
import argparse

import torch
import sklearn.metrics as metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import SyntheticDataset, TrajectoryDataset
from knn import PyNN
from pca import PCA


# Parser to define variables in the terminal
parser = argparse.ArgumentParser()
parser.add_argument('-v','--var', default=0.99, type=float, help = 'Desired cummulative variance of the eigenvectors')
parser.add_argument('-d','--dist', default='euc',choices=['euc', 'max', 'cab', 'sqd'], type=str, help = 'Distance used in the KNN')
parser.add_argument('-k',default=2, type=int, help='Number of neighbors')
parser.add_argument('-e', '--epsilon', default='baseline', type=str, help='Epsilon (privacy budget) of the desired dataset. If baseline means no differential privacy guarantees.')
parser.add_argument('--dataset', default='geolife', help='Name of the tataset to be used in the attack.')
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

print(f'Loading data...')
data_test = pd.read_csv(file_path_test, sep=" ", header=None)
data_test = torch.unique(torch.Tensor(data_test.values), dim=0)

data_validation = pd.read_csv(file_path_validation, sep=" ", header=None)
data_validation = torch.unique(torch.Tensor(data_validation.values), dim=0)

data_negative  = torch.cat([data_test, data_validation])

data_synthetic = pd.read_csv(file_path_synthetic, sep=" ", header=None)
data_synthetic = torch.unique(torch.Tensor(data_synthetic.values), dim=0)

data_train = pd.read_csv(file_path_train, sep=" ", header=None)
data_positive = torch.unique(torch.Tensor(data_train.values), dim=0)

# Perform PCA
pca = PCA()
pca.perform_svd(data_synthetic)

n_comp = pca.get_n_components(desired_var)

print(f'Number of components for a variance of {desired_var} is: {n_comp}', flush = True)

# Apply pca
negative_pca = pca.apply(data_negative, n_comp)
synthetic_pca = pca.apply(data_synthetic, n_comp)
positive_pca = pca.apply(data_positive, n_comp)

# Create the positive, negative and synthetic datasets
negative = TrajectoryDataset(negative_pca, torch.zeros(len(negative_pca)))
synthetic = SyntheticDataset(synthetic_pca)
positive = TrajectoryDataset(positive_pca, torch.ones(len(positive_pca)))

# Perform KNN:

knn = PyNN(dist_knn)
knn.fit(synthetic.data)

pos_prob, pos_idx = knn.predict(positive, K)
neg_prob, neg_idx = knn.predict(negative, K)

# Plot resuts

preds = torch.cat([pos_prob, neg_prob])
y_test = torch.cat([positive.labels, negative.labels]).type(torch.uint8)

all_data = torch.cat([data_positive, data_negative], dim=0)
results = torch.cat([all_data, torch.cat([preds.unsqueeze(-1), y_test.unsqueeze(-1)], dim=-1)], dim=-1)

preds= preds.tolist()
y_test = y_test.tolist()

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(threshold, index = i)})
th = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.scatter(th.fpr.values, th.tpr.values, c='r', label = f'Best threshold: {th.thresholds.values[0]:.2f}')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


images_pth = os.listdir(f'{data_path}/{epsilon}/images/')
file_name = f'{opt.attack_model}_k{K}_comp{n_comp}_{dist_knn}_roc_'
i = 0
length = len(file_name) #The length of characters that the image shall have without the {i}.png
images_pth = [pth[0:length] for pth in images_pth]
for pth in images_pth:
    if pth == file_name:
        i += 1

print(f'Saving image as: {data_path}/{epsilon}/images/k{K}_comp{n_comp}_{dist_knn}_roc_{i}.png')
plt.savefig(f'{data_path}/{epsilon}/images/k{K}_comp{n_comp}_{dist_knn}_roc_{i}.png')

print(f'Saving file as: {data_path}/{epsilon}/predictions/k{K}_comp{n_comp}_{dist_knn}_res_{i}.pt')
torch.save(results, f'{data_path}/{epsilon}/predictions/k{K}_comp{n_comp}_{dist_knn}_res_{i}.pt')

