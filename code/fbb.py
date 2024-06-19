import os
import argparse

import torch
import sklearn.metrics as metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

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
data_positive = data_positive[0:len(data_negative)]

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
synthetic = SyntheticDataset(synthetic_pca)
negative = TrajectoryDataset(negative_pca, torch.zeros(len(negative_pca)))
positive = TrajectoryDataset(positive_pca, torch.ones(len(positive_pca)))
if opt.attack_model == 'none':
    # Perform KNN:

    knn = PyNN(dist_knn)
    knn.fit(synthetic.data)

    pos_dist, pos_idx, _ = knn.predict(positive, K)
    pos_prob  = torch.mean(1- (pos_dist-pos_dist.min()) / (pos_dist.max()-pos_dist.min()), dim=1)
    neg_dist, neg_idx, _ = knn.predict(negative, K)
    neg_prob  = torch.mean(1- (neg_dist-neg_dist.min()) / (neg_dist.max()-neg_dist.min()), dim=1)
    
    preds = torch.cat([pos_prob, neg_prob])
    y_test = torch.cat([positive.labels, negative.labels]).type(torch.uint8)
    
    all_data = torch.cat([data_positive, data_negative], dim=0)
    
else:
    from torch.utils.data import random_split
    import math

    #Split synthetic dataset training KNN and testing the whole model
    if opt.dataset == 'geolife':
        #Has to be specific for each dataset
        synthetic_1, synthetic_2, _ = random_split(synthetic, [math.ceil(0.5*len(synthetic)), math.floor(0.5*len(negative)), len(synthetic)-math.ceil(0.5*len(synthetic))- math.floor(0.5*len(negative))])
        negative_train, negative_test = random_split(negative, [math.ceil(0.5*len(negative)), math.floor(0.5*len(negative))]) #negative has to be split into 70-30
        positive_test, _ = random_split(positive, [math.ceil(0.5*len(negative)), len(positive)-math.ceil(0.5*len(negative))])
    
    training_dataset =TrajectoryDataset(torch.cat([synthetic_2.dataset[synthetic_2.indices], negative_train.dataset[negative_train.indices][0]]), torch.cat([torch.ones(len(synthetic_2)), torch.zeros(len(negative_train))]))
    #half negative half positive
    testing_dataset = TrajectoryDataset(torch.cat([negative_test.dataset[negative_test.indices][0], positive.data]), torch.cat([torch.zeros(len(positive)), torch.ones(len(negative_test))]))
    
    #Fit KNN
    knn = PyNN(dist_knn)
    knn.fit(synthetic_1.dataset[synthetic_1.indices])
    
    # Predict knn for synthetic data and real data (We know this is a 0)
    train_dist, train_idx, train_labels = knn.predict(training_dataset, K)
    if 'SVM' in opt.attack_model:
        kernel = opt.attack_model.split('SVM')[0]
        model = svm.SVC(kernel=kernel, probability=True)
    if 'rf' in opt.attack_model:
        model = RandomForestClassifier(n_estimators=100,random_state=0)
    #Train model

    model.fit(train_dist.tolist(), train_labels.tolist())
    
    #Test model
    test_dist, test_idx, test_labels = knn.predict(testing_dataset, K)
    preds = model.predict(test_dist.tolist())
    preds = torch.Tensor(preds)
    y_test = testing_dataset.labels
    
    all_data = testing_dataset.data
    
    
# Plot resuts
results = torch.cat([all_data, torch.cat([preds.unsqueeze(-1), y_test.unsqueeze(-1)], dim=-1)], dim=-1)

preds= preds.tolist()
y_test = y_test.tolist()

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(threshold, index = i)})
th = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.scatter(th.fpr.values[0], th.tpr.values[0], c='r', label = 'Maximum tpr')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

images_pth = os.listdir(f'{data_path}/{epsilon}/images/')
file_path = f'{opt.attack_model}_k{K}_comp{n_comp}_{dist_knn}_'
i = 0
length = len(file_path) #The length of characters that the image shall have without the _roc_{i}.png
images_pth = [pth[0:length] for pth in images_pth]
for pth in images_pth:
    if pth == file_path:
        i += 1

print(f'Saving image as: {data_path}/{epsilon}/images/{file_path}roc_{i}.png')
plt.savefig(f'{data_path}/{epsilon}/images/{file_path}roc_{i}.png')

print(f'Saving file as: {data_path}/{epsilon}/predictions/{file_path}res_{i}.pt')
torch.save(results, f'{data_path}/{epsilon}/predictions/{file_path}res_{i}.pt')
