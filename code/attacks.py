from knn import PyNN
import torch
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from torch.utils.data import random_split
import math
from datasets import SyntheticDataset, TrajectoryDataset

def KNN_attack(dist_knn,synthetic,positive,negative,K,data_positive,data_negative):
    knn = PyNN(dist_knn)
    knn.fit(synthetic.data)

    pos_dist, pos_idx, _ = knn.predict(positive, K)
    pos_prob  = torch.mean(1- (pos_dist-pos_dist.min()) / (pos_dist.max()-pos_dist.min()), dim=1)
    neg_dist, neg_idx, _ = knn.predict(negative, K)
    neg_prob  = torch.mean(1- (neg_dist-neg_dist.min()) / (neg_dist.max()-neg_dist.min()), dim=1)
    
    preds = torch.cat([pos_prob, neg_prob])
    y_test = torch.cat([positive.labels, negative.labels]).type(torch.uint8)
    
    return torch.cat([data_positive, data_negative], dim=0),preds,y_test


def SVD_attack(dist_knn,synthetic,positive,negative,K,data_positive,data_negative,attack_model):
    #Split synthetic dataset training KNN and testing the whole model 
    synthetic_1, synthetic_2 = random_split(synthetic, [math.ceil(0.5*len(synthetic)), math.floor(0.5*len(synthetic))])
    negative_train, negative_test = random_split(negative, [math.ceil(0.7*len(negative)), math.floor(0.3*len(negative))]) #negative has to be split into 70-30

    training_dataset = TrajectoryDataset(torch.cat([synthetic_2.dataset[synthetic_2.indices], negative_train.dataset[negative_train.indices][0]]), torch.cat([torch.ones(len(synthetic_2)), torch.zeros(len(negative_train))]))
    #half negative half positive
    testing_dataset = TrajectoryDataset(torch.cat([negative_test.dataset[negative_test.indices][0], positive.data]), torch.cat([torch.zeros(len(positive)), torch.ones(len(negative_test))]))
    
    #Fit KNN
    knn = PyNN(dist_knn)
    knn.fit(synthetic_1.dataset[synthetic_1.indices])
    
    # Predict knn for synthetic data and real data (We know this is a 0)
    train_dist, train_idx, train_labels = knn.predict(training_dataset, K)
    if 'SVM' in attack_model:
        kernel =attack_model.split('SVM')[0]
        model = svm.SVC(kernel=kernel, probability=True)
    if 'rf' in attack_model:
        model = RandomForestClassifier(n_estimators=100,random_state=0)
    #Train model

    model.fit(train_dist.tolist(), train_labels.tolist())
    
    #Test model
    test_dist, test_idx, test_labels = knn.predict(testing_dataset, K)
    preds = model.predict(test_dist.tolist())
    preds = torch.Tensor(preds)
    y_test = testing_dataset.labels
    
    return testing_dataset.data,preds,y_test
