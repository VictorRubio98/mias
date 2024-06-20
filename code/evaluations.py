import os
import argparse

import torch

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='geolife', help='Name of the tataset to be used in the attack.')

opt = parser.parse_args()

epsilons = ['baseline','epsilon100','epsilon70','epsilon50','epsilon20','epsilon10','epsilon5','epsilon0']
privacy_gain = []

for e in epsilons:
    data_path = f'data/{opt.dataset}'
    predictions_path = f'{data_path}/{e}/predictions'
    for file in os.listdir(path=predictions_path):
        results = torch.load(os.path.join(predictions_path, file))
        PG = calculatePG(results,results_to_compare)
        print(f'Privacy gain among {epsilon} and {e} is {PG}!' )
        privacy_gain.append(PG)
        break
    break

def calculatePG(baseline,e_private):
    # Procedure for baseline
    #baseline = torch.load(baseline_results)
    #e_private = torch.load(e_private_results) no need since we work already with tensors.
    
    baseline_p_label = baseline[baseline[:, 49] == 1][:, 48]
    baseline_n_label = baseline[baseline[:, 49] == 0][:, 48]
    decision_threshold = (baseline_p_label.mean() + baseline_n_label.mean()) / 2
    
    # Procedure for e-private
    e_private_p_label = e_private[e_private[:, 49] == 1][:, 48]
    e_private_n_label = e_private[e_private[:, 49] == 0][:, 48]
    decision_threshold_e = (e_private_p_label.mean() + e_private_n_label.mean()) / 2
    
    # Calculate probabilities for baseline
    probTP_b = (baseline_p_label > decision_threshold).float().mean().item()
    probFP_b = (baseline_n_label > decision_threshold).float().mean().item()
    
    # Calculate probabilities for e-private
    probTP_e = (e_private_p_label > decision_threshold).float().mean().item()
    probFP_e = (e_private_n_label > decision_threshold_e).float().mean().item()
    
    # Calculate privacy loss
    privacy_gain = (probTP_b - probFP_b) - (probTP_e - probFP_e)
    print(f"Baseline Prob TP: {probTP_b}")
    print(f"Baseline Prob FP: {probFP_b}")
    print(f"E-Private Prob TP: {probTP_e}")
    print(f"E-Private Prob FP: {probFP_e}")
    print(f"Privacy Gain: {privacy_gain}")
    return torch.tensor(privacy_gain,dtype=float)