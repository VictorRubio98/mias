def calculatePG(baseline,e_private):
    # Procedure for baseline
    #baseline = torch.load(baseline_results)
    #e_private = torch.load(e_private_results) no need since we work already with tensors.
    
    baseline_p_label = baseline[baseline[:, -1] == 1][:, -2]
    baseline_n_label = baseline[baseline[:, -1] == 0][:, -2]
    decision_threshold = (baseline_p_label.mean() + baseline_n_label.mean()) / 2
    
    # Procedure for e-private
    e_private_p_label = e_private[e_private[:, -1] == 1][:, -2]
    e_private_n_label = e_private[e_private[:, -1] == 0][:, -2]
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

def PGPlotting(data_path,file_path_synthetic,n_comp,attack_model,dist_knn,positive,negative,K):
  epsilons = ['baseline','epsilon100','epsilon70','epsilon50','epsilon20','epsilon10','epsilon5','epsilon0']
  privacy_gain = []
  for epsilon_plot in epsilons:
      file_path_synthetic = f'{data_path}/{epsilon_plot}/gene.data'
      data_synthetic = pd.read_csv(file_path_synthetic, sep=" ", header=None)
      data_synthetic = torch.unique(torch.Tensor(data_synthetic.values), dim=0)
      synthetic_pca = pca.apply(data_synthetic, n_comp)
      synthetic = SyntheticDataset(synthetic_pca)
      if attack_model == 'none':
          all_data,preds,y_test = KNN_attack(dist_knn,synthetic,positive,negative,K)
      else:
          all_data,preds,y_test = SVD_attack(dist_knn,synthetic,positive,negative,K)
          
      results_to_compare = torch.cat([all_data, torch.cat([preds.unsqueeze(-1), y_test.unsqueeze(-1)], dim=-1)], dim=-1)
      PG = calculatePG(results,results_to_compare)
      print(f'Privacy gain among {epsilon} and {epsilon_plot} is {PG}!' )
      privacy_gain.append(PG)
  
  return torch.tensor(privacy_gain)
