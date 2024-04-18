import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA

def compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2):
    avg_sigma = (sigma1 + sigma2) / 2
    first_part = torch.sum((mu1 - mu2)**2 / avg_sigma) / 8
    second_part = torch.sum(torch.log(avg_sigma))
    second_part -= 0.5 * (torch.sum(torch.log(sigma1)))
    second_part -= 0.5 * (torch.sum(torch.log(sigma2)))
    return first_part + 0.5 * second_part


def get_bhattacharyya_distance(per_class_stats, c1, c2):
    mu1 = per_class_stats[c1]['mean']
    mu2 = per_class_stats[c2]['mean']
    sigma1 = per_class_stats[c1]['variance']
    sigma2 = per_class_stats[c2]['variance']
    sigma1 = torch.mean(sigma1)
    sigma2 = torch.mean(sigma2)
    return compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2)


def compute_per_class_mean_and_variance(features, target_labels, device, unique_labels):
    per_class_stats = {}
    for label in unique_labels:
        label = int(label)
        per_class_stats[label] = {}
        class_ids = torch.eq(target_labels, label)
        class_features = features[class_ids]
        mean = torch.mean(class_features, dim=0).squeeze()
        if class_features.shape[0] > 1:
            variance = torch.var(class_features, dim=0)
        else:
            variance = torch.var(class_features, dim=0, unbiased=False)
        # print(mean, variance)
        per_class_stats[label]['mean'] = mean
        per_class_stats[label]['variance'] = torch.maximum(variance, torch.Tensor([1e-4]).to(device))
    # for c, stat in per_class_stats.items():
    #     sum_mean = torch.sum(stat['mean'])
    #     sum_var = torch.sum(stat['variance'])
    #     print(f'{c}: mean{sum_mean}  var{sum_var}')
    return per_class_stats



def get_gbc_score(features, target_labels, device, pca_dim=64):

    if features.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        features = pca.fit_transform(features.cpu().detach().numpy())
        features = torch.from_numpy(features).to(device)


    unique_labels = torch.unique(target_labels)
    unique_labels = list(unique_labels)

    per_class_stats = compute_per_class_mean_and_variance(
      features, target_labels, device, unique_labels)
   
    per_class_bhattacharyya_distance = []
    for c1 in unique_labels:
        temp_metric = []
        for c2 in unique_labels:
            if c1 != c2:
                bhattacharyya_distance = get_bhattacharyya_distance(per_class_stats, int(c1), int(c2))
                temp_metric.append(torch.exp(-bhattacharyya_distance))
        per_class_bhattacharyya_distance.append(sum(temp_metric))
    gbc = -sum(per_class_bhattacharyya_distance)
    
    return float(gbc)