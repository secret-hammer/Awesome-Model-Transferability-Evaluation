from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import torch

def get_nleep_score(
    features,
    target_labels,
    device,
    pca_dim=512,
    random_state=123
):
    features = features.cpu().detach().numpy()
    # if features.shape[1] > pca_dim:
    pca = PCA(n_components=0.8, random_state=random_state)
    features = pca.fit_transform(features)
    
    print('NLEEP:', features.shape[1])
    
    unique_target_labels = torch.unique(target_labels).flatten()
    num_components_gmm = (int(unique_target_labels.shape[0])) * 5
    while num_components_gmm >= features.shape[0]:
        num_components_gmm -= int(unique_target_labels.shape[0])
    print(num_components_gmm)
    gmm = GaussianMixture(
      n_components=num_components_gmm, random_state=random_state, covariance_type='spherical', reg_covar=1e-5).fit(features)
    gmm_predictions = gmm.predict_proba(features)
    nleep = get_leep_score(gmm_predictions.astype('float32'), target_labels, device)
    return nleep

def get_leep_score(features, labels, device):

    features = torch.tensor(features).to(device)
    unique_labels = torch.unique(labels).flatten()
    max_unique_label, _ = torch.max(unique_labels, dim=0)
    joint_distribution = torch.zeros(int(max_unique_label.item())+1, features.shape[1]).to(device)
    joint_distribution.scatter_add_(0, labels.squeeze().unsqueeze(-1).repeat(1, features.size(-1)), features.squeeze())

    joint_distribution /= features.shape[0]
    
    marginal_distribution = torch.sum(joint_distribution, axis=0)
    
    conditional_distribution = torch.div(joint_distribution, marginal_distribution)
    
    total_sum = torch.sum(torch.log(torch.sum(conditional_distribution[labels] * features, axis=1)))
    return float((total_sum / features.shape[0]).item())
    



if __name__ == '__main__':
    """Compute NLEEP on the target training data."""
    features, labels = torch.ones(10, 512), torch.tensor([1, 2, 3, 1 ,2, 3, 1, 2, 3, 1], dtype=torch.int64)
    num_components_gmm = int((torch.max(labels).item() + 1) * 5)
    nleep = get_nleep_score(features, labels, 5)
    print(float(nleep))