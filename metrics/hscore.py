import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

def get_covariance(matrix):
    zero_mean_matrix = matrix - torch.mean(matrix, axis=0, keepdims=True)
    cov = torch.matmul(zero_mean_matrix.T, zero_mean_matrix) / (
        matrix.shape[0] - 1)
    return cov
    
def get_hscore(features, target_labels, device, pca_dim):
    """Compute H score metric based on Bao et al. 2019.

        Args:
            features: source features from the target data.
            target_labels: ground truth labels in the target label space.

        Returns:
            hscore: transferability metric score.
    """
    if features.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        features = pca.fit_transform(features.cpu().detach().numpy())
        features = torch.from_numpy(features).to(device)

    covariance_features = get_covariance(features)
    inter_class_features = torch.zeros_like(features)
    unique_labels = torch.unique(target_labels)
    
    for label in list(unique_labels):
        label_indices = torch.where(torch.eq(target_labels, label))[0]
        label_mean = torch.mean(torch.index_select(features, dim=0, index=label_indices), dim=0, keepdims=True)
        label_mean_repeated = label_mean.repeat(label_indices.shape[0], 1)
        inter_class_features.scatter_(src=label_mean_repeated, index=label_indices.view(-1, 1).repeat(1, label_mean_repeated.shape[1]), dim=0)
    
    inter_class_covariance = get_covariance(inter_class_features)
    hscore = torch.trace(torch.mm(torch.linalg.pinv(covariance_features, rcond=1e-5),inter_class_covariance))
    
    return float(hscore)

def get_regularized_h_score(features, target_labels, pca_dim, device='cpu'):
    r"""
    Regularized H-score in `Newer is not always better: Rethinking transferability metrics, their peculiarities, stability and performance (NeurIPS 2021) 
    <https://openreview.net/pdf?id=iz_Wwmfquno>`_.
    
    The  regularized H-Score :math:`\mathcal{H}_{\alpha}` can be described as:
    .. math::
        \mathcal{H}_{\alpha}=\operatorname{tr}\left(\operatorname{cov}_{\alpha}(f)^{-1}\left(1-\alpha \right)\operatorname{cov}\left(\mathbb{E}[f \mid y]\right)\right)
    
    where :math:`f` is the features extracted by the model to be ranked, :math:`y` is the groud-truth label vector and :math:`\operatorname{cov}_{\alpha}` the  Ledoit-Wolf 
    covariance estimator with shrinkage parameter :math:`\alpha`
    Args:
        features (torch.tensor):features extracted by pre-trained model.
        labels (torch.tensor):  groud-truth labels.
    Shape:
        - features: (N, F), with number of samples N and feature dimension F.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar.
    """
    if features.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        features = pca.fit_transform(features.cpu().detach().numpy())

    if type(features) != np.ndarray:
        features = features.cpu().detach().numpy()
    target_labels = target_labels.detach().cpu().numpy()
    f = features.astype('float64')
    f = f - np.mean(f, axis=0, keepdims=True)  # Center the features for correct Ledoit-Wolf Estimation
    y = target_labels

    C = int(y.max() + 1)
    g = np.zeros_like(f)

    cov = LedoitWolf(assume_centered=False).fit(f)
    alpha = cov.shrinkage_
    covf_alpha = cov.covariance_

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = np.cov(g, rowvar=False)
    score = np.trace(np.dot(np.linalg.pinv(covf_alpha, rcond=1e-15), (1 - alpha) * covg))

    return score


# if __name__ == '__main__':
#     src_x_list = []
#     src_y_list = []

#     NUM_SAMPLE_SCR = 200

#     # suppose the feature dimension is 512, and the label is in range [0,10].
#     for i in range(NUM_SAMPLE_SCR):
#         src_x_list.append(np.random.randn(512))
#         src_y_list.append(np.random.randint(0,20))
    
#     src_x_torch = torch.tensor(np.array(src_x_list), dtype=torch.float)
#     src_y_torch = torch.tensor(np.array(src_y_list), dtype=torch.int)
#     h1 = get_hscore(src_x_torch, src_y_torch)
#     h2 = hscore_np.getHscore(np.array(src_x_list), np.array(src_y_list))
    
#     print(h1)
#     print(h2)
    