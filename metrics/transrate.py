import torch
import numpy as np
from sklearn.decomposition import PCA


def coding_rate(features, device, eps=1e-4):
    f = features
    n, d = f.shape
    (_, rate) = torch.linalg.slogdet((torch.eye(d).to(device) + 1 / (n * eps) * f.T @ f))
    return 0.5 * rate


def get_transrate_score(features, target_labels, device, pca_dim, eps=1e-4):
    r"""
    TransRate in `Frustratingly easy transferability estimation (ICML 2022) 
    <https://proceedings.mlr.press/v162/huang22d/huang22d.pdf>`_.
    
    The TransRate :math:`TrR` can be described as:

    .. math::
        TrR= R\left(f, \espilon \right) - R\left(f, \espilon \mid y \right) 
    
    where :math:`f` is the features extracted by the model to be ranked, :math:`y` is the groud-truth label vector, 
    :math:`R` is the coding rate with distortion rate :math:`\epsilon`

    Args:
        features (np.ndarray):features extracted by pre-trained model.
        labels (np.ndarray):  groud-truth labels.
        eps (float, optional): distortion rare (Default: 1e-4)

    Shape:
        - features: (N, F), with number of samples N and feature dimension F.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar.
    """
    if features.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        features = pca.fit_transform(features.cpu().detach().numpy())
        features = torch.from_numpy(features).to(device)
    
    print(features.shape[1])
    f = features
    y = target_labels
    f = f - torch.mean(f, dim=0, keepdims=True)
    f /= torch.sqrt(torch.trace(f @ f.T))
    Rf = coding_rate(f, device, eps)
    Rfy = 0.0
    C = int(y.max() + 1)
    for i in range(C):
        Rfy += coding_rate(f[(y == i).flatten()], device, eps) * f[(y == i).flatten()].shape[0]
    return float(Rf - Rfy / f.shape[0])