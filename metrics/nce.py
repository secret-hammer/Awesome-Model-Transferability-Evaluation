import numpy as np
import torch
import torch.nn.functional as F

def get_nce_score(logits: torch.Tensor, target_labels: torch.Tensor, device):
    r"""
    Negative Conditional Entropy in `Transferability and Hardness of Supervised 
    Classification Tasks (ICCV 2019) <https://arxiv.org/pdf/1908.08142v1.pdf>`_.
    
    The NCE :math:`\mathcal{H}` can be described as

    Shape:
        - source_labels: (N, C_s) with source class number :math:`C_s`.
        - target_labels: (N) with target class number :math:`C_t`.
    """
    
    prob = F.softmax(logits,dim=1)
    source_labels = torch.max(prob, dim=1)[1]
    
    C_t = int(torch.max(target_labels) + 1)
    C_s = int(torch.max(source_labels) + 1)
    N = source_labels.shape[0]

    joint_distribution = torch.zeros((C_t, C_s))  # placeholder for the joint_distribution distribution, shape [C_t, C_s]
    for s, t in zip(source_labels, target_labels):
        s = int(s)
        t = int(t)
        joint_distribution[t, s] += 1.0 / N
    marginal_distribution = joint_distribution.sum(dim=0, keepdims=True)
    
    condition_distribution = (joint_distribution / marginal_distribution).T  # P(y | z), shape [C_s, C_t]
    mask = marginal_distribution.reshape(-1) != 0  # valid Z, shape [C_s]
    condition_distribution = condition_distribution[mask] + 1e-20  # remove NaN where p(z) = 0, add 1e-20 to avoid log (0)
    entropy_y_given_z = torch.sum(- condition_distribution * torch.log(condition_distribution), dim=1, keepdims=True)
    conditional_entropy = torch.sum(entropy_y_given_z * marginal_distribution.reshape((-1, 1))[mask])

    return float(-conditional_entropy.item())


if __name__ == '__main__':
    features = torch.randn(128, 20).cuda()
    source_labels = torch.arange(start=0, end=20).cuda()
    target_labels = torch.arange(start=0, end=10).cuda()
    a = get_nce_score(logits=features, target_labels=target_labels)
    