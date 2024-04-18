import torch
import numpy as np
import ot
import geomloss


def get_emd_score(features_src, features_tar, labels_src=None, labels_tar=None, device='cpu', alpha=0.1, easy=False):
    
    if not easy:
        assert labels_src is not None and labels_tar is not None
    
        class_feature_src = torch.zeros(int(torch.max(labels_src.flatten()).item())+1, features_src.shape[1]).to(device)
        class_feature_tar = torch.zeros(int(torch.max(labels_tar.flatten()).item())+1, features_tar.shape[1]).to(device)
        
        assert class_feature_src.shape[1] == class_feature_tar.shape[1]
        
        class_feature_src.scatter_add_(0, labels_src.squeeze().unsqueeze(-1).repeat(1, features_src.size(-1)), features_src.squeeze())
        class_feature_tar.scatter_add_(0, labels_tar.squeeze().unsqueeze(-1).repeat(1, features_tar.size(-1)), features_tar.squeeze())
        
        _, src_label_counts = torch.unique(labels_src, return_counts=True)
        _, tar_label_counts = torch.unique(labels_tar, return_counts=True)

        src_label_counts = src_label_counts.view(src_label_counts.shape[0], 1)
        tar_label_counts = tar_label_counts.view(tar_label_counts.shape[0], 1)

        class_feature_src = torch.div(class_feature_src, src_label_counts)
        class_feature_tar = torch.div(class_feature_tar, tar_label_counts)
    else:
        class_feature_src = features_src
        class_feature_tar = features_tar

    cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)
    C = cost_function(class_feature_src, class_feature_tar).cpu().detach().numpy()

    P = ot.emd(ot.unif(class_feature_src.shape[0]), ot.unif(class_feature_tar.shape[0]), C, numItermax=1000000)
    # P = ot.sinkhorn(ot.unif(feature1.shape[0]), ot.unif(feature2.shape[0]), C, reg= .5, numItermax=1000000, method='sinkhorn')
    
    EMD = np.sum(P*np.array(C)) / np.sum(P)

    result = 10 * np.exp(-alpha * EMD)

    return float(result)



if __name__ == '__main__':
    src_x_list = []
    src_y_list = []
    tar_x_list = []
    tar_y_list = []

    NUM_SAMPLE_SCR = 100
    NUM_SAMPLE_TAR = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # suppose the feature dimension is 512, and the label is in range [0,10].
    for i in range(NUM_SAMPLE_SCR):
        src_x_list.append(np.random.randn(512))
        src_y_list.append(np.random.randint(0,10))
        
    
    for i in range(NUM_SAMPLE_TAR):
        tar_x_list.append(np.random.randn(512))
        tar_y_list.append(np.random.randint(0,10))
    
    # the shape of x is n*512, and the shape of y is n*1 
    src_x = torch.tensor(np.array(src_x_list), dtype=torch.float).to(device)
    tar_x = torch.tensor(np.array(tar_x_list), dtype=torch.float).to(device)
    src_y = torch.tensor(np.array(src_y_list)[:,np.newaxis]).to(device)
    tar_y = torch.tensor(np.array(tar_y_list)[:,np.newaxis]).to(device)

    print(get_emd_score(src_x, tar_x, src_y, tar_y, device))
