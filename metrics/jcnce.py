import numpy as np
import ot
import geomloss
import torch
import math
from sklearn.decomposition import PCA

# return: C : [Ns, Ny]
def compute_cost(X_src, X_tar, Y_src, Y_tar, device, const=0.5):
    C = torch.zeros((X_src.shape[0], X_tar.shape[0])).to(device)
    c1 = lambda x, y: geomloss.utils.squared_distances(x, y)
    C1 = c1(X_src, X_tar)
    Cw = torch.zeros_like(C1).to(device)
    
    unique_src_label = torch.unique(Y_src)
    unique_tar_label = torch.unique(Y_tar)

    al_src = torch.zeros((unique_src_label.shape[0], X_src.shape[1]))
    al_tar = torch.zeros((unique_tar_label.shape[0], X_tar.shape[1]))

    for i, y1 in zip(range(unique_src_label.shape[0]), unique_src_label):
        y1_idx = torch.where(Y_src==y1)
        al_src[i] = torch.mean(X_src[y1_idx[0]], dim=0)

    for i, y2 in zip(range(unique_tar_label.shape[0]), unique_tar_label):
        y2_idx = torch.where(Y_tar==y2)
        al_tar[i] = torch.mean(X_tar[y2_idx[0]], dim=0)
    
    C_al = c1(al_src, al_tar).cpu().detach().numpy()
    P_al = ot.emd(ot.unif(al_src.shape[0]), ot.unif(al_tar.shape[0]), C_al, numItermax=1000000)
    W_al = P_al * np.array(C_al)
    W_al = torch.tensor(W_al).to(device)
    
    for i, y1 in zip(range(unique_src_label.shape[0]), unique_src_label):
        for j, y2 in zip(range(unique_tar_label.shape[0]), unique_tar_label):
            y1_index = torch.where(unique_src_label == y1)[0].item()
            y2_index = torch.where(unique_tar_label == y2)[0].item()
            w =  W_al[y1_index, y2_index]
            y1_idx = torch.where(Y_src==y1)[0]
            y2_idx = torch.where(Y_tar==y2)[0]
            
            w_matrix = torch.full((y1_idx.shape[0], y2_idx.shape[0]), w).to(device)
            w_matrix_tar = torch.zeros((y1_idx.shape[0], Y_tar.shape[0])).to(device)
            w_matrix_tar.scatter_add_(1, y2_idx.reshape(1, -1).repeat(y1_idx.shape[0], 1), w_matrix)
            Cw.scatter_add_(0, y1_idx.reshape(-1, 1).repeat(1, y2_idx.shape[0]),w_matrix_tar)
    
    C = const * C1 + (1 - const) * Cw
    
    return C.cpu().detach().numpy()
        
def compute_coupling(C):
    
    P = ot.emd(ot.unif(C.shape[0]), ot.unif(C.shape[1]), C, numItermax=2000000)

    return P

# Y_src: [Ns, 1]
# Y_tar: [Nt, 1]
def compute_CE(P, Y_src, Y_tar):
    Y_src = Y_src.cpu().detach().numpy()
    Y_tar = Y_tar.cpu().detach().numpy()
    src_label_set = set(sorted(list(Y_src.flatten())))
    tar_label_set = set(sorted(list(Y_tar.flatten())))

    # joint distribution of source and target label
    P_src_tar = np.zeros((np.max(Y_src)+1,np.max(Y_tar)+1))

    for y1 in src_label_set:
        y1_idx = np.where(Y_src==y1)
        for y2 in tar_label_set:
            y2_idx = np.where(Y_tar==y2)

            RR = y1_idx[0].repeat(y2_idx[0].shape[0])
            
            CC = np.tile(y2_idx[0], y1_idx[0].shape[0])

            P_src_tar[y1,y2] = np.sum(P[RR,CC])

    # marginal distribution of source label
    P_src = np.sum(P_src_tar,axis=1)

    ce = 0.0
    for y1 in src_label_set:
        P_y1 = P_src[y1]
        for y2 in tar_label_set:
            
            if P_src_tar[y1,y2] != 0:
                ce += -(P_src_tar[y1,y2] * math.log(P_src_tar[y1,y2] / P_y1))
    return ce

def get_jcnce_score(features_src, features_tar, labels_src, labels_tar, device, pca_dim):
    if features_tar.shape[1] != features_src.shape[1] or features_tar.shape[1] > pca_dim or features_src.shape[1] > pca_dim:
        n_components = pca_dim
        assert features_src.shape[1] >= n_components and  features_tar.shape[1] >= n_components
        pca = PCA(n_components=n_components)
        features_tar = pca.fit_transform(features_tar.cpu().detach().numpy())
        features_tar = torch.from_numpy(features_tar).to(device)
        features_src = pca.fit_transform(features_src.cpu().detach().numpy())
        features_src = torch.from_numpy(features_src).to(device)

    print(features_src.shape[1], features_tar.shape[1])

    C = compute_cost(features_src, features_tar, labels_src, labels_tar, device=device)
    P = compute_coupling(C)
    score = -compute_CE(P, labels_src, labels_tar)
    return float(score)

def test():
    #-----------start: randomly generate the testing data-----------
    src_x_list = []
    src_y_list = []
    tar_x_list = []
    tar_y_list = []

    NUM_SAMPLE_SCR = 100
    NUM_SAMPLE_TAR = 20

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
    #-----------end: randomly generate the testing data------------

    score = get_jcnce_score(src_x, tar_x, src_y, tar_y, device)

    print ('JC_NCE: %.4f'%(score))

    

if __name__ == '__main__':
    test()