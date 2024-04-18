import torch
import numpy as np
import ot
import geomloss
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

zero_threshold = 0.00000001


class Kuhn_Munkras(object):
    def __init__(self, weight):
        self.weight= weight
        self.n = weight.shape[0]
        self.lx = [None] * self.n
        self.ly = [None] * self.n
        self.sx = [None] * self.n
        self.sy = [None] * self.n
        self.match = [None] * self.n
        
    def search_path(self, u):
        self.sx[u] = True
        for v in range(self.n):
            if self.sy[v] == False and self.lx[u] + self.ly[v] == self.weight[u][v]:
                self.sy[v]=True
                if self.match[v] == -1 or self.search_path(self.match[v]):
                    self.match[v] = u
                    return True
        return False
    
    def Kuhn_munkras(self):
        for i in range(self.n):
            self.ly[i] = 0
            self.lx[i] = -0x7fffffff
            for j in range(self.n):
                if self.lx[i] < self.weight[i][j]:
                    self.lx[i] = self.weight[i][j]
        
        self.match = [-1] * self.n
        for u in tqdm(range(self.n)):
            while True:
                self.sx = [False] * self.n
                self.sy = [False] * self.n
                if self.search_path(u):
                    break
                inc = 0x7fffffff
                for i in range(self.n):
                    if self.sx[i]:
                        for j in range(self.n):
                            if self.sy[j] == False and ((self.lx[i] + self.ly[j] - self.weight[i][j]) < inc):
                                inc = self.lx[i] + self.ly[j] - self.weight[i][j]
                if inc == 0:
                    print('fuck!')
                for i in range(self.n):
                    if self.sx[i]:
                        self.lx[i] -= inc
                    if self.sy[i]:
                        self.ly[i] += inc
                        
        sum = 0.0
        for i in range(self.n):
            if self.match[i] >= 0:
                sum += self.weight[self.match[i]][i]
        return sum / self.n


def get_ids_score(features_src, features_tar, device, pca_dim=512):

    if features_src.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        features_src = pca.fit_transform(features_src.cpu().detach().numpy())
        features_src = torch.from_numpy(features_src).to(device)
        features_tar = pca.fit_transform(features_tar.cpu().detach().numpy())
        features_tar = torch.from_numpy(features_tar).to(device)
    
    mean_src = torch.mean(features_src, dim=1)
    mean_tar = torch.mean(features_tar, dim=1)
    std_src = torch.std(features_src, dim=1)
    std_tar = torch.std(features_tar, dim=1)
    
    features_src = features_src.sub_(mean_src[:, None]).div_(std_src[:, None])
    features_tar = features_tar.sub_(mean_tar[:, None]).div_(std_tar[:, None])
    print(features_src)
    print(features_tar)
    cost_function = lambda x, y: (-euclidean_distances(x, y))
    D = cost_function(features_src.cpu().detach().numpy(), features_tar.cpu().detach().numpy())
    print(D)
    print(D.shape)
    
    result = Kuhn_Munkras(D).Kuhn_munkras()

    return float(result)

if __name__ == '__main__':
    src_x_list = []
    src_y_list = []
    tar_x_list = []
    tar_y_list = []

    NUM_SAMPLE_SCR = 1000
    NUM_SAMPLE_TAR = 1000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # suppose the feature dimension is 512, and the label is in range [0,10].
    for i in range(NUM_SAMPLE_SCR):
        src_x_list.append(np.random.randn(512))
        # src_y_list.append(np.random.randint(0,10))
        
    
    for i in range(NUM_SAMPLE_TAR):
        tar_x_list.append(np.random.randn(512))
        # tar_y_list.append(np.random.randint(0,10))
    
    # the shape of x is n*512, and the shape of y is n*1 
    src_x = torch.tensor(np.array(src_x_list), dtype=torch.float).to(device)
    tar_x = torch.tensor(np.array(tar_x_list), dtype=torch.float).to(device)
    # src_y = torch.tensor(np.array(src_y_list)[:,np.newaxis]).to(device)
    # tar_y = torch.tensor(np.array(tar_y_list)[:,np.newaxis]).to(device)

    
    print(get_ids_score(src_x, tar_x, device))
