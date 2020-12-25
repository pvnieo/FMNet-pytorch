# stdlib
from os import listdir
from os.path import isfile, join
from itertools import permutations
# 3p
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class FAUSTDataset(Dataset):
    """FAUST dataset"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
        self.combinations = list(permutations(range(len(self.samples)), 2))

    def loader(self, path):
        """
        pos: num_vertices * 3
        evecs: num_vertices * n_basis
        evecs_trans: n_basis * num_vertices
        feat: num_vertices * n_features
        dist: num_vertices * num_vertices
        """
        mat = sio.loadmat(path)
        return (torch.Tensor(mat['feat']).float(), torch.Tensor(mat['evecs']).float(),
                torch.Tensor(mat['evecs_trans']).float(), torch.Tensor(mat['geod_dist']).float())

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, index):
        idx1, idx2 = self.combinations[index]
        path1, path2 = self.samples[idx1], self.samples[idx2]
        feat_x, evecs_x, evecs_trans_x, dist_x = self.loader(path1)
        feat_y, evecs_y, evecs_trans_y, dist_y = self.loader(path2)

        return [feat_x, evecs_x, evecs_trans_x, dist_x, feat_y, evecs_y, evecs_trans_y, dist_y]
