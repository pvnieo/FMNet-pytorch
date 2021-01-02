# stdlib
from os import listdir
from os.path import isfile, join
from itertools import permutations
# 3p
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class FAUSTDataset(Dataset):
    """FAUST dataset"""
    def __init__(self, root, dim_basis=100, transform=None):
        self.root = root
        self.dim_basis = dim_basis
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
        return (torch.Tensor(mat['feat']).float(), torch.Tensor(mat['evecs'])[:, :self.dim_basis].float(),
                torch.Tensor(mat['evecs_trans'])[:self.dim_basis, :].float(),
                torch.Tensor(mat['geod_dist']).float(), torch.Tensor(mat['vts']).long())

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, index):
        idx1, idx2 = self.combinations[index]
        path1, path2 = self.samples[idx1], self.samples[idx2]

        feat_x, evecs_x, evecs_trans_x, dist_x, vts_x = self.loader(path1)
        feat_x, evecs_x, evecs_trans_x, dist_x = feat_x[vts_x], evecs_x[vts_x], evecs_trans_x[:, vts_x], dist_x[vts_x][:, vts_x]
        feat_y, evecs_y, evecs_trans_y, dist_y, vts_y = self.loader(path2)
        feat_y, evecs_y, evecs_trans_y, dist_y = feat_y[vts_y], evecs_y[vts_y], evecs_trans_y[:, vts_y], dist_y[vts_y][:, vts_y]
        if self.transform is not None:
            feat_x, evecs_x, evecs_trans_x, dist_x = self.transform((feat_x, evecs_x, evecs_trans_x, dist_x))
            feat_y, evecs_y, evecs_trans_y, dist_y = self.transform((feat_y, evecs_y, evecs_trans_y, dist_y))

        return [feat_x, evecs_x, evecs_trans_x, dist_x, feat_y, evecs_y, evecs_trans_y, dist_y]


class RandomSampling(object):
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices

    def __call__(self, sample):
        feat_x, evecs_x, evecs_trans_x, dist_x = sample
        vertices = np.random.choice(feat_x.size(0), self.num_vertices)
        feat_x = feat_x[vertices, :]
        evecs_x = evecs_x[vertices, :]
        evecs_trans_x = evecs_trans_x[:, vertices]
        dist_x = dist_x[vertices, :][:, vertices]

        return feat_x, evecs_x, evecs_trans_x, dist_x
