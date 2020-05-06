# stdlib
from os import listdir
from os.path import isfile, join
from itertools import permutations
# 3p
import scipy.io as sio
import torch
import torch.nn as nn
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
        pcd: num_vertices * 3
        evecs: num_vertices * n_basis
        feat: num_vertices * n_features
        dist: num_vertices * num_vertices
        """
        mat = sio.loadmat(path)
        return torch.Tensor(mat['feat']).float(), torch.Tensor(mat['evecs']).float(), torch.Tensor(mat['dist']).float()

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, index):
        idx1, idx2 = self.combinations[index]
        path1, path2 = self.samples[idx1], self.samples[idx2]
        feat_x, evecs_x, dist_x = self.loader(path1)
        feat_y, evecs_y, dist_y = self.loader(path2)

        return [feat_x, evecs_x, dist_x, feat_y, evecs_y, dist_y]


class SoftErrorLoss(nn.Module):
    """
    Calculate soft error loss as defined is FMNet paper.
    """
    def __init__(self):
        super().__init__()

    def forward(self, P, geodesic_dist):
        """Compute soft error loss

        Arguments:
            P {torch.Tensor} -- soft correspondence matrix. Shape: batch_size x num_vertices x num_vertices.
            geodesic_dist {torch.Tensor} -- geodesic distances on Y. Shape: batch_size x num_vertices x num_vertices.

        Returns:
            float -- total loss
        """
        loss = torch.sqrt(((P * geodesic_dist) ** 2).sum((1, 2)))
        return torch.mean(loss)


if __name__ == "__main__":
    dataroot = "./data/faust/train"
    dataset = FAUSTDataset(dataroot)
    print(len(dataset))
    print(len(dataset[0]))
    bs, n_points = 10, 1000
    P = torch.rand(bs, n_points, n_points)
    dist_y = torch.rand(bs, n_points, n_points)
    criterion = SoftErrorLoss()
    print(criterion(P, dist_y))
