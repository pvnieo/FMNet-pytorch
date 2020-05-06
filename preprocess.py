"""
Preprocessing codes are provided by You Yang: qq456cvb@github
"""
from timeit import default_timer as timer
import os
import numpy as np
import scipy.io as sio
from scipy.linalg import eigh
from sklearn import neighbors
from sklearn.utils.graph import graph_shortest_path
import plyfile
import data.shot.shot as shot

NUM_EIGENS = 100
NUM_POINTS = 6890
NUM_POINTS_SAMPLE = 6890  # 2048
NORMAL_R = 0.1
SHOT_R = 0.1
KNN = 20
DESCRIPTOR = 'shot'
MODELS_DIR = 'data/MPI-FAUST/training/registrations'
SAVE_DIR = 'data/faust/train'


def normalize_adj(adj):
    rowsum = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def normalized_laplacian(adj):
    adj_normalized = normalize_adj(adj)
    norm_laplacian = np.eye(adj.shape[0]) - adj_normalized
    return norm_laplacian


if __name__ == '__main__':
    cnt = 0
    for root, dirs, files in os.walk(MODELS_DIR):
        print(root, dirs, files)
        for file in files:
            if file[-4:] == '.png':
                continue
            cnt += 1
            file_path = os.path.join(root, file)
            ply = plyfile.PlyData.read(file_path)
            x = ply.elements[0].data
            x = np.stack((x['x'], x['y'], x['z']), axis=-1)
            np.random.seed(cnt)  # assure different randoms
            random_indices = np.random.choice(NUM_POINTS, NUM_POINTS_SAMPLE)
            x = x[random_indices]

            x = x - np.mean(x, axis=0, keepdims=True)
            x = x / np.max(np.linalg.norm(x, axis=1))

            t = timer()
            graph_x_csr = neighbors.kneighbors_graph(x, KNN, mode='distance', include_self=False)
            print('knn time:', timer() - t)

            t = timer()
            graph_x = graph_x_csr.toarray()  # derive a n*n matrix, each element is distance(if knn) or 0(if not knn)
            graph_x = np.exp(- graph_x ** 2 / np.mean(graph_x[:, -1]) ** 2)
            graph_x = graph_x + (graph_x.T - graph_x) * np.greater(graph_x.T, graph_x).astype(
                np.float)

            laplacian_x = normalized_laplacian(graph_x)
            print('laplacian time:', timer() - t)

            t = timer()
            _, eigen_x = eigh(laplacian_x, eigvals=(laplacian_x.shape[0] - NUM_EIGENS, laplacian_x.shape[0] - 1))
            print('eigen function time:', timer() - t)

            t = timer()
            geodesic_x = graph_shortest_path(graph_x_csr, directed=False)  # directed=False means geodesic, not l2 distance
            print('geodesic matrix time:', timer() - t)

            t = timer()

            if DESCRIPTOR == 'shot':
                feature_x = shot.compute(x, NORMAL_R, SHOT_R).reshape(-1, 352)

            print('shot time:', timer() - t)
            feature_x[np.where(np.isnan(feature_x))] = 0

            print(os.path.join(SAVE_DIR, '{}.mat'.format(file[:-4])))
            sio.savemat(os.path.join(SAVE_DIR, '{}.mat'.format(file[:-4])),
                        {'pcd': x,
                         'evecs': eigen_x,
                         'feat': feature_x,
                         'dist': geodesic_x,
                         })
