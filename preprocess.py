"""
Preprocessing codes are provided by You Yang: qq456cvb@github
"""
# stdlib
import argparse
from timeit import default_timer as timer
import os
# 3p
import numpy as np
import scipy.io as sio
from scipy.linalg import eigh
from sklearn import neighbors
from sklearn.utils.graph import graph_shortest_path
import plyfile
# project
import data.shot.shot as shot


NORMAL_R = 0.1
SHOT_R = 0.1
KNN = 20
DESCRIPTOR = 'shot'

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


def main(args):
    for root, dirs, files in os.walk(args.dataroot):
        print(root, dirs, files)
        for file in files:
            if file[-4:] == '.png':
                continue
            to_save = {}
            file_path = os.path.join(root, file)
            ply = plyfile.PlyData.read(file_path)
            x = ply.elements[0].data
            x = np.stack((x['x'], x['y'], x['z']), axis=-1)

            x = x - np.mean(x, axis=0, keepdims=True)
            x = x / np.max(np.linalg.norm(x, axis=1))
            to_save["pcd"] = x

            if not args.no_dec:
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
                _, eigen_x = eigh(laplacian_x, eigvals=(laplacian_x.shape[0] - args.num_eigen, laplacian_x.shape[0] - 1))
                print('eigen function time:', timer() - t)
                to_save["evecs"] = eigen_x

            if not args.no_geo:
                t = timer()
                geodesic_x = graph_shortest_path(graph_x_csr, directed=False)  # directed=False means geodesic, not l2 distance
                print('geodesic matrix time:', timer() - t)
                to_save["dist"] = geodesic_x

            if not args.no_shot:
                t = timer()

                if DESCRIPTOR == 'shot':
                    feature_x = shot.compute(x, NORMAL_R, SHOT_R).reshape(-1, 352)

                print('shot time:', timer() - t)
                feature_x[np.where(np.isnan(feature_x))] = 0
                to_save["feat"] = feature_x

            print(os.path.join(args.save_dir, '{}.mat'.format(file[:-4])))
            sio.savemat(os.path.join(args.save_dir, '{}.mat'.format(file[:-4])), to_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Preprocess data for FMNet training.
                       Compute Laplacian eigen decomposition, shot features, and geodesic distance for each shape."""
    )
    parser.add_argument('-d', '--dataroot', required=False,
                        default="./data/MPI-FAUST/training/registrations", help='root directory of the dataset')
    parser.add_argument('-sd', '--save-dir', required=False,
                        default="./data/faust/train", help='root directory to save the computed matrices')
    parser.add_argument("-ne", "--num-eigen", type=int, default=100, help="number of eigenvectors kept.")
    parser.add_argument("--no-shot", action='store_true', help="Do not compute shot features.")
    parser.add_argument("--no-geo", action='store_true', help="Do not compute geodesic distances.")
    parser.add_argument("--no-dec", action='store_true', help="Do not compute Laplacian eigen decomposition.")

    args = parser.parse_args()
    main(args)
