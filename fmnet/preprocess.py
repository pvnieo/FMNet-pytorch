# stdlib
import argparse
from pathlib import Path
# 3p
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from sklearn import neighbors
from sklearn.utils.graph import graph_shortest_path
import trimesh
import networkx as nx
# project
import utils.shot.shot as shot
from utils.io import read_mesh
from utils.laplace_decomposition import laplace_decomposition

# SHOT's hyperparameters
NORMAL_R = 0.1
SHOT_R = 0.1
KNN = 20


def compute_geodesic_matrix(verts, faces, NN):
    # get adjacency matrix
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertex_adjacency = mesh.vertex_adjacency_graph
    vertex_adjacency_matrix = nx.adjacency_matrix(vertex_adjacency, range(verts.shape[0]))
    # get adjacency distance matrix
    graph_x_csr = neighbors.kneighbors_graph(verts, n_neighbors=NN, mode='distance', include_self=False)
    distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
    distance_adj[vertex_adjacency_matrix != 0] = graph_x_csr[vertex_adjacency_matrix != 0]
    # compute geodesic matrix
    geodesic_x = graph_shortest_path(distance_adj, directed=False)
    return geodesic_x


def process_mesh(mesh, corres_root, save_dir, args):
    new_name = mesh.stem

    verts, faces = read_mesh(mesh)
    # center shape
    verts -= np.mean(verts, axis=0)

    # compute decomposition
    evals, evecs, evecs_trans, old_sqrt_area = laplace_decomposition(verts, faces, args.num_eigen)

    # normalize area and save
    verts /= old_sqrt_area

    # recompute decomposition and save eigenvalues
    evals, evecs, evecs_trans, sqrt_area = laplace_decomposition(verts, faces, args.num_eigen)
    print(f"shape {mesh.stem} ==> old sqrt area: {old_sqrt_area :.8f} | new sqrt area: {sqrt_area :.8f}")

    to_save = {"pos": verts, "faces": faces,
               "evals": evals, "evecs": evecs, "evecs_trans": evecs_trans}

    # compute geodesic matrix
    geodesic_x = compute_geodesic_matrix(verts, faces, args.nn)
    to_save["geod_dist"] = geodesic_x

    # compute shot descriptors
    shot_features = shot.compute(verts, NORMAL_R, SHOT_R).reshape(-1, 352)
    to_save["feat"] = shot_features

    # add correspandance
    if corres_root is not None:
        to_save["vts"] = np.loadtxt(corres_root / f"{new_name}.vts", dtype=np.int32)

    # save
    sio.savemat(save_dir / f"{new_name}.mat", to_save)


def main(args):
    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    meshes_root = Path(args.dataroot / "shapes")
    corres_root = Path(args.dataroot / "correspondences") if Path(args.dataroot / "correspondences").is_dir() else None

    meshes = list(meshes_root.iterdir())
    _ = Parallel(n_jobs=args.njobs)(delayed(process_mesh)(mesh, corres_root, save_root, args)
                                    for mesh in tqdm(meshes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Preprocess data for FMNet training.
                       Compute Laplacian eigen decomposition, shot features, and geodesic distance for each shape."""
    )
    parser.add_argument('-d', '--dataroot', required=False,
                        default="../data/faust/raw", help='root directory of the dataset')
    parser.add_argument('-sd', '--save-dir', required=False,
                        default="../data/faust/processed", help='root directory to save the processed dataset')
    parser.add_argument("-ne", "--num-eigen", type=int, default=100, help="number of eigenvectors kept.")
    parser.add_argument("-nj", "--njobs", type=int, default=-2, help="Number of parallel processes to use.")
    parser.add_argument("--nn", type=int, default=20,
                        help="Number of Neighbor to consider when computing geodesic matrix.")

    args = parser.parse_args()
    main(args)
