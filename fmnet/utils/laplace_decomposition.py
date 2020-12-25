# 3p
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


def cotangent(p):
    return np.cos(p) / np.sin(p)


# source: https://github.com/JM-data/PyFuncMap/blob/master/utils/MeshProcess.py
def cotangent_laplacian(verts, faces):
    '''
    Compute the cotangent matrix and the weight matrix of a shape
    :param S: a shape with VERT and TRIV
    :return: cotangent matrix W, and the weight matrix A
    '''
    T1 = faces[:, 0]
    T2 = faces[:, 1]
    T3 = faces[:, 2]

    V1 = verts[T1, :]
    V2 = verts[T2, :]
    V3 = verts[T3, :]

    nv = verts.shape[0]

    L1 = np.linalg.norm(V2 - V3, axis=1)
    L2 = np.linalg.norm(V1 - V3, axis=1)
    L3 = np.linalg.norm(V1 - V2, axis=1)
    L = np.column_stack((L1, L2, L3))  # Edges of each triangle

    Cos1 = (L2 ** 2 + L3 ** 2 - L1 ** 2) / (2 * L2 * L3)
    Cos2 = (L1 ** 2 + L3 ** 2 - L2 ** 2) / (2 * L1 * L3)
    Cos3 = (L1 ** 2 + L2 ** 2 - L3 ** 2) / (2 * L1 * L2)
    Cos = np.column_stack((Cos1, Cos2, Cos3))  # Cosines of opposite edges for each triangle
    Ang = np.arccos(Cos)  # Angles

    I = np.concatenate((T1, T2, T3))
    J = np.concatenate((T2, T3, T1))
    w = 0.5 * cotangent(np.concatenate((Ang[:, 2], Ang[:, 0], Ang[:, 1]))).astype(float)
    In = np.concatenate((I, J, I, J))
    Jn = np.concatenate((J, I, I, J))
    wn = np.concatenate((-w, -w, w, w))
    W = csr_matrix((wn, (In, Jn)), [nv, nv])  # Sparse Cotangent Weight Matrix

    cA = cotangent(Ang) / 2  # Half cotangent of all angles
    At = 1 / 4 * (L[:, [1, 2, 0]] ** 2 * cA[:, [1, 2, 0]] + L[:, [2, 0, 1]] ** 2 * cA[:, [2, 0, 1]]).astype(float)  # Voronoi Area

    Ar = np.linalg.norm(np.cross(V1 - V2, V1 - V3), axis=1)

    # Use Ar is ever cot is negative instead of At
    locs = cA[:, 0] < 0
    At[locs, 0] = Ar[locs] / 4
    At[locs, 1] = Ar[locs] / 8
    At[locs, 2] = Ar[locs] / 8

    locs = cA[:, 1] < 0
    At[locs, 0] = Ar[locs] / 8
    At[locs, 1] = Ar[locs] / 4
    At[locs, 2] = Ar[locs] / 8

    locs = cA[:, 2] < 0
    At[locs, 0] = Ar[locs] / 8
    At[locs, 1] = Ar[locs] / 8
    At[locs, 2] = Ar[locs] / 4

    Jn = np.zeros(I.shape[0])
    An = np.concatenate((At[:, 0], At[:, 1], At[:, 2]))
    Area = csr_matrix((An, (I, Jn)), [nv, 1])  # Sparse Vector of Area Weights

    In = np.arange(nv)
    A = csr_matrix((np.squeeze(np.array(Area.todense())), (In, In)),
                   [nv, nv])  # Sparse Matrix of Area Weights
    return W, A


def laplace_decomposition(verts, faces, neig=50):
    W, A = cotangent_laplacian(verts, faces)

    # sqrt_area = np.sqrt(A.diagonal().sum())

    try:
        # compute basis
        evals, evecs = eigsh(W, neig, A, 1e-6)
    except Exception as e:
        print(e)
        # add small diagonal matrix to W in case there is some isolated vertices
        c = np.ones(verts.shape[0]) * 1e-8
        damping = diags(c)
        W += damping
        # compute basis
        evals, evecs = eigsh(W, neig, A, 1e-6)

    evecs = np.array(evecs, ndmin=2)
    evecs_trans = evecs.T @ A
    evals = np.array(evals)
    return evals, evecs, evecs_trans, np.sqrt(A.diagonal().sum())
