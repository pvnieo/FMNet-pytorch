# stdlib
from pathlib import Path
# 3p
import numpy as np
from plyfile import PlyData


def read_off(file):
    file = open(file, "r")
    if file.readline().strip() != "OFF":
        raise "Not a valid OFF header"

    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(" ")])
    verts = [[float(s) for s in file.readline().strip().split(" ")] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(" ")][1:] for i_face in range(n_faces)]

    return np.array(verts), np.array(faces)


def write_off(file, verts, faces):
    file = open(file, "w")
    file.write("OFF\n")
    file.write(f"{verts.shape[0]} {faces.shape[0]} {0}\n")
    for x in verts:
        file.write(f"{' '.join(map(str, x))}\n")
    for x in faces:
        file.write(f"{len(x)} {' '.join(map(str, x))}\n")


def read_ply(file):
    plydata = PlyData.read(file)
    verts = np.vstack([[float(y) for y in x] for x in plydata["vertex"]])
    faces = np.array([list(x) for x in plydata['face'].data['vertex_indices']])

    return verts, faces


def read_txt(file):
    verts = np.loadtxt(file)

    return verts, None


def read_mesh(file):
    file = Path(file)
    if file.suffix == ".off":
        return read_off(file)
    elif file.suffix == ".ply":
        return read_ply(file)
    elif file.suffix == ".txt":
        return read_txt(file)
    else:
        raise "File extention not implemented yet!"


def write_mesh(file, verts, faces):
    file = Path(file)
    if file.suffix == ".off":
        write_off(file, verts, faces)
    else:
        raise "File extention not implemented yet!"
