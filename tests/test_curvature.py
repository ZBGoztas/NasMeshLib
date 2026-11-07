"""Validate curvature on analytic shapes: sphere, cylinder, plane.

Run from the repo root:  python -m pytest tests  (or: python tests/test_curvature.py)
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ParserNas import ParserNas  # noqa: E402
from utils.geometry_features import GeometryFeatures  # noqa: E402
from utils.topology_extractor import TopologyExtractor  # noqa: E402


def write_nas(path, P, F, quads=False):
    with open(path, "w") as fh:
        fh.write("BEGIN BULK\n")
        for k, p in enumerate(P):
            fh.write(f"GRID,{k + 1},,{p[0]:.8f},{p[1]:.8f},{p[2]:.8f}\n")
        card = "CQUAD4" if quads else "CTRIA3"
        for k, f in enumerate(F):
            fh.write(f"{card},{k + 1},1," + ",".join(str(i + 1) for i in f) + "\n")
        fh.write("ENDDATA\n")


def sphere(R=10.0, n=3000):
    from scipy.spatial import ConvexHull
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    P = R * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    F = ConvexHull(P).simplices
    for k, f in enumerate(F):  # orient outward
        a, b, c = P[f]
        if np.dot(np.cross(b - a, c - a), a) < 0:
            F[k] = f[[0, 2, 1]]
    return P, F


def cylinder_quads(R=5.0, h=20.0, nu=60, nv=40):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, h, nv)
    P = np.array([[R * np.cos(a), R * np.sin(a), z] for z in v for a in u])
    F = [[j * nu + i, j * nu + (i + 1) % nu, (j + 1) * nu + (i + 1) % nu, (j + 1) * nu + i]
         for j in range(nv - 1) for i in range(nu)]
    return P, np.array(F)


def plane(n=20):
    xs = np.linspace(0, 1, n)
    P = np.array([[x, y, 0.0] for y in xs for x in xs])
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i
            F += [[a, b, c], [a, c, d]]
    return P, np.array(F)


def features(P, F, quads=False):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mesh.nas")
        write_nas(path, P, F, quads)
        data = ParserNas().parse_file(path)
    g = GeometryFeatures("test").extract_geometry_features(data["nodes"], data["elements"])
    H = np.array(list(g["mean_curvature"].values()))
    K = np.array(list(g["gaussian_curvature"].values()))
    return data, H, K


def test_parser_and_topology():
    P, F = sphere()
    data, _, _ = features(P, F)
    assert len(data["nodes"]) == len(P) and len(data["elements"]) == len(F)
    topo = TopologyExtractor().extract_topology("sphere", data["nodes"], data["elements"])
    assert topo["edge_count"] == 3 * len(F) // 2  # closed triangle mesh: E = 3F/2


def test_sphere():
    R = 10.0
    _, H, K = features(*sphere(R))
    assert abs(np.median(H) - 1 / R) / (1 / R) < 0.02
    assert abs(np.median(K) - 1 / R ** 2) / (1 / R ** 2) < 0.02


def test_cylinder_quads():
    R = 5.0
    P, F = cylinder_quads(R)
    _, H, K = features(P, F, quads=True)
    interior = (P[:, 2] > 1.0) & (P[:, 2] < 19.0)
    assert abs(np.median(H[interior]) - 1 / (2 * R)) / (1 / (2 * R)) < 0.02
    assert np.max(np.abs(K[interior])) < 1e-6


def test_plane():
    P, F = plane()
    _, H, K = features(P, F)
    interior = (P[:, 0] > 0) & (P[:, 0] < 1) & (P[:, 1] > 0) & (P[:, 1] < 1)
    assert np.max(np.abs(H[interior])) < 1e-9 and np.max(np.abs(K[interior])) < 1e-9


if __name__ == "__main__":
    for t in (test_parser_and_topology, test_sphere, test_cylinder_quads, test_plane):
        t()
        print("PASS", t.__name__)
