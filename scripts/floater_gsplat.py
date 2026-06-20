"""Floater metrics for gsplat PLYs, computed in gsplat's NORMALIZED world frame.

gsplat trains with normalize_world_space=True, so its PLY coordinates differ from
the original COLMAP frame. Using the original cameras for the hull/scene-scale gives
garbage (e.g. 99.9% "outside hull"). This loads gsplat's Parser to get the normalized
camera poses and computes floater(a)/(b) + the floater(a)&outside-hull "ghost" count
in the matching frame.

Usage:
  python floater_gsplat.py <gsplat_examples_dir> <data_dir> <ply1> [<ply2> ...]
"""
import sys, os, json
import numpy as np

GSPLAT_EXAMPLES = sys.argv[1]
DATA_DIR = sys.argv[2]
PLYS = sys.argv[3:]

sys.path.insert(0, GSPLAT_EXAMPLES)                       # gsplat datasets.colmap.Parser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this repo's colmap_io / geom

from datasets.colmap import Parser  # noqa: E402
import colmap_io as cio  # noqa: E402
import geom  # noqa: E402

parser = Parser(data_dir=DATA_DIR, factor=1, normalize=True, test_every=8)
cc = parser.camtoworlds[:, :3, 3].astype(np.float64)  # normalized camera centers
diag = float(geom.scene_diagonal(cc))
sigma = 0.02 * diag


def metrics(ply):
    v = cio.read_ply_vertices(ply)
    n = len(v["x"])
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], float)))
    sc = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(float)).max(1)
    fa = (op < 0.05) & (sc > sigma)
    outside = ~geom.inside_hull_mask(xyz, cc, dilate=0.25)
    return {"n": n, "floater_a": int(fa.sum()), "floater_b_outside": int(outside.sum()),
            "ghost_a_and_outside": int((fa & outside).sum())}


out = {"scene_diag": diag, "sigma": sigma,
       "runs": {os.path.basename(os.path.dirname(os.path.dirname(p))): metrics(p) for p in PLYS}}
print(json.dumps(out, indent=2, ensure_ascii=False))
