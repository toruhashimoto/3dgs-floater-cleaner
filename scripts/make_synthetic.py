"""
make_synthetic.py — 自己検証用の合成 COLMAP データと合成 3DGS .ply を生成。

GPU を使わずにスクリプト群（cloud_metrics / sor_clean / floater_metrics / compare）
の動作を end-to-end で検証するための「正解が分かっている」最小データを作る。

生成物:
  <out>/sparse/0/{cameras.txt, images.txt, points3D.txt}
     - カメラ: 原点を見る球面配置（凸包が原点付近を内包）
     - 物体点: 原点近傍の密な殻（= 車体相当, ROI 内）
     - 外れ点: 遠方/上空の疎な点（= SfM ノイズ, SOR が除去すべき）
  <out>/fake_gs_baseline.ply, <out>/fake_gs_cleaned.ply
     - 既知個数のフローター（低不透明・大スケール・凸包外）を仕込んだ 3DGS 風 .ply
     - cleaned 側はフローターを減らしてある（compare の判定確認用）

使い方:
  python make_synthetic.py --out <dir> [--seed 0]
"""
from __future__ import annotations
import argparse
import os
import numpy as np

import colmap_io as cio


def rotmat2qvec(R):
    """回転行列 → COLMAP クォータニオン (qw,qx,qy,qz)。"""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    qvec = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if qvec[0] < 0:
        qvec = -qvec
    return qvec


def look_at_R(C, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    """カメラ位置 C から target を見る world->camera 回転（OpenCV: x右 y下 z前）。"""
    z = target - C
    z = z / (np.linalg.norm(z) + 1e-12)
    if abs(np.dot(z, up)) > 0.999:
        up = np.array([0, 1.0, 0])
    x = np.cross(z, up); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)  # 行が camera 軸 = world->cam
    return R


def fibonacci_sphere(n, radius):
    pts = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        r = np.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        pts.append([np.cos(theta) * r, y, np.sin(theta) * r])
    return np.array(pts) * radius


def write_cameras(path, n_cams, w=1600, h=1000, f=1200.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid in range(1, n_cams + 1):
            fp.write(f"{cid} PINHOLE {w} {h} {f} {f} {w/2} {h/2}\n")


def write_images(path, centers):
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("# Image list\n# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i, C in enumerate(centers, start=1):
            R = look_at_R(np.asarray(C, dtype=np.float64))
            q = rotmat2qvec(R)
            t = -R @ np.asarray(C, dtype=np.float64)
            fp.write(f"{i} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                     f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {i} img_{i:04d}.jpg\n")
            fp.write("\n")  # 2D点行（空）


def write_points3D(path, xyz, rgb):
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("# 3D point list\n"
                 "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
                 f"# Number of points: {len(xyz)}\n")
        for i, (p, c) in enumerate(zip(xyz, rgb), start=1):
            fp.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                     f"{int(c[0])} {int(c[1])} {int(c[2])} 0.5 1 0\n")


def make_gaussian_ply(path, n_object, n_floaters, scene_diag, rng,
                      object_radius=1.2, floater_radius=9.0):
    """3DGS 風の .ply を生成。floater は低不透明・大スケール・凸包外に配置。"""
    # 物体ガウシアン: 原点近傍, 高不透明(logit大), 小スケール(log小)
    o_xyz = rng.normal(0, object_radius * 0.4, size=(n_object, 3))
    o_op = np.full(n_object, 4.0)              # sigmoid(4)=0.982
    o_scale = np.full((n_object, 3), np.log(scene_diag * 0.002))  # 小

    # フローター: 遠方(凸包外), 低不透明(logit負), 大スケール(log大)
    dirs = rng.normal(size=(n_floaters, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    f_xyz = dirs * floater_radius
    f_op = np.full(n_floaters, -4.0)           # sigmoid(-4)=0.018 < 0.05
    f_scale = np.full((n_floaters, 3), np.log(scene_diag * 0.05))  # 大(>2%)

    xyz = np.vstack([o_xyz, f_xyz]).astype(np.float32)
    op = np.concatenate([o_op, f_op]).astype(np.float32)
    scale = np.vstack([o_scale, f_scale]).astype(np.float32)
    n = xyz.shape[0]
    rot = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    fdc = np.zeros((n, 3), dtype=np.float32)

    props = (["x", "y", "z"] + ["f_dc_0", "f_dc_1", "f_dc_2"] + ["opacity"]
             + ["scale_0", "scale_1", "scale_2"] + ["rot_0", "rot_1", "rot_2", "rot_3"])
    cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2], fdc[:, 0], fdc[:, 1], fdc[:, 2], op,
            scale[:, 0], scale[:, 1], scale[:, 2],
            rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]]
    data = np.stack(cols, axis=1).astype("<f4")

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {p}" for p in props]
    header += ["end_header"]
    with open(path, "wb") as fp:
        fp.write(("\n".join(header) + "\n").encode("ascii"))
        fp.write(data.tobytes())
    return n, n_floaters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cams", type=int, default=60)
    ap.add_argument("--n-object", type=int, default=4000)
    ap.add_argument("--n-outliers", type=int, default=300)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    sp = os.path.join(args.out, "sparse", "0")
    os.makedirs(sp, exist_ok=True)

    # カメラ（球面, 原点を内包）
    centers = fibonacci_sphere(args.n_cams, radius=5.0)
    write_cameras(os.path.join(sp, "cameras.txt"), args.n_cams)
    write_images(os.path.join(sp, "images.txt"), centers)

    # 物体点（原点近傍の密な殻 = ROI 内）
    u = rng.normal(size=(args.n_object, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    obj = u * (1.0 + rng.normal(0, 0.05, size=(args.n_object, 1)))  # 半径~1の殻
    obj_rgb = rng.integers(80, 200, size=(args.n_object, 3))

    # 外れ点（遠方/上空の疎なノイズ = SOR が除去すべき）
    out_dirs = rng.normal(size=(args.n_outliers, 3))
    out_dirs /= np.linalg.norm(out_dirs, axis=1, keepdims=True) + 1e-9
    radii = rng.uniform(6.0, 14.0, size=(args.n_outliers, 1))
    outl = out_dirs * radii
    outl_rgb = rng.integers(150, 255, size=(args.n_outliers, 3))

    xyz = np.vstack([obj, outl])
    rgb = np.vstack([obj_rgb, outl_rgb])
    write_points3D(os.path.join(sp, "points3D.txt"), xyz, rgb)

    # 合成 3DGS .ply（baseline は floater 多, cleaned は少）
    scene_diag = float(np.linalg.norm(centers.max(0) - centers.min(0)))
    nb, fb = make_gaussian_ply(os.path.join(args.out, "fake_gs_baseline.ply"),
                               n_object=5000, n_floaters=400, scene_diag=scene_diag, rng=rng)
    nc, fc = make_gaussian_ply(os.path.join(args.out, "fake_gs_cleaned.ply"),
                               n_object=5000, n_floaters=120, scene_diag=scene_diag, rng=rng)

    print(f"[make_synthetic] -> {args.out}")
    print(f"  cameras={args.n_cams}  object_pts={args.n_object}  outliers={args.n_outliers}")
    print(f"  total points3D={len(xyz):,}  (expected SOR removes ~outliers)")
    print(f"  fake_gs_baseline: {nb:,} gaussians ({fb} injected floaters)")
    print(f"  fake_gs_cleaned : {nc:,} gaussians ({fc} injected floaters)")
    print(f"  scene_diag(cameras)={scene_diag:.3f}")


if __name__ == "__main__":
    main()
