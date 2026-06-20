"""
cloud_metrics.py — COLMAP sparse 点群のメトリクス（T1/T2）。

§4「点群メトリクス」の操作的定義を実装:
  - 総点数
  - 点群 bbox（min/max/size/diagonal）
  - カメラ位置の凸包に対する「外側点」の割合（dilate=0 と 0.25 の2種）
  - 近傍密度ヒストグラム（kNN 平均距離の分布）
  - 再投影誤差（ERROR）の統計

使い方:
  python cloud_metrics.py --sparse <sparse/0> --out metrics/baseline.json [--knn 20] [--hist-bins 30]
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np

import colmap_io as cio
import geom


def compute_metrics(sparse_dir, knn=20, hist_bins=30):
    cams_p = os.path.join(sparse_dir, "cameras.txt")
    imgs_p = os.path.join(sparse_dir, "images.txt")
    pts_p = os.path.join(sparse_dir, "points3D.txt")

    images = cio.read_images_text(imgs_p)
    cam_centers = cio.camera_centers(images)
    pts = cio.read_points3D_text(pts_p)
    xyz = pts["xyz"]
    n = xyz.shape[0]

    mn, mx, size, diag = geom.bbox(xyz)

    # カメラ凸包に対する外側点の割合
    try:
        outside_frac = geom.outside_hull_fraction(xyz, cam_centers, dilate=0.0)
        outside_frac_d25 = geom.outside_hull_fraction(xyz, cam_centers, dilate=0.25)
    except Exception as e:
        outside_frac = outside_frac_d25 = None
        print(f"[warn] convex hull fraction skipped: {e}")

    # 近傍密度（kNN 平均距離）
    d = geom.knn_mean_distances(xyz, k=knn)
    hist_counts, hist_edges = np.histogram(d, bins=hist_bins)

    metrics = {
        "source": os.path.abspath(sparse_dir),
        "num_points": int(n),
        "num_images": int(len(images)),
        "bbox": {
            "min": mn.tolist(), "max": mx.tolist(),
            "size": size.tolist(), "diagonal": diag,
        },
        "camera_centers_bbox_diagonal": geom.scene_diagonal(cam_centers) if len(cam_centers) else None,
        "outside_camera_hull_fraction": outside_frac,
        "outside_camera_hull_fraction_dilate25": outside_frac_d25,
        "knn_mean_distance": {
            "k": int(knn),
            "mean": float(d.mean()), "std": float(d.std()),
            "min": float(d.min()), "max": float(d.max()),
            "median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)),
            "p99": float(np.percentile(d, 99)),
            "histogram_counts": hist_counts.tolist(),
            "histogram_edges": hist_edges.tolist(),
        },
        "reproj_error": {
            "mean": float(pts["error"].mean()) if n else None,
            "median": float(np.median(pts["error"])) if n else None,
            "p95": float(np.percentile(pts["error"], 95)) if n else None,
        },
    }
    return metrics


def main():
    ap = argparse.ArgumentParser(description="COLMAP sparse 点群メトリクス")
    ap.add_argument("--sparse", required=True, help="sparse/0 ディレクトリ")
    ap.add_argument("--out", required=True, help="出力 json パス")
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--hist-bins", type=int, default=30)
    args = ap.parse_args()

    m = compute_metrics(args.sparse, knn=args.knn, hist_bins=args.hist_bins)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

    print(f"[cloud_metrics] {args.sparse}")
    print(f"  num_points = {m['num_points']:,}")
    print(f"  num_images = {m['num_images']:,}")
    print(f"  bbox size  = {np.array(m['bbox']['size']).round(3).tolist()} "
          f"(diag {m['bbox']['diagonal']:.3f})")
    of = m["outside_camera_hull_fraction"]
    print(f"  outside camera hull = {ofx(of)}  "
          f"(dilate0.25 {ofx(m['outside_camera_hull_fraction_dilate25'])})")
    print(f"  kNN mean dist = {m['knn_mean_distance']['mean']:.4f} "
          f"(p99 {m['knn_mean_distance']['p99']:.4f})")
    print(f"  -> {args.out}")


def ofx(v):
    return "n/a" if v is None else f"{v*100:.2f}%"


if __name__ == "__main__":
    main()
