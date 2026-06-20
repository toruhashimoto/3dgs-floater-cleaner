"""
hull_clean.py — 学習前クリーンアップ（実験②: カメラ凸包外側点の明示除去）。

SOR（孤立点のみ除去）が効かなかったことを受け、より積極的に
「(膨張)カメラ凸包の外側にある点」を init から丸ごと除去して A/B するためのツール。
SOR と同じく faithful 書き戻し（保持点の生行をそのまま）で、唯一変数=points3D に保つ。

生成:
  <outdir>/hull_sor/sparse/0/{cameras,images,points3D}.txt   （inside-hull のみ保持）
  <outdir>/metrics/hull_sweep.json                            （expand 別の保持/除去数）
  <outdir>/metrics/hull_removed.ply                           （除去点のみ）

使い方:
  python hull_clean.py --sparse F:/RealityScan/sano/sparse/0 \
      --outdir F:/RealityScan/sano/exp --expand 0.25
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import numpy as np

import colmap_io as cio
import geom


def main():
    ap = argparse.ArgumentParser(description="カメラ凸包外側点の明示除去（実験②）")
    ap.add_argument("--sparse", required=True, help="入力 sparse/0")
    ap.add_argument("--outdir", required=True, help="出力ベース（metrics/ と hull_sor/ を作る）")
    ap.add_argument("--expand", type=float, default=0.25,
                    help="カメラ凸包の膨張率（この外側を除去）。floater(b) と同じ既定 0.25")
    ap.add_argument("--sweep", type=str, default=None,
                    help="例: 0.0,0.25,0.5 — 保持/除去数のみ確認（hull_sor は --expand で生成）")
    args = ap.parse_args()

    images = cio.read_images_text(os.path.join(args.sparse, "images.txt"))
    cam_centers = cio.camera_centers(images)
    pts = cio.read_points3D_text(os.path.join(args.sparse, "points3D.txt"))
    xyz = pts["xyz"]
    n = xyz.shape[0]

    metrics_dir = os.path.join(args.outdir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    expands = [float(x) for x in args.sweep.split(",")] if args.sweep else [args.expand]
    table = []
    print(f"[hull_clean] points={n:,}  cameras={len(cam_centers):,}  open3d={geom.have_open3d()}")
    print(f"  {'expand':>7} {'inside':>10} {'outside':>10} {'outside%':>9}")
    for e in expands:
        inside = geom.inside_hull_mask(xyz, cam_centers, dilate=e)
        n_in = int(np.count_nonzero(inside))
        n_out = n - n_in
        print(f"  {e:>7.2f} {n_in:>10,} {n_out:>10,} {n_out / n * 100:>8.2f}%")
        table.append({"expand": e, "inside": n_in, "outside": n_out,
                      "outside_pct": n_out / n * 100.0})
    with open(os.path.join(metrics_dir, "hull_sweep.json"), "w", encoding="utf-8") as f:
        json.dump({"sparse": os.path.abspath(args.sparse), "total_points": n,
                   "cameras": int(len(cam_centers)), "table": table}, f,
                  ensure_ascii=False, indent=2)

    # 採用 expand で hull_sor を生成（inside-hull のみ保持）
    keep = geom.inside_hull_mask(xyz, cam_centers, dilate=args.expand)
    removed = ~keep
    dst_sparse = os.path.join(args.outdir, "hull_sor", "sparse", "0")
    os.makedirs(dst_sparse, exist_ok=True)
    for fn in ("cameras.txt", "images.txt"):
        src = os.path.join(args.sparse, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_sparse, fn))
    kept = cio.write_points3D_text(os.path.join(dst_sparse, "points3D.txt"),
                                   pts["raw"], keep, header=pts["header"])
    ply_path = os.path.join(metrics_dir, "hull_removed.ply")
    cio.write_ply_xyzrgb(ply_path, xyz[removed], pts["rgb"][removed], binary=True)

    print(f"\n  ADOPTED expand={args.expand}")
    print(f"  hull_sor points3D -> {dst_sparse}  (kept {kept:,} / removed {int(np.count_nonzero(removed)):,})")
    print(f"  hull_removed.ply -> {ply_path}")
    with open(os.path.join(metrics_dir, "hull_adopted.json"), "w", encoding="utf-8") as f:
        json.dump({"expand": args.expand, "total_points": n, "kept": int(kept),
                   "removed": int(np.count_nonzero(removed)),
                   "removed_pct": float(np.count_nonzero(removed) / n * 100.0),
                   "cleaned_sparse": os.path.abspath(dst_sparse),
                   "removed_ply": os.path.abspath(ply_path)}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
