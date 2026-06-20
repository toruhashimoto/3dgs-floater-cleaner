"""
floater_metrics.py — 学習後フローター指標（T1 実装 / T5 で使用）。

入力は 3DGS 学習出力の .ply（gaussian splat）。COLMAP sparse はシーンスケールと
カメラ凸包の算出に使う。§4「フローター指標（学習後）」を実装:

  (a) opacity < tau_op (既定 0.05) かつ max_scale > sigma
      (既定: シーン対角の 2%) のガウシアン数。
  (b) カメラ凸包を expand 率だけ膨張させた領域の外側に中心を持つガウシアン数。

3DGS .ply の慣例:
  opacity は logit 値で格納 → 実不透明度 = sigmoid(opacity)
  scale_* は log 値で格納 → 実スケール = exp(scale_*)
  これらが見つからない場合は素の値として扱い、警告する。

使い方:
  python floater_metrics.py --ply <out/point_cloud.ply> --sparse <sparse/0> \
      --out metrics/floater_baseline.json [--tau-op 0.05] [--scale-frac 0.02] [--expand 0.25]
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np

import colmap_io as cio
import geom


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_floater_metrics(ply_path, sparse_dir, tau_op=0.05, scale_frac=0.02,
                            expand=0.25, scene_ref="cameras"):
    v = cio.read_ply_vertices(ply_path)
    n = len(next(iter(v.values())))
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    notes = []

    # --- opacity ---
    if "opacity" in v:
        opacity = _sigmoid(np.asarray(v["opacity"], dtype=np.float64))
    else:
        opacity = None
        notes.append("opacity プロパティ無し → 指標(a) は不透明度条件を無視")

    # --- scale ---
    scale_keys = [k for k in ("scale_0", "scale_1", "scale_2") if k in v]
    if scale_keys:
        scales = np.stack([np.asarray(v[k], dtype=np.float64) for k in scale_keys], axis=1)
        scales = np.exp(scales)  # log -> 実スケール
        max_scale = scales.max(axis=1)
    else:
        max_scale = None
        notes.append("scale_* プロパティ無し → 指標(a) はスケール条件を無視")

    # --- シーンスケール基準 ---
    images = cio.read_images_text(os.path.join(sparse_dir, "images.txt"))
    cam_centers = cio.camera_centers(images)
    if scene_ref == "cameras" and len(cam_centers) >= 2:
        scene_diag = geom.scene_diagonal(cam_centers)
    else:
        scene_diag = geom.scene_diagonal(xyz)
    sigma = scale_frac * scene_diag

    # --- 指標 (a): 低不透明度 かつ 大スケール ---
    cond = np.ones(n, dtype=bool)
    if opacity is not None:
        cond &= (opacity < tau_op)
    if max_scale is not None:
        cond &= (max_scale > sigma)
    metric_a = int(np.count_nonzero(cond))

    # --- 指標 (b): 膨張カメラ凸包の外側 ---
    try:
        inside = geom.inside_hull_mask(xyz, cam_centers, dilate=expand)
        metric_b = int(np.count_nonzero(~inside))
    except Exception as e:
        metric_b = None
        notes.append(f"指標(b) 凸包判定失敗: {e}")

    result = {
        "ply": os.path.abspath(ply_path),
        "total_gaussians": int(n),
        "params": {"tau_op": tau_op, "scale_frac": scale_frac,
                   "expand": expand, "scene_ref": scene_ref},
        "scene_diagonal": scene_diag,
        "scale_threshold_sigma": sigma,
        "metric_a_lowopacity_bigscale": metric_a,
        "metric_a_fraction": metric_a / n if n else 0.0,
        "metric_b_outside_dilated_hull": metric_b,
        "metric_b_fraction": (metric_b / n) if (metric_b is not None and n) else None,
        "notes": notes,
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="学習後フローター指標")
    ap.add_argument("--ply", required=True, help="3DGS 学習出力 .ply")
    ap.add_argument("--sparse", required=True, help="sparse/0（カメラ用）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau-op", type=float, default=0.05)
    ap.add_argument("--scale-frac", type=float, default=0.02)
    ap.add_argument("--expand", type=float, default=0.25)
    ap.add_argument("--scene-ref", choices=["cameras", "points"], default="cameras")
    args = ap.parse_args()

    r = compute_floater_metrics(args.ply, args.sparse, tau_op=args.tau_op,
                                scale_frac=args.scale_frac, expand=args.expand,
                                scene_ref=args.scene_ref)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)

    print(f"[floater_metrics] {args.ply}")
    print(f"  total_gaussians = {r['total_gaussians']:,}")
    print(f"  metric(a) low-opacity & big-scale = {r['metric_a_lowopacity_bigscale']:,} "
          f"({r['metric_a_fraction']*100:.3f}%)")
    mb = r["metric_b_outside_dilated_hull"]
    print(f"  metric(b) outside dilated hull   = {('n/a' if mb is None else f'{mb:,}')}")
    if r["notes"]:
        print("  notes:", "; ".join(r["notes"]))
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
