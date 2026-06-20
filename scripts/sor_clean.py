"""
sor_clean.py — 学習前クリーンアップ（T3）。

COLMAP points3D に Statistical Outlier Removal (SOR) を適用し、
  - cleaned_sor/sparse/0/  に保持点を書き戻し（cameras/images はコピー）
  - removed_points.ply に「削除点のみ」を出力
を行う。総削除率と「車体ROI内削除率」を算出する。

ROI（車体）の既定推定:
  カメラ凸包内部にある点だけを取り、その各軸 roi-lo–roi-hi パーセンタイルで
  囲った bbox を ROI とする（§4「シーン中心bbox（暫定）」）。
  --roi xmin,ymin,zmin,xmax,ymax,zmax で明示指定も可能。

2つのモード:
  単発:  --std-ratio 2.0            → cleaned_sor を生成
  スイープ: --sweep 1.5,2.0,3.0     → 各比率の削除率/ROI内削除率を表に。
           --adopt 2.0 を併用すると、その比率で cleaned_sor も生成。

使い方例:
  python sor_clean.py --sparse <in/sparse/0> --outdir <exp> \
      --nb-neighbors 20 --sweep 1.5,2.0,3.0 --adopt 2.0 \
      --roi-max-removal 1.0
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import numpy as np

import colmap_io as cio
import geom


def estimate_roi(xyz, cam_centers, lo, hi):
    """カメラ凸包内部の点から ROI(bbox) を推定。返り値 (mn, mx, method)。"""
    try:
        inside = geom.inside_hull_mask(xyz, cam_centers, dilate=0.0)
        sub = xyz[inside]
        if sub.shape[0] < 50:  # 内部点が少なすぎる→全点で代替
            sub = xyz
            method = f"percentile[{lo},{hi}] of ALL points (hull interior too small)"
        else:
            method = f"percentile[{lo},{hi}] of points inside camera hull"
    except Exception:
        sub = xyz
        method = f"percentile[{lo},{hi}] of ALL points (hull failed)"
    mn, mx = geom.percentile_bbox(sub, lo=lo, hi=hi)
    return mn, mx, method


def run_one(xyz, nb_neighbors, std_ratio, roi_min, roi_max):
    """1つの std_ratio で SOR を実行し、統計を返す。"""
    keep = geom.statistical_outlier_mask(xyz, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    removed = ~keep
    n = xyz.shape[0]
    n_removed = int(np.count_nonzero(removed))

    in_roi = geom.inside_bbox_mask(xyz, roi_min, roi_max)
    n_roi = int(np.count_nonzero(in_roi))
    n_roi_removed = int(np.count_nonzero(removed & in_roi))
    roi_removal_rate = (n_roi_removed / n_roi * 100.0) if n_roi else 0.0

    return {
        "std_ratio": std_ratio,
        "nb_neighbors": nb_neighbors,
        "total_points": n,
        "removed": n_removed,
        "total_removal_rate_pct": n_removed / n * 100.0 if n else 0.0,
        "roi_points": n_roi,
        "roi_removed": n_roi_removed,
        "roi_removal_rate_pct": roi_removal_rate,
        "keep_mask": keep,  # 呼び出し側で使用（json には出さない）
        "removed_mask": removed,
    }


def write_cleaned(in_sparse, outdir, pts, keep_mask):
    """cleaned_sor/sparse/0 に保持点を書き戻し、cameras/images をコピー。"""
    dst_sparse = os.path.join(outdir, "sparse", "0")
    os.makedirs(dst_sparse, exist_ok=True)
    # cameras / images はそのままコピー
    for fn in ("cameras.txt", "images.txt"):
        src = os.path.join(in_sparse, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_sparse, fn))
    kept = cio.write_points3D_text(
        os.path.join(dst_sparse, "points3D.txt"),
        pts["raw"], keep_mask, header=pts["header"])
    return dst_sparse, kept


def main():
    ap = argparse.ArgumentParser(description="学習前 SOR クリーンアップ")
    ap.add_argument("--sparse", required=True, help="入力 sparse/0")
    ap.add_argument("--outdir", required=True, help="出力ベース（metrics/ と cleaned_sor/ を作る）")
    ap.add_argument("--nb-neighbors", type=int, default=20)
    ap.add_argument("--std-ratio", type=float, default=None, help="単発モードの比率")
    ap.add_argument("--sweep", type=str, default=None, help="例: 1.5,2.0,3.0")
    ap.add_argument("--adopt", type=float, default=None, help="cleaned_sor を生成する比率")
    ap.add_argument("--roi", type=str, default=None,
                    help="xmin,ymin,zmin,xmax,ymax,zmax を明示指定")
    ap.add_argument("--roi-lo", type=float, default=2.5)
    ap.add_argument("--roi-hi", type=float, default=97.5)
    ap.add_argument("--roi-max-removal", type=float, default=1.0,
                    help="ROI内削除率の許容上限(%)")
    args = ap.parse_args()

    images = cio.read_images_text(os.path.join(args.sparse, "images.txt"))
    cam_centers = cio.camera_centers(images)
    pts = cio.read_points3D_text(os.path.join(args.sparse, "points3D.txt"))
    xyz = pts["xyz"]

    # ROI 決定
    if args.roi:
        v = [float(x) for x in args.roi.split(",")]
        roi_min = np.array(v[:3]); roi_max = np.array(v[3:])
        roi_method = "user-specified"
    else:
        roi_min, roi_max, roi_method = estimate_roi(xyz, cam_centers, args.roi_lo, args.roi_hi)

    metrics_dir = os.path.join(args.outdir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    ratios = []
    if args.sweep:
        ratios = [float(x) for x in args.sweep.split(",")]
    elif args.std_ratio is not None:
        ratios = [args.std_ratio]
    else:
        ratios = [2.0]

    print(f"[sor_clean] points={xyz.shape[0]:,}  nb_neighbors={args.nb_neighbors}")
    print(f"  ROI method: {roi_method}")
    print(f"  ROI bbox min={np.round(roi_min,3).tolist()} max={np.round(roi_max,3).tolist()}")
    print(f"  Open3D used: {geom.have_open3d()}")
    print(f"  {'std_ratio':>9} {'removed':>10} {'total%':>8} {'ROI%':>7} {'gate(<= '+str(args.roi_max_removal)+'%)':>14}")

    table = []
    results = {}
    for r in ratios:
        res = run_one(xyz, args.nb_neighbors, r, roi_min, roi_max)
        results[r] = res
        gate = "OK" if res["roi_removal_rate_pct"] <= args.roi_max_removal else "OVER"
        print(f"  {r:>9.2f} {res['removed']:>10,} {res['total_removal_rate_pct']:>7.2f}% "
              f"{res['roi_removal_rate_pct']:>6.2f}% {gate:>14}")
        row = {k: v for k, v in res.items() if k not in ("keep_mask", "removed_mask")}
        row["roi_gate_pass"] = (res["roi_removal_rate_pct"] <= args.roi_max_removal)
        table.append(row)

    sweep_out = {
        "sparse": os.path.abspath(args.sparse),
        "nb_neighbors": args.nb_neighbors,
        "roi_method": roi_method,
        "roi_min": roi_min.tolist(), "roi_max": roi_max.tolist(),
        "roi_max_removal_pct": args.roi_max_removal,
        "open3d_used": geom.have_open3d(),
        "table": table,
    }
    with open(os.path.join(metrics_dir, "sor_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(sweep_out, f, ensure_ascii=False, indent=2)

    # cleaned_sor 生成（adopt 指定 or 単発 std_ratio）
    adopt = args.adopt if args.adopt is not None else (args.std_ratio if not args.sweep else None)
    if adopt is not None:
        if adopt not in results:
            res = run_one(xyz, args.nb_neighbors, adopt, roi_min, roi_max)
        else:
            res = results[adopt]
        cleaned_dir = os.path.join(args.outdir, "cleaned_sor")
        dst_sparse, kept = write_cleaned(args.sparse, cleaned_dir, pts, res["keep_mask"])
        # removed_points.ply
        removed_xyz = xyz[res["removed_mask"]]
        removed_rgb = pts["rgb"][res["removed_mask"]]
        ply_path = os.path.join(metrics_dir, "removed_points.ply")
        cio.write_ply_xyzrgb(ply_path, removed_xyz, removed_rgb, binary=True)
        print(f"\n  ADOPTED std_ratio={adopt}")
        print(f"  cleaned points3D -> {dst_sparse}  (kept {kept:,})")
        print(f"  removed_points.ply -> {ply_path}  ({removed_xyz.shape[0]:,} pts)")
        adopt_summary = {k: v for k, v in res.items() if k not in ("keep_mask", "removed_mask")}
        with open(os.path.join(metrics_dir, "adopted.json"), "w", encoding="utf-8") as f:
            json.dump({"adopted_std_ratio": adopt, **adopt_summary,
                       "cleaned_sparse": os.path.abspath(dst_sparse),
                       "removed_ply": os.path.abspath(ply_path)},
                      f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
