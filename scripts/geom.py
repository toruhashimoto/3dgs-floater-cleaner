"""
geom.py — 点群の幾何ユーティリティ（SOR・凸包内外判定・近傍距離・シーンスケール）。

SOR は Open3D が入っていれば Open3D の statistical_outlier_removal を使用し、
無ければ scipy.cKDTree で「Open3D と同一アルゴリズム」のフォールバックを使う。

Open3D の SOR アルゴリズム（再現条件）:
  各点について、自分を除く nb_neighbors 個の最近傍までの平均距離 d_i を求める。
  全点の d_i の平均 mu と標準偏差 sigma を取り、
  d_i > mu + std_ratio * sigma の点を外れ点として除去する。
"""
from __future__ import annotations
import numpy as np

try:
    from scipy.spatial import cKDTree, ConvexHull, Delaunay
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

import os as _os_env
# SOR_DISABLE_OPEN3D=1 で Open3D を完全に無効化（壊れた/半インストール時の SIGBUS 回避）。
if _os_env.environ.get("SOR_DISABLE_OPEN3D", "0") == "1":
    _HAVE_O3D = False
else:
    try:
        import open3d as o3d
        _HAVE_O3D = True
    except Exception:
        _HAVE_O3D = False


def have_open3d():
    return _HAVE_O3D


def knn_mean_distances(points, k=20):
    """各点の、自分を除く最近傍 k 点までの平均距離 (N,) を返す（scipy.cKDTree）。"""
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy が必要です (pip install scipy)")
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    k_eff = min(k, n - 1)
    tree = cKDTree(points)
    # 自分自身(距離0)を含めて k+1 個取得し、先頭を除外
    # workers は環境変数 SOR_WORKERS で上書き可（既定1: 制限環境での segfault 回避）
    import os as _os
    _w = int(_os.environ.get("SOR_WORKERS", "1"))
    dist, _ = tree.query(points, k=k_eff + 1, workers=_w)
    dist = np.atleast_2d(dist)
    return dist[:, 1:].mean(axis=1)


def statistical_outlier_mask(points, nb_neighbors=20, std_ratio=2.0):
    """
    SOR の keep_mask (True=保持) を返す。
    Open3D があればそれを使い、無ければ scipy で同一定義を計算する。
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n == 0:
        return np.ones(0, dtype=bool)

    if _HAVE_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, keep_idx = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(keep_idx, dtype=np.int64)] = True
        return mask

    # scipy フォールバック（Open3D と同一アルゴリズム）
    d = knn_mean_distances(points, k=nb_neighbors)
    mu = float(d.mean())
    sigma = float(d.std())
    thr = mu + std_ratio * sigma
    return d <= thr


def bbox(points):
    """min(3,), max(3,), size(3,), diagonal(float) を返す。"""
    points = np.asarray(points, dtype=np.float64)
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    size = mx - mn
    diag = float(np.linalg.norm(size))
    return mn, mx, size, diag


def scene_diagonal(points):
    """点群（またはカメラ中心群）の bbox 対角長。シーンスケールの基準。"""
    _, _, _, diag = bbox(points)
    return diag


def dilate_points(points, factor):
    """重心まわりに points を (1+factor) 倍に膨張させた座標を返す。"""
    points = np.asarray(points, dtype=np.float64)
    c = points.mean(axis=0)
    return c + (points - c) * (1.0 + factor)


def inside_hull_mask(query_points, hull_points, dilate=0.0):
    """
    query_points が hull_points の凸包内部にあるか (N,) bool。
    dilate>0 なら凸包を重心まわりに膨張させてから判定。
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy が必要です")
    query_points = np.asarray(query_points, dtype=np.float64)
    hp = np.asarray(hull_points, dtype=np.float64)
    if dilate and dilate != 0.0:
        hp = dilate_points(hp, dilate)
    try:
        dela = Delaunay(hp)
        return dela.find_simplex(query_points) >= 0
    except Exception:
        # 退化（共面など）時は bbox 内外で代替
        mn, mx = hp.min(axis=0), hp.max(axis=0)
        return np.all((query_points >= mn) & (query_points <= mx), axis=1)


def outside_hull_fraction(query_points, hull_points, dilate=0.0):
    """凸包外側にある点の割合 (0-1)。"""
    inside = inside_hull_mask(query_points, hull_points, dilate=dilate)
    n = len(inside)
    if n == 0:
        return 0.0
    return float(np.count_nonzero(~inside)) / n


def percentile_bbox(points, lo=2.5, hi=97.5):
    """各軸の lo–hi パーセンタイルで囲った bbox (min(3,), max(3,))。ROI 自動推定用。"""
    points = np.asarray(points, dtype=np.float64)
    mn = np.percentile(points, lo, axis=0)
    mx = np.percentile(points, hi, axis=0)
    return mn, mx


def inside_bbox_mask(points, mn, mx):
    """points が [mn, mx] の bbox 内にあるか (N,) bool。"""
    points = np.asarray(points, dtype=np.float64)
    mn = np.asarray(mn, dtype=np.float64)
    mx = np.asarray(mx, dtype=np.float64)
    return np.all((points >= mn) & (points <= mx), axis=1)
