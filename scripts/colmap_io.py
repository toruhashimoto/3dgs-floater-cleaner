"""
colmap_io.py — COLMAP (text format) と PLY の読み書きユーティリティ。

sano-floater-cleanup の全スクリプトが共有する基盤モジュール。
RealityScan が出力する COLMAP text 形式
  sparse/0/cameras.txt, images.txt, points3D.txt
を読み、フィルタ済み points3D の書き戻しと removed_points.ply の出力を行う。

設計方針:
- points3D は「生の行テキスト」を保持し、保持点はそのまま書き戻す
  （track 情報を一切破壊しない＝再現性・後段互換性のため）。
- 依存は numpy のみ（scipy は幾何ユーティリティ側でのみ使用）。
"""
from __future__ import annotations
import os
import numpy as np


# -----------------------------------------------------------------------------
# 回転・姿勢
# -----------------------------------------------------------------------------
def qvec2rotmat(qvec):
    """COLMAP のクォータニオン (qw, qx, qy, qz) → world->camera 回転行列 R。"""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# -----------------------------------------------------------------------------
# cameras.txt
# -----------------------------------------------------------------------------
def read_cameras_text(path):
    """cameras.txt を読む。{camera_id: dict(model,width,height,params)} を返す。"""
    cameras = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            cam_id = int(elems[0])
            cameras[cam_id] = {
                "model": elems[1],
                "width": int(elems[2]),
                "height": int(elems[3]),
                "params": np.array([float(v) for v in elems[4:]], dtype=np.float64),
            }
    return cameras


# -----------------------------------------------------------------------------
# images.txt
# -----------------------------------------------------------------------------
def read_images_text(path):
    """
    images.txt を読む。COLMAP は 1画像につき2行（姿勢行 + 2D点行）。
    返り値: {image_id: dict(qvec, tvec, camera_id, name, center)}
      center = -R^T t （ワールド座標でのカメラ中心）
    2D点行（巨大になりうる）は読み飛ばす。
    """
    images = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        elems = line.split()
        # 姿勢行: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if len(elems) >= 10 and elems[0].isdigit():
            image_id = int(elems[0])
            qvec = np.array([float(v) for v in elems[1:5]], dtype=np.float64)
            tvec = np.array([float(v) for v in elems[5:8]], dtype=np.float64)
            camera_id = int(elems[8])
            name = elems[9]
            R = qvec2rotmat(qvec)
            center = -R.T @ tvec
            images[image_id] = {
                "qvec": qvec, "tvec": tvec, "camera_id": camera_id,
                "name": name, "center": center,
            }
            i += 2  # 次の2D点行をスキップ
        else:
            i += 1
    return images


def camera_centers(images):
    """images dict から Nx3 のカメラ中心配列を返す。"""
    if not images:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array([images[k]["center"] for k in sorted(images.keys())],
                    dtype=np.float64)


# -----------------------------------------------------------------------------
# points3D.txt
# -----------------------------------------------------------------------------
def read_points3D_text(path):
    """
    points3D.txt を読む。
    返り値 dict:
      ids   : (N,) int64
      xyz   : (N,3) float64
      rgb   : (N,3) uint8
      error : (N,) float64
      raw   : list[str]  各点の生の行（改行なし）。保持点の faithful 書き戻し用。
      header: list[str]  先頭のコメント行（# ...）
    """
    ids, xyz, rgb, error, raw = [], [], [], [], []
    header = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.rstrip("\n")
            if s.startswith("#"):
                header.append(s)
                continue
            if not s.strip():
                continue
            elems = s.split()
            # POINT3D_ID X Y Z R G B ERROR TRACK[...]
            ids.append(int(elems[0]))
            xyz.append((float(elems[1]), float(elems[2]), float(elems[3])))
            rgb.append((int(elems[4]), int(elems[5]), int(elems[6])))
            error.append(float(elems[7]))
            raw.append(s)
    return {
        "ids": np.array(ids, dtype=np.int64),
        "xyz": np.array(xyz, dtype=np.float64).reshape(-1, 3),
        "rgb": np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        "error": np.array(error, dtype=np.float64),
        "raw": raw,
        "header": header,
    }


def write_points3D_text(path, raw_lines, keep_mask, header=None):
    """
    保持点の生の行をそのまま書き戻す（track を保持）。
    raw_lines: read_points3D_text の 'raw'
    keep_mask: (N,) bool。True の点だけ残す。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    kept = int(np.count_nonzero(keep_mask))
    if header is None:
        header = [
            "# 3D point list with one line of data per point:",
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        ]
    # COLMAP 慣例の点数コメントを更新して付与
    header = [h for h in header if not h.lower().startswith("# number of points")]
    header = header + [f"# Number of points: {kept}, mean track length: 0"]
    with open(path, "w", encoding="utf-8") as f:
        for h in header:
            f.write(h + "\n")
        for raw, keep in zip(raw_lines, keep_mask):
            if keep:
                f.write(raw + "\n")
    return kept


# -----------------------------------------------------------------------------
# PLY 出力（removed_points.ply など）
# -----------------------------------------------------------------------------
def write_ply_xyzrgb(path, xyz, rgb=None, binary=True):
    """xyz (N,3) と任意の rgb (N,3 uint8) を PLY で書き出す。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = xyz.shape[0]
    has_rgb = rgb is not None
    if has_rgb:
        rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)

    header = ["ply"]
    header.append("format binary_little_endian 1.0" if binary else "format ascii 1.0")
    header.append(f"element vertex {n}")
    header += ["property float x", "property float y", "property float z"]
    if has_rgb:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")

    if binary:
        with open(path, "wb") as f:
            f.write(("\n".join(header) + "\n").encode("ascii"))
            if has_rgb:
                dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                               ("r", "u1"), ("g", "u1"), ("b", "u1")])
                arr = np.empty(n, dtype=dt)
                arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            else:
                dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
                arr = np.empty(n, dtype=dt)
                arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            f.write(arr.tobytes())
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
            for k in range(n):
                if has_rgb:
                    f.write(f"{xyz[k,0]} {xyz[k,1]} {xyz[k,2]} "
                            f"{rgb[k,0]} {rgb[k,1]} {rgb[k,2]}\n")
                else:
                    f.write(f"{xyz[k,0]} {xyz[k,1]} {xyz[k,2]}\n")
    return n


def read_ply_vertices(path):
    """
    PLY を読み、頂点プロパティを {name: ndarray} で返す（ascii / binary_little_endian 対応）。
    3DGS の出力 .ply（x,y,z,opacity,scale_*,rot_*,f_dc_* ...）を解析するために使う。
    """
    with open(path, "rb") as f:
        # ヘッダ読取
        magic = f.readline().strip()
        if magic != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        fmt = None
        n_vert = 0
        props = []  # (name, dtype_char)
        in_vertex = False
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                parts = line.split()
                if parts[1] == "vertex":
                    n_vert = int(parts[2])
                    in_vertex = True
                else:
                    in_vertex = False
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                props.append((parts[2], parts[1]))
            elif line.startswith("end_header"):
                break

        ply_to_np = {
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
            "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
            "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
            "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
        }
        names = [p[0] for p in props]

        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n_vert)
            data = np.atleast_2d(data)
            return {nm: data[:, i] for i, nm in enumerate(names)}
        else:
            dt = np.dtype([(p[0], ply_to_np[p[1]]) for p in props])
            buf = f.read(dt.itemsize * n_vert)
            arr = np.frombuffer(buf, dtype=dt, count=n_vert)
            return {nm: np.asarray(arr[nm]) for nm in names}
