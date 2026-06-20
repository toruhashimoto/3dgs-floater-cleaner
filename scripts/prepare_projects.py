"""
prepare_projects.py — A/B 学習用の COLMAP プロジェクトを準備する（T4 前段）。

cleaned_sor は points3D だけが違うので、1.6GB の images/ を複製せず
リンク（Windows=ジャンクション / Unix=シンボリックリンク, 失敗時のみコピー）で共有する。

生成:
  <orig>/exp/cleaned_proj/
      images -> <orig>/images           （リンク）
      sparse/0/{cameras,images,points3D}.txt  （cleaned のもの）

baseline 側はオリジナル <orig> をそのまま学習に使えるので準備不要。

使い方:
  python prepare_projects.py --orig F:/RealityScan/sano \
      --cleaned-sparse F:/RealityScan/sano/exp/cleaned_sor/sparse/0 \
      --proj F:/RealityScan/sano/exp/cleaned_proj
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys


def link_dir(src, dst):
    """src ディレクトリを dst にリンク。戻り値: 'symlink'|'junction'|'copy'。"""
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.path.exists(dst) or os.path.islink(dst):
        return "exists"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # 1) symlink
    try:
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    except Exception:
        pass
    # 2) Windows junction（管理者権限不要）
    if os.name == "nt":
        try:
            subprocess.check_call(["cmd", "/c", "mklink", "/J", dst, src],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "junction"
        except Exception:
            pass
    # 3) コピー（最終手段, 容量大）
    shutil.copytree(src, dst)
    return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="オリジナル sano（images/ を持つ）")
    ap.add_argument("--cleaned-sparse", required=True, help="cleaned_sor/sparse/0")
    ap.add_argument("--proj", required=True, help="作成する学習プロジェクト dir")
    args = ap.parse_args()

    images_src = os.path.join(args.orig, "images")
    if not os.path.isdir(images_src):
        print(f"[error] images/ が見つかりません: {images_src}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.proj, exist_ok=True)
    # images をリンク
    method = link_dir(images_src, os.path.join(args.proj, "images"))
    # sparse/0 に cleaned をコピー（小さいので実体コピーで安全）
    dst_sparse = os.path.join(args.proj, "sparse", "0")
    os.makedirs(dst_sparse, exist_ok=True)
    for fn in ("cameras.txt", "images.txt", "points3D.txt"):
        s = os.path.join(args.cleaned_sparse, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst_sparse, fn))

    print(f"[prepare_projects] proj={args.proj}")
    print(f"  images: {method} -> {images_src}")
    print(f"  sparse/0: cleaned copied")
    print(f"  学習: gaussian_splatting_cuda -d {args.proj} -o <out> --strategy mcmc -i <iter> --max-cap <N>")


if __name__ == "__main__":
    main()
