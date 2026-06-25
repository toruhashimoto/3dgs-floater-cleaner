"""prune_missing_images.py — sparse モデルが参照するが images/ に存在しない
画像エントリを sparse/0/images.txt から除去する修復ユーティリティ。

用途:
  別環境へのコピー漏れ等で画像が数枚欠けていると、LichtFeld が
    [error] Failed to load dataset: [Path not found] Image 'XXXXX.jpeg' was not found ...
  で起動できない。本スクリプトは欠落エントリ（姿勢行＋POINTS2D行の2行）を
  取り除き、残りの画像で学習可能な状態にする。失うのは欠落画像のビューのみで、
  数枚であれば 3DGS 学習へはほぼ無影響。

安全策:
  - 元 images.txt は images.txt.bak へバックアップしてから書き換える
    （既存 .bak は上書きしない＝最初の原本を常に保全。二重実行も安全）。
  - points3D.txt は変更しない（3DGS 初期化は XYZ/RGB のみを使用し、
    track 内の dangling な IMAGE_ID は無害）。
  - 出力は newline="\n"（COLMAP テキストの LF を維持）。

使い方:
  python scripts/prune_missing_images.py <colmap_dir>            # 実行（バックアップ後に書換）
  python scripts/prune_missing_images.py <colmap_dir> --dry-run  # 確認のみ（書換なし）

  <colmap_dir> は images/ と sparse/0/ を含むフォルダ
  （例: D:/path/to/your_dataset/colmap）。

GUI からの利用:
  app/desktop_app.py は学習開始時に欠落を検出すると、確認のうえ apply_prune() を
  呼んで自動修復する（CLI と同一ロジック）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def _present_image_set(images_dir: str) -> set[str]:
    return {f.lower() for f in os.listdir(images_dir)}


def scan(images_txt: str, present: set[str]):
    """images.txt を1パスでストリーミング走査。
    返り値: (header_comments: list[str], total: int, missing: list[str])
    巨大な POINTS2D 行も含め全行を読むが、保持はヘッダコメントと欠落名のみ（低メモリ）。
    """
    header: list[str] = []
    total = 0
    missing: list[str] = []
    body_started = False
    with open(images_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s[0] == "#":
                if not body_started:
                    header.append(s)
                continue
            last = s.rsplit(None, 1)[-1]   # 姿勢行=画像名 / POINTS2D行=整数
            if last.lower().endswith(IMAGE_EXTS):
                body_started = True
                total += 1
                if last.lower() not in present:
                    missing.append(last)
    return header, total, missing


def _rewrite(images_txt: str, out_path: str, present: set[str], kept: int) -> None:
    """欠落画像の2行を落として out_path に書き出す。
    姿勢行の直後の行（POINTS2D 行・空行でも）をその画像の2行目として扱い対で処理する。
    """
    missing_label = None  # 直前の姿勢行が欠落画像なら True 相当（名前を保持）
    pending_pose: str | None = None
    with open(images_txt, "r", encoding="utf-8", errors="ignore") as fin, \
            open(out_path, "w", encoding="utf-8", newline="\n") as fout:
        # 更新したヘッダを先頭に出力
        for h in _new_header(images_txt, kept):
            fout.write(h + "\n")
        for line in fin:
            if pending_pose is None:
                s = line.strip()
                if not s or s[0] == "#":
                    continue  # ヘッダ・空行は再生成済みヘッダに集約
                last = s.rsplit(None, 1)[-1]
                if last.lower().endswith(IMAGE_EXTS):
                    pending_pose = line
                    missing_label = last.lower() not in present
                # 画像名で終わらない非コメント行は想定外 → スキップ
            else:
                # この行は pending_pose の POINTS2D 行（空行含む）
                if not missing_label:
                    fout.write(pending_pose if pending_pose.endswith("\n") else pending_pose + "\n")
                    fout.write(line if line.endswith("\n") else line + "\n")
                pending_pose = None
                missing_label = None


def _new_header(images_txt: str, kept: int) -> list[str]:
    """元のコメントヘッダを取り出し、画像枚数コメントを kept に差し替えて返す。"""
    header: list[str] = []
    with open(images_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s[0] != "#":
                break
            header.append(s)
    out, replaced = [], False
    for h in header:
        if h.lower().startswith("# number of images"):
            out.append(f"# Number of images: {kept}")
            replaced = True
        else:
            out.append(h)
    if not replaced:
        out.append(f"# Number of images: {kept}")
    return out


def analyze(colmap_dir: str) -> dict:
    """書き換えずに欠落状況を調べる。
    返り値 dict:
      ok(bool), error(str|None), images_txt(str|None),
      total(int 参照画像数), present(int images/ 実在数),
      missing(list[str] 欠落画像名), kept(int 除去後の参照数)
    """
    sparse = os.path.join(colmap_dir, "sparse", "0")
    images_dir = os.path.join(colmap_dir, "images")
    images_txt = os.path.join(sparse, "images.txt")
    for p in (images_dir, sparse, images_txt):
        if not os.path.exists(p):
            return {"ok": False, "error": f"見つかりません: {p}", "images_txt": None,
                    "total": 0, "present": 0, "missing": [], "kept": 0}
    present = _present_image_set(images_dir)
    _header, total, missing = scan(images_txt, present)
    return {"ok": True, "error": None, "images_txt": images_txt,
            "total": total, "present": len(present),
            "missing": missing, "kept": total - len(missing)}


def apply_prune(colmap_dir: str) -> dict:
    """欠落エントリを images.txt から除去して書き換える（GUI/CLI 共通の中核）。
    返り値 dict: changed(bool), kept(int), removed(list[str]), backup(str|None)
      欠落が無ければ changed=False で何もしない。
      images.txt.bak が既に在れば上書きせず最初の原本を保全する。
    """
    a = analyze(colmap_dir)
    if not a["ok"]:
        raise FileNotFoundError(a["error"])
    if not a["missing"]:
        return {"changed": False, "kept": a["kept"], "removed": [], "backup": None}
    images_txt = a["images_txt"]
    present = _present_image_set(os.path.join(colmap_dir, "images"))
    bak = images_txt + ".bak"
    backup = None
    if not os.path.exists(bak):      # 既存 .bak は上書きしない（最初の原本を保全）
        shutil.copy2(images_txt, bak)
        backup = bak
    tmp = images_txt + ".tmp"
    _rewrite(images_txt, tmp, present, a["kept"])
    os.replace(tmp, images_txt)
    return {"changed": True, "kept": a["kept"], "removed": a["missing"], "backup": backup}


def prune(colmap_dir: str, dry_run: bool = False) -> int:
    """CLI エントリ: 状況を表示し、--dry-run でなければ apply_prune を実行。"""
    a = analyze(colmap_dir)
    if not a["ok"]:
        print(f"[error] {a['error']}", file=sys.stderr)
        return 2
    print(f"参照画像: {a['total']} 枚 / images/ 実在: {a['present']} 個")
    if not a["missing"]:
        print("欠落なし。修復は不要です。")
        return 0
    m = a["missing"]
    show = "、".join(m[:12]) + (f" ほか{len(m) - 12}枚" if len(m) > 12 else "")
    print(f"欠落 {len(m)} 枚: {show}")
    print(f"→ 除去後の参照画像: {a['kept']} 枚")
    if dry_run:
        print("[dry-run] 書き換えは行いませんでした。")
        return 0
    r = apply_prune(colmap_dir)
    where = (f"（原本は {os.path.basename(r['backup'])} に保存）"
             if r["backup"] else "（既存 .bak を保持）")
    print(f"完了: images.txt を {r['kept']} 枚に更新{where}。")
    print("points3D.txt は変更していません。LichtFeld の学習を再実行してください。")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="COLMAP images.txt から欠落画像エントリを除去")
    ap.add_argument("colmap_dir", help="images/ と sparse/0/ を含むフォルダ")
    ap.add_argument("--dry-run", action="store_true", help="確認のみ（書き換えない）")
    args = ap.parse_args(argv)
    return prune(args.colmap_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
