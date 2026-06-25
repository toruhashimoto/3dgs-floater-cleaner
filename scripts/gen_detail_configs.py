"""configs/lichtfeld_mrnf_*.json を DETAIL プリセットから生成（コードと同期・再現可能）。

desktop_app の DETAIL / build_config / STRATEGY を唯一の真実として、5段の詳細度プリセットを
独立した LichtFeld config(JSON) に書き出す。GUI を使わない CLI/バッチ実行用:

  LichtFeld-Studio --headless --config configs/lichtfeld_mrnf_105k_5M.json \
      -d <COLMAPデータ> -o <出力> -r 1 --use-error-map --use-edge-map

メモ:
  - scale_reg は推奨の 0.02 を焼き込む（フローター抑制の既定。必要なら各ファイルを編集）。
  - --use-error-map / --use-edge-map は LichtFeld の CLI フラグ（config キーではない）なので
    高精細化したい場合は実行時に付与する（GUI の「MRNF ディテール強化」と同等）。
  - eval は無効（全画像で学習し最終 iter の .ply を保存する apply モード）。

使い方:  python scripts/gen_detail_configs.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "app"))
import desktop_app as da  # noqa: E402

SCALE_REG = 0.02  # 推奨強度（フローター抑制の既定）


def slug(iters: int, cap: int) -> str:
    """iter/cap から安定したファイル名を作る（ラベル文字列には依存しない）。"""
    k = f"{iters // 1000}k"
    if cap % 1_000_000 == 0:
        m = f"{cap // 1_000_000}M"
    else:
        m = f"{cap / 1_000_000:.1f}M".replace(".", "p")
    return f"lichtfeld_mrnf_{k}_{m}.json"


def generate(out_dir: str | None = None) -> list[str]:
    """DETAIL の各プリセットを out_dir(既定 configs/) に独立 config として書き出す。"""
    out_dir = out_dir or os.path.join(_REPO, "configs")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    with tempfile.TemporaryDirectory() as tmp:
        for iters, cap in da.DETAIL.values():
            # build_config が strategy=mrnf / max_cap / stop_refine 比例延長 / eval 無効を反映
            src = da.build_config(SCALE_REG, iters, tmp, max_cap=cap, strategy=da.STRATEGY)
            dst = os.path.join(out_dir, slug(iters, cap))
            shutil.copyfile(src, dst)
            written.append(dst)
    return written


if __name__ == "__main__":
    for p in generate():
        print("wrote", os.path.relpath(p, _REPO))
