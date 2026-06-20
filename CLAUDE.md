# CLAUDE.md — sano-floater-cleanup リポジトリ案内（Claude Code 用）

**まず `CODE_TASK.md` を読むこと。** それが現在の作業指示（T4 A/B学習 → T5 判定 → GitHub push）。
現状の確定結果は `RESULTS.md`。

## このリポジトリは何か
RealityScan の COLMAP 出力 `sano` に対し、**学習前点群クリーンアップ(SOR)が 3DGS のフローターを減らすかを A/B 学習で測定**するツール一式 + Gradio アプリ。CPU 工程(T0–T3)は実データで完了済み、残りは GPU 工程(T4/T5)。

## リポジトリ地図
- `scripts/colmap_io.py` … COLMAP text / PLY I/O（points3D は track 保持で faithful 書き戻し）
- `scripts/geom.py` … SOR・凸包内外・kNN。**Open3D があれば使用、無ければ scipy 同一実装**（`SOR_DISABLE_OPEN3D=1` で強制無効化）
- `scripts/cloud_metrics.py`(T2) / `sor_clean.py`(T3) / `floater_metrics.py`(T1/T5) / `compare.py`(T5)
- `scripts/prepare_projects.py` … cleaned 学習プロジェクト作成（images をリンク, 1.6GB複製回避）
- `scripts/make_synthetic.py` … GPU不要の自己検証用 合成データ生成
- `configs/clean.yaml`(sano実測値) / `configs/train.yaml`(A/B学習テンプレ)
- `app/app.py` … Gradio UI（`run.bat`/`run.sh` で起動）
- `push_to_github.bat`/`.sh` … private push ヘルパー

## パス（重要）
- データ（読取専用）：`F:\RealityScan\sano`（`images/` + `sparse/0/`）
- CPU工程の出力：`F:\RealityScan\sano\exp\`（baseline / cleaned_sor / metrics）
- 採用クリーンアップ：nb_neighbors=20, **std_ratio=2.0**

## 必読ルール（測定タスクの原則）
1. **「効果なし」は正当な結論。** 小改善を成功と早合点しない。`ΔPSNR≈0 & floater不変` なら「SOR有効でない」と明記して打ち切る。
2. **A/B の唯一変数は点群クリーンアップの有無。** seed/iter/strategy/max_cap/resize/holdout は baseline と cleaned で完全一致。
3. **LichtFeld の seed/eval/holdout フラグは版依存。`--help` で確認し、勝手に発明しない。**
4. 失敗時は修正前に「事実→仮説2つ以上→最小確認」（`CODE_TASK.md` §4）。
5. 元データと `exp/baseline`・`exp/cleaned_sor` は変更禁止。データ・学習出力はコミットしない（`.gitignore` 済み）。

## 既知の所見（CPU工程）
258,394点 / 3,297画像。**点の87.5%がカメラ凸包の外**。SOR(std=2.0)の削除は0.31%(804点)で、削除点は92.9%がROI外＝空中ノイズ中心。→ A/B の差は小さい可能性が現実的。効果が無ければ上流（学習戦略・初期化・撮り方）が本命（別タスク）。
