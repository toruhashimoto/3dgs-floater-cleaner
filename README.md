# 3DGS Floater Cleaner（3DGSフロータークリーナー）

3D Gaussian Splatting (3DGS) の **フローター（視点を変えると見える、空中に浮いた不要なガウシアン）を抑えて再構成する** Windows デスクトップツール。RealityScan で撮影・アライメントしたデータ（COLMAP 形式エクスポート）を選び、**1クリックでフローターの少ない `.ply` を出力**します。

> One-click Windows tool that reconstructs 3D Gaussian Splatting scenes with far fewer floaters — by **training with a validated geometry regularizer (`scale_reg`)**, not by pre-train point-cloud cleanup. Input is a RealityScan→COLMAP export.

**ダウンロード → [Releases](https://github.com/toruhashimoto/3dgs-floater-cleaner/releases)**

---

## これは何をするツール？

3DGS の学習結果には、どの視点からも観測が薄い領域に「フローター（浮遊ゴースト）」が現れます。本ツールは LichtFeld を **MRNF 戦略**で学習し、**詳細度プリセット（30k/1M〜150k/8M・既定 105k/5M）** で高精細に再構成しつつ、学習時の幾何正則化 **`scale_reg`** で浮遊フローターを抑えます。

> `scale_reg=0.02` による floater **−75〜82%**（PSNR/SSIM ほぼ不変）は **MCMC での検証値**です。既定の MRNF は高精細を優先する設定で、floater 数は学習後に計測表示しますが低減は保証しません（検証の詳細は [FINDINGS.md](FINDINGS.md)）。

- **入力**：RealityScan のアライメントを **COLMAP 形式でエクスポートしたフォルダ**（`images/` + `sparse/0/`）
- **出力**：低フローターの 3DGS `.ply`
- **学習エンジン**：[LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)（**MRNF**、詳細度 30k/1M〜150k/8M）をヘッドレス実行

---

## クイックスタート

1. [Releases](https://github.com/toruhashimoto/3dgs-floater-cleaner/releases) から取得して展開（または本リポジトリをクローン）。
2. `app\run_desktop.bat` をダブルクリック（初回のみ Python 仮想環境を自動作成）。
   - デスクトップにショートカットを作るなら `powershell -ExecutionPolicy Bypass -File app\install_desktop_shortcut.ps1`
3. **データフォルダ**（RealityScan→COLMAP：`images/` + `sparse/0`）を選択。
4. 既定のまま「学習開始」（詳細度=**最高詳細 105k/5M**〔既定〕/ 戦略=MRNF / 強度=標準 `scale_reg=0.02` / floater計測 ON）。
5. 完了後：**「SuperSplatで開く」**で編集・SOG 出力、**「出力フォルダを開く」**で `.ply` を確認。

### 必要環境
- Windows + NVIDIA GPU（CUDA）
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)（自動検出。`%LICHTFELD_EXE%` か GUI の参照欄で指定可）
- 標準 Python 3.x（Tk 同梱版）。floater 計測用の numpy/scipy はランチャが自動導入。

---

## 機能

- **詳細度プリセット5段**：30k/1M・60k/2M・90k/3.5M・**105k/5M〔既定〕**・150k/8M。`max_cap`（最大ガウシアン数）と密度化停止点 `stop_refine` を iterations に応じ自動調整
- 幾何正則化 `scale_reg` でフローター抑制（強度：**標準 0.02** / 強め 0.04 / オフ）。MCMC で −75〜82% を確認（MRNF では未検証・計測のみ）
- **MRNF ディテール強化トグル**（既定 ON）：`--use-error-map` / `--use-edge-map` で SSIM 誤差・Sobel エッジ領域に密度化を集中させ高精細化
- **実%進捗バー**（再描画 4Hz 上限で低負荷）、学習後に **floater 数を自動計測**して表示
- **SuperSplat 連携**：完了後ボタンで出力 `.ply` を SuperSplat（編集・SOG 出力可）で開く。ローカル exe があれば直接起動、無ければ web 版を開き `.ply` をエクスプローラで選択表示（ドラッグ&ドロップ）
- **before/after 比較**（任意・既定 OFF・時間2倍）：baseline と並べて floater 削減率を表示

---

## なぜこの方式なのか（検証で得た根拠）

「学習前に点群を綺麗にすればフローターが減る」という直感は、A/B 検証の結果 **否定** されました。

- **学習前の点群クリーンアップ（SOR・カメラ凸包外点の除去）は無効**。MCMC が学習中に背景構造を作り直すため、初期点をいじっても吸収される（washes out）。
- **効くのは学習時の `scale_reg`**（各ガウシアンの世界スケールへの L1 ペナルティ）。大きく拡散したフローターを縮めて表面に密着 or 消滅させる。`scale_reg=0.02` で floater **−75〜82%**、PSNR/SSIM ほぼ不変。`opacity_reg` は併用しない（品質が落ち、指標を交絡させる）。
- 2つの独立データセット × 短時間/本番（15000 / 30000 iter）× ノイズフロア測定で再現を確認。
- カメラ凸包の外側に広がる「背景構造」は除去対象のフローターではなく正当なシーンで、点群除去でも深度正則化でも減りません（撮影/ROI の問題）。

詳細レポート：**[FINDINGS.md](FINDINGS.md)**

> 注：`sano` / `toyota` は検証に使った**データセット名**にすぎず、ツールの対象・用途とは無関係です。任意の RealityScan→COLMAP データに使えます。

---

## リポジトリ構成

```
app/
  desktop_app.py              デスクトップ本体（Tkinter）
  run_desktop.bat             ダブルクリック起動ランチャ
  install_desktop_shortcut.ps1 デスクトップショートカット作成（任意）
  app.py                      旧 Gradio 検証 UI（研究記録として残置）
configs/
  lichtfeld_mrnf_30k_1M.json … 150k_8M.json  詳細度プリセット5段（MRNF, scale_reg=0.02）
  lichtfeld_scalereg02_prod30k.json  build_config のテンプレ兼検証 base（MCMC/30k）
  lichtfeld_scalereg02.json          検証用（MCMC/短時間 15000 iter）
scripts/
  gen_detail_configs.py        詳細度プリセット → 独立 config(JSON) を生成
  （ほか検証ハーネス：floater 指標 / A-B 比較 / COLMAP・PLY I/O 等）
tests/                         ユニットテスト
FINDINGS.md                    検証の結論（floater(a) は scale_reg で解決 / floater(b) は背景）
GSPLAT_SETUP.md                深度正則化(gsplat)実験の再現メモ（上級・任意）
docs/superpowers/specs/        ツールの設計書
```
データ（画像・COLMAP）と学習出力はリポジトリに含めません（`.gitignore`）。

---

## 開発者向け：検証の再現

研究ハーネス（点群クリーンアップ・A/B 学習比較・floater 指標）は `scripts/` にあります。ユニットテストは `python tests/test_desktop_app.py`。検証手順と数値は [FINDINGS.md](FINDINGS.md)、Blackwell GPU で gsplat を用いた深度実験は [GSPLAT_SETUP.md](GSPLAT_SETUP.md) を参照。

## クレジット
- 学習エンジン：[LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)（MrNeRF）
- ビューア／編集・SOG 出力：[SuperSplat](https://github.com/playcanvas/supersplat)（PlayCanvas）

各ソフトウェアのライセンスはそれぞれのプロジェクトに従ってください。
