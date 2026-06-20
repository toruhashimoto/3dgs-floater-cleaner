# 3DGS Floater Cleaner（3DGSフロータークリーナー）

3D Gaussian Splatting (3DGS) の **フローター（視点を変えると見える、空中に浮いた不要なガウシアン）を抑えて再構成する** Windows デスクトップツール。RealityScan で撮影・アライメントしたデータ（COLMAP 形式エクスポート）を選び、**1クリックでフローターの少ない `.ply` を出力**します。

> One-click Windows tool that reconstructs 3D Gaussian Splatting scenes with far fewer floaters — by **training with a validated geometry regularizer (`scale_reg`)**, not by pre-train point-cloud cleanup. Input is a RealityScan→COLMAP export.

**ダウンロード → [Releases](https://github.com/toruhashimoto/sano-floater-cleanup/releases)**

---

## これは何をするツール？

3DGS の学習結果には、どの視点からも観測が薄い領域に「フローター（浮遊ゴースト）」が現れます。本ツールは、検証で有効と確認した **学習時の幾何正則化 `scale_reg`** を使って学習し、**浮遊フローターを 75〜82% 低減**します（品質指標 PSNR/SSIM はほぼ不変）。

- **入力**：RealityScan のアライメントを **COLMAP 形式でエクスポートしたフォルダ**（`images/` + `sparse/0/`）
- **出力**：低フローターの 3DGS `.ply`
- **学習エンジン**：[LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)（MCMC）をヘッドレス実行

---

## クイックスタート

1. [Releases](https://github.com/toruhashimoto/sano-floater-cleanup/releases) から取得して展開（または本リポジトリをクローン）。
2. `app\run_desktop.bat` をダブルクリック（初回のみ Python 仮想環境を自動作成）。
   - デスクトップにショートカットを作るなら `powershell -ExecutionPolicy Bypass -File app\install_desktop_shortcut.ps1`
3. **データフォルダ**（RealityScan→COLMAP：`images/` + `sparse/0`）を選択。
4. 既定のまま「学習開始」（強度=標準 `scale_reg=0.02` / 品質=本番 30000 / floater計測 ON）。
5. 完了後：**「SuperSplatで開く」**で編集・SOG 出力、**「出力フォルダを開く」**で `.ply` を確認。

### 必要環境
- Windows + NVIDIA GPU（CUDA）
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)（自動検出。`%LICHTFELD_EXE%` か GUI の参照欄で指定可）
- 標準 Python 3.x（Tk 同梱版）。floater 計測用の numpy/scipy はランチャが自動導入。

---

## 機能

- 検証済み `scale_reg` 設定で低フローター `.ply` を出力（強度プリセット：**標準 0.02** / 強め 0.04 / オフ）
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
  lichtfeld_scalereg02_prod30k.json  推奨学習設定（scale_reg=0.02, 30000 iter）
  lichtfeld_scalereg02.json          同（短時間 15000 iter）
scripts/                       検証ハーネス（floater 指標 / A-B 比較 / COLMAP・PLY I/O 等）
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
