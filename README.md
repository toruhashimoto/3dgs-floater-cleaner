# sano-floater-cleanup

学習前点群クリーンアップ（Statistical Outlier Removal, SOR）による **3D Gaussian Splatting (3DGS) のフローター低減を、再現可能な A/B 学習比較で定量検証する**ためのツール一式とデスクトップアプリ（Gradio）。

対象データは RealityScan が COLMAP text 形式で出力した `sano`（cameras / images / points3D）。

> （初期の主目的。**検証は完了済み** → 下記「結論」を参照）

---

## ★ 結論（2026-06-20）と実用ツール

検証の結果（詳細は **[FINDINGS.md](FINDINGS.md)**）:
- **学習前 SOR / カメラ凸包外除去（init 側介入）は効かない** — MCMC が背景を再成長させ washes out（sano/toyota で確認）。
- **効くのは学習時の幾何正則化 `scale_reg`** — LichtFeld で `scale_reg=0.02` にすると floater(a) を **−75〜82%**、PSNR/SSIM はほぼ不変（15k/30k・2データセットで再現）。`opacity_reg` は上げない（PSNR 劣化）。
- floater(b)（凸包外 ~82–85%）は**正当な背景**で、深度正則化（外部 gsplat）でも減らない（`GSPLAT_SETUP.md`）。

→ この成果を使う **デスクトップツール「FloaterClean Trainer」** を同梱（下記）。

## FloaterClean Trainer（ワンクリック低フローター学習・推奨）

RealityScan を COLMAP 形式でエクスポートしたフォルダ（`images/` + `sparse/0`、`F:\RealityScan\sano` と同構造）を選び、検証済み `scale_reg` 設定で LichtFeld をヘッドレス学習 → 低フローターの `.ply` を出力。ブラウザ不要のネイティブ窓（Tkinter）。

```
app\run_desktop.bat               # ダブルクリックで起動（初回は venv 自動作成）
app\install_desktop_shortcut.ps1  # 任意: デスクトップにショートカット作成
```
- **入力**: RealityScan→COLMAP エクスポートフォルダ（アライメント再構成済み前提）。
- **強度**: 標準 `scale_reg=0.02`（推奨）/ 強め 0.04 / オフ。**品質**: 本番 30000（既定）/ 短時間 15000。
- 学習後に **floater(a) 数を計測表示**（既定 ON）。LichtFeld 実行ファイルは自動検出（`%LICHTFELD_EXE%` → `F:\LichtFeld-Studio\...` → PATH、手動指定可）。
- **前提**: NVIDIA GPU + LichtFeld Studio。設計書: `docs/superpowers/specs/2026-06-20-floaterclean-desktop-tool-design.md`。

> 旧 `app/app.py`（Gradio・SOR 中心）は検証記録として残置。実用は FloaterClean Trainer を使用。

---

## 何ができるか（工程と GPU 要否）

| 工程 | 内容 | GPU | このリポジトリ |
|---|---|:---:|:---:|
| T0 | データ実体の確認 | 不要 | ✅ アプリ/CLI |
| T2 | baseline 点群メトリクス | 不要 | ✅ |
| T3 | SOR スイープ・採用・removed_points.ply | 不要 | ✅ |
| T4 | A/B 学習（LichtFeld Studio, MCMC） | **必要** | ⚙️ コマンド生成のみ |
| T5 | フローター指標・PSNR/SSIM 比較・判定 | 不要 | ✅（学習出力を入力） |

CPU で完結する T0–T3 と T5 はこのツールで実行できる。**T4 の学習だけは CUDA GPU が必要**で、アプリが生成するコマンドをユーザーが実行する。

---

## セットアップと起動

Windows はリポジトリ直下の `run.bat` をダブルクリック（venv 作成 → 依存導入 → アプリ起動）。Linux/macOS は `./run.sh`。

手動の場合:

```bash
pip install -r requirements.txt
python app/app.py     # http://127.0.0.1:7860 が開く
```

Open3D は任意。未導入でも scipy による同一アルゴリズムの SOR で動作する（ブリーフ §5 T3 の Open3D `statistical_outlier_removal` と同一定義）。

---

## GitHub へ push（private）

このフォルダはそのまま push できる状態（コードのみ。データ・実験出力は `.gitignore` 済み）。
`gh` CLI があれば private リポジトリ作成から push まで自動:

```bash
# Windows: ダブルクリック
push_to_github.bat
# Linux/macOS:
./push_to_github.sh
```

手動の場合:

```bash
cd sano-floater-cleanup
git init && git add . && git commit -m "initial: sano-floater-cleanup"
git branch -M main
gh repo create sano-floater-cleanup --private --source=. --remote=origin --push
# gh が無ければ GitHub で空の private repo を作成後:
#   git remote add origin https://github.com/<account>/sano-floater-cleanup.git && git push -u origin main
```

## アプリの使い方

1. **T0 データ確認** — `sparse/0` のパスを入れて点数・bbox・カメラ数を確認。
2. **T2 メトリクス** — kNN 平均距離ヒストグラム、カメラ凸包外側点の割合などを計算し `metrics/baseline.json` に保存。
3. **T3 SOR** — `std_ratio` をスイープして総削除率と **ROI 内削除率**（既定上限 1.0%）を表示。採用比率を決めて `cleaned_sor` と `removed_points.ply` を生成。3D 散布図で削除点が空中ノイズ中心であることを目視確認。
4. **T4 学習コマンド生成** — `cleaned 学習プロジェクト` を準備（`images/` をリンクして 1.6GB の複製を回避）し、baseline と cleaned を**同一設定**で回す 2 本のコマンドを生成。これを GPU マシンで実行。
5. **T5 A/B 比較** — 学習出力（`train_baseline` / `train_cleaned`）を指定し、フローター指標・総ガウシアン数・PSNR/SSIM を 1 表に集約して §1.5 の判定を表示。

---

## CLI（アプリを使わない場合）

```bash
# T2 メトリクス
python scripts/cloud_metrics.py --sparse F:/RealityScan/sano/sparse/0 \
    --out F:/RealityScan/sano/exp/metrics/baseline.json

# T3 SOR スイープ + 採用(std_ratio=2.0)
python scripts/sor_clean.py --sparse F:/RealityScan/sano/sparse/0 \
    --outdir F:/RealityScan/sano/exp --nb-neighbors 20 \
    --sweep 1.5,2.0,3.0 --adopt 2.0 --roi-max-removal 1.0

# T4 前: cleaned 学習プロジェクト準備（images をリンク）
python scripts/prepare_projects.py --orig F:/RealityScan/sano \
    --cleaned-sparse F:/RealityScan/sano/exp/cleaned_sor/sparse/0 \
    --proj F:/RealityScan/sano/exp/cleaned_proj

# T4 学習（GPU。フラグは各自のビルドで --help 確認）
gaussian_splatting_cuda -d F:/RealityScan/sano               -o F:/RealityScan/sano/exp/train_baseline --strategy mcmc -i 15000 --max-cap 1000000
gaussian_splatting_cuda -d F:/RealityScan/sano/exp/cleaned_proj -o F:/RealityScan/sano/exp/train_cleaned  --strategy mcmc -i 15000 --max-cap 1000000

# T5 比較
python scripts/floater_metrics.py --ply F:/.../train_baseline/point_cloud.ply \
    --sparse F:/RealityScan/sano/sparse/0 --out F:/.../metrics/floater_baseline.json
python scripts/compare.py --baseline F:/.../train_baseline --cleaned F:/.../train_cleaned \
    --sparse F:/RealityScan/sano/sparse/0 --out F:/.../metrics/report.md
```

---

## 構成

```
scripts/
  colmap_io.py        COLMAP text / PLY の読み書き（track 保持の faithful 書き戻し）
  geom.py             SOR・凸包内外判定・kNN・シーンスケール（Open3D / scipy フォールバック）
  cloud_metrics.py    T1/T2 点群メトリクス
  sor_clean.py        T3 SOR クリーンアップ＋スイープ＋ROI ゲート
  floater_metrics.py  T1/T5 学習後フローター指標（opacity×scale, 凸包外側）
  compare.py          T5 A/B 比較と §1.5 判定
  prepare_projects.py T4 前の学習プロジェクト準備（images リンク）
  make_synthetic.py   GPU 不要の自己検証用 合成 COLMAP / 合成 3DGS .ply 生成
configs/
  clean.yaml          SOR パラメータ（sano 実測値つき）
  train.yaml          A/B 学習設定テンプレート
app/app.py            Gradio デスクトップ UI
run.bat / run.sh      起動スクリプト
```

---

## 指標の定義（操作的定義, ブリーフ §4）

- **フローター指標(a)**: `sigmoid(opacity) < τ_op`(既定 0.05) かつ `max(exp(scale)) > σ`(既定: シーン対角の 2%) のガウシアン数。
- **フローター指標(b)**: カメラ位置の凸包を 25% 膨張させた領域の外側に中心を持つガウシアン数。
- **品質**: 保留ビューの PSNR / SSIM、総ガウシアン数。
- **判定(§1.5)**: フローター指標(a) が改善し、**かつ** ΔPSNR ≥ -0.3dB なら PASS。

3DGS の .ply は opacity を logit、scale を log で格納する慣例に従い、`floater_metrics.py` 内で `sigmoid` / `exp` を適用する。

---

## sano 実データの確定結果（2026-06-20, CPU 工程）

詳細は [RESULTS.md](RESULTS.md)。要点:

- 258,394 点 / 3,297 画像。**スパース点の 87.5% がカメラ凸包の外側**（被写体周囲を超えた遠方/背景構造が多い）。
- SOR(nb=20): std_ratio 1.5 / 2.0 / 3.0 で総削除 0.44% / 0.31% / 0.22%、**ROI 内削除はいずれも 0.00%**。採用 std=2.0（804 点削除）。
- removed_points.ply は **92.9% が ROI 対角の外**、ROI 中心からの距離 中央値 114m / 最大 782m。空中・遠方ノイズ中心であることを確認。
- **重要**: フローター低減効果は T4/T5 の A/B 学習でしか確認できない（未実施＝要 GPU）。「点が削れた」ことを成功と早合点しない。
