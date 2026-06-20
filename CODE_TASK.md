# Claude Code タスクブリーフ：sano フローター低減 A/B 学習の実行と判定（T4–T5 + push）

このリポジトリの CPU 工程（T0–T3）は実データで完了済み。**残りは GPU が要る T4（A/B学習）と T5（比較・判定）、および GitHub への push。** Claude Code はこのブリーフ単体で自走できるよう書いてある。

> 最重要マインドセット：これは「フローターを消す」タスクではなく **「学習前 SOR がフローターをどれだけ減らすかを測る」測定タスク**。**「効果なし」も正当な結論**であり、それを正直に出すことがゴール。小さな改善を成功と早合点しない。

---

## 0. 前提・現状（着手前に把握）

- 実行環境：**GPU 搭載マシン**（NVIDIA）。CUDA で 3DGS 学習が回ること。
- リポジトリ：`D:\Claude\Photogrammetry\フォトグラメトリー探索\sano-floater-cleanup`
- 元データ（読み取り専用）：`F:\RealityScan\sano`（COLMAP text：`images/` + `sparse/0/`、3,297画像 / 258,394点）
- **CPU 工程の成果（生成済み, `F:\RealityScan\sano\exp\`）**
  - `baseline/sparse/0/`：固定済み COLMAP baseline
  - `cleaned_sor/sparse/0/`：SOR 後 points3D（**採用 nb_neighbors=20, std_ratio=2.0**、257,590点保持 / 804点削除）
  - `metrics/`：`baseline.json` / `sor_sweep.json` / `adopted.json` / `removed_points.ply`
- ツール：`scripts/`（colmap_io, geom, cloud_metrics, sor_clean, floater_metrics, compare, prepare_projects, make_synthetic）、`app/app.py`（Gradio）、`configs/`（clean.yaml, train.yaml）
- 既知の数値・所見は `RESULTS.md` を参照。

---

## 1. 完了条件（Acceptance Criteria）

以下を**すべて**満たしたら完了。満たせない項目は「未達」として明記すること。

1. **Stage 0** で GPU と 3DGS トレーナの実体・実際のCLIフラグ（seed / eval / holdout の指定方法）が確定し、ログに記録されている。
2. `baseline` と `cleaned_sor` が **同一設定（同 strategy・同 iter・同 seed・同 max_cap・同 resize・同 保留ビュー分割）** で学習完走し、各 `.ply` と学習ログが所定ディレクトリに揃っている。**唯一の変数は点群クリーンアップの有無**。
3. `compare.py`（または app の T5）で **フローター指標(a)(b)・総ガウシアン数・PSNR/SSIM** を 1 表に集約し、§1.5 判定（フローター改善 **かつ** ΔPSNR ≥ -0.3dB なら PASS）を下した `metrics/report.md` が生成されている。
4. §7 チェックリストを**実体確認の結果として**記載（存在するはず、ではなく中身を開いて確認）。
5. リポジトリが GitHub（**private**, 名称 `sano-floater-cleanup`）に push されている。実スキャンデータ・学習出力は `.gitignore` 済みでコミットに含めない。

---

## 2. 対象ファイル・影響範囲

**読み取り専用（変更禁止）**
- `F:\RealityScan\sano\**`（元データ）、`exp\baseline\**`、`exp\cleaned_sor\**`

**新規作成（書き込み先）**
- `F:\RealityScan\sano\exp\cleaned_proj\`（cleaned 学習用プロジェクト：images をリンク）
- `F:\RealityScan\sano\exp\train_baseline\`、`exp\train_cleaned\`（学習出力）
- `F:\RealityScan\sano\exp\metrics\report.md` / `report.json`（比較結果）

**影響範囲**：元データに副作用なし。学習は GPU を占有するため、まず短iterスモークで配線確認 → 本 A/B、の順で行う。

---

## 3. 実行手順

### Stage 0：環境とトレーナの確定（コードを書く前に）
1. `nvidia-smi` で GPU を確認。
2. 3DGS トレーナ（**LichtFeld Studio = `gaussian_splatting_cuda`**）の実体を確認：
   - PATH かフルパスで起動し、**必ず `gaussian_splatting_cuda --help` を実行して実際のフラグを取得**する。
   - **確定済みフラグ（調査済み）**：`-d <COLMAP>`, `-o <out>`, `-i <iter>`(既定30000), `-r <resize>`, `--strategy mcmc|default`, `--max-cap <N>`。
   - **要確認フラグ（版依存・絶対に勝手に発明しない）**：`seed` の固定方法、`eval`/`test` の有効化、保留ビュー（holdout）の指定方法、PSNR/SSIM の出力先。`--help` の実出力から確定し、`exp/metrics/trainer_help.txt` に保存。
   - トレーナが未導入なら、ビルド/入手手順（https://github.com/MrNeRF/LichtFeld-Studio ）を提示して**停止し、橋本さんに依頼**。代替（Jawset Postshot 等が導入済み）を使う場合は CLI の差異を Stage 0 で吸収すること。

### Stage T4：A/B 短時間学習
1. cleaned 学習プロジェクトを準備（**images を複製せずリンク**）:
   ```
   python scripts\prepare_projects.py --orig F:\RealityScan\sano ^
       --cleaned-sparse F:\RealityScan\sano\exp\cleaned_sor\sparse\0 ^
       --proj F:\RealityScan\sano\exp\cleaned_proj
   ```
2. **配線スモーク**（短iter, 例 `-i 3000`）で baseline / cleaned 両方が完走することを先に確認。
3. **本 A/B**（`configs/train.yaml` の設定。既定 strategy=mcmc, iter=15000, max_cap=1,000,000, resize=1, seed=0。両者で完全一致させる）:
   ```
   gaussian_splatting_cuda -d F:\RealityScan\sano                -o F:\RealityScan\sano\exp\train_baseline --strategy mcmc -i 15000 --max-cap 1000000 -r 1 <seed/eval フラグ>
   gaussian_splatting_cuda -d F:\RealityScan\sano\exp\cleaned_proj -o F:\RealityScan\sano\exp\train_cleaned  --strategy mcmc -i 15000 --max-cap 1000000 -r 1 <seed/eval フラグ>
   ```
   - 両学習のログを保存。完走（`.ply` 出力 + 反復到達 + 総ガウシアン数）を実体確認。

### Stage T5：比較と結論
```
python scripts\compare.py --baseline F:\RealityScan\sano\exp\train_baseline ^
    --cleaned F:\RealityScan\sano\exp\train_cleaned ^
    --sparse F:\RealityScan\sano\sparse\0 ^
    --out F:\RealityScan\sano\exp\metrics\report.md
```
- `compare.py` は出力ディレクトリ内の json/csv/log から PSNR/SSIM を自動抽出する。**抽出できない場合**は、Stage 0 で判明した eval 出力の場所を確認し、`--baseline-psnr/--cleaned-psnr/--baseline-ssim/--cleaned-ssim` で手入力する（推測値を入れない）。
- 指標(a)：`sigmoid(opacity) < 0.05` かつ `max(exp(scale)) > シーン対角の2%`。指標(b)：カメラ凸包25%膨張の外側。3DGS の opacity=logit / scale=log 格納は `floater_metrics.py` が処理済み。

---

## 4. 失敗時プロトコル（§6：修正コードの前に原因仮説）

いずれかが失敗・想定外の結果になったら、**修正の前に**以下を出力して一旦停止：
1. 観測事実（エラーログ / 数値 / 出力ファイルの実体）。
2. 原因仮説を最低2つ（可能性順）。各仮説に「真なら観測されるはずの証拠」を併記。
3. 仮説を切り分ける最小の確認（1コマンド/1スクリプト）。
4. 確認結果を踏まえて修正に着手。

**特に注意：「点が削れた／学習が回った」を成功と早合点しない。フローター低減は T5 の比較でしか確認できない。**

---

## 5. この実データ固有の落とし穴（CPU工程で判明済み・必読）

- **被写体は小さい（ROI 約29m×4m×13m）が、点群 bbox は対角1365**。遠方外れ点で膨張している。シーンスケール基準は `floater_metrics.py` が**カメラ中心 bbox**を使う（点群bboxを使うと外れ点で歪む）。Stage T5 でスケール基準の妥当性を一度確認すること。
- **スパース点の 87.5% がカメラ凸包の外**。フローター源の多くは「凸包外の密な背景構造」で、これは SOR では落ちない。
- **SOR(std=2.0) の削除は全体の 0.31%（804点）と小さい。** したがって **A/B の差が小さい／無い可能性が現実的にある。**
- **判定が「ΔPSNR≈0 かつ floater指標ほぼ不変」なら、結論は「この条件では学習前SORは有効でない」と明記して打ち切る。** std を上げる延命はしない（ROI を削り始めるだけ）。
- 効果が無い場合の次の本命（**やるなら別タスクとして明示・勝手に膨らませない**）：① 学習戦略軸（MCMC vs 無印 densification）の比較、② カメラ凸包外側点の明示除去、③ 深度事前分布での初期化（FFGS / Clean-GS 系）。点群クリーンアップより上流（撮り方・初期化・戦略）が支配的という仮説の検証。

---

## 6. GitHub への push（private）

`.gitignore` でデータ・学習出力は除外済み（コードのみ commit）。
```
cd /d D:\Claude\Photogrammetry\フォトグラメトリー探索\sano-floater-cleanup
push_to_github.bat          REM gh があれば private 作成+push まで自動
```
`gh` 不在時は GitHub で空の private `sano-floater-cleanup` を作成 → `git remote add origin ...` → `git push -u origin main`。**report.md / 学習出力はリポジトリに入れない**（`F:\sano\exp` に残す。共有が必要なら別途）。

---

## 7. 完了報告フォーマット（実体確認の結果を記載）

- [ ] GPU（`nvidia-smi`）とトレーナ実体・**実際に使った seed/eval/holdout フラグ**
- [ ] baseline / cleaned の学習完走（iter到達・総ガウシアン数・出力 `.ply` パス）
- [ ] `metrics/report.md` の比較表（floater(a)(b) / 総ガウシアン数 / PSNR / SSIM）
- [ ] §1.5 判定（PASS / FAIL / 効果なし）と**一文での結論**
- [ ] スケール基準・ROI・指標定義の妥当性レビュー所見
- [ ] GitHub リポジトリ URL（private）
- [ ] 未達項目・打ち切り箇所の明示

---

## 付録：唯一変数を守るためのチェック
baseline と cleaned で次が**完全一致**していること：strategy / iter / max_cap / resize / seed / 保留ビュー分割 / トレーナのバージョン。違いは **points3D（=点群クリーンアップの有無）だけ**。
