# FloaterClean Trainer — デスクトップツール 設計書

作成日: 2026-06-20 / 対象リポジトリ: sano-floater-cleanup

## 1. 背景・目的
本プロジェクトの検証で、3DGS のフローター低減に **効くのは学習前クリーンアップ(SOR)ではなく、学習時の幾何正則化 `scale_reg`** であることが判明した（sano/toyota・15k/30k で `scale_reg=0.02` が floater(a) を −75〜82%、品質コストほぼ0）。本ツールはこの成果を**日常的に使える形**にする：

> COLMAP/RealityScan データを選んで1クリック → 検証済み `scale_reg` 設定で LichtFeld 学習 → フローターを抑えた `.ply` を出力。

UI はブラウザ不要の **ネイティブ Tkinter 窓**（Python 標準同梱）。「適用専用」（A/B 検証は対象外）。

## 2. スコープ
**やること**
- データ選択 → 検証 → `scale_reg` 設定で LichtFeld をヘッドレス学習 → `.ply` 出力。
- LichtFeld 実行ファイルの自動検出（手動指定フォールバック）。
- 学習進捗のログ表示＋進捗バー。
- 学習後に出力 `.ply` の **floater(a) 数を計測して表示**（既定 ON）。

**やらないこと（YAGNI）**
- SOR / 凸包外除去（無効と判明）。A/B 検証・ノイズフロア（別ツール領域）。PSNR/SSIM holdout（本番=全画像学習と両立しないため適用モードでは出さない）。深度正則化 / gsplat（floater(b) は本質的に残るため対象外）。

## 3. 既定値（ユーザー承認済み）
- 品質の既定 = **本番 30000 iter**（選択肢に短時間 15000）。
- 強度プリセット = **標準 `scale_reg=0.02`（既定）** / 強め `0.04` / オフ `0.0042`（baseline 相当）。
- **学習後 floater(a) 計測 = 既定 ON**。

## 3.1 入力データ形式（確定）
入力は **RealityScan でアライメント後、COLMAP 形式でエクスポートしたフォルダ**をそのまま選択する。構造は検証で使った `F:\RealityScan\sano` と同一：
```
<選択フォルダ>/
  images/                         # 元画像（RealityScan が参照したもの）
  sparse/0/
    cameras.txt                   # RealityScan→COLMAP は text 形式・通常 PINHOLE
    images.txt
    points3D.txt
```
- `validate_dataset` はこの構造を厳密に確認：`images/`(>0 枚) と `sparse/0/{cameras,images,points3D}` の存在（`.txt` 優先、`.bin` も可）。欠落時は開始不可＋具体メッセージ（例「sparse/0/points3D.txt が見つかりません。RealityScan を COLMAP 形式でエクスポートしてください」）。
- LichtFeld は `--data-path <選択フォルダ>` を受け、`<フォルダ>/images` と `<フォルダ>/sparse/0` を読む（検証時と同じ）。サブフォルダ探索や再構成は行わない（既にアライメント済み前提）。

## 4. アーキテクチャ
単一 Tkinter アプリ `app/desktop_app.py` ＋ ランチャ。各責務を小さな関数/クラスに分離：

| ユニット | 役割 | 入力 → 出力 | 依存 |
|---|---|---|---|
| `locate_lichtfeld()` | LichtFeld exe 解決 | (任意の手動パス) → exe パス or None | os/env |
| `validate_dataset(dir)` | データ妥当性 | dir → (ok, message, has_images, sparse_path) | os |
| `build_config(base, quality, preset, out)` | 一時 config 生成 | 設定 → tmp json パス | json（検証済み base を読み override） |
| `run_training(exe, cfg, data, out, opts, on_line)` | 学習 subprocess | 引数 → returncode（行ごとに on_line コールバック） | subprocess |
| `measure_floaters(ply, sparse)` | floater(a) 計測 | ply, sparse → dict（無依存なら skip） | floater_metrics（numpy/scipy 任意） |
| `DesktopApp(tk)` | GUI 配線 | — | tkinter |

**設計方針**：GUI（`DesktopApp`）はロジック関数を呼ぶだけ。各ロジック関数は GUI 非依存で単体テスト可能（subprocess/FS はパラメータ注入）。

## 5. config 生成の詳細
- ベース = `configs/lichtfeld_scalereg02_prod30k.json`（検証済み・単一の真実源）。これを読み、UI 選択で override して出力先に一時 config（例 `<out>/_run_config.json`）を書く：
  - `scale_reg` = プリセット（0.02 / 0.04 / 0.0042）
  - `iterations` = 品質（30000 / 15000）
  - `eval_steps` = `[iterations]`、`save_steps` = `[iterations]`
  - `enable_eval=false`、`enable_save_eval_images=false`（適用モードは全画像学習・holdout 無し）
  - その他（strategy=mcmc, max_cap=1000000, opacity_reg=0.0042, headless 等）はベースのまま。**`opacity_reg` は上げない**（PSNR を壊し floater(a) 定義を交絡させるため固定）。
- `scale_reg` は LichtFeld の CLI フラグに無く config キーのみ → 必ず `--config` 経由。`--strategy` は config と競合するので CLI で渡さない。

## 6. LichtFeld 起動コマンド
```
<exe> --headless --config <tmp.json> --data-path <dataset> --output-path <out> -r 1
```
- 進捗：標準出力の `... <iter>/<total> ...` 行を正規表現で拾い進捗バー更新。最終 `.ply`（`splat_<iter>.ply`）を出力先から検出。
- 任意（上級）：`--undistort`（SIMPLE_RADIAL データ用、既定 OFF。PINHOLE は不要）。
- 文字コード：`cmd /c` 経由でログをファイルにも保存（PowerShell の native stderr ラップ回避）。`PYTHONUTF8=1`。

## 7. GUI レイアウト（1画面・Tkinter）
- 行1: データフォルダ [Entry][参照]
- 行2: 出力フォルダ [Entry][参照]（既定 `<data>\exp\train_floaterclean`）
- 行3: 強度 [Radio: 標準0.02 / 強め0.04 / オフ] ／ 品質 [Radio: 本番30000 / 短時間15000]
- 行4: [✓] 学習後に floater 数を計測（既定 ON）　[ ] 歪み補正(--undistort)
- 行5: [学習開始]（実行中は無効化）／[中止]
- 行6: 進捗バー（determinate, iter ベース）＋ ステータスラベル
- 行7: ログ（ScrolledText, 読取専用, 自動スクロール）
- 完了時: 「✅ 完了：<ply パス>　floater(a)=N」＋[出力フォルダを開く]

学習は別スレッドで実行（GUI 非ブロッキング）。`on_line` は `tk.after` 経由でメインスレッドに反映。

## 8. エラー処理
- LichtFeld 未検出 → ダイアログ「実行ファイルを指定してください」＋参照。
- データ不正（images/ or sparse/0 欠落）→ 具体メッセージ、開始ボタン無効。
- GPU 無し（`nvidia-smi` 不在/失敗）→ 警告（続行可、ただし失敗想定）。
- 学習が非ゼロ終了 → 末尾ログ＋ログファイルパス提示。
- floater 計測の依存（numpy/scipy）欠如 → 計測のみスキップし学習結果は提示。

## 9. ランチャ（windows-launcher-scripts 準拠）
- `run_desktop.bat`：文字コード安全（chcp/UTF-8 配慮）、ダブルクリック起動。python は tkinter 同梱の環境を使用。floater 計測用に numpy/scipy を確保（無ければ計測のみ無効化）。
- `install_desktop_shortcut.ps1`（任意）：デスクトップにショートカット作成。非 ASCII パスは PowerShell に直書きせず Python 側で処理（cp932 mojibake 回避）。

## 10. テスト
- `validate_dataset`：正常/images欠落/sparse欠落/bin・txt 両対応 をユニットテスト。
- `build_config`：各プリセット×品質で scale_reg/iterations/eval_steps が期待値、ベースの他キーが保持されることを検証。
- `locate_lichtfeld`：env/既定パス/PATH/未検出 の分岐。
- 統合（手動・GPU 必要）：sano で短時間(15000)・標準プリセットを1回流し、`.ply` 出力＋floater 計測表示まで確認。

## 11. ファイル構成
- `app/desktop_app.py`（新規, Tkinter）
- `app/run_desktop.bat`（新規ランチャ）
- `app/install_desktop_shortcut.ps1`（新規・任意）
- `tests/test_desktop_app.py`（新規・ロジック関数のユニットテスト）
- 既存 `configs/lichtfeld_scalereg02_prod30k.json` を config ベースに使用（変更なし）。
- 既存 `app/app.py`（SOR 中心の旧ツール）は当面残置（別物）。README に新ツールの位置づけを追記。
