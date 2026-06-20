# FINDINGS — 3DGS フローター低減 A/B（T4/T5, GPU 工程）

実データ sano（3,297枚）/ toyota（952枚）。トレーナ: LichtFeld Studio v0.5.2（MCMC, iter15000, max-cap=100万）。
A/B の唯一変数のみを変え、他（seed以外＝LichtFeldはseed固定不可・後述）は完全一致。判定は holdout PSNR/SSIM ＋ floater 指標(a)(b)。
※ データ・学習出力はリポジトリに含めない（`.gitignore`）。詳細レポートは各データの `exp/metrics/`（FINDING_scale_reg.md 等）に保存。

## 結論サマリ
**学習前の点群クリーンアップ（SOR・凸包外除去）はフローターを減らさない（MCMC が再成長させ washes out）。一方、学習時の幾何正則化 `scale_reg` を強めるとフローターが劇的に減り、品質コストはほぼゼロ。これが本検証の中心的成果。**

## 介入別の結果（sano、floater(a)=低不透明・大スケール）
| 介入 | 種別 | ΔPSNR | floater(a) | 判定 |
|---|---|---:|---:|:--:|
| ① SOR（std=2.0, 0.31%除去） | 学習前 init | −0.014 | −1.2% | 効果なし（ノイズ内）|
| ② カメラ凸包外除去（84%除去） | 学習前 init | −1.11 | **+56%** | 失敗・悪化 |
| reg bundle（opacity+scale=0.02） | 学習時 | −1.47 | +16% | 失敗（opacity_reg が原因）|
| **scale_reg=0.02（単独）** | **学習時** | **−0.004** | **−82%** | **成功（コストゼロ）**|

- ノイズフロア（baseline 2本）: ΔPSNR ±0.187dB / floater(a) ±715。①の差はノイズ未満＝効果なし。
- MCMC は seed 固定フラグ無し（`mcmc.cpp` が wall-clock seed）。判定は必ず baseline 2本でノイズフロアを取り、効果量と比較する。

## scale_reg スイープ（sano, isolated）と toyota 再現
| scale_reg | sano PSNR | sano floater(a) | toyota PSNR | toyota floater(a) |
|---:|---:|---:|---:|---:|
| 0.0042 (base) | 25.428 | 17,041 | 24.767 | 14,244 |
| 0.01 | 25.483 (+0.06) | 6,569 (−62%) | — | — |
| **0.02 推奨** | 25.424 (−0.00) | 3,067 (**−82%**) | 24.776 (+0.01) | 3,567 (**−75%**) |
| 0.04 | 25.244 (−0.18) | 1,665 (−90%) | — | — |

→ 2データセットで一致：**scale_reg=0.02 で floater(a) −75〜82%、PSNR/SSIM 不変**。0.04 で −90%（PSNR −0.18dB, 許容内）。

## 本番長（iter=30000）での確定（標準採用の検証）
| iter=30000 | PSNR | SSIM | floater(a) | floater(b) | Δfloater(a) | ΔPSNR |
|---|---:|---:|---:|---:|---:|---:|
| sano base | 26.5701 | 0.8803 | 1,976 | 854,205 | — | — |
| sano scale_reg=0.02 | 26.4875 | 0.8809 | 144 | 844,819 | **−92.7%** | −0.083 |
| toyota base | 25.6468 | 0.8775 | 4,312 | 278,285 | — | — |
| toyota scale_reg=0.02 | 25.6361 | 0.8781 | 476 | 274,248 | **−89.0%** | −0.011 |

- 本番長でも効果は健在：floater(a) ほぼ消滅（−89〜93%）、SSIM 微増、ΔPSNR はノイズフロア内（sano −0.083, toyota −0.011）。
- 注記：30k では学習自体が長い分 baseline の floater(a) も自然に減る（15k 17,041→30k 1,976）。scale_reg はそこから「最後の一掃」を担う。15k では劇的削減（−82%, コスト0）、30k では near-elimination（−93%, 微コスト）。どちらも有効。
- **結論：scale_reg=0.02 を本番（iter=30000）標準として採用可**。コストはノイズ内・SSIM 不変以上。よりPSNRを優先するなら 0.01（効果は緩むがコストさらに小）も選択肢。

## 推奨運用
- フローター抑制したい場合 **`--config configs/lichtfeld_scalereg02.json`（scale_reg=0.02）** を採用。他設定は baseline と同一。
- `scale_reg` は CLI フラグではなく config キー。`opacity_reg` は混ぜない（PSNR を壊し、floater(a) 定義を交絡させる）。
- floater(b)（凸包外の背景構造）は scale_reg では動かない＝対象外。そこまで踏み込むなら深度正則化（外部 gsplat を sm_120 AOT ビルド + Depth-Anything-V2）が higher-ceiling だが、floater(a) は本設定で解決済み。

## メカニズム
`scale_reg * mean(exp(scale))` を loss に加え全ガウシアンのスケールを縮める。大スケールの拡散フローターが潰れて表面密着 or 消滅。幾何のみを締めるため不透明度・色を弄らず再構成品質を保つ（opacity_reg と違い PSNR を壊さない）。
