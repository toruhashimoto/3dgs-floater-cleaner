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

---

# floater(b)（凸包外の背景構造）と 深度正則化（gsplat, 別トレーナ）

floater(a) を解決した後、floater(b)（カメラ凸包外の Gaussian, sano で全体の ~82–85%）を深度正則化で減らせるか検証。LichtFeld は深度監督を持たないため、外部 **gsplat 1.5.3（nerfstudio simple_trainer, MCMC）** を Blackwell でビルドして A/B（唯一変数=`--depth_loss`、SfM疎点深度の視差L1監督。外部単眼深度は不要）。詳細は `F:\RealityScan\sano\exp`(無)→ `F:\RealityScan\gsplat_work\FINDING_floater_b_depth.md`、構築は `GSPLAT_SETUP.md`。

## 一文結論
**深度正則化（gsplat sparse-SfM depth_loss、λ=0.01/0.2）は floater(b) を低減しない（全指標が gsplat の run-to-run ノイズ未満）。floater(b) の主体はワイド背景の正当な構造で、SfM 点（＝その背景）に整合させる depth_loss では除去できない。加えて gsplat の MCMC は floater(a) を元々ごく少量しか生まない。**

## 結果（gsplat, 正規化フレーム, 15000 step, sano）
| run | PSNR | floater(a) | floater(b) 凸包外 | ghost(a∩外) |
|---|---:|---:|---:|---:|
| baseline | 25.538 | 1,656 | 819,166 (82%) | 1,638 |
| baseline2 (noise) | 25.610 | 1,798 | 817,647 | 1,787 |
| depth λ=0.01 | 25.619 | 1,760 | 818,588 | 1,753 |
| depth λ=0.2 | 25.472 | 1,584 | 820,346 | 1,567 |

ノイズフロア(±): PSNR 0.072 / floater(a) 142 / floater(b) 1,519。depth の Δ は λ=0.01/0.2 とも全てノイズ内 → **効果なし**。

## 通しの結論
- **floater(a)（自由空間の浮遊フローター）= 除去可能**：LichtFeld の学習時 `scale_reg=0.02` で −75〜82%（コスト0、本書上段）。
- **floater(b)（凸包外）= 大半が正当な背景**で「フローター」ではない。点群クリーンアップ（SOR/凸包除去）も深度正則化（gsplat depth_loss）も減らさない（全てノイズ未満）。floater(b) を消したいのは品質改善でなく ROI クロップ（背景切り）という別目的。
- 実用結論：浮遊フローター対策は **LichtFeld + scale_reg=0.02** で足りる。トレーナを gsplat に替えると floater(a) は元々少ないが、floater(b) は本質的にどちらでも残る（シーン構造）。
