"""
compare.py — A/B 比較と結論（T5）。

baseline / cleaned それぞれの 3DGS 学習出力について
  - フローター指標（floater_metrics）
  - 総ガウシアン数
  - PSNR / SSIM（出力ディレクトリ内の json/csv/log から自動抽出、無ければ手入力）
を 1 表に集約し、§1.5 の判定を下して metrics/report.md と compare.json を出力する。

判定（既定）:
  PASS = フローター指標(a) が改善（cleaned < baseline） かつ ΔPSNR >= -0.3 dB
  ※ PSNR が取得できない場合は「フローター改善のみ確認・品質未検証」と明記。

使い方:
  python compare.py --baseline <train_baseline> --cleaned <train_cleaned> \
      --sparse <sparse/0> --out metrics/report.md \
      [--baseline-psnr 28.1 --cleaned-psnr 28.0 ...]
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re

import floater_metrics as fm


PLY_CANDIDATES = [
    "point_cloud.ply", "splat.ply", "gaussians.ply", "output.ply",
    "point_cloud/iteration_*/point_cloud.ply", "**/point_cloud.ply", "**/*.ply",
]


def find_ply(d):
    for pat in PLY_CANDIDATES:
        hits = sorted(glob.glob(os.path.join(d, pat), recursive=True))
        if hits:
            # iteration_* は最大反復を優先
            hits.sort(key=lambda p: _iter_key(p))
            return hits[-1]
    return None


def _iter_key(p):
    m = re.search(r"iteration_(\d+)", p)
    return int(m.group(1)) if m else 0


def read_csv_final_metrics(d):
    """LichtFeld の metrics.csv（iteration,psnr,ssim,time_per_image,num_gaussians）の
    最終行＝最終 iteration の値を返す。最終モデル（保存 .ply）に対応する確定値であり、
    metrics_report.txt の "Best PSNR"（regex 先頭一致で誤取得しうる）より faithful。
    見つからなければ None。"""
    hits = sorted(glob.glob(os.path.join(d, "**", "metrics.csv"), recursive=True))
    for csv_path in hits:
        try:
            rows = [r for r in open(csv_path, encoding="utf-8",
                                   errors="ignore").read().splitlines() if r.strip()]
        except Exception:
            continue
        if len(rows) < 2:
            continue
        header = [h.strip().lower() for h in rows[0].split(",")]
        last = [c.strip() for c in rows[-1].split(",")]
        if len(last) != len(header):
            continue
        rec = dict(zip(header, last))
        try:
            ng = rec.get("num_gaussians")
            return {
                "psnr": float(rec["psnr"]),
                "ssim": float(rec["ssim"]),
                "num_gaussians": int(float(ng)) if ng else None,
                "iteration": int(float(rec.get("iteration", "0"))),
                "src": csv_path,
            }
        except Exception:
            continue
    return None


def find_metric(d, keys):
    """ディレクトリ内の json/csv/log から PSNR/SSIM 等の数値を探す。"""
    # 1) json
    for jp in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        val = _search_json(data, keys)
        if val is not None:
            return val, jp
    # 2) csv / txt / log を grep
    for ext in ("*.csv", "*.txt", "*.log"):
        for fp in glob.glob(os.path.join(d, "**", ext), recursive=True):
            try:
                txt = open(fp, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for key in keys:
                m = re.search(rf"{re.escape(key)}\s*[:=]\s*([0-9]+\.?[0-9]*)",
                              txt, re.IGNORECASE)
                if m:
                    return float(m.group(1)), fp
    return None, None


def _search_json(obj, keys, depth=0):
    if depth > 6:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in [x.lower() for x in keys] \
                    and isinstance(v, (int, float)):
                return float(v)
        for v in obj.values():
            r = _search_json(v, keys, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _search_json(v, keys, depth + 1)
            if r is not None:
                return r
    return None


def analyze_side(name, train_dir, sparse, args):
    ply = find_ply(train_dir)
    if ply is None:
        return {"name": name, "train_dir": os.path.abspath(train_dir),
                "error": "学習出力 .ply が見つかりません", "ply": None}
    fmr = fm.compute_floater_metrics(ply, sparse, tau_op=args.tau_op,
                                     scale_frac=args.scale_frac, expand=args.expand)
    # 最終 iter の PSNR/SSIM は metrics.csv の最終行を最優先（faithful）。
    csv_final = read_csv_final_metrics(train_dir)
    if csv_final is not None:
        psnr, psnr_src = csv_final["psnr"], csv_final["src"]
        ssim, ssim_src = csv_final["ssim"], csv_final["src"]
    else:
        psnr, psnr_src = find_metric(train_dir, ["psnr"])
        ssim, ssim_src = find_metric(train_dir, ["ssim"])
    return {
        "name": name,
        "train_dir": os.path.abspath(train_dir),
        "ply": ply,
        "total_gaussians": fmr["total_gaussians"],
        "floater_a": fmr["metric_a_lowopacity_bigscale"],
        "floater_b": fmr["metric_b_outside_dilated_hull"],
        "scene_diagonal": fmr.get("scene_diagonal"),
        "scale_threshold_sigma": fmr.get("scale_threshold_sigma"),
        "floater_notes": fmr.get("notes"),
        "psnr": psnr, "psnr_src": psnr_src,
        "ssim": ssim, "ssim_src": ssim_src,
        "eval_iteration": (csv_final or {}).get("iteration"),
        "gaussians_csv": (csv_final or {}).get("num_gaussians"),
    }


def main():
    ap = argparse.ArgumentParser(description="A/B 比較と結論")
    ap.add_argument("--baseline", required=True, help="train_baseline ディレクトリ")
    ap.add_argument("--cleaned", required=True, help="train_cleaned ディレクトリ")
    ap.add_argument("--sparse", required=True, help="sparse/0（カメラ）")
    ap.add_argument("--out", default="metrics/report.md")
    ap.add_argument("--tau-op", type=float, default=0.05)
    ap.add_argument("--scale-frac", type=float, default=0.02)
    ap.add_argument("--expand", type=float, default=0.25)
    ap.add_argument("--delta-psnr-min", type=float, default=-0.3)
    # 手入力オーバーライド
    ap.add_argument("--baseline-psnr", type=float, default=None)
    ap.add_argument("--cleaned-psnr", type=float, default=None)
    ap.add_argument("--baseline-ssim", type=float, default=None)
    ap.add_argument("--cleaned-ssim", type=float, default=None)
    args = ap.parse_args()

    b = analyze_side("baseline", args.baseline, args.sparse, args)
    c = analyze_side("cleaned_sor", args.cleaned, args.sparse, args)
    if args.baseline_psnr is not None: b["psnr"] = args.baseline_psnr
    if args.cleaned_psnr is not None: c["psnr"] = args.cleaned_psnr
    if args.baseline_ssim is not None: b["ssim"] = args.baseline_ssim
    if args.cleaned_ssim is not None: c["ssim"] = args.cleaned_ssim

    # 判定
    verdict_lines = []
    floater_improved = None
    if "error" not in b and "error" not in c:
        floater_improved = c["floater_a"] < b["floater_a"]
    dpsnr = None
    if b.get("psnr") is not None and c.get("psnr") is not None:
        dpsnr = c["psnr"] - b["psnr"]

    if floater_improved is None:
        verdict = "判定不能（学習出力が揃っていない）"
    elif dpsnr is None:
        verdict = ("フローター指標(a)は" + ("改善" if floater_improved else "改善せず")
                   + "。PSNR未取得のため品質は未検証（要手動入力）")
    else:
        quality_ok = dpsnr >= args.delta_psnr_min
        if floater_improved and quality_ok:
            verdict = f"PASS: フローター改善 かつ ΔPSNR={dpsnr:+.2f}dB >= {args.delta_psnr_min}"
        elif floater_improved and not quality_ok:
            verdict = f"FAIL(品質劣化): フローターは改善も ΔPSNR={dpsnr:+.2f}dB < {args.delta_psnr_min}"
        else:
            verdict = f"FAIL(改善なし): フローター指標(a)が改善せず（ΔPSNR={dpsnr:+.2f}dB）"

    # 出力 json
    out_json = {"baseline": b, "cleaned": c,
                "delta_psnr": dpsnr, "floater_a_improved": floater_improved,
                "verdict": verdict, "params": vars(args)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    # report.md
    def fmtnum(x, n=2):
        return "—" if x is None else (f"{x:,.{n}f}" if isinstance(x, float) else f"{x:,}")
    md = []
    md.append("# sano フローター低減 A/B 比較レポート\n")
    md.append(f"- 判定: **{verdict}**")
    md.append(f"- floater 指標(a) 定義: opacity < {args.tau_op} かつ max_scale > {args.scale_frac*100:.0f}% of scene diag")
    md.append(f"- floater 指標(b) 定義: カメラ凸包を {args.expand*100:.0f}% 膨張させた外側\n")
    md.append("| 指標 | baseline | cleaned_sor | 差分 |")
    md.append("|---|---:|---:|---:|")
    def row(label, kb, kc, lower_better=True, n=2):
        vb, vc = b.get(kb), c.get(kc if kc else kb)
        if isinstance(vb, (int, float)) and isinstance(vc, (int, float)):
            diff = vc - vb
            arrow = ""
            if diff != 0:
                good = (diff < 0) if lower_better else (diff > 0)
                arrow = " ✅" if good else " ⚠️"
            diff_s = (f"{diff:+,.{n}f}" if isinstance(diff, float) else f"{diff:+,}") + arrow
        else:
            diff_s = "—"
        md.append(f"| {label} | {fmtnum(vb,n)} | {fmtnum(vc,n)} | {diff_s} |")
    row("総ガウシアン数", "total_gaussians", "total_gaussians", lower_better=True, n=0)
    row("floater(a) 低不透明・大スケール", "floater_a", "floater_a", lower_better=True, n=0)
    row("floater(b) 凸包外側", "floater_b", "floater_b", lower_better=True, n=0)
    row("PSNR (dB)", "psnr", "psnr", lower_better=False, n=2)
    row("SSIM", "ssim", "ssim", lower_better=False, n=4)
    md.append("")
    md.append(f"- baseline ply: `{b.get('ply')}`")
    md.append(f"- cleaned ply : `{c.get('ply')}`")
    if b.get("psnr_src"): md.append(f"- baseline PSNR 出典: `{b['psnr_src']}`")
    if c.get("psnr_src"): md.append(f"- cleaned PSNR 出典: `{c['psnr_src']}`")
    md.append("")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("[compare] verdict:", verdict)
    print(f"  baseline floater(a)={b.get('floater_a')}  cleaned floater(a)={c.get('floater_a')}")
    print(f"  -> {args.out}\n  -> {json_path}")


if __name__ == "__main__":
    main()
