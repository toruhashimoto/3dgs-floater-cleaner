"""
app.py — sano フローター低減クリーンアップ・デスクトップアプリ（Gradio ローカルUI）。

GPU 不要の工程（T0 データ確認 / T2 メトリクス / T3 SORスイープ・採用 / 可視化）を
ブラウザUIで実行し、GPU が要る T4 学習はワンクリックでコマンド生成、T5 比較は学習
出力を指定して集計する。

起動: python app/app.py   （ブラウザで http://127.0.0.1:7860 が開く）
"""
from __future__ import annotations
import os
import sys
import json
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# scripts/ をインポートパスに追加
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), "scripts")
sys.path.insert(0, _SCRIPTS)

import colmap_io as cio          # noqa: E402
import geom                       # noqa: E402
import cloud_metrics as cm        # noqa: E402
import sor_clean as sc            # noqa: E402
import floater_metrics as fmet    # noqa: E402
import compare as cmp_            # noqa: E402

DEFAULT_SPARSE = "F:/RealityScan/sano/sparse/0"
DEFAULT_OUTDIR = "F:/RealityScan/sano/exp"
DEFAULT_ORIG = "F:/RealityScan/sano"


# ---------------------------------------------------------------- T0
def summarize_data(sparse_dir):
    try:
        cams = cio.read_cameras_text(os.path.join(sparse_dir, "cameras.txt"))
        imgs = cio.read_images_text(os.path.join(sparse_dir, "images.txt"))
        pts = cio.read_points3D_text(os.path.join(sparse_dir, "points3D.txt"))
        mn, mx, size, diag = geom.bbox(pts["xyz"])
        md = (f"### データ確認 (T0)\n"
              f"- sparse: `{sparse_dir}`\n"
              f"- カメラ: **{len(cams):,}**　登録画像: **{len(imgs):,}**　3D点: **{pts['xyz'].shape[0]:,}**\n"
              f"- bbox size: {np.round(size,2).tolist()}　(対角 {diag:.1f})\n"
              f"- Open3D 使用: **{geom.have_open3d()}**（False の場合 scipy 同一実装で SOR）")
        return md
    except Exception as e:
        return f"⚠️ 読み込み失敗: {e}"


# ---------------------------------------------------------------- T2
def run_metrics(sparse_dir, outdir, knn):
    try:
        m = cm.compute_metrics(sparse_dir, knn=int(knn), hist_bins=40)
        os.makedirs(os.path.join(outdir, "metrics"), exist_ok=True)
        outp = os.path.join(outdir, "metrics", "baseline.json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        k = m["knn_mean_distance"]
        md = (f"### 点群メトリクス (T2) → `{outp}`\n"
              f"- 総点数 **{m['num_points']:,}**　画像 {m['num_images']:,}\n"
              f"- bbox 対角 {m['bbox']['diagonal']:.1f}\n"
              f"- カメラ凸包の外側点 **{_pct(m['outside_camera_hull_fraction'])}**"
              f"（25%膨張後 {_pct(m['outside_camera_hull_fraction_dilate25'])}）\n"
              f"- kNN平均距離 mean {k['mean']:.4f} / median {k['median']:.4f} / p99 {k['p99']:.4f}")
        # ヒストグラム
        edges = np.array(k["histogram_edges"]); counts = np.array(k["histogram_counts"])
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=(edges[1] - edges[0]) * 0.9,
               color="#3b7dd8")
        ax.set_title(f"kNN(k={int(knn)}) mean-distance histogram")
        ax.set_xlabel("mean distance to neighbors"); ax.set_ylabel("count")
        fig.tight_layout()
        return md, fig
    except Exception as e:
        return f"⚠️ 失敗: {e}", None


# ---------------------------------------------------------------- T3
def run_sweep(sparse_dir, outdir, nb, sweep_str, roi_mode, roi_lo, roi_hi,
              roi_bbox, max_removal):
    try:
        imgs = cio.read_images_text(os.path.join(sparse_dir, "images.txt"))
        cc = cio.camera_centers(imgs)
        pts = cio.read_points3D_text(os.path.join(sparse_dir, "points3D.txt"))
        xyz = pts["xyz"]
        if roi_mode == "manual" and roi_bbox.strip():
            v = [float(x) for x in roi_bbox.split(",")]
            roi_min, roi_max = np.array(v[:3]), np.array(v[3:])
            roi_method = "user-specified"
        else:
            roi_min, roi_max, roi_method = sc.estimate_roi(xyz, cc, float(roi_lo), float(roi_hi))
        ratios = [float(x) for x in sweep_str.split(",")]
        rows = []
        for r in ratios:
            res = sc.run_one(xyz, int(nb), r, roi_min, roi_max)
            rows.append([r, res["removed"], round(res["total_removal_rate_pct"], 3),
                         round(res["roi_removal_rate_pct"], 3),
                         "OK" if res["roi_removal_rate_pct"] <= float(max_removal) else "OVER"])
        gate_md = (f"ROI法: {roi_method}\nROI bbox min={np.round(roi_min,2).tolist()} "
                   f"max={np.round(roi_max,2).tolist()}\nOpen3D使用: {geom.have_open3d()}")
        return rows, gate_md
    except Exception as e:
        return [], f"⚠️ 失敗: {e}"


def adopt_clean(sparse_dir, outdir, nb, adopt_ratio, roi_mode, roi_lo, roi_hi, roi_bbox):
    try:
        imgs = cio.read_images_text(os.path.join(sparse_dir, "images.txt"))
        cc = cio.camera_centers(imgs)
        pts = cio.read_points3D_text(os.path.join(sparse_dir, "points3D.txt"))
        xyz = pts["xyz"]
        if roi_mode == "manual" and roi_bbox.strip():
            v = [float(x) for x in roi_bbox.split(",")]
            roi_min, roi_max = np.array(v[:3]), np.array(v[3:])
        else:
            roi_min, roi_max, _ = sc.estimate_roi(xyz, cc, float(roi_lo), float(roi_hi))
        res = sc.run_one(xyz, int(nb), float(adopt_ratio), roi_min, roi_max)
        cleaned_dir = os.path.join(outdir, "cleaned_sor")
        dst_sparse, kept = sc.write_cleaned(sparse_dir, cleaned_dir, pts, res["keep_mask"])
        removed_xyz = xyz[res["removed_mask"]]
        ply = os.path.join(outdir, "metrics", "removed_points.ply")
        cio.write_ply_xyzrgb(ply, removed_xyz, pts["rgb"][res["removed_mask"]], binary=True)

        # 可視化: 削除点(赤) + 保持点サブサンプル(灰) + ROI箱
        fig = plt.figure(figsize=(6, 4.6))
        ax = fig.add_subplot(111, projection="3d")
        keep_xyz = xyz[res["keep_mask"]]
        sub = keep_xyz[np.random.default_rng(0).choice(
            keep_xyz.shape[0], size=min(4000, keep_xyz.shape[0]), replace=False)]
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, c="#cccccc", alpha=0.4, label="kept")
        ax.scatter(removed_xyz[:, 0], removed_xyz[:, 1], removed_xyz[:, 2],
                   s=6, c="red", label=f"removed ({removed_xyz.shape[0]})")
        _draw_box(ax, roi_min, roi_max)
        ax.set_title(f"removed points (std_ratio={adopt_ratio})")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        md = (f"✅ 採用 std_ratio={adopt_ratio}\n"
              f"- cleaned: `{dst_sparse}` （保持 {kept:,} / 削除 {removed_xyz.shape[0]:,}）\n"
              f"- removed_points.ply: `{ply}`")
        return md, fig
    except Exception as e:
        return f"⚠️ 失敗: {e}", None


# ---------------------------------------------------------------- T4
def gen_commands(binary, orig, cleaned_proj, out_base, strategy, iters, max_cap, resize):
    base = (f'{binary} -d "{orig}" -o "{out_base}/train_baseline" '
            f'--strategy {strategy} -i {iters} --max-cap {max_cap} -r {resize}')
    clean = (f'{binary} -d "{cleaned_proj}" -o "{out_base}/train_cleaned" '
             f'--strategy {strategy} -i {iters} --max-cap {max_cap} -r {resize}')
    return (f"# baseline 学習\n{base}\n\n# cleaned 学習（同一設定）\n{clean}\n\n"
            f"# ※ seed/eval/holdout のフラグは `gaussian_splatting_cuda --help` で要確認")


def do_prepare(orig, cleaned_sparse, proj):
    try:
        from prepare_projects import link_dir
        import shutil
        os.makedirs(proj, exist_ok=True)
        m = link_dir(os.path.join(orig, "images"), os.path.join(proj, "images"))
        dst = os.path.join(proj, "sparse", "0"); os.makedirs(dst, exist_ok=True)
        for fn in ("cameras.txt", "images.txt", "points3D.txt"):
            s = os.path.join(cleaned_sparse, fn)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, fn))
        return f"✅ 準備完了: `{proj}`（images: {m}, sparse: cleaned コピー）"
    except Exception as e:
        return f"⚠️ 失敗: {e}"


# ---------------------------------------------------------------- T5
def run_compare(baseline_dir, cleaned_dir, sparse_dir, outdir,
                tau_op, scale_frac, expand, dpsnr_min,
                b_psnr, c_psnr, b_ssim, c_ssim):
    try:
        args = SimpleNamespace(tau_op=float(tau_op), scale_frac=float(scale_frac),
                               expand=float(expand))
        b = cmp_.analyze_side("baseline", baseline_dir, sparse_dir, args)
        c = cmp_.analyze_side("cleaned_sor", cleaned_dir, sparse_dir, args)
        for side, p, s in ((b, b_psnr, b_ssim), (c, c_psnr, c_ssim)):
            if p not in (None, "", "None"):
                try: side["psnr"] = float(p)
                except Exception: pass
            if s not in (None, "", "None"):
                try: side["ssim"] = float(s)
                except Exception: pass
        if "error" in b or "error" in c:
            return [], f"⚠️ 学習出力 .ply が見つかりません（baseline/cleaned の train_* を確認）"
        improved = c["floater_a"] < b["floater_a"]
        dp = (c["psnr"] - b["psnr"]) if (b.get("psnr") and c.get("psnr")) else None
        rows = [
            ["総ガウシアン数", b["total_gaussians"], c["total_gaussians"],
             c["total_gaussians"] - b["total_gaussians"]],
            ["floater(a)", b["floater_a"], c["floater_a"], c["floater_a"] - b["floater_a"]],
            ["floater(b)", b["floater_b"], c["floater_b"],
             (c["floater_b"] - b["floater_b"]) if (b["floater_b"] is not None and c["floater_b"] is not None) else None],
            ["PSNR", b.get("psnr"), c.get("psnr"), (round(dp, 3) if dp is not None else None)],
            ["SSIM", b.get("ssim"), c.get("ssim"), None],
        ]
        if improved and dp is not None:
            verdict = (f"PASS: フローター改善 かつ ΔPSNR={dp:+.2f} ≥ {dpsnr_min}"
                       if dp >= float(dpsnr_min) else
                       f"FAIL(品質劣化): ΔPSNR={dp:+.2f} < {dpsnr_min}")
        elif improved:
            verdict = "フローター改善（PSNR未取得→品質未検証, 手入力可）"
        else:
            verdict = "FAIL: フローター指標(a) が改善せず"
        # 保存
        os.makedirs(os.path.join(outdir, "metrics"), exist_ok=True)
        with open(os.path.join(outdir, "metrics", "report.json"), "w", encoding="utf-8") as f:
            json.dump({"baseline": b, "cleaned": c, "verdict": verdict}, f,
                      ensure_ascii=False, indent=2, default=str)
        return rows, f"### 判定: {verdict}"
    except Exception as e:
        return [], f"⚠️ 失敗: {e}"


# ---------------------------------------------------------------- helpers
def _pct(v):
    return "n/a" if v is None else f"{v*100:.2f}%"


def _draw_box(ax, mn, mx):
    mn = np.asarray(mn); mx = np.asarray(mx)
    import itertools
    corners = np.array(list(itertools.product(*zip(mn, mx))))
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), c="green", lw=1.0)


# ---------------------------------------------------------------- UI
def build_ui():
    with gr.Blocks(title="sano フローター低減クリーンアップ") as demo:
        gr.Markdown("# sano フローター低減クリーンアップ\n"
                    "学習前点群クリーンアップ → A/B 学習比較の検証ツール。"
                    "GPU不要の工程はこのアプリで実行、GPU学習(T4)はコマンド生成で実行。")
        with gr.Row():
            sparse_in = gr.Textbox(DEFAULT_SPARSE, label="COLMAP sparse/0")
            outdir_in = gr.Textbox(DEFAULT_OUTDIR, label="出力 exp ディレクトリ")

        with gr.Tab("T0 データ確認"):
            b0 = gr.Button("読み込んで確認")
            md0 = gr.Markdown()
            b0.click(summarize_data, [sparse_in], [md0])

        with gr.Tab("T2 メトリクス"):
            knn_in = gr.Slider(5, 50, value=20, step=1, label="kNN k")
            b2 = gr.Button("メトリクス計算")
            md2 = gr.Markdown(); plot2 = gr.Plot()
            b2.click(run_metrics, [sparse_in, outdir_in, knn_in], [md2, plot2])

        with gr.Tab("T3 SOR クリーンアップ"):
            with gr.Row():
                nb_in = gr.Number(20, label="nb_neighbors", precision=0)
                sweep_in = gr.Textbox("1.5,2.0,3.0", label="std_ratio sweep")
                maxrem_in = gr.Number(1.0, label="ROI内削除率 上限(%)")
            with gr.Row():
                roi_mode = gr.Radio(["auto", "manual"], value="auto", label="ROI")
                roi_lo = gr.Number(2.5, label="auto percentile lo")
                roi_hi = gr.Number(97.5, label="auto percentile hi")
            roi_bbox = gr.Textbox("", label="manual ROI bbox: xmin,ymin,zmin,xmax,ymax,zmax")
            b3 = gr.Button("スイープ実行")
            tbl3 = gr.Dataframe(headers=["std_ratio", "removed", "total%", "ROI%", "gate"],
                                label="スイープ結果")
            gate3 = gr.Markdown()
            b3.click(run_sweep, [sparse_in, outdir_in, nb_in, sweep_in, roi_mode,
                                 roi_lo, roi_hi, roi_bbox, maxrem_in], [tbl3, gate3])
            gr.Markdown("---")
            adopt_in = gr.Number(2.0, label="採用 std_ratio")
            b3b = gr.Button("採用して cleaned_sor を生成", variant="primary")
            md3 = gr.Markdown(); plot3 = gr.Plot()
            b3b.click(adopt_clean, [sparse_in, outdir_in, nb_in, adopt_in, roi_mode,
                                    roi_lo, roi_hi, roi_bbox], [md3, plot3])

        with gr.Tab("T4 学習コマンド生成"):
            with gr.Row():
                bin_in = gr.Textbox("gaussian_splatting_cuda", label="LichtFeld 実行ファイル")
                orig_in = gr.Textbox(DEFAULT_ORIG, label="オリジナル sano (images/ 有)")
            with gr.Row():
                proj_in = gr.Textbox(DEFAULT_OUTDIR + "/cleaned_proj", label="cleaned 学習プロジェクト")
                cleaned_sparse_in = gr.Textbox(DEFAULT_OUTDIR + "/cleaned_sor/sparse/0",
                                               label="cleaned_sor/sparse/0")
            bprep = gr.Button("cleaned 学習プロジェクトを準備 (images をリンク)")
            md_prep = gr.Markdown()
            bprep.click(do_prepare, [orig_in, cleaned_sparse_in, proj_in], [md_prep])
            with gr.Row():
                strat_in = gr.Dropdown(["mcmc", "default"], value="mcmc", label="strategy")
                iter_in = gr.Number(15000, label="iter", precision=0)
                cap_in = gr.Number(1000000, label="max-cap", precision=0)
                rz_in = gr.Number(1, label="resize", precision=0)
            bcmd = gr.Button("コマンド生成", variant="primary")
            cmd_out = gr.Code(label="学習コマンド（同一設定で baseline / cleaned）")
            bcmd.click(gen_commands, [bin_in, orig_in, proj_in, outdir_in, strat_in,
                                      iter_in, cap_in, rz_in], [cmd_out])

        with gr.Tab("T5 A/B 比較"):
            with gr.Row():
                bdir = gr.Textbox(DEFAULT_OUTDIR + "/train_baseline", label="train_baseline")
                cdir = gr.Textbox(DEFAULT_OUTDIR + "/train_cleaned", label="train_cleaned")
            with gr.Row():
                tau = gr.Number(0.05, label="tau_op"); sf = gr.Number(0.02, label="scale_frac")
                ex = gr.Number(0.25, label="expand"); dpm = gr.Number(-0.3, label="ΔPSNR下限")
            with gr.Row():
                bp = gr.Textbox("", label="baseline PSNR(任意)"); cp = gr.Textbox("", label="cleaned PSNR(任意)")
                bs = gr.Textbox("", label="baseline SSIM(任意)"); cs = gr.Textbox("", label="cleaned SSIM(任意)")
            b5 = gr.Button("比較を実行", variant="primary")
            tbl5 = gr.Dataframe(headers=["指標", "baseline", "cleaned", "差分"], label="比較表")
            md5 = gr.Markdown()
            b5.click(run_compare, [bdir, cdir, sparse_in, outdir_in, tau, sf, ex, dpm,
                                   bp, cp, bs, cs], [tbl5, md5])
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
