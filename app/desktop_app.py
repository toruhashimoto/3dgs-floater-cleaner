"""FloaterClean Trainer — ワンクリック低フローター 3DGS 学習（Tkinter デスクトップ）。

RealityScan のアライメントを COLMAP 形式でエクスポートしたフォルダ（images/ + sparse/0、
F:/RealityScan/sano と同一構造）を選び、検証済み `scale_reg` 設定で LichtFeld をヘッドレス
学習して、フローターを抑えた .ply を出力する。学習後に floater(a) 数を計測表示する。

設計: docs/superpowers/specs/2026-06-20-floaterclean-desktop-tool-design.md
起動: app/run_desktop.bat（ダブルクリック）
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_REPO, "scripts")
_CONFIGS = os.path.join(_REPO, "configs")
BASE_CONFIG = os.path.join(_CONFIGS, "lichtfeld_scalereg02_prod30k.json")

DEFAULT_LICHTFELD = r"F:\LichtFeld-Studio\build\Release\LichtFeld-Studio.exe"

# プリセット（強度）と品質
PRESETS = {
    "標準 (scale_reg=0.02・推奨)": 0.02,
    "強め (0.04・floater最大減/PSNR微低)": 0.04,
    "オフ (0.0042・baseline相当)": 0.0042,
}
QUALITY = {
    "本番 (30000 iter)": 30000,
    "短時間 (15000 iter)": 15000,
}


# --------------------------------------------------------------------------- #
# GUI 非依存のロジック（単体テスト対象）                                          #
# --------------------------------------------------------------------------- #
def locate_lichtfeld(explicit: str | None = None) -> str | None:
    """LichtFeld 実行ファイルを解決。手動指定 → $LICHTFELD_EXE → 既定パス → PATH。"""
    for cand in (explicit, os.environ.get("LICHTFELD_EXE"), DEFAULT_LICHTFELD):
        if cand and os.path.isfile(cand):
            return cand
    for name in ("LichtFeld-Studio", "LichtFeld-Studio.exe", "gaussian_splatting_cuda"):
        hit = shutil.which(name)
        if hit:
            return hit
    return None


def _has_member(sparse_dir: str, stem: str) -> bool:
    return (os.path.isfile(os.path.join(sparse_dir, stem + ".txt"))
            or os.path.isfile(os.path.join(sparse_dir, stem + ".bin")))


def validate_dataset(data_dir: str):
    """RealityScan→COLMAP エクスポート構造を検証。
    返り値: (ok: bool, message: str, sparse_dir: str | None)"""
    if not data_dir or not os.path.isdir(data_dir):
        return False, "データフォルダを選択してください。", None
    images = os.path.join(data_dir, "images")
    if not os.path.isdir(images):
        return False, f"images/ が見つかりません: {images}", None
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    has_img = any(f.lower().endswith(exts) for f in os.listdir(images))
    if not has_img:
        return False, "images/ に画像がありません。", None
    sparse = os.path.join(data_dir, "sparse", "0")
    if not os.path.isdir(sparse):
        return False, (f"sparse/0/ が見つかりません: {sparse}\n"
                       "RealityScan を COLMAP 形式でエクスポートしてください。"), None
    for stem in ("cameras", "images", "points3D"):
        if not _has_member(sparse, stem):
            return False, (f"sparse/0/{stem}.txt が見つかりません。\n"
                           "RealityScan を COLMAP 形式でエクスポートしてください。"), None
    return True, "OK", sparse


def build_config(scale_reg: float, iterations: int, out_dir: str,
                 base_path: str = BASE_CONFIG) -> str:
    """検証済み base config を読み、scale_reg / iterations を反映した一時 config を out_dir に書く。
    適用モードは全画像学習（holdout 無し）なので eval は無効化。返り値: 生成 config パス。"""
    with open(base_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["scale_reg"] = float(scale_reg)
    cfg["iterations"] = int(iterations)
    cfg["eval_steps"] = [int(iterations)]
    cfg["save_steps"] = [int(iterations)]
    cfg["enable_eval"] = False
    cfg["enable_save_eval_images"] = False
    # opacity_reg はベース値(0.0042)のまま固定（上げると PSNR 劣化・floater 定義交絡）
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "_run_config.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return out


def build_command(exe: str, config_path: str, data_dir: str, out_dir: str,
                  undistort: bool = False) -> list[str]:
    cmd = [exe, "--headless", "--config", config_path,
           "--data-path", data_dir, "--output-path", out_dir, "-r", "1"]
    if undistort:
        cmd.append("--undistort")
    return cmd


def find_output_ply(out_dir: str) -> str | None:
    """出力ディレクトリ内で最大 iteration の splat .ply を返す。"""
    plys = glob.glob(os.path.join(out_dir, "**", "*.ply"), recursive=True)
    if not plys:
        return None

    def it(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    plys.sort(key=it)
    return plys[-1]


def measure_floaters(ply: str, sparse_dir: str):
    """出力 .ply の floater(a) 数を計測。numpy/scipy 無ければ None。"""
    try:
        if _SCRIPTS not in sys.path:
            sys.path.insert(0, _SCRIPTS)
        import floater_metrics as fm  # noqa
        r = fm.compute_floater_metrics(ply, sparse_dir)
        return {"floater_a": r["metric_a_lowopacity_bigscale"],
                "total": r["total_gaussians"],
                "scene_diag": r.get("scene_diagonal")}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def run_training(cmd: list[str], log_path: str, on_line, should_stop) -> int:
    """LichtFeld を subprocess 実行。各行を on_line(line) に渡しログにも書く。
    should_stop() が True を返すと終了。返り値: returncode（中止時 -1）。"""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        lf.write(" ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env)
        try:
            for raw in iter(proc.stdout.readline, b""):
                if should_stop():
                    proc.terminate()
                    on_line("[中止] 学習を停止しました。")
                    proc.wait(timeout=10)
                    return -1
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line:
                    lf.write(line + "\n")
                    lf.flush()
                    on_line(line)
        finally:
            if proc.stdout:
                proc.stdout.close()
        return proc.wait()


# --------------------------------------------------------------------------- #
# Tkinter GUI                                                                  #
# --------------------------------------------------------------------------- #
def _has_gpu() -> bool:
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:  # noqa: BLE001
        return False


class DesktopApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk, scrolledtext

        self.tk = tk
        self.root = root
        self._stop = threading.Event()
        self._worker = None
        root.title("FloaterClean Trainer — 低フローター 3DGS 学習")
        root.geometry("820x620")
        pad = dict(padx=8, pady=4)

        frm = ttk.Frame(root)
        frm.pack(fill="x", **pad)

        # データフォルダ
        ttk.Label(frm, text="データフォルダ (RealityScan→COLMAP: images/ + sparse/0)").grid(row=0, column=0, sticky="w", columnspan=3)
        self.data_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.data_var, width=80).grid(row=1, column=0, columnspan=2, sticky="we")
        ttk.Button(frm, text="参照…", command=self._pick_data).grid(row=1, column=2, sticky="e")

        # 出力フォルダ
        ttk.Label(frm, text="出力フォルダ").grid(row=2, column=0, sticky="w", columnspan=3)
        self.out_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.out_var, width=80).grid(row=3, column=0, columnspan=2, sticky="we")
        ttk.Button(frm, text="参照…", command=self._pick_out).grid(row=3, column=2, sticky="e")

        # LichtFeld
        ttk.Label(frm, text="LichtFeld 実行ファイル").grid(row=4, column=0, sticky="w", columnspan=3)
        self.exe_var = tk.StringVar(value=locate_lichtfeld() or "")
        ttk.Entry(frm, textvariable=self.exe_var, width=80).grid(row=5, column=0, columnspan=2, sticky="we")
        ttk.Button(frm, text="参照…", command=self._pick_exe).grid(row=5, column=2, sticky="e")

        frm.columnconfigure(0, weight=1)

        # オプション行
        opt = ttk.Frame(root)
        opt.pack(fill="x", **pad)
        ttk.Label(opt, text="強度").grid(row=0, column=0, sticky="w")
        self.preset_var = tk.StringVar(value=list(PRESETS)[0])
        ttk.OptionMenu(opt, self.preset_var, list(PRESETS)[0], *PRESETS).grid(row=0, column=1, sticky="w")
        ttk.Label(opt, text="  品質").grid(row=0, column=2, sticky="w")
        self.quality_var = tk.StringVar(value=list(QUALITY)[0])  # 本番30000 が既定
        ttk.OptionMenu(opt, self.quality_var, list(QUALITY)[0], *QUALITY).grid(row=0, column=3, sticky="w")

        self.measure_var = tk.BooleanVar(value=True)   # floater 計測 既定 ON
        ttk.Checkbutton(opt, text="学習後に floater 数を計測", variable=self.measure_var).grid(row=1, column=0, columnspan=2, sticky="w")
        self.undistort_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="歪み補正 (--undistort)", variable=self.undistort_var).grid(row=1, column=2, columnspan=2, sticky="w")

        # ボタン
        btns = ttk.Frame(root)
        btns.pack(fill="x", **pad)
        self.start_btn = ttk.Button(btns, text="学習開始", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="中止", command=self._cancel, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.open_btn = ttk.Button(btns, text="出力フォルダを開く", command=self._open_out, state="disabled")
        self.open_btn.pack(side="left")

        # 進捗
        self.prog = ttk.Progressbar(root, mode="indeterminate")
        self.prog.pack(fill="x", **pad)
        self.status = tk.StringVar(value="待機中" + ("" if _has_gpu() else "  ⚠️ GPU(nvidia-smi)未検出"))
        ttk.Label(root, textvariable=self.status).pack(fill="x", padx=8)

        # ログ
        self.log = scrolledtext.ScrolledText(root, height=20, state="disabled", wrap="none")
        self.log.pack(fill="both", expand=True, padx=8, pady=6)

        self.data_var.trace_add("write", lambda *_: self._autofill_out())

    # ---- UI helpers ----
    def _pick_data(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="RealityScan→COLMAP フォルダを選択")
        if d:
            self.data_var.set(d)

    def _pick_out(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="出力フォルダを選択")
        if d:
            self.out_var.set(d)

    def _pick_exe(self):
        from tkinter import filedialog
        f = filedialog.askopenfilename(title="LichtFeld-Studio.exe を選択",
                                       filetypes=[("exe", "*.exe"), ("all", "*.*")])
        if f:
            self.exe_var.set(f)

    def _autofill_out(self):
        d = self.data_var.get().strip()
        if d and not self.out_var.get().strip():
            self.out_var.set(os.path.join(d, "exp", "train_floaterclean"))

    def _open_out(self):
        out = self.out_var.get().strip()
        if out and os.path.isdir(out):
            os.startfile(out)  # noqa: B606 (Windows)

    def _logln(self, s):
        self.log.configure(state="normal")
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_running(self, running):
        self.start_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        if running:
            self.prog.start(12)
        else:
            self.prog.stop()

    # ---- actions ----
    def _start(self):
        data = self.data_var.get().strip()
        out = self.out_var.get().strip()
        exe = self.exe_var.get().strip()
        ok, msg, sparse = validate_dataset(data)
        if not ok:
            from tkinter import messagebox
            messagebox.showerror("データ不正", msg)
            return
        if not exe or not os.path.isfile(exe):
            from tkinter import messagebox
            messagebox.showerror("LichtFeld 未検出", "LichtFeld 実行ファイルを指定してください。")
            return
        if not out:
            out = os.path.join(data, "exp", "train_floaterclean")
            self.out_var.set(out)
        scale_reg = PRESETS[self.preset_var.get()]
        iters = QUALITY[self.quality_var.get()]
        undistort = self.undistort_var.get()
        measure = self.measure_var.get()

        self._stop.clear()
        self._set_running(True)
        self.open_btn.configure(state="disabled")
        self.status.set(f"学習中… scale_reg={scale_reg} iter={iters}")
        self._logln(f"=== FloaterClean: scale_reg={scale_reg}, iter={iters}, data={data} ===")

        def work():
            try:
                cfg = build_config(scale_reg, iters, out)
                cmd = build_command(exe, cfg, data, out, undistort)
                self._post(self._logln, "$ " + " ".join(cmd))
                rc = run_training(cmd, os.path.join(out, "train.log"),
                                  on_line=lambda ln: self._post(self._logln, ln),
                                  should_stop=self._stop.is_set)
                if rc == -1:
                    self._post(self.status.set, "中止しました。")
                    return
                if rc != 0:
                    self._post(self.status.set, f"⚠️ 学習が異常終了 (code {rc})。ログ確認。")
                    return
                ply = find_output_ply(out)
                tail = f"✅ 完了: {ply}" if ply else "✅ 完了（.ply 未検出）"
                if ply and measure:
                    self._post(self.status.set, "floater 計測中…")
                    m = measure_floaters(ply, sparse)
                    if m and "floater_a" in m:
                        tail += f"  | floater(a)={m['floater_a']:,} / GS {m['total']:,}"
                    elif m and "error" in m:
                        tail += f"  | floater計測スキップ ({m['error'].splitlines()[0][:60]})"
                self._post(self.status.set, tail)
                self._post(self._logln, tail)
                self._post(lambda: self.open_btn.configure(state="normal"))
            except Exception as e:  # noqa: BLE001
                self._post(self.status.set, f"⚠️ エラー: {e}")
                self._post(self._logln, f"⚠️ エラー: {e}")
            finally:
                self._post(lambda: self._set_running(False))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _cancel(self):
        self._stop.set()
        self.status.set("中止要求中…")

    def _post(self, fn, *args):
        """ワーカースレッドから GUI スレッドへ安全に反映。"""
        self.root.after(0, lambda: fn(*args))


def main():
    import tkinter as tk
    root = tk.Tk()
    DesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
