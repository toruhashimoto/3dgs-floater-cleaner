"""Unit tests for app/desktop_app.py logic (GUI-independent).

Run:  python tests/test_desktop_app.py     (prints PASS/FAIL, exit 0 if all pass)
   or: python -m pytest tests/test_desktop_app.py
"""
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "app"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import desktop_app as da  # noqa: E402
import prune_missing_images as pmi  # noqa: E402


def _make_dataset(root, with_sparse=True, with_images=True, ext="txt"):
    if with_images:
        os.makedirs(os.path.join(root, "images"))
        open(os.path.join(root, "images", "0001.jpg"), "w").close()
    if with_sparse:
        s = os.path.join(root, "sparse", "0")
        os.makedirs(s)
        for stem in ("cameras", "images", "points3D"):
            open(os.path.join(s, f"{stem}.{ext}"), "w").close()


def _make_colmap(root, referenced, present):
    """images/ に present の空画像を作り、sparse/0 に referenced を参照する
    images.txt（COLMAP 2行/画像）と空 cameras/points3D を作る。返り値: sparse_dir。"""
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    for n in present:
        open(os.path.join(root, "images", n), "w").close()
    s = os.path.join(root, "sparse", "0")
    os.makedirs(s, exist_ok=True)
    open(os.path.join(s, "cameras.txt"), "w").close()
    open(os.path.join(s, "points3D.txt"), "w").close()
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(referenced)}, mean observations per image: 1",
    ]
    for i, n in enumerate(referenced, start=1):
        lines.append(f"{i} 1 0 0 0 0 0 0 7 {n}")     # 姿勢行
        lines.append(f"{100 + i}.5 {200 + i}.5 {i}")  # POINTS2D 行
    with open(os.path.join(s, "images.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return s


def test_validate_dataset_ok():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d)
        ok, msg, sparse = da.validate_dataset(d)
        assert ok, msg
        assert sparse.endswith(os.path.join("sparse", "0"))


def test_validate_dataset_ok_bin():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, ext="bin")
        ok, _, _ = da.validate_dataset(d)
        assert ok


def test_validate_dataset_missing_sparse():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, with_sparse=False)
        ok, msg, _ = da.validate_dataset(d)
        assert not ok and "sparse" in msg.lower()


def test_validate_dataset_missing_images():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, with_images=False)
        ok, msg, _ = da.validate_dataset(d)
        assert not ok and "images" in msg.lower()


def test_validate_dataset_detects_missing_referenced_image():
    # この回帰がまさに新環境の不具合: sparse は 3 枚参照だが images/ に 2 枚しかない。
    with tempfile.TemporaryDirectory() as d:
        _make_colmap(d, referenced=["0001.jpg", "0002.jpg", "0003.jpg"],
                     present=["0001.jpg", "0003.jpg"])
        ok, msg, sparse = da.validate_dataset(d)
        assert not ok
        assert sparse is None
        assert "0002.jpg" in msg     # 欠落名を提示
        assert "1/3" in msg          # 3 枚中 1 枚欠落


def test_validate_dataset_ok_when_all_referenced_present():
    with tempfile.TemporaryDirectory() as d:
        _make_colmap(d, referenced=["0001.jpg", "0002.jpg"],
                     present=["0001.jpg", "0002.jpg"])
        ok, msg, sparse = da.validate_dataset(d)
        assert ok, msg
        assert sparse.endswith(os.path.join("sparse", "0"))


def test_referenced_image_names_parses_pose_lines_only():
    with tempfile.TemporaryDirectory() as d:
        s = _make_colmap(d, referenced=["0001.jpg", "0002.jpg"], present=[])
        assert da.referenced_image_names(s) == ["0001.jpg", "0002.jpg"]


def test_prune_removes_missing_entry_and_backs_up():
    with tempfile.TemporaryDirectory() as d:
        s = _make_colmap(d, referenced=["0001.jpg", "0002.jpg", "0003.jpg"],
                         present=["0001.jpg", "0003.jpg"])
        assert pmi.prune(d, dry_run=False) == 0
        # 原本はバックアップされ、images.txt は欠落エントリを除いて自己整合。
        assert os.path.isfile(os.path.join(s, "images.txt.bak"))
        assert da.referenced_image_names(s) == ["0001.jpg", "0003.jpg"]
        txt = open(os.path.join(s, "images.txt"), encoding="utf-8").read()
        assert "# Number of images: 2" in txt
        assert "0002.jpg" not in txt
        # POINTS2D 行は保持点のものだけ残る（0002 の POINTS2D=102.5.. は消える）
        assert "101.5" in txt and "103.5" in txt and "102.5" not in txt
        ok, msg, _ = da.validate_dataset(d)   # 修復後は検証を通る
        assert ok, msg


def test_prune_dry_run_does_not_write():
    with tempfile.TemporaryDirectory() as d:
        s = _make_colmap(d, referenced=["0001.jpg", "0002.jpg"], present=["0001.jpg"])
        before = open(os.path.join(s, "images.txt"), encoding="utf-8").read()
        assert pmi.prune(d, dry_run=True) == 0
        after = open(os.path.join(s, "images.txt"), encoding="utf-8").read()
        assert before == after
        assert not os.path.isfile(os.path.join(s, "images.txt.bak"))


def test_prune_no_missing_is_noop():
    with tempfile.TemporaryDirectory() as d:
        s = _make_colmap(d, referenced=["0001.jpg"], present=["0001.jpg"])
        assert pmi.prune(d, dry_run=False) == 0
        assert not os.path.isfile(os.path.join(s, "images.txt.bak"))  # 書換なし


# ---- GUI 自動修復のための検出関数 ----
def test_missing_referenced_images_partial():
    with tempfile.TemporaryDirectory() as d:
        _make_colmap(d, referenced=["0001.jpg", "0002.jpg", "0003.jpg"],
                     present=["0001.jpg", "0003.jpg"])
        assert da.missing_referenced_images(d) == ["0002.jpg"]


def test_missing_referenced_images_all_present_empty():
    with tempfile.TemporaryDirectory() as d:
        _make_colmap(d, referenced=["0001.jpg"], present=["0001.jpg"])
        assert da.missing_referenced_images(d) == []


def test_missing_referenced_images_no_images_empty():
    # images/ に画像皆無は通常の構造エラー扱い → 修復対象にしない
    with tempfile.TemporaryDirectory() as d:
        _make_colmap(d, referenced=["0001.jpg", "0002.jpg"], present=[])
        assert da.missing_referenced_images(d) == []


def test_missing_referenced_images_all_missing_empty():
    # 参照が「全部」欠落＝フォルダ取り違え等の異常 → prune では直さない
    with tempfile.TemporaryDirectory() as d:
        _make_colmap(d, referenced=["0001.jpg", "0002.jpg"], present=["other.jpg"])
        assert da.missing_referenced_images(d) == []


# ---- 再利用可能な修復コア (analyze / apply_prune) ----
def test_analyze_reports_counts():
    with tempfile.TemporaryDirectory() as d:
        _make_colmap(d, referenced=["0001.jpg", "0002.jpg", "0003.jpg"],
                     present=["0001.jpg", "0003.jpg"])
        a = pmi.analyze(d)
        assert a["ok"] and a["total"] == 3 and a["present"] == 2
        assert a["missing"] == ["0002.jpg"] and a["kept"] == 2


def test_apply_prune_idempotent():
    with tempfile.TemporaryDirectory() as d:
        s = _make_colmap(d, referenced=["0001.jpg", "0002.jpg"], present=["0001.jpg"])
        r1 = pmi.apply_prune(d)
        assert r1["changed"] and r1["removed"] == ["0002.jpg"] and r1["kept"] == 1
        assert r1["backup"] == os.path.join(s, "images.txt.bak")
        # 2 回目は欠落が無いので何もしない
        r2 = pmi.apply_prune(d)
        assert r2["changed"] is False and r2["removed"] == []


def test_apply_prune_does_not_overwrite_existing_bak():
    with tempfile.TemporaryDirectory() as d:
        s = _make_colmap(d, referenced=["0001.jpg", "0002.jpg"], present=["0001.jpg"])
        bak = os.path.join(s, "images.txt.bak")
        with open(bak, "w", newline="\n") as f:
            f.write("SENTINEL-ORIGINAL\n")          # 既存 .bak（最初の原本想定）
        r = pmi.apply_prune(d)
        assert r["changed"] and r["backup"] is None  # 新規バックアップは作らない
        assert open(bak, encoding="utf-8").read() == "SENTINEL-ORIGINAL\n"
        assert da.referenced_image_names(s) == ["0001.jpg"]  # 本体は修復済み


def test_build_config_overrides_and_preserves():
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.json")
        json.dump({"iterations": 30000, "scale_reg": 0.02, "opacity_reg": 0.0042,
                   "strategy": "mcmc", "max_cap": 1000000, "eval_steps": [30000],
                   "save_steps": [30000], "enable_eval": True,
                   "enable_save_eval_images": True}, open(base, "w"))
        out = os.path.join(d, "out")
        p = da.build_config(0.04, 15000, out, base_path=base)
        cfg = json.load(open(p, encoding="utf-8"))
        assert cfg["scale_reg"] == 0.04
        assert cfg["iterations"] == 15000
        assert cfg["eval_steps"] == [15000] and cfg["save_steps"] == [15000]
        assert cfg["enable_eval"] is False and cfg["enable_save_eval_images"] is False
        # preserved from base (NOT touched):
        assert cfg["opacity_reg"] == 0.0042
        assert cfg["strategy"] == "mcmc" and cfg["max_cap"] == 1000000


def test_build_config_real_base():
    # the committed production base must exist and produce a valid config
    assert os.path.isfile(da.BASE_CONFIG), da.BASE_CONFIG
    with tempfile.TemporaryDirectory() as d:
        # strategy 未指定なら検証済み base の値(mcmc)を保持（base ファイルは pristine）
        cfg = json.load(open(da.build_config(0.02, 30000, d), encoding="utf-8"))
        assert cfg["scale_reg"] == 0.02 and cfg["strategy"] == "mcmc"
        # アプリ経路: strategy / max_cap を上書きできる（既定は MRNF）
        cfg2 = json.load(open(da.build_config(0.02, 30000, os.path.join(d, "2"),
                                              max_cap=5_000_000, strategy=da.STRATEGY),
                              encoding="utf-8"))
        assert cfg2["strategy"] == "mrnf" and cfg2["max_cap"] == 5_000_000


def test_build_config_max_cap_and_stop_refine_scaling():
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.json")
        json.dump({"iterations": 30000, "scale_reg": 0.02, "opacity_reg": 0.0042,
                   "strategy": "mcmc", "max_cap": 1000000, "stop_refine": 25000,
                   "eval_steps": [30000], "save_steps": [30000],
                   "enable_eval": True, "enable_save_eval_images": True}, open(base, "w"))
        # 105k/5M・MRNF: iterations を 3.5x にすると stop_refine も 25000→87500 に比例延長
        cfg = json.load(open(da.build_config(0.02, 105000, os.path.join(d, "hi"),
                                             max_cap=5_000_000, strategy="mrnf",
                                             base_path=base), encoding="utf-8"))
        assert cfg["iterations"] == 105000
        assert cfg["max_cap"] == 5_000_000
        assert cfg["strategy"] == "mrnf"
        assert cfg["stop_refine"] == 87500          # round(25000 * 105000/30000)
        # 係数1.0（iterations=base）なら stop_refine 不変＝検証済み挙動を維持
        cfg2 = json.load(open(da.build_config(0.02, 30000, os.path.join(d, "std"),
                                              max_cap=1_000_000, base_path=base),
                              encoding="utf-8"))
        assert cfg2["stop_refine"] == 25000 and cfg2["max_cap"] == 1_000_000


def test_detail_presets():
    assert len(da.DETAIL) == 5
    assert da.DETAIL_DEFAULT in da.DETAIL
    assert da.DETAIL[da.DETAIL_DEFAULT] == (105000, 5_000_000)   # 既定=指定の高詳細
    iters = [v[0] for v in da.DETAIL.values()]
    caps = [v[1] for v in da.DETAIL.values()]
    assert iters == sorted(iters) and caps == sorted(caps)       # 単調増加の階段
    assert da.STRATEGY == "mrnf"


def test_build_command():
    cmd = da.build_command("L.exe", "c.json", "D:/data", "D:/out")
    assert cmd[:2] == ["L.exe", "--headless"]
    assert "--config" in cmd and "--data-path" in cmd and "--output-path" in cmd
    assert "-r" in cmd and "1" in cmd
    assert "--undistort" not in cmd
    assert "--undistort" in da.build_command("L.exe", "c.json", "d", "o", undistort=True)
    # MRNF ディテール強化フラグ（既定 OFF、detail_maps=True で両方付与）
    assert "--use-error-map" not in cmd and "--use-edge-map" not in cmd
    c2 = da.build_command("L.exe", "c.json", "d", "o", detail_maps=True)
    assert "--use-error-map" in c2 and "--use-edge-map" in c2


def test_gen_detail_configs():
    import gen_detail_configs as gen
    with tempfile.TemporaryDirectory() as d:
        files = gen.generate(d)
        assert len(files) == len(da.DETAIL) == 5
        seen = set()
        for f in files:
            assert os.path.isfile(f)
            c = json.load(open(f, encoding="utf-8"))
            assert c["strategy"] == "mrnf" and c["scale_reg"] == 0.02
            assert (c["iterations"], c["max_cap"]) in da.DETAIL.values()
            assert c["enable_eval"] is False
            seen.add((c["iterations"], c["max_cap"]))
        assert seen == set(da.DETAIL.values())   # 5段すべて網羅・重複なし
        # 既定プリセット(105k/5M)のファイルが存在し中身が一致
        p = os.path.join(d, "lichtfeld_mrnf_105k_5M.json")
        assert os.path.isfile(p)
        c = json.load(open(p, encoding="utf-8"))
        assert c["iterations"] == 105000 and c["max_cap"] == 5_000_000 and c["stop_refine"] == 87500


def test_locate_lichtfeld_explicit():
    with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tf:
        path = tf.name
    try:
        assert da.locate_lichtfeld(path) == path
    finally:
        os.unlink(path)
    assert da.locate_lichtfeld(r"X:\nope\missing.exe") in (None,) or \
        os.path.isfile(da.locate_lichtfeld(r"X:\nope\missing.exe") or "")


def test_find_output_ply_picks_max_iter():
    with tempfile.TemporaryDirectory() as d:
        sub = os.path.join(d, "ply")
        os.makedirs(sub)
        open(os.path.join(sub, "point_cloud_700.ply"), "w").close()
        open(os.path.join(d, "splat_15000.ply"), "w").close()
        assert os.path.basename(da.find_output_ply(d)) == "splat_15000.ply"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
