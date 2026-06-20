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

import desktop_app as da  # noqa: E402


def _make_dataset(root, with_sparse=True, with_images=True, ext="txt"):
    if with_images:
        os.makedirs(os.path.join(root, "images"))
        open(os.path.join(root, "images", "0001.jpg"), "w").close()
    if with_sparse:
        s = os.path.join(root, "sparse", "0")
        os.makedirs(s)
        for stem in ("cameras", "images", "points3D"):
            open(os.path.join(s, f"{stem}.{ext}"), "w").close()


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
        p = da.build_config(0.02, 30000, d)
        cfg = json.load(open(p, encoding="utf-8"))
        assert cfg["scale_reg"] == 0.02 and cfg["strategy"] == "mcmc"


def test_build_command():
    cmd = da.build_command("L.exe", "c.json", "D:/data", "D:/out")
    assert cmd[:2] == ["L.exe", "--headless"]
    assert "--config" in cmd and "--data-path" in cmd and "--output-path" in cmd
    assert "-r" in cmd and "1" in cmd
    assert "--undistort" not in cmd
    assert "--undistort" in da.build_command("L.exe", "c.json", "d", "o", undistort=True)


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
