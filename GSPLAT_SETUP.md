# gsplat を Blackwell(RTX 50系/sm_120) + Windows + torch 2.9 で動かす（深度正則化 A/B 用）

floater(b) 検証で gsplat(nerfstudio simple_trainer) を使うために必要だった環境構築の記録。
同じ落とし穴に再度はまらないための手順。データ/学習出力は本リポジトリに含めない（`.gitignore`）。

## 前提
- GPU: RTX 5070 Ti (Blackwell, compute 12.0 / sm_120)
- venv に **torch 2.9.1+cu128**（`torch.cuda.get_arch_list()` に `sm_120` を含むこと）
- CUDA Toolkit **12.8**（torch の cu128 と一致, nvcc）, VS2022 BuildTools (cl 14.4x)

## ビルド環境（コンパイル時に必須の env）
```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
set "VSLANG=1033"   REM 重要: cl/nvcc 出力を英語化。日本語(cp932)だと torch の oem デコードが UnicodeDecodeError でクラッシュ
```

## 4つの落とし穴と対処
1. **gsplat の JIT が MSVC で D8021**：`gsplat/cuda/_backend.py` が `extra_cflags=["-O3","-Wno-attributes"]`（GCC用）を cl に渡す。
   → Windows 分岐で `extra_cflags = ["/O2"]` に差し替え（`os.name=="nt"`）。`extra_cuda_cflags`(-O3)は nvcc が受理するので温存。
2. **torch のエラー表示が cp932 でクラッシュ**：`SUBPROCESS_DECODE_ARGS=('oem',)` ハードコードで、nvcc/cl 出力をデコードできず本当のエラーが隠れる。
   → `VSLANG=1033` で出力を英語化。実エラーが必要なら torch_extensions の build dir で `ninja` を手動実行して直接捕捉。
3. **fused-ssim が torch2.9 でコンパイル不可**（`compiled_autograd.h` C2872 'std' ambiguous）。
   → `examples/fused_ssim.py` に**純torch の微分可能 SSIM shim** を置いて代替（`scripts/fused_ssim_shim.py` 参照）。両A/Bアームで同一SSIMなら妥当。
   インストール時は `--no-build-isolation`（pip 隔離環境に torch が無く `No module named torch` になる）。
4. **pycolmap(rmbrualla) が Py2/numpy2 非互換**：`np.uint64(-1)`(numpy2でOverflow) と txt ローダの `map()`(Py3でiterator)。
   → `scene_manager.py` の `np.uint64(-1)`→`np.uint64(2**64-1)`、`map(...)`→`list(map(...))`（cameras/images/points3D の txt ローダ計8箇所）。
   binary ローダは `struct.unpack('L', read(8))` が Windows で 4byte 扱いになり別途壊れるので、**text モデルを使う**（上記patchで可）。

## A/B 実行（gsplat-vs-gsplat, 唯一変数=depth_loss）
```bat
cd <gsplat_repo>\examples
python simple_trainer.py mcmc --data-dir F:\RealityScan\sano --data-factor 1 ^
  --result-dir <out> --max-steps 15000 --eval-steps 15000 --save-steps 15000 --ply-steps 15000 ^
  --save-ply --disable-viewer --disable-video --test-every 8 [--depth-loss --depth-lambda 0.01]
```
- `init_type=sfm`（既定）で COLMAP 疎点から初期化。`--depth-loss` は SfM 疎点深度の視差L1監督（外部単眼深度不要）。
- **注意**：gsplat は `normalize_world_space=True`（既定）で world を正規化するため、PLY 座標は元 COLMAP 系と別。floater 指標は **Parser の正規化カメラ**で算出すること（`scripts/floater_gsplat.py`）。
