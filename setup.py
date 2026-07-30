#!/usr/bin/env python
# ---------------------------------------------------------------------------
# BitVideo-1.58 / bit_gemm  --  build script
#
# The CUDA extension (`bitvideo._C`) is OPTIONAL.  BitVideo ships a complete
# pure-PyTorch reference path plus an optional Triton path, so the package is
# fully usable on machines without `nvcc`.  When a CUDA toolkit is present the
# hand written W1.58A8 kernels are compiled and become the default back-end.
#
#   Environment variables
#   ---------------------
#   BITVIDEO_SKIP_CUDA=1      never attempt to build the CUDA extension
#   BITVIDEO_FORCE_CUDA=1     fail loudly instead of degrading gracefully
#   BITVIDEO_CUDA_ARCH="80;86;89;90;100;120"
#                             explicit SM list (default: auto-detect + fat set)
#   BITVIDEO_DEBUG=1          -G -lineinfo, no fast math, assertions on
#   MAX_JOBS=<n>              parallel nvcc jobs (honoured by torch)
#
#   Usage
#   -----
#   pip install -e .                       # editable dev install
#   python setup.py build_ext --inplace    # just build the kernels
#   BITVIDEO_SKIP_CUDA=1 pip install -e .  # python-only install
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from setuptools import setup

HERE = Path(__file__).parent.resolve()
CUDA_DIR = HERE / "bitvideo" / "cuda"

IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _find_nvcc() -> str | None:
    """Locate nvcc via CUDA_HOME/CUDA_PATH or PATH."""
    exe = "nvcc.exe" if IS_WINDOWS else "nvcc"
    for var in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT"):
        root = os.environ.get(var)
        if root:
            cand = Path(root) / "bin" / exe
            if cand.exists():
                return str(cand)
    from shutil import which

    return which("nvcc")


def _nvcc_version(nvcc: str) -> tuple[int, int]:
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return (0, 0)
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


# Minimum nvcc (major, minor) required for each SM target.
_SM_REQUIREMENTS: dict[str, tuple[int, int]] = {
    "80": (11, 0),   # A100
    "86": (11, 1),   # RTX 30xx / A10 / A40
    "87": (11, 4),   # Orin
    "89": (11, 8),   # RTX 40xx / L40 / L4
    "90": (12, 0),   # H100 / H200
    "100": (12, 8),  # B100 / B200  (datacenter Blackwell)
    "101": (12, 8),  # GB10
    "120": (12, 8),  # RTX 50xx     (consumer Blackwell)
}


def _resolve_arch_list(nvcc_ver: tuple[int, int]) -> list[str]:
    """Return the list of SM numbers we will emit code for."""
    explicit = os.environ.get("BITVIDEO_CUDA_ARCH")
    if explicit:
        wanted = [a.strip() for a in re.split(r"[;,\s]+", explicit) if a.strip()]
    else:
        wanted = ["80", "86", "89", "90", "100", "120"]
        # Add the local device so a source build is always optimal for the host.
        try:
            import torch

            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                local = f"{major}{minor}"
                if local not in wanted:
                    wanted.append(local)
        except Exception:
            pass

    usable, dropped = [], []
    for a in wanted:
        req = _SM_REQUIREMENTS.get(a, (12, 8))
        if nvcc_ver >= req:
            usable.append(a)
        else:
            dropped.append(a)
    if dropped:
        print(f"[bitvideo] nvcc {nvcc_ver[0]}.{nvcc_ver[1]} too old for SM {dropped}; skipping.")
    if not usable:
        usable = ["80"]
    # Sort numerically so the PTX fallback is generated from the newest arch.
    usable.sort(key=int)
    return usable


def _gencode_flags(archs: list[str]) -> list[str]:
    flags: list[str] = []
    for a in archs:
        flags += [f"-gencode=arch=compute_{a},code=sm_{a}"]
    # Forward-compatible PTX from the highest architecture we can build.
    flags += [f"-gencode=arch=compute_{archs[-1]},code=compute_{archs[-1]}"]
    return flags


# ---------------------------------------------------------------------------
# extension construction
# ---------------------------------------------------------------------------
def build_extensions() -> tuple[list, dict]:
    """Return (ext_modules, cmdclass).  Empty/no-op when CUDA is unavailable."""
    if _env_flag("BITVIDEO_SKIP_CUDA"):
        print("[bitvideo] BITVIDEO_SKIP_CUDA=1 -> building python-only package.")
        return [], {}

    force = _env_flag("BITVIDEO_FORCE_CUDA")

    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception as exc:  # pragma: no cover - torch is a build requirement
        msg = f"[bitvideo] torch unavailable at build time ({exc})."
        if force:
            raise RuntimeError(msg) from exc
        print(msg + " Building python-only package.")
        return [], {}

    nvcc = _find_nvcc()
    if nvcc is None:
        msg = (
            "[bitvideo] nvcc not found.  Set CUDA_HOME or add nvcc to PATH to build the\n"
            "           fused W1.58A8 kernels.  The package still works via the\n"
            "           Triton / PyTorch fallbacks."
        )
        if force:
            raise RuntimeError(msg)
        print(msg)
        return [], {}

    nvcc_ver = _nvcc_version(nvcc)
    if nvcc_ver < (12, 0):
        msg = f"[bitvideo] CUDA 12.0+ required, found {nvcc_ver[0]}.{nvcc_ver[1]}."
        if force:
            raise RuntimeError(msg)
        print(msg + "  Building python-only package.")
        return [], {}

    archs = _resolve_arch_list(nvcc_ver)
    print(f"[bitvideo] nvcc {nvcc_ver[0]}.{nvcc_ver[1]}  torch {torch.__version__}")
    print(f"[bitvideo] building for SM: {', '.join(archs)}")

    debug = _env_flag("BITVIDEO_DEBUG")

    sources = [
        str(CUDA_DIR / "kernels.cpp"),
        str(CUDA_DIR / "bit_gemm.cpp"),
        str(CUDA_DIR / "bit_packing.cpp"),
        str(CUDA_DIR / "bit_gemm_kernel.cu"),
        str(CUDA_DIR / "bit_gemm_launch.cu"),
        str(CUDA_DIR / "ternary_gemm.cu"),
        str(CUDA_DIR / "int8_gemm.cu"),
    ]
    missing = [s for s in sources if not Path(s).exists()]
    if missing:
        raise FileNotFoundError(f"[bitvideo] missing CUDA sources: {missing}")

    cxx_flags: list[str] = []
    nvcc_flags: list[str] = [
        "-O3" if not debug else "-O0",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--use_fast_math" if not debug else "-lineinfo",
        "-lineinfo",
        # Keep register pressure predictable across the whole extension; the
        # per-kernel __launch_bounds__ do the fine-grained work.
        "--ptxas-options=-v",
    ]
    if debug:
        nvcc_flags += ["-G", "-DBITVIDEO_DEBUG=1"]
        nvcc_flags = [f for f in nvcc_flags if f != "--use_fast_math"]

    if IS_WINDOWS:
        cxx_flags += ["/O2", "/std:c++17", "/EHsc", "/bigobj", "/DNOMINMAX", "/wd4624", "/wd4067"]
        nvcc_flags += ["-Xcompiler", "/bigobj", "-Xcompiler", "/wd4624", "-DNOMINMAX"]
    else:
        cxx_flags += ["-O3", "-std=c++17", "-fvisibility=hidden", "-Wno-unused-function"]
        nvcc_flags += ["-Xcompiler", "-fPIC", "-std=c++17"]

    nvcc_flags += _gencode_flags(archs)

    ext = CUDAExtension(
        name="bitvideo._C",
        sources=sources,
        include_dirs=[str(CUDA_DIR)],
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        define_macros=[("BITVIDEO_WITH_CUDA", "1")],
    )

    class TolerantBuildExtension(BuildExtension):
        """BuildExtension that degrades to a python-only wheel on failure."""

        def run(self):  # noqa: D102
            try:
                super().run()
            except Exception as exc:
                if force:
                    raise
                print("=" * 78, file=sys.stderr)
                print(f"[bitvideo] CUDA extension build FAILED: {exc}", file=sys.stderr)
                print(
                    "[bitvideo] Continuing with the python-only install.  Re-run with\n"
                    "           BITVIDEO_FORCE_CUDA=1 to see the full error.",
                    file=sys.stderr,
                )
                print("=" * 78, file=sys.stderr)

    return [ext], {"build_ext": TolerantBuildExtension.with_options(use_ninja=not IS_WINDOWS)}


ext_modules, cmdclass = build_extensions()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
