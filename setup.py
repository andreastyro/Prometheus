"""
Build script for the prometheus Python extension.

Compiles all C++ source into a Python-importable module (prometheus.pyd on
Windows, prometheus.so on Linux/macOS).

Build option, set as an environment variable before building:
    PROMETHEUS_USE_OPENMP=1     parallelise across CPU cores

The BLAS library is not a build option: OpenBLAS or Intel MKL is loaded at
runtime if present (prometheus.set_blas(), or PROMETHEUS_BLAS=...).

Install locally:
    pip install -e . --no-build-isolation

Build a wheel:
    pip wheel . --no-build-isolation
"""

from setuptools import setup, Extension
import pybind11
import glob
import os
import sys

src_root = os.path.dirname(os.path.abspath(__file__))

sources = (
    glob.glob(os.path.join(src_root, "src/**/*.cpp"), recursive=True) +
    [os.path.join(src_root, "python/bindings.cpp")]
)

# ── Compiler flags ────────────────────────────────────────────────────────────
if sys.platform == "win32":
    compile_args = ["/std:c++17", "/O2", "/EHsc"]
    openmp_flag  = ["/openmp"]
else:
    compile_args = ["-std=c++17", "-O2", "-march=native"]
    openmp_flag  = ["-fopenmp"]

# ── Build options ─────────────────────────────────────────────────────────────
use_openmp = os.environ.get("PROMETHEUS_USE_OPENMP", "0") == "1"

define_macros  = []
include_dirs   = [os.path.join(src_root, "include"), pybind11.get_include()]
libraries      = ["dl"] if sys.platform.startswith("linux") else []   # dlopen
library_dirs   = []
extra_link     = []

if use_openmp:
    define_macros.append(("PROMETHEUS_USE_OPENMP", None))
    compile_args += openmp_flag
    if sys.platform != "win32":
        extra_link += ["-fopenmp"]

ext = Extension(
    "prometheus",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=extra_link,
)

setup(
    name="prometheus",
    version="0.1.0",
    ext_modules=[ext],
    # prometheus_onnx.py sits beside setup.py and provides save_onnx(). It was
    # not listed here, so pip install gave users the extension but not the
    # exporter, and `import prometheus_onnx` failed anywhere outside the
    # source tree.
    py_modules=["prometheus_onnx"],
    # MKL is loaded at runtime when present; this just makes it one command
    extras_require={"mkl": ["mkl"]},
)
