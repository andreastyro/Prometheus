"""
Build script for the prometheus Python extension.

Compiles all C++ source into a Python-importable module (prometheus.pyd on
Windows, prometheus.so on Linux/macOS).

Backend selection — set environment variables before building:
    PROMETHEUS_USE_OPENBLAS=1   use OpenBLAS (best on AMD, default in CI)
    PROMETHEUS_USE_MKL=1        use Intel MKL (best on Intel)
    PROMETHEUS_USE_OPENMP=1     parallelise across CPU cores

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

# ── Backend selection via environment variables ───────────────────────────────
use_openblas = os.environ.get("PROMETHEUS_USE_OPENBLAS", "0") == "1"
use_mkl      = os.environ.get("PROMETHEUS_USE_MKL",      "0") == "1"
use_openmp   = os.environ.get("PROMETHEUS_USE_OPENMP",   "0") == "1"

define_macros  = []
include_dirs   = [os.path.join(src_root, "include"), pybind11.get_include()]
libraries      = []
library_dirs   = []
extra_link     = []

if use_openblas:
    define_macros.append(("PROMETHEUS_USE_OPENBLAS", None))
    # Linux/macOS: openblas is usually in a standard location
    # Windows CI: set OPENBLAS_ROOT to the extracted OpenBLAS directory
    openblas_root = os.environ.get("OPENBLAS_ROOT", "")
    if openblas_root:
        include_dirs.append(os.path.join(openblas_root, "include"))
        library_dirs.append(os.path.join(openblas_root, "lib"))
    libraries.append("openblas")

elif use_mkl:
    define_macros.append(("PROMETHEUS_USE_MKL", None))
    mkl_root = os.environ.get("MKLROOT", os.environ.get("MKL_ROOT", ""))
    if mkl_root:
        include_dirs.append(os.path.join(mkl_root, "include"))
        library_dirs.append(os.path.join(mkl_root, "lib", "intel64"))
    libraries += ["mkl_rt"]  # single dynamic library interface

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
)
