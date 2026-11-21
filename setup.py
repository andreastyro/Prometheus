"""
Build script for the prometheus Python extension.

Compiles all C++ source into a Python-importable module (prometheus.pyd on
Windows, prometheus.so on Linux/macOS).

Build options, set as environment variables before building:
    PROMETHEUS_USE_OPENMP=1     parallelise across CPU cores
    PROMETHEUS_NATIVE=1         tune for this machine's CPU (-march=native /
                                /arch:AVX2). Off by default: a wheel built
                                that way crashes on older CPUs.

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
import platform
import sys

src_root = os.path.dirname(os.path.abspath(__file__))

sources = (
    glob.glob(os.path.join(src_root, "src/**/*.cpp"), recursive=True) +
    [os.path.join(src_root, "python/bindings.cpp")]
)

# ── Compiler flags ────────────────────────────────────────────────────────────
# Portable by default. -march=native uses every instruction the build machine
# has, so a wheel built with it dies with an illegal instruction on an older
# CPU. The BLAS libraries pick their kernels at runtime anyway; what native
# tuning buys here is wider vectors in the elementwise loops, which a local
# build can ask for.
native = os.environ.get("PROMETHEUS_NATIVE", "0") == "1"

if sys.platform == "win32":
    compile_args = ["/std:c++17", "/O2", "/EHsc"] + (["/arch:AVX2"] if native else [])
    openmp_flag  = ["/openmp"]
else:
    compile_args = ["-std=c++17", "-O2"] + (["-march=native"] if native else [])
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

# The compiled core is prometheus._C; python/ is installed as the prometheus
# package around it, so the pure-Python models ship too.
ext = Extension(
    "prometheus._C",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=extra_link,
    # setuptools rebuilds only when a source is newer than the built module;
    # without the headers here, a change to one alone was silently not built
    depends=(glob.glob(os.path.join(src_root, "include/**/*.hpp"), recursive=True) +
             glob.glob(os.path.join(src_root, "src/**/*.hpp"), recursive=True)),
)

# Compile the ~40 sources in parallel. setuptools can, but does not unless
# asked; one at a time a rebuild took minutes. PROMETHEUS_BUILD_JOBS overrides.
from setuptools.command.build_ext import build_ext as _build_ext


import re
from concurrent.futures import ThreadPoolExecutor

_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
_include_cache = {}


def _local_includes(path):
    """Every header of this project `path` includes, transitively."""
    if path in _include_cache:
        return _include_cache[path]
    _include_cache[path] = set()            # cycles stop here
    found = set()
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return found
    for name in _INCLUDE.findall(text):
        for base in (os.path.dirname(path), os.path.join(src_root, "include"), os.path.join(src_root, "src")):
            cand = os.path.normpath(os.path.join(base, name))
            if os.path.isfile(cand):
                found.add(cand)
                found |= _local_includes(cand)
                break
    _include_cache[path] = found
    return found


def _stale(source, obj):
    """The object is missing, or older than the source or any header it includes."""
    if not os.path.exists(obj):
        return True
    t = os.path.getmtime(obj)
    return any(os.path.getmtime(f) > t for f in [source, *_local_includes(source)])


class build_ext(_build_ext):
    """Per-file incremental, parallel compilation.

    setuptools compiles every source of an extension, one at a time, whenever
    any of them is out of date. Here only the sources whose object file is
    older than the source or a project header it includes are recompiled,
    PROMETHEUS_BUILD_JOBS (default: every core) at once; the rest are linked
    from the objects already built.
    """

    def finalize_options(self):
        super().finalize_options()
        if not self.parallel:
            self.parallel = int(os.environ.get("PROMETHEUS_BUILD_JOBS", os.cpu_count() or 1))

    def build_extensions(self):
        compiler = self.compiler
        original = compiler.compile
        jobs = max(1, int(self.parallel or 1))
        # MSVC finds its toolchain on first use: once, before the threads start
        if hasattr(compiler, "initialized") and not compiler.initialized:
            compiler.initialize()

        def compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0,
                    extra_preargs=None, extra_postargs=None, depends=None):
            objects = compiler.object_filenames(sources, output_dir=output_dir)
            # objects built with other flags (OpenMP on / off, native) are all stale
            stamp = os.path.join(output_dir or self.build_temp, "prometheus-flags.txt")
            flags = repr((sorted(map(str, macros or [])), list(extra_preargs or []), list(extra_postargs or []),
                          sys.version))
            same = os.path.exists(stamp) and open(stamp).read() == flags
            stale = [s for s, o in zip(sources, objects) if not same or _stale(s, o)]
            def arch(s):
                # the AVX2 / AVX-512 builds of the hot loops (src/cpu/simd.cpp picks one at runtime)
                x86 = platform.machine().lower() in ("x86_64", "amd64", "i386", "i686", "x86")
                if not x86:
                    return []
                if s.endswith("simd_avx2.cpp"):
                    return ["/arch:AVX2"] if sys.platform == "win32" else ["-mavx2", "-mfma"]
                if s.endswith("simd_avx512.cpp"):
                    return ["/arch:AVX512"] if sys.platform == "win32" else \
                        ["-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl", "-mfma"]
                return []
            if stale:
                one = lambda s: original([s], output_dir, macros, include_dirs, debug,
                                         extra_preargs, list(extra_postargs or []) + arch(s), depends)
                if jobs > 1 and len(stale) > 1:
                    with ThreadPoolExecutor(jobs) as pool:
                        list(pool.map(one, stale))
                else:
                    for s in stale:
                        one(s)
            os.makedirs(os.path.dirname(stamp), exist_ok=True)
            with open(stamp, "w") as f:
                f.write(flags)
            print("prometheus: compiled %d of %d sources" % (len(stale), len(sources)))
            return objects

        compiler.compile = compile
        super().build_extensions()


setup(
    name="prometheus",
    cmdclass={"build_ext": build_ext},
    version="0.1.0",
    ext_modules=[ext],
    packages=["prometheus", "prometheus.models", "prometheus.compiler"],
    package_dir={"prometheus": "python"},
    # prometheus_onnx.py sits beside setup.py and provides save_onnx(). It was
    # not listed here, so pip install gave users the extension but not the
    # exporter, and `import prometheus_onnx` failed anywhere outside the
    # source tree.
    py_modules=["prometheus_onnx"],
    # MKL is loaded at runtime when present; this just makes it one command
    extras_require={"mkl": ["mkl"]},
)
