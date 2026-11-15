"""Build the C++ GPU test (tests/test_cuda.cpp) from the objects pip already
compiled for the Python module: the library is compiled once, by one
compiler, and the test only adds its own file and a link.

    python -m pip install . --no-build-isolation     # compiles the library (incrementally)
    python tests/build_gpu_tests.py [--run]          # compiles test_cuda.cpp, links build/gpu_tests/test_cuda

The compiler, flags and defines are setup.py's own (read by running it with
setup() captured), so the test links against exactly what the module runs.
--run runs it afterwards; PROMETHEUS_CUDA_PATH points it at NVRTC / cuBLAS
when they are not on the library path.
"""
import os
import runpy
import subprocess
import sys

import setuptools

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)

# setup.py's extension, without building anything
captured = {}
run_after = "--run" in sys.argv[1:]
real_setup = setuptools.setup
setuptools.setup = lambda **kw: captured.update(kw)
try:
    sys.argv = ["setup.py"]
    setup_globals = runpy.run_path(os.path.join(root, "setup.py"), run_name="__main__")
finally:
    setuptools.setup = real_setup
ext = captured["ext_modules"][0]

from setuptools import Distribution
from setuptools.command.build_ext import build_ext
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler

cmd = build_ext(Distribution({"ext_modules": [ext]}))
cmd.finalize_options()
cc = new_compiler()
customize_compiler(cc)

# the library's objects, where pip's build left them
library_sources = [s for s in ext.sources if not s.endswith("bindings.cpp")]
objects = cc.object_filenames(library_sources, output_dir=cmd.build_temp)
missing = [o for o in objects if not os.path.exists(o)]
if missing:
    sys.exit("build_gpu_tests: %d library objects missing under %s -- run "
             "`python -m pip install . --no-build-isolation` first" % (len(missing), cmd.build_temp))

out_dir = os.path.join(root, "build", "gpu_tests")
os.makedirs(out_dir, exist_ok=True)
test_src = os.path.join(root, "tests", "test_cuda.cpp")
test_obj = cc.object_filenames([test_src], output_dir=out_dir)
if setup_globals["_stale"](test_src, test_obj[0]):        # setup.py's own dependency check
    test_obj = cc.compile([test_src], output_dir=out_dir, macros=ext.define_macros, include_dirs=ext.include_dirs,
                          extra_postargs=ext.extra_compile_args)
# target_lang: link with the C++ driver (gcc alone leaves out libstdc++)
cc.link_executable(test_obj + objects, "test_cuda", output_dir=out_dir, libraries=ext.libraries,
                   library_dirs=ext.library_dirs, extra_postargs=ext.extra_link_args, target_lang="c++")
exe = os.path.join(out_dir, "test_cuda" + (".exe" if sys.platform == "win32" else ""))
print("built", exe)
if run_after:
    sys.exit(subprocess.call([exe]))
