"""Fail if the README names Python API that the module does not export.

The README drifts when a commit renames or removes an op and nobody edits
the docs: it went on listing broadcast_add and reshape_op after both were
gone. This checks two places a reader takes names from:

  - every `p.name` / `prometheus.name` inside a ```python block
  - every backticked identifier in the "Implemented" reference section

A name passes if the module exports it, or if any exported class has it as a
method or attribute (so Tensor and layer methods such as `detach` or
`forward_with_state` count). Names written with a dot, slash or space --
file names, C++ paths, prose -- are not API and are skipped, as are lines
marked "(C++)", which list API that exists only on the C++ side.

Run from the repository root: python tests/check_readme_api.py
"""
import importlib
import os
import re
import sys

import prometheus as p

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()

module_names = set(dir(p))
member_names = set()
for name in module_names:
    obj = getattr(p, name)
    if isinstance(obj, type):
        member_names.update(dir(obj))
known = module_names | member_names

wanted = {}   # name -> where the README uses it

for block in re.findall(r"```python\n(.*?)```", README, re.S):
    for name in re.findall(r"\b(?:p|prometheus)\.([A-Za-z_]\w*)", block):
        wanted.setdefault(name, "python example")

impl = re.search(r"^## Implemented$(.*?)^---$", README, re.S | re.M)
if not impl:
    sys.exit("check_readme_api: no '## Implemented' section found")
# a line marked "(C++)" lists C++-only API, which the module does not bind
impl_text = "\n".join(l for l in impl.group(1).split("\n") if "(C++)" not in l)
for token in re.findall(r"`([^`]+)`", impl_text):
    token = token.strip()
    if token.endswith("()"):
        token = token[:-2]
    if re.fullmatch(r"[A-Za-z_]\w*", token):
        wanted.setdefault(token, "Implemented section")

def exists(name):
    """An export, a class member, or a submodule (prometheus.models): a
    submodule is not in dir() until imported, so import it -- which also
    proves it imports."""
    if name in known:
        return True
    try:
        importlib.import_module("prometheus." + name)
        return True
    except ImportError:
        return False


missing = sorted((n, where) for n, where in wanted.items() if not exists(n))

print("README API check: %d names referenced, %d missing" % (len(wanted), len(missing)))
for name, where in missing:
    print("  MISSING  %-28s (%s)" % (name, where))
sys.exit(1 if missing else 0)
