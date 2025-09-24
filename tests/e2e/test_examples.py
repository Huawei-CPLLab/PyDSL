import importlib
import shutil
import subprocess
import glob

from helper import run

import pytest

has_triton = (
    importlib.util.find_spec("triton") is not None
    and importlib.util.find_spec("triton.language") is not None
    and shutil.which("triton-adapter-opt") is not None
)

# === Gather all example scripts ===
all_examples = glob.glob("examples/**/*.py", recursive=True)

# Expected failures
EXPECTED_FAILURES = {
    "examples/mlp.py",  # cannot import name 'mesh' from 'mlir.dialects'
    "examples/autotune/tune_transform.py",  # NotImplementedError
    "examples/autotune/tune_combined.py",  # NotImplementedError
    "examples/autotune/heat_demo.py",  # NotImplementedError
}

# Conditionally add more failures
if not has_triton:
    EXPECTED_FAILURES.add("examples/triton_interop.py")

# Prepare pytest parameters
params = [
    pytest.param(ex, marks=pytest.mark.xfail(reason="Known issue"))
    if ex in EXPECTED_FAILURES
    else ex
    for ex in all_examples
]


# === Pytest test ===
@pytest.mark.parametrize("example", params)
def test_example_runs(example: str) -> None:
    """Run example scripts and assert they complete successfully."""
    print(f"Running {example}...")
    result = subprocess.run(
        ["python", example], capture_output=True, text=True
    )
    assert result.returncode == 0, f"{example} failed:\n{result.stderr}"


def make_runner(example):
    def f():
        return test_example_runs(example)

    f.__name__ = f"test {example}"
    return f


if __name__ == "__main__":
    run(make_runner("examples/affine_explicit.py"))
    run(make_runner("examples/affine_implicit.py"))
    run(make_runner("examples/function.py"))
    run(make_runner("examples/hello_memref.py"))
    run(make_runner("examples/memref.py"))
    # run(make_runner("examples/mlp.py")) # cannot import name 'mesh' from 'mlir.dialects'
    run(make_runner("examples/simple.py"))
    run(make_runner("examples/triton_interop.py"))
    run(make_runner("examples/tuple.py"))
    run(make_runner("examples/with_recursively.py"))
