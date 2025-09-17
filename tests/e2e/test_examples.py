import subprocess
import glob

from helper import run

import pytest

# === Gather all example scripts ===
all_examples = glob.glob("examples/*.py")

# List of examples that are expected to fail
EXPECTED_FAILURES = {
    "examples/mlp.py"  # cannot import name 'mesh' from 'mlir.dialects'
}

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
    f = lambda: test_example_runs(example)
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
