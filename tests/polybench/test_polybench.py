from helper import run
import sys
import pytest
import importlib.util
from pathlib import Path

# Get paths relative to this test file
TEST_DIR = Path(__file__).parent
BENCHMARKS_DIR = TEST_DIR / "benchmarks"
GOLDEN_DIR = TEST_DIR / "golden_files"
POLYBENCH_DIR = TEST_DIR / "polybench_so_files"
name_discrepancies = {
    "jacobi1d": "jacobi-1d",
    "jacobi2d": "jacobi-2d",
    "heat": "heat-3d",
    "fdtd": "fdtd-2d",
    "seidal": "seidel-2d",
}
blacklisted_files = ["__init__.py"]
# Discover benchmark files
benchmark_files = [
    f
    for f in BENCHMARKS_DIR.glob("*.py")
    if f.name not in blacklisted_files and f.is_file()
]


# Parametrize tests with benchmark files
@pytest.mark.parametrize(
    "benchmark_path", benchmark_files, ids=lambda p: p.stem
)
def test_benchmark(benchmark_path):
    """
    A test which runs a specific polybench testcase from benchmarks and
    checks that it is giving expected output. The output from the benchmark
    will be checked against the text file in golden_files if there isn't a
    `polybench_so_files` directory. If there is a `polybench_so_files`
    directory, then it will run the polybench C implementation and check
    against its output.
    """
    # Get corresponding golden file
    benchmark_name = name_discrepancies.get(
        benchmark_path.stem, benchmark_path.stem
    )
    golden_path = GOLDEN_DIR / f"{benchmark_name}.txt"
    polybench_path = POLYBENCH_DIR / f"{benchmark_name}.so"
    polybench_exists = polybench_path.exists()
    # Run benchmark and capture output
    spec = importlib.util.spec_from_file_location(
        "benchmark_module", benchmark_path.as_posix()
    )
    benchmark_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_module)
    result = benchmark_module.main(
        current_dataset="SMALL_DATASET",
        output_array=True,
        c_test=polybench_exists,
        ctest_obj=polybench_path.as_posix() if polybench_exists else "",
    )

    if polybench_exists:
        assert "results are correct." == result["c_correctness"]
        assert result["c_perf"] >= 0.0
    else:
        # Check golden file exists
        if not golden_path.exists():
            pytest.skip(f"Golden file missing: {golden_path.name}")

        # Load golden file content
        golden_content = golden_path.read_text().strip()

        # Compare outputs
        assert result["array"].strip() == golden_content
        assert result["perf"] > 0.0


if __name__ == "__main__":
    for file in benchmark_files:
        benchmark = lambda: test_benchmark(file)
        benchmark.__name__ = f"test_{file.stem}"
        run(benchmark)
