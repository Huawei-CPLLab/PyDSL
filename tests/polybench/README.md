# Overview

This test suite implements the [PolyBench](https://sourceforge.net/projects/polybench/) test cases in PyDSL.
This comprises 30 tests designed to be a benchmark for various application domains.
Currently, this test suite does not have a nussinov implementation, since LLVM-19
does not have MLIR python bindings for the affine.if op.

## Usage

Generally these testcases will just be run as normal when a user runs `hatch test`.
In this case it will run the PyDSL implementation of each benchmark, and then
check it against the hardcoded expected output found in `golden_files`. If you
want to run a particular benchmark yourself to see the output, you can simply
run `python benchmark_name.py` in the `benchmarks` directory,
where `benchmark_name` is the name of the benchmark you want to run, and it
will print the SMALL_DATASET output array as well as the amount of time the
benchmark took to run.

### testing against PolyBenchC

These testcases can also check against the original PolyBench C implementation.
In order to do so, first download the [PolyBench/C](https://sourceforge.net/projects/polybench/) test suite, and then
place the `PolyBenchC-4.2.1-master` folder in the `polybench` directory (make sure that it has 
utilities, linear-algebra, etc as immediate subdirectories). 
Then you can simply run `bash generate_polybench_folder.sh PolyBenchC-4.2.1-master`
and the script will automatically compile all the required .so files and store them in
a directory called `polybench_so_files`. Now simply by calling `hatch test` the
testcases will be instead checked against the actual C implementations of PolyBench,
instead of simply against the hardcoded golden files.
