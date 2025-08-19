Condensed build instructions for setting up PyDSL on a clean installation of
Ubuntu 24.04, intended for developers of PyDSL.
If you are not on a new installation, probably some parts of the first block
of commands are not necessary.

If you have already installed PyDSL using the instructions for users, the only
other thing you need to do is to install Hatch using `pip install hatch`.
Hatch is the project manager we use for PyDSL.
It provides utilities like `hatch test` (equivalent to `pytest`),
`hatch fmt -f`, and `hatch build`, described in more detail in
[dev_docs.md](/dev/dev_docs.md).

You should make sure to set up environment variables properly.
If you are running the script below, you can simply copy each `export`
statement to your `~/.bashrc`.
Making an environment script with the `export` statements and sourcing that
when you start up a shell isn't quite ideal, because then your IDE might not
find the MLIR Python binding files for code highlighting and exploration.

This almost works out of the box, but depending on your hardware and the ratio
of RAM to number of threads, you may have to decrease the number of jobs when
running `ninja` to prevent them from running out of memory.
This can be done with the `-j` flag, e.g. `-j16`.
You can call and stop the `ninja` command multiple times with different `-j`
values, and it will resume the tasks that haven't been completed yet.

```sh
# Non-Python dependencies
sudo apt-get update
sudo apt install clang ninja-build cmake python3-pip python-is-python3 python3.12-venv

# Python virtual environment
python -m venv ~/.venv
export PATH=$HOME/.venv/bin:$PATH

# Clone repository
cd ~
time git clone https://github.com/Huawei-CPLLab/PyDSL.git --recurse-submodules --shallow-submodules

python -m pip install -r PyDSL/requirements.txt

# MLIR Python bindings
mkdir ~/tmp
cd PyDSL/llvm-project
time cmake -G Ninja ../llvm \
    -DCMAKE_C_COMPILER="$(which clang)" \
    -DCMAKE_CXX_COMPILER="$(which clang++)" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PIC=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE="$(which python3)" \
    >~/tmp/1.out 2>~/tmp/1.err

time ninja check-mlir >>~/tmp/2.out

# Install PyDSL
cd ~/PyDSL
pip install -e .
export PATH=$HOME/PyDSL/llvm-project/build/bin:$PATH
export PYDSL_LLVM=$HOME/PyDSL/llvm-project/build
export PYTHONPATH=$HOME/PyDSL/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
export PYTHONPATH=$HOME/PyDSL/tests:$PYTHONPATH
pip install hatch
hatch test
```
