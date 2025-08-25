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

# Environment variables

You should make sure to set up environment variables properly.
If you are running the script below, you can simply copy each `export`
statement to your `~/.bashrc`.
Making an environment script with the `export` statements and sourcing that
when you start up a shell isn't quite ideal, because then your IDE might not
find the MLIR Python binding files for code highlighting and exploration.

# Memory of ninja

The most ressource intensive step is `ninja check-mlir`, which uses both a lot
of compute and memory.
Depending on the amount of available RAM on your system, you may run out of
memory, causing the command to fail.
To combat this, you can decrease the number of parallel jobs used by `ninja`
using the `-j` option, which we set to `-j16` by default.
`ninja` can be run multiple times, and it will continue the tasks it has not
yet finnished.

## Allocating memory on WSL

In Windows, place a file `.wslconfig` in your home directory containing
```
[wsl2]
memory=10GB
swap=8GB
```
Then run `wsl --shutdown` to restart WSL, then in WSL use `htop` to cheeck
whether the new configuration file was applied:
```sh
sudo apt install htop
htop
```

# Script

If you have ensured you have enough memory, you should be able to simply
copy-paste the below script into a Linux terminal.

This script was primarily developed for Ubuntu 24.04 in WSL 2.

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

# MLIR Python bindings
cd ~PyDSL/llvm-project
pip install -r ../requirements.txt
mkdir build
cd build
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
    -DPython3_EXECUTABLE="$(which python3)"

mkdir ~/tmp
time ninja check-mlir -j16 >>~/tmp/1.out

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

On an Intel(R) Core(TM) Ultra 7 155H, the amount of time `ninja check-mlir`
took is:
```sh
~/PyDSL/llvm-project/build$ time ninja check-mlir -j16 >>~/tmp/2.out

real    33m15.995s
user    488m24.590s
sys     26m13.884s
```

The second most time consuming step is cloning the repository, which might take
5-15min depending on your network speed.

# Notes

- This script makes a venv for storing Python packages. You can also use Conda
if you prefer.
- We redirect the output of `ninja check-mlir` to a file, since there can be
quite a lot of output, and sending it to the terminal where it gets flushed
every time is a bit slower.
    - Append mode (`>>` instead of `>`) is used in case you run the command
    multiple times.
    - We use `~/tmp` instead of `/tmp`, since `/tmp` is cleared on a restart
    (which you might do when messing with amount of allocated memory in WSL).
