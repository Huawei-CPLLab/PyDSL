FROM ubuntu:24.04

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto

COPY llvm-project ./llvm-project
COPY requirements.txt ./requirements.txt

RUN apt-get update && \
    apt-get install -y software-properties-common git build-essential ninja-build cmake clang-19

# install python 3.12, which should be shipped by ubuntu 24.04
RUN apt-get install -y python3-full python3-dev python3-pip

# create a virtual environment to use editable install later on
RUN python3 -m venv pydsl-venv
ENV VIRTUAL_ENV=/root/pydsl-venv
ENV PATH=/root/pydsl-venv/bin:$PATH

# install python dependencies
RUN pip install -r requirements.txt && \
    pip install pytest hatch

# create symlinks for clang and clang++
RUN ln -s /usr/bin/clang-19 /usr/bin/clang && \
    ln -s /usr/bin/clang++-19 /usr/bin/clang++

RUN mkdir -p llvm-project/build && \
    cd llvm-project/build && \
    cmake -G Ninja ../llvm \
        -DCMAKE_C_COMPILER="$(which clang-19)" \
        -DCMAKE_CXX_COMPILER="$(which clang++-19)" \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PIC=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE="$(which python3.12)" && \
    ninja && \
    cd ../..

ENV PYDSL_LLVM=/root/llvm-project/build
ENV PATH=$PYDSL_LLVM/bin:$PATH
ENV PYTHONPATH=$PYDSL_LLVM/tools/mlir/python_packages/mlir_core
CMD ["bash"]
