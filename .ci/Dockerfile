# common environment setup
FROM ubuntu:24.04 AS base

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto

RUN apt-get update && \
    apt-get install -y software-properties-common git build-essential ninja-build cmake clang-19 && \
    apt-get install -y python3-full python3-dev python3-pip && \
    apt-get install unzip && \
    apt-get clean

# create symlinks for clang and clang++
RUN ln -s /usr/bin/clang-19 /usr/bin/clang && \
    ln -s /usr/bin/clang++-19 /usr/bin/clang++

# create a virtual environment to use editable install later on
RUN python3 -m venv pydsl-venv
ENV VIRTUAL_ENV=/root/pydsl-venv
ENV PATH=/root/pydsl-venv/bin:$PATH

# install python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt && \
    pip install pytest hatch && \
    pip install triton && \
    rm requirements.txt

# build LLVM and MLIR in a separate stage so that we avoid including the source code, etc in the final image
FROM base AS builder

COPY llvm-project ./llvm-project

RUN mkdir -p llvm-project/build && \
    cd llvm-project/build && \
    cmake -G Ninja ../llvm \
        -DCMAKE_INSTALL_PREFIX=/root/llvm-build \
        -DCMAKE_C_COMPILER="$(which clang-19)" \
        -DCMAKE_CXX_COMPILER="$(which clang++-19)" \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=OFF \
        -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PIC=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE="$(which python3.12)" && \
    ninja && \
    ninja install

# final image with all the necessary components
FROM base
COPY --from=builder /root/llvm-build /root/llvm-build

# get a copy of triton-adapter-opt to use with triton
RUN pip download triton-ascend --python-version=3.11 --no-deps && \
    unzip triton_ascend*.whl -d extracted_wheel && \
    mkdir triton_bin && \
    cp extracted_wheel/triton/backends/huawei/triton-adapter-opt triton_bin/triton-adapter-opt
RUN rm -rf extracted_wheel triton_ascend*.whl
ENV PATH=/root/triton_bin:$PATH

ENV PYDSL_LLVM=/root/llvm-build
ENV PATH=$PYDSL_LLVM/bin:$PATH
ENV PYTHONPATH=$PYDSL_LLVM/python_packages/mlir_core
CMD ["bash"]
