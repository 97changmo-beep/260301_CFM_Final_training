########################################################################
# CFM-ID 4.0 with OpenMP Parallel Training + Prediction
# Based on wishartlab/cfmid:latest (pre-built RDKit, InChI, MPICH, LP Solve)
# Modified for: transfer learning with OpenMP parallelization
#
# NOTE: Building from scratch fails because InChI download URL (inchi-trust.org)
#   is dead since ~2025. Using wishartlab/cfmid as base avoids this issue.
########################################################################
FROM wishartlab/cfmid:latest AS base
LABEL author="DESTNI-DTB Project"
LABEL name="CFM-ID 4 OpenMP Training + Prediction Image"

ARG MODEL_DIR="/trained_models_cfmid4.0"

############################################################################################
# Rebuild CFM-ID 4 with TRAINING + OpenMP patches
# Base image already has: RDKit 2017_09_3, InChI, MPICH 3.2, LP Solve 5.5, Boost
# We only need to: install build tools, apply OpenMP patches, rebuild, cleanup
############################################################################################

# Copy patched source files
COPY patches/EmModel.cpp /tmp/patches/EmModel.cpp
COPY patches/NNParam.cpp /tmp/patches/NNParam.cpp
COPY patches/cfm_code_CMakeLists.txt /tmp/patches/cfm_code_CMakeLists.txt
COPY patches/cfm-train-main.cpp /tmp/patches/cfm-train-main.cpp
COPY patches/cfm-predict-main.cpp /tmp/patches/cfm-predict-main.cpp

ARG MSML_VERSION="4.4.10"
RUN apk update && apk add --no-cache wget tar g++ gcc make cmake boost-dev libgomp \
    && cd /tmp \
    && mkdir cfmid \
    && wget -O cfmid.tar.gz https://bitbucket.org/wishartlab/cfm-id-code/get/CFM-ID_${MSML_VERSION}.tar.gz \
    && tar xf cfmid.tar.gz -C cfmid \
    && mv cfmid/wishartlab* cfmid/v4 \
    && mv cfmid/v4/cfm cfm \
    # Apply OpenMP patches (EmModel: parallel E/M-step, NNParam: thread-safe, main: parallel graph gen)
    && cp /tmp/patches/EmModel.cpp /tmp/cfm/cfm-code/EmModel.cpp \
    && cp /tmp/patches/NNParam.cpp /tmp/cfm/cfm-code/NNParam.cpp \
    && cp /tmp/patches/cfm_code_CMakeLists.txt /tmp/cfm/cfm-code/CMakeLists.txt \
    && cp /tmp/patches/cfm-train-main.cpp /tmp/cfm/cfm-train/main.cpp \
    && cp /tmp/patches/cfm-predict-main.cpp /tmp/cfm/cfm-predict/main.cpp \
    && cd /tmp/cfm \
    && mkdir build \
    && cd build \
    && cmake .. -DINCLUDE_TESTS=OFF\
             -DINCLUDE_TRAIN=ON\
             -DLPSOLVE_INCLUDE_DIR=/usr/local/include/lp_solve\
             -DLPSOLVE_LIBRARY_DIR=/usr/local/lib\
             -DRDKIT_INCLUDE_DIR=/usr/local/include/rdkit\
             -DRDKIT_INCLUDE_EXT_DIR=/usr/local/include/rdkit/External\
             -DRDKIT_LIBRARY_DIR=/usr/local/lib\
             -DCMAKE_CXX_STANDARD=14 \
             -DCMAKE_CXX_FLAGS="-fopenmp" \
    && make install -j $(nproc) \
    # Build mimalloc: musl libc malloc uses global lock → 32 threads spin-wait
    && cd /tmp \
    && wget -O mimalloc.tar.gz https://github.com/microsoft/mimalloc/archive/refs/tags/v2.1.2.tar.gz \
    && tar xf mimalloc.tar.gz \
    && cd mimalloc-2.1.2 \
    && mkdir -p out && cd out \
    && cmake .. -DMI_BUILD_TESTS=OFF -DMI_BUILD_SHARED=ON -DMI_BUILD_STATIC=OFF \
    && make -j $(nproc) \
    && make install \
    && apk del g++ gcc make cmake boost-dev wget tar \
    && apk add --no-cache libgomp \
    && mkdir -p /opt/cfm/ \
    && cp -f /tmp/cfm/bin/* /opt/cfm/bin/ \
    && rm -rf /tmp/*

# mimalloc: replace musl's single-lock malloc with multi-threaded allocator
# musl libc malloc uses a global lock → 32 threads spin-wait → slower than 1 thread
ENV LD_PRELOAD=/usr/local/lib64/libmimalloc.so

# OpenMP tuning
ENV OMP_PROC_BIND=false
ENV OMP_PLACES=threads
ENV OMP_SCHEDULE=dynamic

############################################################################################
# Setup working directory (train + predict)
############################################################################################
RUN mkdir -p /cfmid/public/train/spectra \
    && mkdir -p /cfmid/public/train/pretrained \
    && mkdir -p /cfmid/public/predict/output \
    && mkdir -p /cfmid/public/predict/spectra

WORKDIR /cfmid

# Pretrained model symlinks (quote paths to prevent glob interpretation of [M+H]+)
RUN ln -sf "${MODEL_DIR}/cfmid4/[M+H]+" "${MODEL_DIR}/[M+H]+" 2>/dev/null || true;\
    ln -sf "${MODEL_DIR}/cfmid4/[M-H]-" "${MODEL_DIR}/[M-H]-" 2>/dev/null || true;

LABEL cfmid.version="5.0.0-omp"
