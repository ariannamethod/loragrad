# loragrad — Linux build for Railway / generic Linux
#
# Self-contained: vendored notorch.c + notorch.h, OpenBLAS for matrix ops,
# Chuck for optimization. Two binaries:
#   examples/smoke_test       — phase-1 mechanism demo
#   examples/train_loragrad   — phase-2 training run with parliament routing
#
# The container's entrypoint trains both routed and control runs and then
# enters a heartbeat loop so the Railway service stays alive and its logs
# remain reachable.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# OPENBLAS_NUM_THREADS=1 disables OpenBLAS's pthread spawn per sgemv;
# for our small dim (~192) the spawn overhead dominates the matmul
# itself. Combined with -O3 -march=native in the Makefile this gives
# ~7× end-to-end speedup on Railway. Source: Henry session 2026-04-29.
ENV OPENBLAS_NUM_THREADS=1

RUN apt-get update && apt-get install -y \
        build-essential \
        libopenblas-dev \
        pkg-config \
        jq \
        ca-certificates \
        procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace/

RUN make clean && make

CMD ["/bin/bash", "/workspace/launcher.sh"]
