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
