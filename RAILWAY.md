# Railway deployment

Project: **loragrad-train**
Workspace: ariannamethod's Projects (Pro plan)
Repo: `ariannamethod/loragrad`, branch `main`

## IDs

| | |
|---|---|
| Project ID | `ea97d6b3-dd8b-4774-9c63-ca874d9c31e4` |
| Environment ID (production) | `b889450f-735c-4060-b389-f963362b2139` |
| Service ID (`loragrad`) | `1dab31aa-42d3-42eb-8bf7-582db2947842` |
| Volume ID (`/data`) | `7a7ff029-4084-4636-ac32-092f84d35fe1` |
| First deploy | `c2941197-ee1e-4aae-af98-5abbeaed106e` (commit `c7a4f72`) |

## Container

* Base: `ubuntu:22.04`
* Build: `make` (cross-platform Makefile picks OpenBLAS on Linux)
* Entrypoint: `launcher.sh`
* CMD persistence: `/data/runs` mounted from project volume — checkpoints
  and logs survive redeploy.

## What the deploy does

`launcher.sh` runs once on container start:

1. Symlinks `./runs` → `/data/runs` so the trainer's artifacts land on the volume.
2. If `/data/runs/routed.bin` does not exist:
   * `./examples/train_loragrad --routed  --steps=10000 --seed=42 --name=routed`
   * `./examples/train_loragrad --control --steps=10000 --seed=42 --name=control`
3. Enters a heartbeat loop (`sleep 600`) so the service stays healthy and
   the training output is reachable from Railway logs indefinitely.

To re-train from scratch: clear the volume (or change `--name=` so the
guard misses).

## Knobs (env vars on the service)

* `LORAGRAD_STEPS` — training steps per mode (default `10000`).
* `LORAGRAD_SEED`  — random seed (default `42`).

## Caveats

* OpenBLAS on the Railway box is roughly an order of magnitude slower
  than Apple Accelerate on Mac Neo for this workload. A 10 000-step run
  takes ~40 min instead of ~3.
* Crash trap is installed on Linux (`SIGSEGV / SIGBUS / SIGABRT / SIGFPE`
  → `backtrace_symbols_fd` then `_exit`) so silent segfaults do not hide
  in the cloud logs.
* The deploy mutation **must** include `commitSha`. `serviceInstanceDeployV2`
  without it redeploys the previously-deployed commit, not `HEAD`.
