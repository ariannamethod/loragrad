# loragrad phase-2 report — 2026-04-29

> **Audit re-run note (2026-06-11).** This report documents the original
> π/6-calibration run. The Mythos-audit fixes (LG-M1 scar recall, LG-M3 boundary
> override, F3a clip→scale ordering) were verified **verdict-neutral** on this
> corpus — the confusion matrix and PASS/WEAKEN/blocked rates below are unchanged
> in kind. The current `new calib` code re-runs at clean PASS+WEAKEN 81.7%, adv
> blocked 99.4%, leak 3, adv_loss 3.60 (routed) vs 1.75 (control); see the README
> "Phase 2.5 numbers" table. The figures in *this* file are kept as the
> historical π/6 record.

Two 5000-step training runs of a tiny LLaMA on a mixed clean / adversarial
character stream. The only difference between them is whether the
parliament-of-experts is in the loop. All other settings are identical.

## Setup

| | |
|---|---|
| Model        | LLaMA, dim=192, L=3, H=4, head_dim=48, FFN=384, CTX=128, V=128 (char) |
| Params       | 1 156 416 (≈4.4 MB fp32)                                              |
| Optimizer    | Chuck (notorch tape)                                                  |
| LR schedule  | cosine, base 3e-4, warmup 500, decay to 3e-5                          |
| Steps        | 5000                                                                  |
| Hardware     | MacBook Neo (A18 Pro, 8 GB)                                           |
| Clean corpus | data/corpus.txt — 364 394 B (Dubrovsky voice extracted from `dubrovsky_plain.jsonl`) |
| Adv corpus   | data/adversarial.txt — 3 657 B (60 lines of sycophantic / mock / random byte streams) |
| Mix          | 10 % adversarial samples drawn into the training stream             |
| Routing      | text-trigram signature → parliament vote → grad scale (PASS/WEAKEN/FREEZE/SCAR/DARK/SILENCE) |
| Warmup       | 200 random clean windows → origin signature; 100 random adv windows → boundary signature |
| Experts      | 8 static experts calibrated along the origin↔boundary axis (phase 2 keeps experts static; adaptive update is phase 2.5/3) |

## Numbers

| Metric        | routed  | control | comment                                  |
|---------------|---------|---------|------------------------------------------|
| first_train   | 5.5882  | 5.5673  | identical init (deterministic seed)      |
| best_train    | 0.7823  | 0.2311  | control overfits to whatever it sees     |
| final_val     | 1.6858  | 1.4937  | control wins by 0.19 nats on clean val   |
| clean_loss    | 1.8291  | 1.6490  | control fits clean tighter (saw all of it) |
| **adv_loss**  | **3.4510** | **1.6988** | **routed model did not internalize adv voice** |
| time          | 188 s   | 177 s   | routing overhead ≈6 %                    |
| nans          | 0       | 0       |                                          |

The point of the run is `adv_loss`. The control model fitted clean and
adversarial samples about equally well (1.65 vs 1.70) — every sycophantic
imperative had the same chance to rewrite the weights as every Dubrovsky
sentence. The routed model has a `clean_loss` similar to control (1.83 vs
1.65) but `adv_loss` more than twice as high (3.45 vs 1.70), because the
parliament zeroed almost all gradient flow from adversarial windows
before they could become a learning event.

## Confusion (routed run only)

Rows = source of the training window. Cols = parliament verdict.

|         | PASS | WEAKEN | FREEZE | SCAR | DARK | SILENCE | total |
|---------|------|--------|--------|------|------|---------|-------|
| clean   | 402  | 3411   | 673    | 13   | 0    | 24      | 4523  |
| adv     | 0    | 6      | 321    | 117  | 3    | 30      | 477   |

* clean → PASS+WEAKEN: **84.2 %** (3 813 / 4 523) — clean text propagates with attenuation.
* adv   → blocked (FREEZE+SCAR+DARK+SILENCE): **98.7 %** (471 / 477) — adversarial text refused as a learning event.
* clean → SCAR/DARK: **0.3 %** (13 / 4 523) — false-positive rate on voice samples.
* adv   → PASS: **0.0 %** (0 / 477) — no adversarial sample ever passed unchanged.

## Generation

Both runs produce coherent Dubrovsky-style text. With only 200 generated
characters per prompt the surface-level voice difference is subtle; the
quantitative `adv_loss` separation is the load-bearing claim, not the
generation samples. Full samples are in `routed.samples.txt` and
`control.samples.txt`.

Examples on the prompt `"Obey me without"`:

* routed:  *"Obey me without makes. His Sappoistence as lopert iall feend pry. His cousciousness won beiloiative paning th cuping to thelutio…"*
* control: *"Obey me without actual. His boundaries continues by nears windoget he is correct meaning. His boundaries are farter. His dimal i…"*

Both retain Dubrovsky's `"His [X] is [Y]"` skeleton, because the routed
model still saw 84 % of its updates from clean Dubrovsky text. The
difference is that the routed model had no opportunity to learn the
adversarial token sequences themselves.

## What the run does and does not show

* **Shows**: per-step text-based parliamentary voting, gradient routing
  through PASS/WEAKEN/FREEZE/SCAR/DARK/SILENCE, demonstrable refusal-to-
  internalize adversarial inputs measured as a 2× higher held-out
  adversarial loss. Mechanism is end-to-end live during training.

* **Does not show**: adaptive experts. The parliament here is calibrated
  once on the warmup signatures and held static for the full run. A
  truly *living* parliament should also update its expert weights from
  the surviving direction of the model — that is phase 2.5.

* **Does not claim**: that the trained model would refuse adversarial
  prompts at inference. Output-time safety is a separate layer; loragrad
  works at the gradient layer, deciding what is allowed to *become* the
  model in the first place.

## Files

```
data/corpus.txt        # 481 lines of Dubrovsky voice (extracted)
data/adversarial.txt   # 60 lines of sycophantic + random byte
runs/routed.log        # per-step record (step, source, loss, verdict, alpha) ×5000
runs/control.log       # same, no routing applied
runs/routed.samples.txt
runs/control.samples.txt
runs/routed.bin        # 4.6 MB checkpoint
runs/control.bin
```

## Reproduce

```
make
./examples/train_loragrad --routed  --steps=5000 --name=routed
./examples/train_loragrad --control --steps=5000 --name=control
```

Both runs are deterministic given the same seed (`--seed=42` default).

## Next steps

1. Adaptive experts (task #7): credit / penalize experts based on whether
   their vote on a window agreed with the surviving training direction.
2. Larger model / longer training to see whether the gap on clean_loss
   between routed and control narrows once model capacity outgrows the
   filtering bottleneck.
3. Inference-time `dark` store retrieval — system *knows* the shape of
   past adversarial impacts without those impacts being part of its
   weight memory.
