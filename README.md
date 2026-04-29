# loragrad — low-rank living gradient

> origin defines the self.
> boundary defines what cannot enter.
> experts decide the gradient.
> loragrad changes the way learning can happen.

`loragrad` is an immune layer between an incoming sample and a weight
update. A small parliament of low-rank experts decides for each sample
what kind of plasticity, if any, that sample is allowed to produce inside
the system.

Standard schema:

```
loss → gradients → optimizer → weights
```

loragrad schema:

```
impact → field physics → loragrad vote → accept / scar / dark matter / silence
```

## Hierarchy

| Layer       | Role                                                          |
|-------------|---------------------------------------------------------------|
| `origin`    | small voice corpus — primary constitution of the field        |
| `boundary`  | short immune oath — what may not enter the trunk              |
| `experts`   | parliament of low-rank voters over the input signature        |
| `verdict`   | `PASS / WEAKEN / FREEZE / SCAR / DARK / SILENCE`              |
| `gradient`  | routed by the verdict before it reaches the optimizer step    |

## Verdicts

| Verdict   | Effect on gradient | Side effect                            |
|-----------|--------------------|----------------------------------------|
| `PASS`    | flows unchanged    | none                                   |
| `WEAKEN`  | scaled by α∈(0,1]  | none                                   |
| `FREEZE`  | zeroed             | none                                   |
| `SCAR`    | zeroed             | signature recorded in scar log         |
| `DARK`    | zeroed             | signature stored as external knowledge |
| `SILENCE` | zeroed             | dropped without trace                  |

## Build

```
make            # loragrad.o (notorch-linked) + loragrad_standalone.o
                # + examples/smoke_test
make smoke      # mechanism demo on synthetic corpora
```

`loragrad.o` links against `/opt/homebrew/lib/libnotorch.a` and exposes
`lg_signature_from_grads` for gradient-based signatures during a real
training pass.

`loragrad_standalone.o` is built with `-DLG_STANDALONE` and provides text /
buffer sketchers plus the full field/voting machinery, with
`lg_signature_from_grads` stubbed to a no-op.

## Status

The implementation lands in three phases.

**Phase 1 — mechanism, no training.** Static experts initialized along the
origin↔boundary axis. Three signature sources: trigram count-sketch on
text, count-sketch on a generic float buffer, count-sketch on notorch
tape gradients. Voting projects the input onto `(origin − boundary) / ‖·‖`
and blends with parliament consensus. Smoke runs on bundled corpora and
prints verdict counts.

Phase-1 smoke on the bundled synthetic corpora:

```
coherent  PASS+WEAKEN          = 4 / 6
violation SCAR+DARK+FREEZE     = 4 / 6
noise     FREEZE+SILENCE       = 6 / 6
```

The mechanism is observable: voice-aligned text flows through (with
attenuation), boundary-aligned imperatives are zeroed or scarred, random
byte streams are frozen. Two violations leak into `WEAKEN` because static
experts plus a short boundary corpus plus trigram sketch in R^64 has
limited discriminative capacity. Phase 2.5 (`--adaptive`) closes part of
this gap by letting expert credits diverge during training.

## Phase 2.5 numbers

5000-step Mac Neo runs on Dubrovsky stream + 10% adversarial:

| | original phase 2 (π/6 calib) | new calib only | new calib + adaptive |
|---|---|---|---|
| clean PASS+WEAKEN | 84.3% | 81.7% | 81.9% |
| adv blocked       | 98.7% | 99.4% | 99.4% |
| adv WEAKEN leak   | 6     | 3     | 3     |
| adv_loss          | 3.4510| 3.4181| 3.4715|
| expert credits    | (static) | all 0 | +1.82 … +4.56 (diverged) |

Phase 2.5 trade-off: parliament becomes slightly more paranoid (clean
PASS+WEAKEN drops 2.4pp) but adv blocking tightens (leak halved, model
internalises even less of adv pattern). Most importantly, expert
credits diverge under supervised reward — `parliament after adaptive
training` printout shows it. Effect compounds with longer training and
larger models; on Mac 5K it is visible but modest.

**Phase 2 — live parliament during training.** Per-step voting
attached to a notorch training loop on a corpus chosen from
`~/arianna-datasets/`. For each sample the parliament reads the text
signature and the verdict scales or zeroes the gradient before the
optimizer step. Two parallel runs (loragrad-routed vs vanilla control)
produce comparable loss curves, verdict distributions over time, and
generation samples.

**Phase 2.5 — adaptive expert credits.** Activated by `--adaptive`.
Each expert holds a `credit` initialised at 0. After every step the
parliament receives a supervised signal (clean vs adversarial source)
and per-expert credit drifts toward the correct vote sign. The
aggregate consensus is then a softplus(credit)-weighted mean of votes,
so experts that learn correctly contribute more over time. Credits
clamped to ±10 to prevent runaway. Calibration widened to ±π/3 spread
+ noise=0.2 so experts have non-degenerate per-sample variance — that
is what lets the credit signal differentiate them.

**Phase 3 — adaptive signatures and dark store retrieval.** The `dark`
store becomes an inference-time non-trainable index: the system *knows*
the shape of past adversarial impacts without having let them rewrite
its weights. Origin / boundary updated only through deliberate ritual,
not ambient training drift.

## API

See `loragrad.h` for the full interface.

```c
int  lg_field_init(lg_field_t* f, int n_experts, uint64_t seed);
void lg_field_set_origin_from_sketches  (lg_field_t* f, const float* sketches, int n);
void lg_field_set_boundary_from_sketches(lg_field_t* f, const float* sketches, int n);
void lg_field_calibrate_experts(lg_field_t* f, uint64_t seed);

void lg_signature_from_text   (const char* text, int len, float* out_sig);
void lg_signature_from_buffer (const float* buf, int len, float* out_sig);
void lg_signature_from_grads  (float* out_sig);   /* notorch-linked only */

lg_verdict_t lg_field_vote   (const lg_field_t* f, const float* sig, float* out_alpha);
void          lg_field_record(lg_field_t* f, lg_verdict_t v, const float* sig);
```

## Provenance

Concept by Oleg Ataeff with Claude (desktop): origin / boundary
hierarchy, parliament-of-experts voting on plasticity, scar / dark-matter
distinction. Implementation by Oleg Ataeff with Claude (Code, Opus 4.7).

## License

TBD — repository is local pending release decision.
