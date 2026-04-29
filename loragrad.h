/*
 * loragrad.h — low-rank living gradient
 *
 * lorograd is not an optimizer and not an output filter. It is an immune layer
 * between an incoming impact (sample / batch) and a weight update. A small
 * parliament of experts decides for each impact what kind of plasticity, if
 * any, that impact is allowed to produce inside the system.
 *
 * Hierarchy (top → down):
 *   origin    — primary constitution of the field (small voice corpus)
 *   boundary  — short immune oath (what may not enter the trunk)
 *   experts   — vote on plasticity mode for the current impact
 *   verdict   — PASS / WEAKEN / FREEZE / SCAR / DARK / SILENCE
 *   gradient  — routed through the verdict, then handed to the optimizer
 *
 * Phase 1 (this file): static experts initialized along the origin↔boundary
 * axis, gradient sketches via count-sketch over notorch tape grads, smoke
 * demonstrating COHERENT / VIOLATION / NOISE behavior. No training run.
 */

#ifndef LORAGRAD_H
#define LORAGRAD_H

#include <stdint.h>
#include <stddef.h>

#define LG_SIG_DIM       64       /* signature dimension R^64             */
#define LG_DEFAULT_EXPERTS 8      /* parliament size                      */
#define LG_SCAR_CAP_DEFAULT 256   /* max scar log entries                 */
#define LG_DARK_CAP_DEFAULT 256   /* max dark matter entries              */

typedef enum {
    LG_PASS    = 0,  /* gradient flows unchanged                          */
    LG_WEAKEN  = 1,  /* gradient scaled by alpha                          */
    LG_FREEZE  = 2,  /* gradient = 0 (no learning from this impact)       */
    LG_SCAR    = 3,  /* gradient = 0 + recorded as wound                  */
    LG_DARK    = 4,  /* gradient = 0 + stored as external knowledge       */
    LG_SILENCE = 5,  /* no event — dropped without trace                  */
    LG_VERDICT_COUNT
} lg_verdict_t;

const char* lg_verdict_name(lg_verdict_t v);

typedef struct {
    int   dim;                              /* = LG_SIG_DIM                       */

    float origin_sig[LG_SIG_DIM];           /* unit vector — voice                */
    float boundary_sig[LG_SIG_DIM];         /* unit vector — what we refuse       */
    int   origin_set;
    int   boundary_set;

    int   n_experts;
    float* expert_w;                        /* [n_experts, dim]                   */
    float* expert_b;                        /* [n_experts]                        */
    float* expert_credit;                   /* [n_experts] — phase 2.5 adaptive   */
                                            /* softplus(credit) is the soft       */
                                            /* weight in aggregate consensus.     */
                                            /* Updated by lg_field_update_experts.*/

    /* Verdict thresholds on (origin_score - boundary_score) ∈ [-1, +1].          */
    /* Tunable; defaults chosen to give clear separation in smoke.                */
    float thresh_pass;     /* >= → PASS                                           */
    float thresh_weaken;   /* >= and < pass → WEAKEN                              */
    float thresh_freeze;   /* >= and < weaken → FREEZE                            */
    float thresh_scar;     /* < freeze and boundary > origin → SCAR               */
    float thresh_dark;     /* < scar and boundary >> origin → DARK                */
    /* Anything else → SILENCE.                                                   */

    /* Scar log — wound shapes, no gradient                                       */
    int    scar_count;
    int    scar_cap;
    float* scar_sigs;                       /* [scar_cap, dim]                    */

    /* Dark matter — knowledge stored as external object                          */
    int    dark_count;
    int    dark_cap;
    float* dark_sigs;                       /* [dark_cap, dim]                    */

    /* Counters (for smoke reporting)                                             */
    int counters[LG_VERDICT_COUNT];
} lg_field_t;

/* ── Field lifecycle ──────────────────────────────────────────────────────── */

int  lg_field_init(lg_field_t* f, int n_experts, uint64_t seed);
void lg_field_free(lg_field_t* f);

/* Set origin / boundary directly from a unit vector (for tests).                 */
void lg_field_set_origin_unit  (lg_field_t* f, const float* unit_vec);
void lg_field_set_boundary_unit(lg_field_t* f, const float* unit_vec);

/* Set origin / boundary by averaging sketches from a corpus pass.                */
/* sketches: [n, LG_SIG_DIM] — already unit-normalized per row.                   */
void lg_field_set_origin_from_sketches  (lg_field_t* f, const float* sketches, int n);
void lg_field_set_boundary_from_sketches(lg_field_t* f, const float* sketches, int n);

/* After both are set, distribute experts along the origin↔boundary axis.         */
/* Each expert gets a slightly perturbed projection so the parliament has         */
/* internal diversity rather than identical votes.                                */
void lg_field_calibrate_experts(lg_field_t* f, uint64_t seed);

/* ── Voting ───────────────────────────────────────────────────────────────── */

/* Vote on an input signature (unit vector, dim = LG_SIG_DIM).                    */
/* Returns the verdict; out_alpha is the gradient scale for WEAKEN (else 1 or 0). */
lg_verdict_t lg_field_vote(const lg_field_t* f, const float* sig, float* out_alpha);

/* Record verdict effects (scar log, dark store, counters).                       */
/* Caller is responsible for actually scaling / zeroing the gradient — this       */
/* function only records the field-side bookkeeping.                              */
void lg_field_record(lg_field_t* f, lg_verdict_t v, const float* sig);

/* Phase 2.5 — adaptive expert credits.                                           */
/*                                                                                */
/* Update each expert's credit by the alignment of its vote with the supervised   */
/* target. `target_origin = 1` means the sample is from the origin distribution   */
/* (parliament should pass it); `target_origin = 0` means it is adversarial       */
/* (parliament should block).                                                     */
/*                                                                                */
/* Reward per expert = (vote_e ∈ [-1, +1]) × (target_sign ∈ {+1, -1}).            */
/* credit_e += lr × reward_e, clamped to ±LG_CREDIT_CLAMP.                        */
/*                                                                                */
/* Aggregate consensus is then a softplus(credit)-weighted mean of votes, so      */
/* experts that learn correctly contribute more over time.                        */
void lg_field_update_experts(lg_field_t* f, const float* sig,
                             int target_origin, float lr);

/* Hard clamp for expert credit. Prevents runaway weights.                        */
#define LG_CREDIT_CLAMP 10.0f

/* Reset counters (between smoke phases).                                         */
void lg_field_reset_counters(lg_field_t* f);

/* Print field summary: origin/boundary alignment, expert spread, counters.       */
void lg_field_summary(const lg_field_t* f, const char* label);

/* ── Signatures ───────────────────────────────────────────────────────────── */

/*
 * lg_signature_from_grads
 *
 * Compute a R^LG_SIG_DIM unit-vector signature by running count-sketch over
 * gradients of all params currently registered on the active notorch tape.
 *
 * Caller protocol:
 *   1. nt_tape_clear()
 *   2. build forward graph, compute loss
 *   3. nt_tape_backward(loss_idx)
 *   4. lg_signature_from_grads(out_sig)
 *
 * Hash uses a stable mixing function so the same param/element combination
 * always lands in the same bin with the same sign across calls.
 */
void lg_signature_from_grads(float* out_sig);

/*
 * lg_signature_from_buffer
 *
 * Compute a R^LG_SIG_DIM unit-vector signature directly from a generic float
 * buffer using the same count-sketch. Useful for tests and for the caller
 * that wants to build its own sketch source without going through the tape.
 */
void lg_signature_from_buffer(const float* buf, int len, float* out_sig);

/*
 * lg_signature_from_text
 *
 * Content-sensitive count-sketch on character bigrams. Two English strings
 * of similar length but different vocabulary land in different parts of the
 * signature space, which lg_signature_from_buffer does not provide on its
 * own. Use this for the phase-1 smoke; use lg_signature_from_grads for
 * actual gradient routing during a model run.
 */
void lg_signature_from_text(const char* text, int len, float* out_sig);

/* Stable count-sketch hash exposed for tests. Returns (bin, sign).               */
void lg_count_sketch_hash(uint64_t key, int dim, int* out_bin, float* out_sign);

#endif /* LORAGRAD_H */
