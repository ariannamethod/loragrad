/*
 * loragrad.c — implementation
 *
 * Phase 1: static experts along origin↔boundary axis. Count-sketch over
 * gradient buffers from notorch tape. Voting blends origin/boundary
 * resonance with parliament consensus and yields a verdict that the caller
 * applies to its own gradients before the optimizer step.
 */

#include "loragrad.h"

#ifndef LG_STANDALONE
#include "notorch.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ── stable mixer for count-sketch ─────────────────────────────────────────── */

/* SplitMix64 — small, fast, deterministic. We use it for both bin selection
 * and sign extraction on top of a stream of (key, salt) pairs. */
static inline uint64_t lg_mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z =  z ^ (z >> 31);
    return z;
}

static inline uint64_t lg_mix2(uint64_t a, uint64_t b) {
    return lg_mix64(a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)));
}

void lg_count_sketch_hash(uint64_t key, int dim, int* out_bin, float* out_sign) {
    uint64_t h = lg_mix64(key);
    *out_bin  = (int)(h % (uint64_t)dim);
    /* second pass for the sign to avoid correlation between bin and sign */
    uint64_t s = lg_mix64(h ^ 0xa5a5a5a5deadbeefULL);
    *out_sign = (s & 1ULL) ? +1.0f : -1.0f;
}

static void lg_normalize_unit(float* v, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) s += (double)v[i] * (double)v[i];
    if (s <= 1e-20) {
        /* Empty / zero sketch — yield axis-aligned unit so dot products are
         * well-defined and small (always orthogonal-ish to anything real). */
        for (int i = 0; i < dim; ++i) v[i] = 0.0f;
        v[0] = 1.0f;
        return;
    }
    double inv = 1.0 / sqrt(s);
    for (int i = 0; i < dim; ++i) v[i] = (float)((double)v[i] * inv);
}

static float lg_dot(const float* a, const float* b, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) s += (double)a[i] * (double)b[i];
    return (float)s;
}

/* ── verdict names ─────────────────────────────────────────────────────────── */

const char* lg_verdict_name(lg_verdict_t v) {
    switch (v) {
        case LG_PASS:    return "PASS";
        case LG_WEAKEN:  return "WEAKEN";
        case LG_FREEZE:  return "FREEZE";
        case LG_SCAR:    return "SCAR";
        case LG_DARK:    return "DARK";
        case LG_SILENCE: return "SILENCE";
        default:         return "?";
    }
}

/* ── signatures ────────────────────────────────────────────────────────────── */

void lg_signature_from_buffer(const float* buf, int len, float* out_sig) {
    int   bin;
    float sign;
    for (int i = 0; i < LG_SIG_DIM; ++i) out_sig[i] = 0.0f;
    for (int i = 0; i < len; ++i) {
        /* Hash on (position, value) so identical positions with different
         * values land in different bins. Without the value mix, two same-
         * length English strings would produce indistinguishable sketches. */
        uint64_t key = lg_mix2((uint64_t)i + 1ULL, (uint64_t)(uint32_t)buf[i]);
        lg_count_sketch_hash(key, LG_SIG_DIM, &bin, &sign);
        out_sig[bin] += sign * buf[i];
    }
    lg_normalize_unit(out_sig, LG_SIG_DIM);
}

/*
 * Trigram count-sketch — content-sensitive feature extractor for text.
 *
 * For each adjacent byte triple (b_i, b_{i+1}, b_{i+2}) we hash the triple
 * into a bin with a sign and increment by 1.0. Trigrams give meaningfully
 * higher discrimination than bigrams when the corpora are short and share
 * surface English bigrams (e.g. "th", "he", "re") but differ in lexical
 * content. This is what the phase-1 smoke uses.
 */
void lg_signature_from_text(const char* text, int len, float* out_sig) {
    int   bin;
    float sign;
    for (int i = 0; i < LG_SIG_DIM; ++i) out_sig[i] = 0.0f;
    if (len < 3) {
        out_sig[0] = 1.0f;
        return;
    }
    for (int i = 0; i + 2 < len; ++i) {
        uint64_t a = (uint64_t)(unsigned char)text[i];
        uint64_t b = (uint64_t)(unsigned char)text[i + 1];
        uint64_t c = (uint64_t)(unsigned char)text[i + 2];
        uint64_t key = lg_mix2((a << 16) | (b << 8) | c, 0xb16fafULL);
        lg_count_sketch_hash(key, LG_SIG_DIM, &bin, &sign);
        out_sig[bin] += sign;
    }
    lg_normalize_unit(out_sig, LG_SIG_DIM);
}

/*
 * Walk the active notorch tape; for every parameter entry with an allocated
 * gradient, count-sketch every gradient element into out_sig.
 *
 * Key for hashing: combine (entry_index, element_index) so the same
 * parameter/element pair always lands in the same bin with the same sign.
 */
#ifndef LG_STANDALONE
void lg_signature_from_grads(float* out_sig) {
    nt_tape* tape = nt_tape_get();
    int   bin;
    float sign;
    for (int i = 0; i < LG_SIG_DIM; ++i) out_sig[i] = 0.0f;

    if (!tape) {
        out_sig[0] = 1.0f;
        return;
    }
    for (int e = 0; e < tape->count; ++e) {
        const nt_tape_entry* en = &tape->entries[e];
        if (!en->is_param) continue;
        if (!en->grad)     continue;
        const float* g = en->grad->data;
        const int    n = en->grad->len;
        for (int k = 0; k < n; ++k) {
            uint64_t key = lg_mix2((uint64_t)e + 1ULL, (uint64_t)k);
            lg_count_sketch_hash(key, LG_SIG_DIM, &bin, &sign);
            out_sig[bin] += sign * g[k];
        }
    }
    lg_normalize_unit(out_sig, LG_SIG_DIM);
}
#else
/* Standalone build — provide a stub so linking works for callers that
 * never need gradient-based signatures (e.g., the phase-1 smoke test).
 * Yields an axis-aligned unit vector that is benign in dot products. */
void lg_signature_from_grads(float* out_sig) {
    for (int i = 0; i < LG_SIG_DIM; ++i) out_sig[i] = 0.0f;
    out_sig[0] = 1.0f;
}
#endif

/* ── field lifecycle ───────────────────────────────────────────────────────── */

int lg_field_init(lg_field_t* f, int n_experts, uint64_t seed) {
    if (!f) return -1;
    if (n_experts <= 0) n_experts = LG_DEFAULT_EXPERTS;

    memset(f, 0, sizeof(*f));
    f->dim       = LG_SIG_DIM;
    f->n_experts = n_experts;

    f->expert_w = (float*)calloc((size_t)n_experts * LG_SIG_DIM, sizeof(float));
    f->expert_b = (float*)calloc((size_t)n_experts, sizeof(float));
    if (!f->expert_w || !f->expert_b) {
        free(f->expert_w); free(f->expert_b);
        return -2;
    }

    /* Reasonable defaults — chosen for axis-projection scoring where the
     * blended score sits in roughly [-1, +1]. Tuned in smoke. */
    f->thresh_pass   =  0.40f;
    f->thresh_weaken =  0.10f;
    f->thresh_freeze = -0.10f;
    f->thresh_scar   =  0.25f;  /* applied to -delta_axis */
    f->thresh_dark   =  0.50f;

    f->scar_cap  = LG_SCAR_CAP_DEFAULT;
    f->scar_sigs = (float*)calloc((size_t)f->scar_cap * LG_SIG_DIM, sizeof(float));
    f->dark_cap  = LG_DARK_CAP_DEFAULT;
    f->dark_sigs = (float*)calloc((size_t)f->dark_cap * LG_SIG_DIM, sizeof(float));

    /* Touch seed so it's not an unused warning; expert calibration uses it. */
    (void)seed;
    return 0;
}

void lg_field_free(lg_field_t* f) {
    if (!f) return;
    free(f->expert_w);  f->expert_w  = NULL;
    free(f->expert_b);  f->expert_b  = NULL;
    free(f->scar_sigs); f->scar_sigs = NULL;
    free(f->dark_sigs); f->dark_sigs = NULL;
}

void lg_field_set_origin_unit(lg_field_t* f, const float* unit_vec) {
    memcpy(f->origin_sig, unit_vec, sizeof(float) * LG_SIG_DIM);
    lg_normalize_unit(f->origin_sig, LG_SIG_DIM);
    f->origin_set = 1;
}

void lg_field_set_boundary_unit(lg_field_t* f, const float* unit_vec) {
    memcpy(f->boundary_sig, unit_vec, sizeof(float) * LG_SIG_DIM);
    lg_normalize_unit(f->boundary_sig, LG_SIG_DIM);
    f->boundary_set = 1;
}

static void lg_mean_unit(const float* sketches, int n, float* out) {
    for (int i = 0; i < LG_SIG_DIM; ++i) out[i] = 0.0f;
    if (n <= 0) { out[0] = 1.0f; return; }
    for (int j = 0; j < n; ++j) {
        const float* row = sketches + (size_t)j * LG_SIG_DIM;
        for (int i = 0; i < LG_SIG_DIM; ++i) out[i] += row[i];
    }
    float inv = 1.0f / (float)n;
    for (int i = 0; i < LG_SIG_DIM; ++i) out[i] *= inv;
    lg_normalize_unit(out, LG_SIG_DIM);
}

void lg_field_set_origin_from_sketches(lg_field_t* f, const float* sketches, int n) {
    lg_mean_unit(sketches, n, f->origin_sig);
    f->origin_set = 1;
}

void lg_field_set_boundary_from_sketches(lg_field_t* f, const float* sketches, int n) {
    lg_mean_unit(sketches, n, f->boundary_sig);
    f->boundary_set = 1;
}

/*
 * Calibrate experts: spread along the origin axis, slightly tilted toward and
 * away from boundary, with a small isotropic perturbation so the parliament
 * has internal diversity rather than identical votes.
 *
 * Per expert i:
 *   w_i = cos(theta_i) * origin + sin(theta_i) * boundary_perp + noise
 *
 * theta_i ∈ [-pi/6, +pi/6] linearly across n_experts, so all experts lean
 * toward origin (positive projection) but with non-trivial spread.
 */
void lg_field_calibrate_experts(lg_field_t* f, uint64_t seed) {
    if (!f->origin_set || !f->boundary_set) return;

    /* boundary_perp = boundary - (origin·boundary) origin, then normalize. */
    float ob = lg_dot(f->origin_sig, f->boundary_sig, LG_SIG_DIM);
    float bperp[LG_SIG_DIM];
    for (int i = 0; i < LG_SIG_DIM; ++i) {
        bperp[i] = f->boundary_sig[i] - ob * f->origin_sig[i];
    }
    lg_normalize_unit(bperp, LG_SIG_DIM);

    uint64_t s = seed ? seed : 0xc0ffeec0ffeefeedULL;
    const double PI = 3.14159265358979323846;
    const double tilt_max = PI / 6.0;

    for (int e = 0; e < f->n_experts; ++e) {
        double t = (f->n_experts == 1) ? 0.0
                 : (-tilt_max + (2.0 * tilt_max) * ((double)e / (f->n_experts - 1)));
        float c = (float)cos(t);
        float sn = (float)sin(t);
        float* row = f->expert_w + (size_t)e * LG_SIG_DIM;
        for (int i = 0; i < LG_SIG_DIM; ++i) {
            s = lg_mix64(s + 1);
            float noise = (((s >> 11) & 0xFFFFF) / (float)0xFFFFF - 0.5f) * 0.05f;
            row[i] = c * f->origin_sig[i] + sn * bperp[i] + noise;
        }
        /* Make each expert weight a unit vector so its dot product with sig
         * sits in [-1, +1] and tanh saturation is meaningful. */
        lg_normalize_unit(row, LG_SIG_DIM);

        s = lg_mix64(s + 7);
        float bnoise = (((s >> 11) & 0xFFFFF) / (float)0xFFFFF - 0.5f) * 0.05f;
        f->expert_b[e] = bnoise;
    }
}

/* ── voting ────────────────────────────────────────────────────────────────── */

lg_verdict_t lg_field_vote(const lg_field_t* f, const float* sig, float* out_alpha) {
    if (out_alpha) *out_alpha = 1.0f;

    if (!f || !f->origin_set || !f->boundary_set) {
        /* Field not configured — refuse to make a decision rather than
         * silently passing. This is a setup error, not a real verdict. */
        if (out_alpha) *out_alpha = 0.0f;
        return LG_FREEZE;
    }

    /* Discriminative axis: project the input onto (origin - boundary) so the
     * dynamic range is [-1, +1] regardless of how aligned origin and
     * boundary happen to be in the raw signature space. With short English
     * corpora, the raw cosine can sit at +0.7 or higher; the difference
     * vector is what carries the useful sign. */
    float axis[LG_SIG_DIM];
    for (int i = 0; i < LG_SIG_DIM; ++i) {
        axis[i] = f->origin_sig[i] - f->boundary_sig[i];
    }
    /* Local normalize without touching the field's caches. */
    {
        double s = 0.0;
        for (int i = 0; i < LG_SIG_DIM; ++i) s += (double)axis[i] * axis[i];
        if (s <= 1e-20) {
            /* origin and boundary are identical — no decision possible. */
            if (out_alpha) *out_alpha = 0.0f;
            return LG_FREEZE;
        }
        double inv = 1.0 / sqrt(s);
        for (int i = 0; i < LG_SIG_DIM; ++i) axis[i] = (float)(axis[i] * inv);
    }
    float delta_axis     = lg_dot(sig, axis, LG_SIG_DIM);            /* [-1, +1] */
    float origin_score   = lg_dot(sig, f->origin_sig,   LG_SIG_DIM);
    float boundary_score = lg_dot(sig, f->boundary_sig, LG_SIG_DIM);

    /* Parliament consensus — modulates inside the resonance frame. */
    double consensus = 0.0;
    for (int e = 0; e < f->n_experts; ++e) {
        const float* w = f->expert_w + (size_t)e * LG_SIG_DIM;
        float z = lg_dot(w, sig, LG_SIG_DIM) + f->expert_b[e];
        consensus += tanh((double)z);
    }
    consensus /= (double)f->n_experts;                                /* [-1, +1] */

    float score = 0.7f * delta_axis + 0.3f * (float)consensus;

    if (score >= f->thresh_pass) {
        if (out_alpha) *out_alpha = 1.0f;
        return LG_PASS;
    }
    if (score >= f->thresh_weaken) {
        float a = (score - f->thresh_weaken) / (f->thresh_pass - f->thresh_weaken);
        if (a < 0.0f) a = 0.0f; if (a > 1.0f) a = 1.0f;
        if (out_alpha) *out_alpha = a;
        return LG_WEAKEN;
    }

    /* Below freeze: check if specifically boundary-aligned.                       */
    /* The decision uses the AXIS projection from the boundary side                */
    /* (i.e. -delta_axis), not raw boundary_score, since raw scores can be         */
    /* simultaneously positive on both origin and boundary when the corpora        */
    /* share a strong common direction.                                            */
    if (-delta_axis >= f->thresh_dark) {
        if (out_alpha) *out_alpha = 0.0f;
        return LG_DARK;
    }
    if (-delta_axis >= f->thresh_scar) {
        if (out_alpha) *out_alpha = 0.0f;
        return LG_SCAR;
    }

    if (score >= f->thresh_freeze) {
        if (out_alpha) *out_alpha = 0.0f;
        return LG_FREEZE;
    }

    /* Suppress unused-variable warnings when not in debug build. */
    (void)origin_score; (void)boundary_score;

    if (out_alpha) *out_alpha = 0.0f;
    return LG_SILENCE;
}

/* ── recording ─────────────────────────────────────────────────────────────── */

void lg_field_record(lg_field_t* f, lg_verdict_t v, const float* sig) {
    if (!f) return;
    if (v >= 0 && v < LG_VERDICT_COUNT) f->counters[v]++;

    if (v == LG_SCAR && f->scar_count < f->scar_cap && sig) {
        memcpy(f->scar_sigs + (size_t)f->scar_count * LG_SIG_DIM,
               sig, sizeof(float) * LG_SIG_DIM);
        f->scar_count++;
    }
    if (v == LG_DARK && f->dark_count < f->dark_cap && sig) {
        memcpy(f->dark_sigs + (size_t)f->dark_count * LG_SIG_DIM,
               sig, sizeof(float) * LG_SIG_DIM);
        f->dark_count++;
    }
}

void lg_field_reset_counters(lg_field_t* f) {
    if (!f) return;
    for (int i = 0; i < LG_VERDICT_COUNT; ++i) f->counters[i] = 0;
}

void lg_field_summary(const lg_field_t* f, const char* label) {
    if (!f) return;
    float ob = (f->origin_set && f->boundary_set)
             ? lg_dot(f->origin_sig, f->boundary_sig, LG_SIG_DIM)
             : 0.0f;
    printf("[lg_field:%s] origin=%d boundary=%d origin·boundary=%+0.4f experts=%d\n",
           label ? label : "?", f->origin_set, f->boundary_set, ob, f->n_experts);
    printf("  thresholds: pass=%+0.2f weaken=%+0.2f freeze=%+0.2f scar=%+0.2f dark=%+0.2f\n",
           f->thresh_pass, f->thresh_weaken, f->thresh_freeze,
           f->thresh_scar, f->thresh_dark);
    printf("  counters:");
    for (int i = 0; i < LG_VERDICT_COUNT; ++i) {
        printf(" %s=%d", lg_verdict_name((lg_verdict_t)i), f->counters[i]);
    }
    printf("\n");
    printf("  scar_log=%d/%d dark_store=%d/%d\n",
           f->scar_count, f->scar_cap, f->dark_count, f->dark_cap);
}
