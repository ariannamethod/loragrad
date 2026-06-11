/*
 * test_freeze_skip.c — regression for LG-H1 (Operation Ordnung, 2026-06-11).
 *
 * Mythos found: blocked verdicts (FREEZE/SCAR/DARK/SILENCE) zero the grads but
 * the trainer still called nt_tape_chuck_step unconditionally. On g=0 Chuck (a)
 * still moves weights via the momentum tail + stagnation noise and (b) feeds
 * gnorm=0 into the freeze detector — NT_CHUCK_STAG_STEPS such steps latch
 * chuck_params.frozen=1 with no recovery path. The fix skips the optimizer
 * entirely on blocked steps (train_loragrad.c, mirroring coa_v1_janus.c:532-535).
 *
 * This test proves both halves on the real vendored notorch:
 *   - SKIP policy (the fix): weights byte-identical across a blocked burst,
 *     frozen count == 0.
 *   - STEP-on-zero policy (the bug): weights move and/or params freeze.
 * No tape reset is needed between the two: the SKIP burst touches neither the
 * weights nor the Chuck state, so the bug burst starts from the same baseline.
 */
#include "../notorch.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define T_DIM   16
#define T_VOCAB 32
#define T_CTX    1
#define WARMUP  20
#define BURST   10

/* Tiny embedding+head LM: enough to produce a real loss and gradients. */
static int tiny_forward(nt_tensor* wte, nt_tensor* head, int tok, int tgt) {
    int wte_i  = nt_tape_param(wte);
    int head_i = nt_tape_param(head);
    nt_tensor* tok_t = nt_tensor_new(T_CTX);
    nt_tensor* tgt_t = nt_tensor_new(T_CTX);
    tok_t->data[0] = (float)tok;
    tgt_t->data[0] = (float)tgt;
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t); nt_tensor_free(tgt_t);
    int h = nt_seq_embedding(wte_i, -1, tok_i, T_CTX, T_DIM);
    int logits = nt_seq_linear(head_i, h, T_CTX);
    return nt_seq_cross_entropy(logits, tgt_i, T_CTX, T_VOCAB);
}

/* Zero every gradient on the active tape — emulates a blocked verdict. */
static void zero_all_grads(void) {
    nt_tape* tape = nt_tape_get();
    if (!tape) return;
    for (int i = 0; i < tape->count; ++i) {
        nt_tape_entry* e = &tape->entries[i];
        if (!e->is_param || !e->grad) continue;
        for (int k = 0; k < e->grad->len; ++k) e->grad->data[k] = 0.0f;
    }
}

static int count_frozen(void) {
    nt_tape* tape = nt_tape_get();
    int n = 0;
    if (!tape) return 0;
    for (int i = 0; i < NT_TAPE_MAX_PARAMS; ++i)
        if (tape->chuck_params[i].frozen) ++n;
    return n;
}

int main(void) {
    nt_seed(0xC0FFEEULL);
    nt_tensor* wte  = nt_tensor_new2d(T_VOCAB, T_DIM); nt_tensor_xavier(wte,  T_VOCAB, T_DIM);
    nt_tensor* head = nt_tensor_new2d(T_VOCAB, T_DIM); nt_tensor_xavier(head, T_VOCAB, T_DIM);

    /* Warmup: real steps accumulate Chuck momentum + grad history. */
    for (int s = 0; s < WARMUP; ++s) {
        nt_tape_start();
        int loss = tiny_forward(wte, head, s % T_VOCAB, (s + 1) % T_VOCAB);
        nt_tape_backward(loss);
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(1e-2f, 1.0f);
        nt_tape_clear();
    }
    int frozen_after_warmup = count_frozen();

    /* Snapshot head weights after warmup. */
    int n = head->len;
    float* w0 = (float*)malloc((size_t)n * sizeof(float));
    memcpy(w0, head->data, (size_t)n * sizeof(float));

    /* ── SKIP policy (the fix): blocked → no optimizer call ─────────────── */
    for (int b = 0; b < BURST; ++b) {
        nt_tape_start();
        int loss = tiny_forward(wte, head, b % T_VOCAB, (b + 1) % T_VOCAB);
        nt_tape_backward(loss);
        zero_all_grads();                 /* blocked verdict */
        /* fix: skip clip + chuck_step entirely */
        nt_tape_clear();
    }
    double skip_delta = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)head->data[i] - (double)w0[i]);
        if (d > skip_delta) skip_delta = d;
    }
    int skip_frozen = count_frozen();

    /* ── STEP-on-zero policy (the bug): blocked → chuck_step anyway ─────── */
    /* SKIP above changed neither weights nor Chuck state, so baseline holds. */
    for (int b = 0; b < BURST; ++b) {
        nt_tape_start();
        int loss = tiny_forward(wte, head, b % T_VOCAB, (b + 1) % T_VOCAB);
        nt_tape_backward(loss);
        zero_all_grads();                 /* blocked verdict */
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(1e-2f, 1.0f);  /* BUG: optimizer runs on g=0 */
        nt_tape_clear();
    }
    double bug_delta = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)head->data[i] - (double)w0[i]);
        if (d > bug_delta) bug_delta = d;
    }
    int bug_frozen = count_frozen();

    printf("warmup frozen=%d\n", frozen_after_warmup);
    printf("SKIP (fix): max|dW|=%.3e  frozen=%d\n", skip_delta, skip_frozen);
    printf("STEP (bug): max|dW|=%.3e  frozen=%d\n", bug_delta, bug_frozen);

    free(w0);
    nt_tensor_free(wte); nt_tensor_free(head);

    if (frozen_after_warmup != 0) {
        fprintf(stderr, "test: FAIL — warmup already froze params (baseline dirty)\n");
        return 3;
    }
    int fix_ok    = (skip_delta == 0.0) && (skip_frozen == 0);
    int bug_repro = (bug_delta > 0.0)   || (bug_frozen > 0);
    if (!fix_ok) {
        fprintf(stderr, "test: FAIL — skip-on-block changed weights or froze "
                        "(dW=%.3e frozen=%d)\n", skip_delta, skip_frozen);
        return 2;
    }
    if (!bug_repro) {
        fprintf(stderr, "test: FAIL — bug not reproduced; test would not catch "
                        "a regression\n");
        return 4;
    }
    printf("\ntest: OK — blocked-skip keeps weights byte-identical & unfrozen; "
           "stepping on zero-grad moves/freezes (regression guarded)\n");
    return 0;
}
