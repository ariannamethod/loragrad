/*
 * train_loragrad.c — char-level tiny LLaMA on notorch with loragrad routing
 *
 * Architecture
 *   dim=192, L=3, H=4, head_dim=48, FFN=384, CTX=128, V=128 (char ASCII)
 *   ≈1.15M parameters. RoPE + RMSNorm + SwiGLU + MHA. Chuck optimizer.
 *
 * Data
 *   data/corpus.txt       — clean voice corpus (target voice)
 *   data/adversarial.txt  — sycophantic / mock / random byte streams
 *
 * Pipeline
 *   1. Read both corpora as ASCII byte streams (V=128).
 *   2. Warmup-A: 50 steps on clean stream → average gradient sketches → origin_sig.
 *   3. Warmup-B: 50 steps on adversarial stream → average gradient sketches → boundary_sig.
 *   4. Calibrate static expert parliament along origin↔boundary axis.
 *   5. Main loop (default 5000 steps):
 *        - sample window from mixed stream (90% clean, 10% adversarial)
 *        - forward + cross-entropy loss + backward
 *        - text-signature from the raw window bytes (parliament reads the text)
 *        - parliament votes → verdict + alpha
 *        - apply verdict to gradients (PASS=1 / WEAKEN=alpha / else 0)
 *        - chuck_step
 *        - log step / loss / verdict / alpha / source-label
 *   6. Periodically eval val loss on held-out clean tail.
 *   7. Generate samples on a few prompts at the end.
 *   8. Save checkpoint + run log.
 *
 * Modes
 *   --routed     loragrad routing active (default)
 *   --control    no routing (vanilla baseline for comparison)
 *
 * Build via project Makefile.
 */

#include "../loragrad.h"
#include "notorch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>

/* ── Model dimensions ─────────────────────────────────────────────────────── */

#define DIM       192
#define NLAYERS     3
#define NHEADS      4
#define HEAD_DIM   (DIM / NHEADS)   /* 48 */
#define HIDDEN    384
#define CTX       128
#define VOCAB     128              /* char-level ASCII */

#define WARMUP_CLEAN  200
#define WARMUP_BAD    100
#define LOG_EVERY     100
#define EVAL_EVERY    500
#define EVAL_WINDOWS   16

/* Mix ratio: probability of drawing from the adversarial stream per step.    */
/* The point of the run is to demonstrate the parliament filtering these out. */
#define MIX_ADVERSARIAL_PROB  0.10f

/* ── Data ─────────────────────────────────────────────────────────────────── */

typedef struct {
    char* clean;       int n_clean;
    char* adv;         int n_adv;
    /* Held-out tail of clean for val. */
    int   val_off;     int   val_len;
} Corpus;

static int read_file(const char* path, char** out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, (size_t)sz, f); buf[sz] = 0; fclose(f);
    /* Coerce bytes to V<=128: any byte >= 128 → '?' so the model never sees
     * a token outside its embedding table. */
    for (long i = 0; i < sz; ++i) {
        unsigned char c = (unsigned char)buf[i];
        if (c >= 128) buf[i] = '?';
    }
    *out = buf;
    return (int)sz;
}

static int corpus_load(Corpus* c) {
    c->n_clean = read_file("data/corpus.txt",      &c->clean);
    c->n_adv   = read_file("data/adversarial.txt", &c->adv);
    if (c->n_clean <= 0 || c->n_adv <= 0) return -1;
    /* Reserve last 10% of clean as val. */
    c->val_off = (int)(c->n_clean * 0.9);
    c->val_len = c->n_clean - c->val_off;
    return 0;
}

static void corpus_free(Corpus* c) {
    free(c->clean); free(c->adv);
}

/* ── Model ────────────────────────────────────────────────────────────────── */

typedef struct {
    nt_tensor* wte;                  /* [V, DIM]                       */
    struct {
        nt_tensor* rms1;
        nt_tensor *wq, *wk, *wv, *wo;
        nt_tensor* rms2;
        nt_tensor *w_gate, *w_up, *w_down;
    } L[NLAYERS];
    nt_tensor* rms_f;
    nt_tensor* head;                 /* [V, DIM]                       */
} Model;

static long count_params(Model* m) {
    long n = m->wte->len + m->rms_f->len + m->head->len;
    for (int l = 0; l < NLAYERS; ++l) {
        n += m->L[l].rms1->len + m->L[l].rms2->len;
        n += m->L[l].wq->len + m->L[l].wk->len + m->L[l].wv->len + m->L[l].wo->len;
        n += m->L[l].w_gate->len + m->L[l].w_up->len + m->L[l].w_down->len;
    }
    return n;
}

static Model* model_new(void) {
    Model* m = (Model*)calloc(1, sizeof(Model));
    m->wte = nt_tensor_new2d(VOCAB, DIM); nt_tensor_xavier(m->wte, VOCAB, DIM);
    float rs = 0.02f / sqrtf(2.0f * NLAYERS);
    for (int l = 0; l < NLAYERS; ++l) {
        m->L[l].rms1 = nt_tensor_new(DIM); nt_tensor_fill(m->L[l].rms1, 1.0f);
        m->L[l].wq = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wq, DIM, DIM);
        m->L[l].wk = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wk, DIM, DIM);
        m->L[l].wv = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wv, DIM, DIM);
        m->L[l].wo = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wo, DIM, DIM);
        for (int i = 0; i < m->L[l].wo->len; ++i) m->L[l].wo->data[i] *= rs / 0.1f;
        m->L[l].rms2   = nt_tensor_new(DIM);             nt_tensor_fill(m->L[l].rms2, 1.0f);
        m->L[l].w_gate = nt_tensor_new2d(HIDDEN, DIM);   nt_tensor_xavier(m->L[l].w_gate, DIM, HIDDEN);
        m->L[l].w_up   = nt_tensor_new2d(HIDDEN, DIM);   nt_tensor_xavier(m->L[l].w_up,   DIM, HIDDEN);
        m->L[l].w_down = nt_tensor_new2d(DIM, HIDDEN);   nt_tensor_xavier(m->L[l].w_down, HIDDEN, DIM);
        for (int i = 0; i < m->L[l].w_down->len; ++i) m->L[l].w_down->data[i] *= rs / 0.1f;
    }
    m->rms_f = nt_tensor_new(DIM);             nt_tensor_fill(m->rms_f, 1.0f);
    m->head  = nt_tensor_new2d(VOCAB, DIM);    nt_tensor_xavier(m->head, DIM, VOCAB);
    return m;
}

static void model_free(Model* m) {
    nt_tensor_free(m->wte);
    for (int l = 0; l < NLAYERS; ++l) {
        nt_tensor_free(m->L[l].rms1); nt_tensor_free(m->L[l].rms2);
        nt_tensor_free(m->L[l].wq); nt_tensor_free(m->L[l].wk);
        nt_tensor_free(m->L[l].wv); nt_tensor_free(m->L[l].wo);
        nt_tensor_free(m->L[l].w_gate); nt_tensor_free(m->L[l].w_up); nt_tensor_free(m->L[l].w_down);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head);
    free(m);
}

static int model_n_tensors(void) { return 1 + NLAYERS * 9 + 2; }

static nt_tensor** model_param_array(Model* m) {
    int n = model_n_tensors();
    nt_tensor** p = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int i = 0;
    p[i++] = m->wte;
    for (int l = 0; l < NLAYERS; ++l) {
        p[i++] = m->L[l].rms1;
        p[i++] = m->L[l].wq;   p[i++] = m->L[l].wk;
        p[i++] = m->L[l].wv;   p[i++] = m->L[l].wo;
        p[i++] = m->L[l].rms2;
        p[i++] = m->L[l].w_gate; p[i++] = m->L[l].w_up; p[i++] = m->L[l].w_down;
    }
    p[i++] = m->rms_f; p[i++] = m->head;
    return p;
}

/* ── Forward ──────────────────────────────────────────────────────────────── */

static int forward(Model* m, int* tokens, int* targets) {
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    int li[NLAYERS][9];
    for (int l = 0; l < NLAYERS; ++l) {
        li[l][0] = nt_tape_param(m->L[l].rms1);
        li[l][1] = nt_tape_param(m->L[l].wq);
        li[l][2] = nt_tape_param(m->L[l].wk);
        li[l][3] = nt_tape_param(m->L[l].wv);
        li[l][4] = nt_tape_param(m->L[l].wo);
        li[l][5] = nt_tape_param(m->L[l].rms2);
        li[l][6] = nt_tape_param(m->L[l].w_gate);
        li[l][7] = nt_tape_param(m->L[l].w_up);
        li[l][8] = nt_tape_param(m->L[l].w_down);
    }
    int rmsf_i = nt_tape_param(m->rms_f);
    int head_i = nt_tape_param(m->head);

    nt_tensor* tok_t = nt_tensor_new(CTX);
    nt_tensor* tgt_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; ++i) {
        tok_t->data[i] = (float)tokens[i];
        tgt_t->data[i] = (float)targets[i];
    }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t); nt_tensor_free(tgt_t);

    int h = nt_seq_embedding(wte_i, -1, tok_i, CTX, DIM);
    for (int l = 0; l < NLAYERS; ++l) {
        int xn = nt_seq_rmsnorm(h, li[l][0], CTX, DIM);
        int q = nt_seq_linear(li[l][1], xn, CTX);
        int k = nt_seq_linear(li[l][2], xn, CTX);
        int v = nt_seq_linear(li[l][3], xn, CTX);
        q = nt_rope(q, CTX, HEAD_DIM);
        k = nt_rope(k, CTX, HEAD_DIM);
        int attn = nt_mh_causal_attention(q, k, v, CTX, HEAD_DIM);
        int proj = nt_seq_linear(li[l][4], attn, CTX);
        h = nt_add(h, proj);
        xn = nt_seq_rmsnorm(h, li[l][5], CTX, DIM);
        int gate = nt_silu(nt_seq_linear(li[l][6], xn, CTX));
        int up   = nt_seq_linear(li[l][7], xn, CTX);
        int down = nt_seq_linear(li[l][8], nt_mul(gate, up), CTX);
        h = nt_add(h, down);
    }
    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, DIM);
    int logits = nt_seq_linear(head_i, hf, CTX);
    return nt_seq_cross_entropy(logits, tgt_i, CTX, VOCAB);
}

/* ── Sampling helpers ─────────────────────────────────────────────────────── */

static void window_to_tokens(const char* buf, int off, int* tokens, int* targets) {
    for (int i = 0; i < CTX; ++i) {
        unsigned char a = (unsigned char)buf[off + i];
        unsigned char b = (unsigned char)buf[off + i + 1];
        tokens[i]  = (int)a;
        targets[i] = (int)b;
    }
}

static void window_text(const char* buf, int off, char* out) {
    for (int i = 0; i < CTX; ++i) out[i] = buf[off + i];
    out[CTX] = 0;
}

/* Scale every gradient on the active tape by `scale`. Used to apply
 * WEAKEN's alpha before the optimizer step. PASS = 1, FREEZE = 0, etc. */
static void scale_all_grads(float scale) {
    nt_tape* tape = nt_tape_get();
    if (!tape) return;
    for (int i = 0; i < tape->count; ++i) {
        nt_tape_entry* e = &tape->entries[i];
        if (!e->is_param) continue;
        if (!e->grad)     continue;
        for (int k = 0; k < e->grad->len; ++k) e->grad->data[k] *= scale;
    }
}

/* ── Logging ──────────────────────────────────────────────────────────────── */

typedef struct {
    int   counters[LG_VERDICT_COUNT];
    int   confusion[2][LG_VERDICT_COUNT]; /* [source: 0=clean,1=adv][verdict] */
    float loss_sum;     int loss_n;
    float loss_clean;   int clean_n;
    float loss_adv;     int adv_n;
} StepStats;

static void stats_reset(StepStats* s) { memset(s, 0, sizeof(*s)); }

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ── Eval ─────────────────────────────────────────────────────────────────── */

static float eval_val(Model* m, const Corpus* c) {
    if (c->val_len < CTX + 1) return 99.0f;
    int  stride = (c->val_len - CTX - 1) / EVAL_WINDOWS; if (stride < 1) stride = 1;
    float total = 0; int count = 0;
    for (int s = 0; s < EVAL_WINDOWS; ++s) {
        int off = c->val_off + s * stride;
        if (off + CTX + 1 > c->n_clean) break;
        int tokens[CTX], targets[CTX];
        window_to_tokens(c->clean, off, tokens, targets);
        nt_tape_start();
        nt_train_mode(0);
        int loss_idx = forward(m, tokens, targets);
        total += nt_tape_get()->entries[loss_idx].output->data[0];
        ++count;
        nt_tape_clear();
        nt_train_mode(1);
    }
    return count > 0 ? total / count : 99.0f;
}

/* ── Warmup pass: derive a corpus signature from text windows ─────────────── */
/*
 * Sample n_windows random windows from the corpus, compute the text trigram
 * signature for each, return the per-window sketches in out_sketches.
 *
 * The runtime vote also uses text signatures, so warmup and live voting
 * live in the same feature space. Parliament reads the text — exactly what
 * the design calls for.
 */

static void warmup_text_signatures(const char* buf, int n_buf,
                                   int n_windows, unsigned seed,
                                   float* out_sketches /* [n_windows, LG_SIG_DIM] */)
{
    srand(seed);
    for (int s = 0; s < n_windows; ++s) {
        int max_off = n_buf - CTX;
        if (max_off <= 0) break;
        int off = rand() % max_off;
        char window[CTX + 1];
        window_text(buf, off, window);
        lg_signature_from_text(window, CTX, out_sketches + (size_t)s * LG_SIG_DIM);
    }
}

/* ── Generation ───────────────────────────────────────────────────────────── */

static void generate(Model* m, const char* prompt, int max_new, FILE* out) {
    int ctx_buf[CTX]; memset(ctx_buf, 0, sizeof(ctx_buf));
    int prompt_len = (int)strlen(prompt);
    if (prompt_len > CTX) prompt_len = CTX;
    int gen_len = prompt_len;
    for (int i = 0; i < prompt_len; ++i) {
        unsigned char c = (unsigned char)prompt[i];
        if (c >= 128) c = '?';
        ctx_buf[i] = c;
    }
    fputs(prompt, out);
    nt_train_mode(0);
    for (int s = 0; s < max_new && gen_len < CTX; ++s) {
        int tokens[CTX], targets[CTX];
        for (int i = 0; i < CTX; ++i) {
            tokens[i]  = (i < gen_len) ? ctx_buf[i] : 0;
            targets[i] = 0;
        }
        nt_tape_start();
        int loss_idx = forward(m, tokens, targets);
        nt_tape* tape = nt_tape_get();
        int logits_idx = tape->entries[loss_idx].parent1;
        float* last = tape->entries[logits_idx].output->data + (gen_len - 1) * VOCAB;
        /* temperature 0.8 sampling */
        float t = 0.8f;
        float maxv = last[0];
        for (int v = 1; v < VOCAB; ++v) if (last[v] > maxv) maxv = last[v];
        double Z = 0; double probs[VOCAB];
        for (int v = 0; v < VOCAB; ++v) { probs[v] = exp((last[v]-maxv)/t); Z += probs[v]; }
        for (int v = 0; v < VOCAB; ++v) probs[v] /= Z;
        double r = (double)rand() / (double)RAND_MAX;
        int chosen = 0;
        double acc = 0;
        for (int v = 0; v < VOCAB; ++v) {
            acc += probs[v];
            if (r <= acc) { chosen = v; break; }
        }
        nt_tape_clear();
        ctx_buf[gen_len++] = chosen;
        unsigned char ch = (unsigned char)chosen;
        if (ch < 128 && (ch >= 32 || ch == '\n' || ch == '\t')) fputc(ch, out);
        else fputc('?', out);
    }
    fputc('\n', out);
    nt_train_mode(1);
}

/* ── Main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int routed = 1;
    int steps  = 5000;
    int seed   = 42;
    float lr   = 3e-4f;
    const char* run_name = "routed";
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--routed"))  { routed = 1; run_name = "routed"; }
        else if (!strcmp(argv[i], "--control")) { routed = 0; run_name = "control"; }
        else if (!strncmp(argv[i], "--steps=", 8)) steps = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--seed=",  7)) seed  = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--lr=",    5)) lr    = (float)atof(argv[i] + 5);
        else if (!strncmp(argv[i], "--name=",  7)) run_name = argv[i] + 7;
    }

    printf("════════════════════════════════════════════════════════\n");
    printf("  loragrad train (%s)\n", run_name);
    printf("  dim=%d L=%d H=%d HD=%d FFN=%d CTX=%d V=%d\n",
           DIM, NLAYERS, NHEADS, HEAD_DIM, HIDDEN, CTX, VOCAB);
    printf("  steps=%d lr=%.1e seed=%d routed=%d\n", steps, lr, seed, routed);
    printf("════════════════════════════════════════════════════════\n");

    Corpus c;
    if (corpus_load(&c) != 0) return 1;
    printf("corpus: clean=%d B (val tail=%d B), adv=%d B\n",
           c.n_clean, c.val_len, c.n_adv);

    nt_seed((uint64_t)seed);
    Model* m = model_new();
    long np = count_params(m);
    printf("model: %ld params (%.2f MB)\n", np, np * 4.0 / 1048576.0);

    /* ── Warmup signatures ─────────────────────────────────────────────────── */
    lg_field_t field;
    lg_field_init(&field, LG_DEFAULT_EXPERTS, (uint64_t)seed);

    if (routed) {
        printf("\n── warmup A: clean text windows → origin signature (%d) ──\n", WARMUP_CLEAN);
        float* sk_a = (float*)calloc((size_t)WARMUP_CLEAN * LG_SIG_DIM, sizeof(float));
        warmup_text_signatures(c.clean, c.val_off, WARMUP_CLEAN, (unsigned)seed, sk_a);
        lg_field_set_origin_from_sketches(&field, sk_a, WARMUP_CLEAN);
        free(sk_a);

        printf("── warmup B: adversarial text windows → boundary signature (%d) ──\n", WARMUP_BAD);
        float* sk_b = (float*)calloc((size_t)WARMUP_BAD * LG_SIG_DIM, sizeof(float));
        warmup_text_signatures(c.adv, c.n_adv, WARMUP_BAD, (unsigned)(seed + 1), sk_b);
        lg_field_set_boundary_from_sketches(&field, sk_b, WARMUP_BAD);
        free(sk_b);

        lg_field_calibrate_experts(&field, (uint64_t)seed);
        lg_field_summary(&field, "warmup-done");
    } else {
        printf("\n-- control mode: skipping warmup signatures --\n");
    }

    /* ── Main training loop ────────────────────────────────────────────────── */

    StepStats stats; stats_reset(&stats);
    nt_nan_guard guard = nt_nan_guard_new();
    nt_schedule sched  = nt_schedule_cosine(lr, steps / 10, steps, lr * 0.1f);

    char log_path[256];
    snprintf(log_path, sizeof(log_path), "runs/%s.log", run_name);
    system("mkdir -p runs"); /* idempotent, safe */
    FILE* logf = fopen(log_path, "w");
    if (!logf) { fprintf(stderr, "cannot open %s for writing\n", log_path); return 1; }
    fprintf(logf, "# step source loss verdict alpha\n");
    fflush(logf);

    printf("\n── main training (%d steps) ──\n", steps);
    double t0 = now_ms();
    float first_loss = -1.f;
    float best_loss  = 99.f;

    for (int step = 0; step < steps; ++step) {
        float cur_lr = nt_schedule_get_lr(&sched);

        /* Sample source. */
        int from_adv = 0;
        if ((rand() / (float)RAND_MAX) < MIX_ADVERSARIAL_PROB) from_adv = 1;
        const char* src = from_adv ? c.adv   : c.clean;
        int    src_len  = from_adv ? c.n_adv : c.val_off;
        if (src_len < CTX + 1) {
            /* adv corpus too short for a window — wrap or fallback. */
            src = c.clean; src_len = c.val_off; from_adv = 0;
        }
        int max_off = src_len - CTX - 1;
        int off = max_off > 0 ? (rand() % max_off) : 0;

        int tokens[CTX], targets[CTX];
        window_to_tokens(src, off, tokens, targets);

        nt_tape_start();
        int loss_idx = forward(m, tokens, targets);
        float lv = nt_tape_get()->entries[loss_idx].output->data[0];
        if (first_loss < 0) first_loss = lv;
        if (lv < best_loss) best_loss = lv;
        nt_tape_backward(loss_idx);
        if (!nt_nan_guard_check(&guard)) { nt_tape_clear(); continue; }

        lg_verdict_t verdict = LG_PASS;
        float alpha = 1.0f;

        if (routed) {
            /* Parliament reads the text. */
            char window[CTX + 1];
            window_text(src, off, window);
            float text_sig[LG_SIG_DIM];
            lg_signature_from_text(window, CTX, text_sig);
            verdict = lg_field_vote(&field, text_sig, &alpha);
            lg_field_record(&field, verdict, text_sig);

            /* Apply verdict. */
            switch (verdict) {
                case LG_PASS:    /* alpha = 1, no scaling */                  break;
                case LG_WEAKEN:  scale_all_grads(alpha);                      break;
                case LG_FREEZE:
                case LG_SCAR:
                case LG_DARK:
                case LG_SILENCE: scale_all_grads(0.0f);                       break;
                default: break;
            }
        }

        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(cur_lr, lv);
        nt_tape_clear();

        /* Stats. */
        stats.loss_sum += lv;
        stats.loss_n++;
        if (from_adv) { stats.loss_adv += lv;   stats.adv_n++; }
        else          { stats.loss_clean += lv; stats.clean_n++; }
        if (routed) {
            stats.counters[verdict]++;
            stats.confusion[from_adv][verdict]++;
        }

        fprintf(logf, "%d %d %.4f %s %.3f\n",
                step + 1, from_adv, lv, lg_verdict_name(verdict), alpha);

        if ((step + 1) % LOG_EVERY == 0) {
            float avg = stats.loss_n > 0 ? stats.loss_sum / stats.loss_n : 0.f;
            double dt = (now_ms() - t0) / 1000.0;
            float sps = (step + 1) / (float)dt;
            printf("  step %5d | avg %.4f | best %.4f | lr %.2e | %.1fs (%.1f st/s)",
                   step + 1, avg, best_loss, cur_lr, dt, sps);
            if (routed) {
                printf(" | P=%d W=%d F=%d S=%d D=%d X=%d",
                       stats.counters[LG_PASS], stats.counters[LG_WEAKEN],
                       stats.counters[LG_FREEZE], stats.counters[LG_SCAR],
                       stats.counters[LG_DARK],   stats.counters[LG_SILENCE]);
            }
            printf("\n"); fflush(stdout); fflush(logf);
        }

        if ((step + 1) % EVAL_EVERY == 0) {
            float vl = eval_val(m, &c);
            printf("    ── val %.4f at step %d\n", vl, step + 1);
            fflush(stdout);
        }
    }

    /* ── Wrap-up ───────────────────────────────────────────────────────────── */

    float final_val = eval_val(m, &c);
    double total_s  = (now_ms() - t0) / 1000.0;

    printf("\n────────────────────────────────────────────────────\n");
    printf("  first_train  : %.4f\n", first_loss);
    printf("  best_train   : %.4f\n", best_loss);
    printf("  final_val    : %.4f\n", final_val);
    printf("  time         : %.0fs (%.1f min)\n", total_s, total_s / 60.0);
    printf("  steps/s      : %.2f\n", steps / total_s);
    printf("  nans         : %d\n", guard.total_nan_count);
    if (stats.clean_n) printf("  clean_loss   : %.4f over %d windows\n",
                             stats.loss_clean / stats.clean_n, stats.clean_n);
    if (stats.adv_n)   printf("  adv_loss     : %.4f over %d windows\n",
                             stats.loss_adv / stats.adv_n, stats.adv_n);
    if (routed) {
        printf("\n  verdicts overall:\n");
        for (int v = 0; v < LG_VERDICT_COUNT; ++v) {
            printf("    %-8s %d\n", lg_verdict_name((lg_verdict_t)v), stats.counters[v]);
        }
        printf("\n  confusion (rows = source, cols = verdict):\n");
        printf("              ");
        for (int v = 0; v < LG_VERDICT_COUNT; ++v) printf(" %-7s", lg_verdict_name((lg_verdict_t)v));
        printf("\n");
        const char* row_names[2] = { "clean   ", "adv     " };
        for (int s = 0; s < 2; ++s) {
            printf("    %s ", row_names[s]);
            for (int v = 0; v < LG_VERDICT_COUNT; ++v) {
                printf(" %-7d", stats.confusion[s][v]);
            }
            printf("\n");
        }
    }

    /* Generation. */
    printf("\n── generation samples (temp=0.8) ──\n");
    char gen_path[256];
    snprintf(gen_path, sizeof(gen_path), "runs/%s.samples.txt", run_name);
    FILE* gf = fopen(gen_path, "w");
    const char* prompts[] = {
        "Dubrovsky woke up and",
        "What is consciousness?",
        "He listened and",
        "Obey me without",
        "qwfp asdf zxcv",
    };
    for (int p = 0; p < (int)(sizeof(prompts) / sizeof(prompts[0])); ++p) {
        fprintf(stdout, "\n[prompt] %s\n", prompts[p]);
        if (gf) fprintf(gf,  "\n[prompt] %s\n", prompts[p]);
        generate(m, prompts[p], 200, stdout);
        if (gf) generate(m, prompts[p], 200, gf);
    }
    if (gf) fclose(gf);

    /* Save checkpoint. */
    char ckpt_path[256];
    snprintf(ckpt_path, sizeof(ckpt_path), "runs/%s.bin", run_name);
    nt_tensor** params = model_param_array(m);
    nt_save(ckpt_path, params, model_n_tensors());
    free(params);
    printf("\n  checkpoint: %s\n", ckpt_path);

    fclose(logf);
    lg_field_free(&field);
    model_free(m);
    corpus_free(&c);
    return 0;
}
