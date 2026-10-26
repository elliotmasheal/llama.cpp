#pragma once

#include <stddef.h>
#include <stdint.h>

struct qrt_layer;

// Speculative decode (design 5.4, M3-3). One cycle verifies k tokens in a single pass,
// keeps the longest draft prefix the verified rows agree with, takes the verified token at
// the first disagreement as the bonus, and rolls the sequence back to the accepted length.
// The accepted tokens are exactly what a serial greedy decode would have produced: a draft
// can only change how fast the cycle runs, never what it emits.
//
// The rollback itself lives in the layers (the DeltaNet state ring and the conv taps), and
// the KV rows are addressed by position, so a full-attention layer needs nothing beyond the
// caller moving its length back.
//
// The MTP draft *pass* (blk.40 plus its prefix) is driven by the decode session, because it
// is device and GGUF row work; what lives here is the draft policy: which candidate a step
// may pick, what a cycle accepts, and how the layers are rolled back.

// How many of the n drafts agree with the verified rows, stopping at the first
// disagreement. A pass carries n + 1 rows: the token the cycle starts from plus one per
// draft, and verified[i] is row i's argmax, which predicts the token fed at row i + 1. So
// draft[i] lines up with verified[i] and verified has n + 1 entries.
uint32_t qrt_spec_accept(const uint32_t * draft, const uint32_t * verified, uint32_t n);

// The tokens one cycle emits: the accepted drafts plus verified[accepted] (the bonus).
// n_out is accepted + 1, in 1..n+1, which is also the number of pass rows that stay.
// Returns -1 on bad arguments.
int qrt_spec_emit(const uint32_t * draft, const uint32_t * verified, uint32_t n, uint32_t * out,
                  uint32_t * n_out);

// Put every layer back at the state after the first `kept` rows of the pass that just ran.
// kept counts the rows that stay, so kept >= 1 and kept <= the rows the pass carried.
// A full-attention layer reports success without touching anything.
int qrt_spec_rollback(struct qrt_layer ** layers, uint32_t n_layers, uint32_t kept,
                      char * err, size_t err_size);

// The restricted draft vocabulary (design 4.1): the token ids a draft step may emit, read
// from the ascending list M1-3a delivered. n_allowed = 0 means the full vocabulary, which is
// the correctness baseline but costs a whole output-head read per draft step.
struct qrt_spec_vocab;
int qrt_spec_vocab_open(const char * path, uint32_t n_allowed, struct qrt_spec_vocab ** out,
                        char * err, size_t err_size);
void qrt_spec_vocab_free(struct qrt_spec_vocab * v);
uint32_t qrt_spec_vocab_count(const struct qrt_spec_vocab * v);
const uint32_t * qrt_spec_vocab_ids(const struct qrt_spec_vocab * v);

// Greedy pick of one draft step over n candidate logits: logits[i] belongs to ids[i], and
// ids NULL means the natural row order (the full vocabulary, the correctness baseline).
// Restricted candidates can only change which token is drafted, never what a cycle emits,
// because the verification pass always reads the whole vocabulary.
uint32_t qrt_spec_draft(const float * logits, const uint32_t * ids, uint32_t n,
                        float * best_logit, float * margin);
