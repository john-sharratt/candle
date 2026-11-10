# The theorem's two direct tests — RTX PRO 5000 72 GB, 2026-11-10

Two experiments named as owed by `docs/unbounded_agents.md` §11.2 ("Empirical
status"), run on BlackWorld (RTX PRO 5000 Blackwell 72 GB, TCC) with the card
to itself, on the working tree above `d5b267051`:

1. **Output fidelity against depth** — does a compressed cache move a model's
   next-token distribution more as the context it reads grows?
   `batch_test::fidelity`, one `#[ignore]`d test per model:
   `quantized_qwen3::tests::fidelity_vs_depth_8b`,
   `quantized_qwen35::tests::fidelity_vs_depth_9b`,
   `quantized_qwen38_moe::tests::fidelity_vs_depth_flash_next`.
2. **Recall at depth past the working set** — the conversation engine with its
   slot capped at 32,768 tokens, a history up to twelve times that, and four
   facts to recall from its middle. `candle-conversation/tests/provenance_depth.rs`.

## 1. Output fidelity against depth

At each depth one needle prompt (four vault codes of random characters spread
through the long-context filler, asked back at the end) is prefilled under each
mode, **sealed as the substrate seals a turn** (the prompt's K/V quantized in
place at the mode's level), and the reference answer is read back
teacher-forced. Each row compares that mode's per-step next-token distribution
with BF16's over the same prefix. The BF16 row compares a second BF16 read with
the first: the model's run-to-run spread.

Every reference read the four codes back correctly at every depth; "recovered"
in the raw rows is false only because each model opens the answer with "Vault"
where the scripted answer opens with the first vault's name.

### Qwen3-8B — classic full attention, every layer reads every token

| Depth | Mode | KL mean (nats) | KL max | Top-1 agree |
|---:|---|---:|---:|---:|
| 4,014 | BF16 (spread) | 0 | 0 | 100.0% |
| 4,014 | C0 | 4.97e-4 | 1.65e-2 | 100.0% |
| 4,014 | C5 | 1.15e-2 | 5.43e-1 | 97.9% |
| 4,014 | C10 | 3.15e-2 | 1.40 | 97.9% |
| 8,024 | C0 / C5 / C10 | 1.61e-2 / 1.90e-2 / 2.07e-2 | 0.73 / 0.48 / 0.85 | 97.9% each |
| 15,934 | C0 / C5 / C10 | 8.18e-3 / 4.04e-2 / 1.09e-2 | 0.39 / 1.81 / 0.28 | 100 / 97.9 / 100% |
| 31,859 | C0 / C5 / C10 | 3.51e-2 / 2.81e-2 / 3.86e-2 | 0.88 / 0.93 / 0.83 | 95.8 / 95.8 / 100% |

### Qwen3.5-9B — DeltaNet hybrid, its full-attention layers read the whole prefix

| Depth | Mode | KL mean (nats) | KL max | Top-1 agree |
|---:|---|---:|---:|---:|
| 4,121 | C0 / C5 / C10 | 4.48e-3 / 6.60e-3 / 6.53e-3 | 0.12 / 0.20 / 0.21 | 97.9% each |
| 16,094 | C0 / C5 / C10 | 1.52e-3 / 1.03e-3 / 1.17e-3 | 0.059 / 0.021 / 0.019 | 97.9% each |
| 64,323 | C0 / C5 / C10 | 6.88e-4 / 9.44e-4 / 1.67e-3 | 0.026 / 0.026 / 0.040 | 100% each |
| 124,479 | C0 / C5 / C10 | 2.56e-3 / 3.57e-3 / 3.51e-3 | 0.069 / 0.114 / 0.087 | 100% each |

BF16 run-to-run spread: 0 at every depth.

### Qwen3.8-Flash-Next — its full-attention layers select a bounded set (QSA indexer)

| Depth | Mode | KL mean (nats) | KL max | Top-1 agree |
|---:|---|---:|---:|---:|
| 4,121 | BF16 (spread) / C0 / C5 / C10 | 0 / 2.24e-4 / 9.84e-3 / 5.93e-2 | 0 / 0.0066 / 0.42 / 1.24 | 100 / 100 / 100 / 97.9% |
| 16,094 | BF16 / C0 / C5 / C10 | 0 / 1.31e-4 / 1.11e-2 / 3.02e-2 | 0 / 0.0030 / 0.47 / 0.40 | 100% each |
| 64,323 | BF16 / C0 / C5 / C10 | 4.26e-5 / 4.41e-4 / 2.06e-2 / 1.08e-1 | 0.0012 / 0.016 / 0.86 / 2.61 | 100 / 100 / 100 / 97.9% |
| 124,479 | BF16 / C0 / C5 / C10 | 2.45e-5 / 1.35e-4 / 7.57e-3 / 1.16e-1 | 0.00055 / 0.0019 / 0.29 / 3.46 | 100 / 100 / 100 / 95.8% |

### Reading

- **Whole-prefix attention does not show output divergence growing with depth
  within the windows measured.** The 9B's C10 divergence is 6.5e-3 nats at 4K
  and 3.5e-3 at 124K — lower, not higher, across a 30× range of exposure; the 8B
  moves within 1.1e-2–3.9e-2 from 4K to 32K with no trend. This is the convex-
  combination argument of §11.1 measured: the stored error a step is exposed to
  grows with depth, the error in its output need not.
- **The bounded-attention model's C10 divergence grows ~2× from 4K to 124K**
  (5.9e-2 → 1.16e-1 mean, worst step 1.24 → 3.46 nats), while its C0 and C5 rows
  stay flat (C5 9.8e-3 → 7.6e-3). Flash-Next is the one model whose attention
  *selects* positions, and the selection is made over the stored keys; at C10 a
  wrong selection at depth is an error mechanism a whole-prefix model does not
  have. That attribution is not instrumented here; what is measured is that the
  growth appears only at the top level and only on the selecting model.
- Runs are deterministic on the 8B and 9B (spread 0); Flash-Next's spread is
  ≤ 4.3e-5 nats, three orders below every compressed row.

## 2. Recall at depth past the working set

The engine (`Model::Qwen38_FlashNext`, argmax, `max_seq_len` 32,768) is given a
history of the repository's own source files, one file a turn, with the four
needle turns (`For the record: the code for the Kirkwall vault is 7QX4-MZ9K.`)
placed together starting at 45% of the history's depth, and is asked for the
four codes back. Each cell is a fresh conversation. Recall counts the codes
given in the order stated. "Needle turns selected" counts the needle turns any
of the question's projections held; the final projection's own count follows
in the notes.

| Arm | History (tok) | Turns | Recalled | Needle turns selected | Projections | Final projection |
|---|---:|---:|---:|---:|---:|---|
| provenance, C10 | 30,000 | 19 | **4/4** | 4/4 | 5 | 17 turns |
| provenance, C10 | 131,072 | 64 | 0/4 | 4/4 | 13 | 17 turns, all 4 needles |
| provenance, C10 | 393,216 | 195 | 0/4 | 4/4 | 11 | 15 turns, 3 needles |
| provenance, C0 | 30,000 | 19 | **4/4** | 4/4 | 7 | 17 turns |
| provenance, C0 | 131,072 | 64 | 1/4 | 4/4 | 17 | 18 turns, all 4 needles |
| provenance, C0 | 393,216 | 195 | 0/4 | 4/4 | 18 | 16 turns, 2 needles |
| one-shot, C10 | 30,000 | 19 | 3/4 | 3/4 | 1 | the oldest 17 turns |
| one-shot, C10 | 131,072 | 64 | 0/4 | 0/4 | 1 | the oldest 14 turns |
| one-shot, C10 | 393,216 | 195 | 0/4 | 0/4 | 1 | the oldest 14 turns |
| rolling window | 30,000 | 19 | 3/4 | 3/4 | 1 | the oldest 17 turns |
| rolling window | 131,072 | 64 | 0/4 | 0/4 | 1 | the oldest 14 turns |
| rolling window | 393,216 | 195 | 0/4 | 0/4 | 1 | the oldest 14 turns |

The failing answers deny the facts rather than garble them: *"No vault codes
were provided in this conversation. The messages contained Rust source files…"*
(provenance C10, 131K, with all four needle turns in its final projection);
*"I was only given one vault code in this conversation: the Kirkwall vault code,
7QX4-MZ9K"* (C0, 131K); *"…the Broome vault, R2VD-8HJ3"* (C0, 393K — Tampere's
code under Broome's name).

### Reading

- **Selection works past the working set.** Mid-decode reprojection put every
  needle turn into a projection at 131K and 393K — out of 64 and 195 turns, into
  a working set of 15–18 — at C10 and at C0 alike. Without it (one-shot, rolling
  window) no needle turn is selected past 30K, and nothing is recalled.
  Within the working set, recall follows exactly what the projection holds: the
  one-shot and rolling arms held three needle turns at 30K and recalled three.
- **Use of what was selected fails past the working set.** With all four needle
  turns in its final projection the model answered that no codes were given
  (C10, 131K) or gave one (C0, 131K). The failure is the same at C0 as at C10,
  so it is not the compression; and it is not the retrieval, which found the
  turns. Turns that arrive by selection from deep in a conversation, between
  unrelated selected turns, are read and then judged absent — the same gap
  seen on the production daemon, where a page the model attended and could
  recall verbatim was still reported missing — and the reason the paper's
  recall claims stop at retrieval.
- The theorem's premise — a bounded working set selected from a history many
  times its size — holds in every provenance cell. Its practical promise, that
  quality does not degrade with depth, does not: recall falls from 4/4 inside
  the working set to 0–1/4 outside it.

### Diagnosis: why selected turns are not used

`why_recall_fails_past_the_working_set` and `recall_with_the_selection_forwarded`
in the same test file run the failing cell (131K of history, C10) with the
capped diagnostic capture (`decode_diag`) armed for the question's decode,
which records for every attention layer's query which pieces of the slot the
QSA index chose:

| Arm | Recalled | Needle pieces in the slot | Queries reading a needle | Best needle rank |
|---|---:|---|---:|---:|
| sparse (production) | 0/4 | 3–4, at 11K–26K of ~31K | 77–81% | 0 |
| dense (every cell attended) | 0/4 | 1–3 | — (no selection) | — |
| fresh (every piece re-placed) | 0/4 | 3–4 | 78–83% | 0 |
| needles last (just before the question) | 0/4 | 3, ending at the slot's end | 83–85% | 0 |
| needles first (start of history) | **4/4** | 4, at 1.8K–6.7K | 100% | 0 |
| needles first, dense | **4/4** | 4 | — | — |
| **the failing selection, forwarded** | **3 of its 3** | the same 3, at the same positions | 100% | 0 |

The failing reply's reasoning lists the history's file turns one by one and
concludes *"At no point did the user tell me any vault codes"*; one run quoted
a needle with another needle's vault name. The forwarded control takes the 17
turns the failing projection selected, in its order, into a fresh conversation
whose slot holds them whole: same tokens, same positions, same C10 storage, but
every turn's K/V computed in this slot's context — and it gives every code the
selection held. Retrieval, QSA selection, splicing, compression, position and
distance from the cursor are each ruled out by an arm above; what differs is the
context the sealed turns were computed in. A turn sealed 59K–106K tokens into a
run of file readings carries that run in its K/V, and placed elsewhere it is
attended but not taken as something the user said — the needles-first turns,
sealed behind almost no context, are read. Forwarding also rebuilds the
recurrent state, so this control alone does not separate K/V from recurrence;
the needles-first and needles-last arms, whose recurrence saw the needles first
and last respectively, argue against the recurrence.

### Isolating the mechanism: the context a sealed turn's K/V was computed in

Two further tests (`recall_against_sealing_depth`, `sealing_or_reading_under_projection`).

**Sealing depth.** 131K of history, C10, production decode, the needles moved:

| Needles at | Stream behind the first needle | Recalled | Needles projected |
|---:|---:|---:|---:|
| 0.00 | 5,039 | 4/4 | 4/4 |
| 0.10 | 18,291 | 4/4 | 4/4 |
| 0.20 | 31,337 | 1/4 | 4/4 |
| 0.30 | 44,227 | 0/4 | 3/4 |
| 0.45 | 65,112 | 0/4 | 2/4 |
| 0.70 | 97,008 | 0/4 | 4/4 |

The slot position of the needles does not decide it (at 0.20 they sat at 2.8K–8.7K
of the slot, where the 0.00 and 0.10 needles also sat, and still failed).

**Sealing against reading.** The needles 65K deep; the plain-prompt dialogue window
(fixed at 32,768 — `Builder::set_layer_window` moves it) set to 262,144, so the
conversation is never projected, or to the production 32,768, once for the inserts
and again for the question:

| Sealed under | Read under | Recalled | Turns held |
|---|---|---:|---:|
| 262K (the true history) | 262K (the true history) | **4/4** | 64 |
| 262K | 32K (a selection) | 0/4 | 15 |
| 32K (a projection) | 262K (the true history) | 1/4 | 64 |
| 32K | 32K (production) | 0/4 | 18 |

with the forwarded control above (sealed and read in the same 32K selection)
recalling every code its selection held. Every match of sealing context and reading
context recalls; every mismatch fails, in either direction. A sealed turn's K/V is
readable in the context it was computed in, and — past ~20–30K tokens of preceding
context at its sealing — not in another. RoPE does not enter: keys are stored
un-rotated and the slot positions are within the first rung. What carries the
context is the activation itself: a token's K and V at an attention layer are
projections of its residual stream, which by the deep layers is dominated by what
earlier attention and the DeltaNet recurrence wrote into it from the context before
it.

### The first design, and why it was replaced

A first run (same day) placed the needles in the first ~32K tokens. Every arm
recalled 4/4 at every depth, the rolling window included — because the one-shot
projection and the append path both fill from the **oldest** turns
(the projections were `[0..16]` in every such cell), which is where the needles
were. The rerun moves them to the middle, where only selection reaches them;
the test as committed is the rerun's. The tables above are copied from the
runs' console output, which the test prints in this form.
