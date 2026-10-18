# Cooperative admission

How a zend cluster places turns across members so that the sum is better than
each member admitting alone: decodes concentrated where weights are resident,
prefill kept off the decode set's card, and every conversation still getting
its fair turn.

The member's own admission (`candle-conversation` `scheduler::admit`) decides
what one card runs. This document is the layer above it: which card a turn goes
to. Code: `coop/` — the cost model, its fit from each member's wave log, and the
cluster simulator that compares the policies below before any of them is built
into the gateway.

## 1. What the members measure

`coop-lab` reads a member's scheduler `wave` lines (one per 2 s window, per
channel: forwards, rows, tokens, attended KV, ms per forward, the window's
expert hit rate) and reports two things: a non-negative least-squares fit of

```text
ms per forward = base + token·T + attention·MKV + resident·GB_hit + link·GB_missed
```

where the expert bytes a forward reads are the expected distinct experts its
`T` tokens select (`49 layers × 512 × (1 − (1 − 10/512)^T) × 2.61 MB`), split by
the hit rate; and the measured step time bucketed by rows per forward, which is
what the simulator uses where the fit explains a channel poorly.

### BlackWorld (RTX PRO 5000 72 GB, PCIe 5.0), builds 3436–3442, 2,829 windows

Decode, by rows per forward:

| rows | decode alone: ms/step | t/s | beside prefill: ms/step | t/s |
|---|---|---|---|---|
| 1 | 53.0 | 19 | 76.6 | 15 |
| 2–3 | 129.0 | 22 | 110.7 | 24 |
| 4–7 | 119.8 | 48 | 168.8 | 30 |
| 8–15 | 94.0 | 139 | 91.4 | 117 |
| 16–31 (~23) | 75.7 | 304 | 81.5 | 270 |

Prefill forwards average ~200 tokens at 77–145 ms: 0.8–2.5K t/s, against
4.8K t/s at full width in the forward gate.

Fits: prefill (all windows) `13.7 ms + 0.218 ms/token + 12.7 ms/MKV + 2.6 ms/GB
missed`, R² 0.79 — the 0.218 ms/token margin is the gate's 4.6K t/s. Decode
alone `38.1 ms + 104 ms/GB missed`, R² 0.72; decode over all windows R² 0.48,
because a decode beside prefill waits on that prefill's expert traffic, which
its own counts do not show.

What this says for placement:

- **Decode throughput is nearly linear in batch size** on a member up to the
  widths seen (~23 rows ≈ 300 t/s, one row ≈ 19 t/s). Twenty-four decodes on
  one card deliver what two cards of twelve each deliver, and leave the
  second card free.
- **Decode beside prefill is 1.1–1.5× slower per step.** Separating the two
  across cards removes that.
- **A single-row decode costs 53 ms in production** against single-digit ms in
  the gate: per-step overhead outside the forward the gate measures. It is the
  member's to fix, and it makes small decode batches the worst use of a card.

WhiteWorld (RTX 3090 24 GB, PCIe 3.0) is to be measured the same way; its gate
figures are ×1 decode 138, ×16 338–544, prefill 2.3K t/s, and its expert zone
is a third of BlackWorld's.

### The link

The members share one 1 GbE path (~110 MB/s) through the gateway, RTT ~1 ms.
What crosses it when a turn's work moves:

- its new KV — on the order of 12 KB a token (12 full-attention layers),
  ~24 MB for a 2K-token prompt, ~0.2 s;
- its recurrent snapshot — 36 DeltaNet layers × 48 value heads × 128 × 128 at
  f32, **113 MB**, about 1 s.

Both replicate anyway: every sealed turn's records go to every member. So a
turn placed on another member costs nothing the substrate was not already
sending, provided that member already holds the turn before it (§2.2).

## 2. Mechanisms

### 2.1 Advertised load figures

Each member says in `CAPACITY`, as it runs, what more work would cost it, from
its own measurements (§1):

- **decode-heavy and prefill-heavy** — how loaded it is on each: the rows it is
  decoding and the prompt tokens it has queued;
- **the marginal value of one more of each** — what one more decoding turn,
  and one more prompt, would do to its own throughput at its current
  residency, from its own fitted curve.

The gateway compares these numbers and nothing else. It models nothing of a
member's insides; the member that knows its card is the one that says what
the card can do.

### 2.2 A lease per turn, placed among the replicated

A conversation's lease (its claim, §5.4 of `replicated_substrate.md`) is held
for **one turn**: its prefill and its decode, the time its state has to be on
a card. At the turn's boundary, once the turn is durable, the lease is
released with `through`, and the next turn is placed afresh.

- **Only members that hold the conversation through its latest turn are
  candidates.** A member that has not yet confirmed replication up to the turn
  just ended is skipped, never waited on. Placement therefore never stalls a
  turn on the link, and a move costs nothing the substrate was not already
  sending.
- Among the candidates, the turn goes by §2.1's figures.
- A turn runs where it is placed, start to finish: no part of it is sent to
  another member mid-turn.

### 2.3 Fairness

Every policy is judged on per-conversation slowdown (the turn's time against
the same turn alone on an idle cluster): its p95 and its Jain index across
conversations, and the longest any turn waited for admission. A policy that
raises throughput by starving the conversations it does not concentrate is
rejected.

### 2.4 Failure

A member dropping hands its claims back (§12.4). A turn running on it starts
again, placed among the members that hold its conversation through the turn
before. A conversation whose only up-to-date copy was on the dropped member
waits for nothing: its next turn is placed among the members that hold it
through the latest turn they have, which is the last turn that member
replicated before it went.

## 3. Experiments

The simulator (`coop/src/sim/`) replays a workload — conversations whose
turns are prefill-then-decode segments, sized from the zend-load rungs —
against members whose step time is their measured curve. Each turn's end
replicates its state to every other member over the link above, FIFO, and a
member becomes a candidate for that conversation once its copy has arrived
(§2.2). The policies:

1. **today** — sticky; a new conversation goes `Best` by
   `power / (1 + in_flight)`;
2. **concentrate decodes** — every turn on the strong card while it reports
   room (rows under its maximum, decode hit rate above 0.9), otherwise `Best`;
3. **earliest advertised finish** — each candidate advertises what this turn
   would cost it now (its prefill backlog plus this prompt at its prefill
   rate, then the reply at the step time one more row gives it), and the turn
   goes to the soonest. §2.1's figures used as the only guide.

Reported per policy: aggregate t/s, first-token p50/p95, longest stall, Jain
index, turns per member, lease moves, and the same with one member dropping
mid-run.

### 3.1 Results (`coop-lab sim`, 24 conversations, seed 1)

BlackWorld as measured; **WhiteWorld assumed** from its gate figures (decode
slope 3.2× BlackWorld's, prefill half its rate, a 9 GB expert zone) until its
wave log is fitted, so every WhiteWorld-dependent figure below moves when it is.

| scenario | policy | t/s | first token p95 | Jain | turns BW/WW | moves |
|---|---|---|---|---|---|---|
| chat | today | 141 | 1.8 s | 0.942 | 60/12 | 0 |
| chat | concentrate | 160 | 2.7 s | 0.997 | 72/0 | 0 |
| chat | earliest finish | **161** | 2.5 s | **0.997** | 71/1 | 1 |
| read | today | 85 | **6.8 s** | 0.995 | 40/8 | 0 |
| read | concentrate | 81 | 9.8 s | 0.997 | 48/0 | 0 |
| read | earliest finish | **86** | 8.7 s | 0.992 | 45/3 | 3 |
| chat + read | today | 94 | 3.0 s | 0.969 | 50/10 | 0 |
| chat + read | earliest finish | **106** | 5.0 s | **0.990** | 59/1 | 1 |

With WhiteWorld dropping at 30 s, today's first-token p95 rises to 22–28 s
(its turns restart); earliest finish stays at 2.7–9.0 s.

What it says so far:

- **Concentrating decodes beats today's power split** by 13% on chat and
  evens the slowdown (Jain 0.94 → 1.00): today's split sends a sixth of the
  conversations to the card that serves them several times slower.
- **Earliest advertised finish is the best single policy** on every mix but
  read's first-token tail, and it is the simplest to build: each member
  computes its own figure and the gateway compares numbers. It moves a lease
  at most three times a run, and only to a member already holding the state.
- Every result assumes the member's own scheduler forms full-width prefills;
  production forwards average ~200 tokens (§1), which no placement fixes.
