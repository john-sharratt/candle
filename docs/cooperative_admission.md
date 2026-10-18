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
- its recurrent snapshot — the GDN store, ~150 MB, **~1.4 s**.

Both replicate anyway: every sealed turn's records go to every member. What a
move costs is the wait for them to have arrived, so a move at a turn boundary
after the user's own pause costs nothing, and a move in the middle of a turn
costs a snapshot crossing.

## 2. Mechanisms

### 2.1 Declared role and capacity

Each member says in `CAPACITY`, as it runs, what it is best used for and how
much more it can take:

- **role** — `decode`, `prefill` or `mixed`, chosen by the member from its
  own measurements (§1): the card with the resident weights takes `decode`
  when another member can take prefill; a card alone is `mixed`;
- **capacity** — tokens per second it can still add on that role at its
  current residency, from its own fitted curve.

The gateway places by these alone. It models nothing of a member's insides;
the member that knows its card is the one that says what the card can do.

### 2.2 Lease transfer at a turn boundary

A conversation lives on the member holding its claim (§5.4 of
`replicated_substrate.md`). Moving it is the claim's existing release with
`through`: the next request goes to another member, which serves once it has
applied the old holder's writes. A transfer carries a **stay** mark — the
conversation is not moved again at its next admission — so a conversation
moves at most once per two turns and never oscillates.

### 2.3 Proxied prefill

The holder keeps the lease and asks another member to run a turn's prefill.

- The executor admits it only once it holds every unit the turn's selection
  names: it has applied the holder's writes through the `oseq` the request
  states. Until then it is not admitted, and the holder may run it itself.
- The holder admits the decode only once the executor's KV chunks and
  recurrent snapshot have replicated back.
- **No recursion.** A proxied request carries the mark; a member receiving one
  runs it or refuses it and never proxies it on. One hop, always.

It pays when the prefill saved outweighs the snapshot's crossing (~1.4 s): a
long prompt — a first turn, a large tool result, a reprojection — on a member
whose card is busy decoding. A follow-up turn's prefill is a few dozen tokens
and is never proxied.

### 2.4 Fairness

Every policy is judged on per-conversation slowdown (the turn's time against
the same turn alone on an idle cluster): its p95 and its Jain index across
conversations, and the longest any turn waited for admission. A policy that
raises throughput by starving the conversations it does not concentrate is
rejected.

### 2.5 Failure

A member dropping hands its claims back (§12.4); its conversations are placed
again as new. A proxied prefill whose executor drops is run by the holder,
which never gave up the lease. A transfer whose target drops before it served
leaves the claim free, and the next request is placed as new.

## 3. Experiments

The simulator replays a workload — conversations with turns of prompt and
output lengths drawn from the zend-load rungs and from production — against
members whose step time is their measured curve, with the link above, under:

1. **today** — sticky, new conversations `Best` by `power / (1 + in_flight)`;
2. **declared roles** (§2.1);
3. **declared roles + lease transfer** (§2.2);
4. **declared roles + proxied prefill** (§2.3).

Reported per policy: aggregate t/s, first-token p50/p95, longest stall, Jain
index, and the same with one member dropping mid-run.
