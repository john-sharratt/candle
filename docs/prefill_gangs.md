# Prefill gangs

How a prefill group divides its budget between the prompts waiting on it, and
why background ingest is divided differently from dialogue.

Code: `candle-conversation/src/scheduler/gang.rs` (the arithmetic),
`scheduler/prefill.rs` (`form_wave_group`, `fit_cohort_to_ground`,
`weights_at_width_beside`, `prefill_gang_for`).

## 1. What the queue looked like

BlackWorld's `code_read` ingest, 938 s of wave windows on 25b70b83
(2026-10-27), the budget filled in order (shortest remaining prompt first):

| Realm | Time | Prefill | Decode | Expert hit |
|---|---|---|---|---|
| one or two prompts prefilling, one or two decoding | 45% | 654 t/s | 15.2 rows/s | 94.8% |
| page chunks alone, nothing decoding | 18% | 1,484 t/s | 0 | 73.4% |
| a burst of 10–55 short turns | 22% | 473 t/s | 45.6 rows/s | 89.1% |
| mixed | 15% | 432 t/s | 31.7 rows/s | 95.6% |

The queue held ~13 file conversations, each a ~14k-token page to prefill and a
summary to decode. Filled in order, one page took the whole budget, finished,
and decoded its summary alone while the next page prefilled — so for 45% of
the time the decode ran one row wide. A decode step costs nearly the same at
one row as at twenty (the step is the expert stream, not the rows), so every
summary token in that realm paid a whole step.

Shared equally instead (the rule before), the pages advanced together and
none reached its first token for minutes. Neither rule lines the decodes up.

## 2. Two regimes

The regime is read from the queue at every group formation:

- **Interactive** — an `Urgent` or `High` prompt leads the group (or a `High`
  section waits). The group carries that band only and its budget is filled in
  order, shortest remaining first: the nearest prompt reaches its first token
  soonest. Nothing below changes this.
- **Background** — only `Normal` and `Low` prompts. Long prompts are prefilled
  in **gangs**.

## 3. Gangs

A *long* prompt is one that cannot finish in one pass (its remainder exceeds
`max_prefill_pass_tokens`). A *short* one finishes in the forward it rides.

**Formation.** When no gang is live and the group is background, a gang is
formed from the long prompts of the highest waiting band, **in arrival order**,
`G` wide. It stays fixed until every member has finished its prefill (or
failed); no prompt joins a gang already formed. A gang narrower than two is no
gang, and the group is filled in order as in the interactive regime.

**Width.** `G` is the widest gang whose decode step costs no more than twice a
lone decode's, by the learned decode model (`WaveRate::decode_step_secs` at the
weight zone's residency, each member routing its own experts):

`step(G) ≤ 2 · step(1)`

so each member decodes at worst half as fast as it would alone, while the rows
per step grow `G`-fold. With the decode model unlearned there is no gang.

**Shares.** Members of a group are laid out *shorts, gang, the rest* (each part
by band, then shortest remaining). Then:

1. the shorts are filled in order from the budget less the gang's **reserve**
   — the lesser of the gang's remainder and half the budget;
2. the gang shares what the shorts left **in proportion to each member's
   remainder**, so its members reach their first tokens in the same forward
   and decode side by side; a member whose proportional share would fall under
   one chunk waits for the next forward instead;
3. the rest are filled in order from what is left.

The same division narrows a cohort when its fit to the ground or its trade
against the running decodes cuts the width, and prices the residency a width
leaves, so the gang stays in step however the cohort is cut.

**Pipelining.** A gang dissolves when its prefill is done, not its decode: the
next gang forms at once and prefills beside the last one's decodes. Prefill
rides the decode forwards (co-batching), so the decode of gang `k` and the
prefill of gang `k+1` share forwards rather than alternating.

## 4. Fairness

- A short prompt waits for nothing: it rides the next forward unless the
  shorts ahead of it already take the half the gang leaves.
- The gang always has half the budget, so a stream of shorts cannot stall it.
- A long prompt outside the gang waits for at most the gang ahead of it, which
  was formed in arrival order — never for one formed after it arrived.
- `Urgent` and `High` work is never inside a background group at all.

The stall the gang adds is the first-token latency of its own members: a
member reaches its first token when the whole gang does, up to `G` times later
than it would alone. That is the trade: background latency for decode rows.
