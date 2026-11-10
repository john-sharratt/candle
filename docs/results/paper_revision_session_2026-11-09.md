# The paper's own revision, measured as a workload — 2026-11-09

Conversation `425bad8d-769b-4e3c-a85b-d2e3eef0fd5c` ("Thoughts on Unbounded
Context"), Qwen3.8-Flash-Next served by `zend` on the RTX PRO 5000 Blackwell
72 GB (BlackWorld), thinking effort *balanced*, tools *comprehensive*. The
session critiqued v2 of `docs/unbounded_agents.md` and then revised it and the
README — the edits committed as `c73d22fa6`. Cited by §9.13.1 of the paper.

**Source.** zend's per-round timing for the conversation, as `timeline_read`
(`view: calls`) reports it: for each tool round, the tokens prefilled and
decoded and the seconds each took. The conversation length at a round is the
sum of every earlier round's prefilled and decoded tokens.

**Which rounds.** Rounds 0–84 are the session's own. A restart during the
session resumed its open tool round on a second member while the first was
still running it (the fault fixed in `e6ccc46d9` and `6c07ea264`: a resumed
turn now takes its conversation's claim, and only once it holds the
conversation complete). The second member's three timed rounds (86–88, decoding
at 12–13 t/s) are not the session's and are left out; its file edits were discarded and the committed files rebuilt
from the session's recorded edits.

## Totals, rounds 0–84

| | Tokens | Seconds | Rate |
|---|---:|---:|---:|
| Prefill | 238,619 | 302.0 | 790 t/s |
| Decode | 82,963 | 833.2 | 99.6 t/s |
| Queued (waiting for the scheduler) | — | 635.5 | — |

- Decoded : prefilled tokens = **1 : 2.88**.
- Rounds 1–4 read the paper whole (~70K tokens prefilled). Over rounds 5–84,
  new input averaged **~2,100 tokens a round** against **~970 decoded**.
- The conversation reached ~321K tokens; every token was prefilled once — the
  earlier turns' K/V is held by the substrate and never prefilled again.

## Decode rate against conversation length

| Rounds | Conversation length at the rounds | Decoded tokens | Seconds | Decode t/s |
|---|---|---:|---:|---:|
| 4–13 | 48K → 130K | 16,236 | 158.5 | 102.4 |
| 14–19 | 130K → 162K | 7,365 | 66.4 | 110.9 |
| 20–35 | 162K → 237K | 23,299 | 212.3 | 109.7 |
| 36–46 | 237K → 256K | 9,852 | 99.5 | 99.0 |
| 47–57 | 256K → 272K | 6,214 | 84.4 | 73.6 |
| 58–75 | 272K → 307K | 14,241 | 144.2 | 98.8 |
| 77–84 | 309K → 321K | 4,762 | 44.8 | 106.3 |

Round 76 (181 tokens in 12.7 s) is left out of its bucket: a single stall with
nothing in the round to explain it. Rounds 47–57 dip and the longer rounds after
them recover, so the dip is not the conversation's length; the round log does
not record what else the card was running at the time.

## What the numbers do not separate

The session had no control run. Whether the ratio is the provenance-selected
working set — earlier reads retrieved into context when they become relevant,
so the model need not read again — or the anchor-addressed file tools, which
let it edit a part without reading the file around it, is not separated here.
The test that separates them is the same task run with provenance selection
replaced by a full window, and with the file tools reading whole files.
