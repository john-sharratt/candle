# The §9.12 dependency battery — production, 2026-11-23 (l)–(r): reprojection during reasoning

Seven runs on BlackWorld, by the method of (j): each question in a fresh private chat, one at a
time, with the prefix "Answer from your memory of the repository only — do not call any tool, do
not read or search any file. If you do not remember, say so.", scored by `score.rs`'s
case-insensitive substring rules.

The question is whether reprojecting *while the model reasons* matters. The held arm keeps every
other reprojection: the turn's opening projection, the boundaries through the answer, and one
reprojection on the committed `</think>`. Only the boundaries inside the think block are passed
over (`reproject_in_thinking: false`, `scheduler/think_gate.rs`).

| Run | Build | Probe | Think block | Before the run |
|---|---|---:|---|---|
| (l) | `98d8f646c` | 256 | reprojects | deploy |
| (m) | `e6ebbb188` (5449) | 128 | held | deploy, normalization drained |
| (n) | `e6ebbb188` | 128 | reprojects | keep restart, drained, 1 min |
| (o) | `e6ebbb188` | 128 | held | keep restart, drained, 1 min |
| (p) | `e6ebbb188` | 128 | reprojects | keep restart, drained, 1 min |
| (q) | `e6ebbb188` | 128 | reprojects | every chat archived or deleted, keep restart, drained, 1 min; each chat closed once scored |
| (r) | `e6ebbb188` | 128 | held | as (q) |

## Results

| Run | Direct | Transitive | Architectural | **All 24** | Full marks |
|---|---:|---:|---:|---:|---:|
| (l) 256, reprojects | 25.0% | 56.3% | 66.7% | **49.3%** | 7 |
| (n) 128, reprojects | 26.0% | 64.6% | 70.8% | **53.8%** | 7 |
| (p) 128, reprojects | 27.1% | 69.8% | 95.8% | **64.2%** | 11 |
| (q) 128, reprojects | 25.0% | 67.7% | 68.8% | **53.8%** | 7 |
| (m) 128, held | 20.8% | 51.0% | 47.9% | **39.9%** | 4 |
| (o) 128, held | 14.6% | 50.0% | 47.9% | **37.5%** | 6 |
| (r) 128, held | 18.8% | 39.6% | 58.3% | **38.9%** | 5 |

At the 128-token probe, three runs per arm on the same build:

| Arm | Mean | Range | Full marks |
|---|---:|---|---:|
| Reprojects in thinking — (n), (p), (q) | **57.3%** | 53.8–64.2% | 7, 11, 7 |
| Held in thinking — (m), (o), (r) | **38.8%** | 37.5–39.9% | 4, 6, 5 |

- **Every held run scores below every reprojecting run.** Under no effect, the chance of the
  three held runs taking the three lowest places of six is 1 in 20 (one-sided permutation,
  p = 0.05). Counting (l) and the earlier runs (h)–(k), which all reproject in thinking and span
  46.9–57.6%, no reprojecting run of the ten has scored as low as any held run.
- **Question by question,** summing each question's recall over the three runs of each arm, the
  reprojecting arm scores higher on 14 questions, the held arm on 4, and 6 tie (sign test,
  p = 0.031 two-sided).
- **The probe's reach is not the cause.** At 128 tokens the reprojecting arm averages 57.3%,
  against (l)'s 49.3% at 256.

Per question, recall summed over the three runs of each arm (reprojecting / held):

| Id | R | H | Id | R | H | Id | R | H |
|---|---:|---:|---|---:|---:|---|---:|---:|
| d1 | 0.5 | 1.0 | t1 | 1.25 | 1.25 | a1 | 3.0 | 3.0 |
| d2 | 1.0 | 0 | t2 | 1.5 | 1.25 | a2 | 3.0 | 2.0 |
| d3 | 1.5 | 2.5 | t3 | 2.25 | 2.5 | a3 | 2.0 | 0.33 |
| d4 | 0.83 | 0.5 | t4 | 1.5 | 2.25 | a4 | 1.0 | 0 |
| d5 | 0.75 | 0 | t5 | 2.0 | 2.0 | a5 | 3.0 | 3.0 |
| d6 | 0 | 0 | t6 | 2.0 | 0 | a6 | 2.5 | 1.0 |
| d7 | 1.67 | 0.33 | t7 | 2.67 | 0.33 | a7 | 1.33 | 1.0 |
| d8 | 0 | 0 | t8 | 3.0 | 1.67 | a8 | 3.0 | 2.0 |

Runs (p), (q) and (r) by question:

- (p): d1 0/2, d2 2/2, d3 1/2, d4 2/6, d5 0/4, d6 0/2, d7 1/3, d8 0/2; t1 1/4, t2 1/4, t3 3/4,
  t4 4/4, t5 2/3, t6 4/4, t7 2/3, t8 3/3; a1 3/3, a2 2/2, a3 3/3, a4 2/2, a5 2/2, a6 2/2, a7 2/3,
  a8 2/2. a3 (`DeltaNet`, `gap-fill`, `reserve_glue_island`) and a4 (keys stored un-rotated, RoPE
  at read) reached full marks for the first time.
- (q): d1 0/2, d2 0/2, d3 1/2, d4 2/6, d5 2/4, d6 0/2, d7 2/3, d8 0/2; t1 1/4, t2 3/4, t3 3/4,
  t4 0/4, t5 2/3, t6 4/4, t7 3/3, t8 3/3; a1 3/3, a2 2/2, a3 2/3, a4 0/2, a5 2/2, a6 1/2, a7 1/3,
  a8 2/2.
- (r): d1 1/2, d2 0/2, d3 2/2, d4 0/6, d5 0/4, d6 0/2, d7 0/3, d8 0/2; t1 0/4, t2 2/4, t3 3/4,
  t4 1/4, t5 2/3, t6 0/4, t7 1/3, t8 2/3; a1 3/3, a2 2/2, a3 1/3, a4 0/2, a5 2/2, a6 0/2, a7 1/3,
  a8 2/2. d5's answer degenerated into a repetition loop ("The file is
  `candle/candle-kernels/src/guides/...`? No." over and over); it is the only loop in the seven
  runs.

(l), (m), (n) and (o) by question:

| Id | (l) | (m) | (n) | (o) |
|---|---:|---:|---:|---:|
| d1 | 0/2 | 1/2 | 1/2 | 0/2 |
| d2 | 0/2 | 0/2 | 0/2 | 0/2 |
| d3 | 1/2 | 1/2 | 1/2 | 2/2 |
| d4 | 2/6 | 2/6 | 1/6 | 1/6 |
| d5 | 0/4 | 0/4 | 1/4 | 0/4 |
| d6 | 0/2 | 0/2 | 0/2 | 0/2 |
| d7 | 2/3 | 1/3 | 2/3 | 0/3 |
| d8 | 1/2 | 0/2 | 0/2 | 0/2 |
| t1 | 1/4 | 3/4 | 3/4 | 2/4 |
| t2 | 2/4 | 1/4 | 2/4 | 2/4 |
| t3 | 3/4 | 3/4 | 3/4 | 4/4 |
| t4 | 4/4 | 4/4 | 2/4 | 4/4 |
| t5 | 1/3 | 2/3 | 2/3 | 2/3 |
| t6 | 4/4 | 0/4 | 0/4 | 0/4 |
| t7 | 0/3 | 0/3 | 3/3 | 0/3 |
| t8 | 2/3 | 2/3 | 3/3 | 1/3 |
| a1 | 3/3 | 3/3 | 3/3 | 3/3 |
| a2 | 2/2 | 0/2 | 2/2 | 2/2 |
| a3 | 0/3 | 0/3 | 1/3 | 0/3 |
| a4 | 0/2 | 0/2 | 0/2 | 0/2 |
| a5 | 2/2 | 2/2 | 2/2 | 2/2 |
| a6 | 2/2 | 1/2 | 2/2 | 1/2 |
| a7 | 1/3 | 1/3 | 1/3 | 1/3 |
| a8 | 2/2 | 2/2 | 2/2 | 0/2 |

In (o), a8's answer said both of its placed reads "came back empty".

## Reading

- **Reprojecting during reasoning is worth about 18 points of recall.** Holding the think block
  on its opening projection takes the battery from 57.3% to 38.8%, with every other reprojection
  in place, the reprojection at `</think>` included.
- **The loss is declines, not errors.** A held answer says what it does not remember: the files
  in its context are the ones the question alone selected, and the reasoning that would have
  pulled the answer files in ran without them. By the `</think>` reprojection, the reasoning that
  needed those files is over.
- **It falls hardest where the answer needs a file the question does not name.** The largest
  per-question gaps are t7 (the glue-island cache, 2.67 against 0.33), t6 (the reprojection
  configuration, 2.0 against 0), a3 and a4 (architectural contracts stated in files no question
  names), and d7 (`PINNED_LAYERS`'s loader). Questions answered from what the question names
  (a1, a5) are level.
- **The probe's reach is free at 128 tokens.** The reprojecting arm at 128 scores above (l) at
  256, so the shorter probe stays.
