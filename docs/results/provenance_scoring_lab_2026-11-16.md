# The provenance scoring lab — code reading and tools, 2026-11-16

What was tried to turn a probe's provenance votes into a ranking of the files that answer a
question, what each change measured, and the same changes checked on tool selection. The
code-reading measurements are the scoring lab (`candle-conversation/examples/provenance_gates`,
`lab`, `lab_formula`, `lab_probes`, `lab_runs`); the tool measurements are rounds R6 and R7 of
`candle-conversation/examples/selection_experiments.rs`.

## 1. Method

**Code reading.** The 23 file-answered questions of the §9.12 dependency battery
(`docs/provenance_expectations/dependency_battery.yaml`), each ranked against the 6,678 files of
BlackWorld's `code_reading` corpus. Every question is probed once through the daemon
(`POST /v1/substrate/zend/project`) and its tokens scanned once on the daemon's resident gallery
(`/scan`): the content competition (every file one case) and the calibration competition (every
calibrated file one case, scored on its calibration questions alone). For a formula that lets a
token vote for more than one file, the same probe is ranked on the CPU
(`window_score::group_ranked`), cached. A formula is a YAML file the lab re-reads on save, so
every variant below scores the same probes and the same votes; no daemon restart and no
re-ingest. Metrics are the rank of the best answer file: MRR, hit@1, hit@5, hit@10. With 23
questions, one question is 0.04 of hit@k, and MRR differences under about 0.08 are noise.

The answer files that are links of the priming chain (the documents every conversation inherits)
are ranked like any other unless a formula excludes them.

**Tools.** The selection-replay fixture's 95-tool catalog (190 captured tool-call turns, two per
tool, `tests/selection_replay_data/export/`), asked leave-one-out as production asks a turn: each
exemplar warms every other tool's background with its own tool left out, then each turn is
probed with its question window (first 64 tokens) and its trailing window (last 256), each
normalized at its own length against the learned backgrounds and max-fused. Reported: Top-1,
Top-5, MRR of the asked tool, and how many *other* tools clear the tools collection's 800 gate
per probe. CPU only, from checked-in signatures.

## 2. Code reading: the formula, round by round

Probe: the question as a chat opens (rounds 1–11), then the question decoded through its
thinking (round 13).

| Round | Change | MRR | hit@1 | hit@10 | Verdict |
|---|---|---:|---:|---:|---|
| 1 | Production's formula: one vote per token, `z × margin` to its best file, groups `[0,1,1]`, calibration ×0.25 on `[0,0,1]` | 0.258 | | 0.48 | baseline |
| 1 | Chain and journals out, documents ×0.5, vote power 0.5, hub divide | 0.354 | | 0.61 | kept, except the hub |
| 1 | Content gate (`content_gated`) | ~0 | | | rejected: code identity sits in groups 1–2 |
| 1 | Length correction | lower | | | rejected: answers are often the big central files |
| 3–4 | **Top-k: each token votes for its best 8 files, file `r` by `z_r × (a_r − a_9)`** | **0.52** | | 0.74 | kept; hit@5 0.39 → 0.74 |
| 3–4 | Top-k without down-weighting documents | | | | documents take 48% of the top 10 |
| 3–4 | Hub correction against the other questions, under top-k | lower | | | rejected |
| 5–6 | **`question_z`: each file as a z-score among its own extension's scoring files on the question** | **0.532** | 0.39 | | kept: replaces the hand-set docs ×0.25 (0.522) and the journals exclusion (0.531 with journals in) |
| 5–6 | Kind's mean taken on the background instead | 0.326 | | | rejected: the background must be the same kind of question |
| 5–6 | Chain exclusion off | 0.479 | | | chain exclusion kept |
| 8–9 | Vote laws: `z` alone, reciprocal rank, softmax, count | below 0.53 | | | rejected: the signal is spiky; summed `z × margin` wins |
| 8–9 | Token weights (top z, spread); aggregates (log sum, token count, count × best) | below 0.53 | | | rejected |
| 8–9 | Calibration votes top-k too | 0.40 | | | rejected: a calibration token matches one file's question cleanly |
| 8–9 | Fold weights | `[1,1,1]` best | | | kept, content and calibration |
| 10 | Content/calibration blend: z-sum, z-max, reciprocal-rank fusion | 0.507 / 0.422 / 0.490 vs 0.539 | | | rejected: the calibration's scale is information |
| 11 | Token runs: a vote boosted, or required, when neighbouring probe tokens match the same file | best 0.537 vs 0.540; runs only 0.32 | | | rejected: the signal is in single-token matches |
| 13 | Probe decoded through its thinking (`--decode 256`; the model stopped itself at 39–255 tokens): the question's tokens alone | 0.408 | | | the templated question scores below the bare one (0.540) |
| 13 | … the thinking's tokens alone | 0.667 | 0.61 | | |
| 13 | … the whole probe | 0.663 | | | |
| 13 | **… the whole probe, calibration ×2.0** | **0.701** | **0.65** | **0.78** | best measured |
| 13 | Production's formula on the decoded probe: question / whole | 0.121 / 0.235 | | | |
| 13 | Best formula at top-1 on the whole probe | 0.508 | | | top-k still needed |

**The formula.** Every round 13 variant uses this:

```yaml
topk: 8                         # each token votes for its best 8 files, z_r × (a_r − a_9)
groups: [1, 1, 1]               # content competition, every fold group
calibration: 1.0                # 2.0 on a probe that has thought
calibration_groups: [1, 1, 1]   # the calibration keeps one vote per token
keep: 0.25                      # needle gate: top quarter of tokens by summed vote
vote_power: 0.5                 # each kept vote square-rooted before summing
exclude_chain: true             # the priming chain's files are not ranked
type_norm: question_z           # a file is a z-score among its own extension on the question
```

**What the gain is made of.** On the same decoded probe, production's formula scores 0.235 and
this one 0.663. That is the formula's contribution. The decoded probe adds about 0.26 over the
opening probe under either formula. Production already reprojects on the thinking block, so that
part is the lab catching up with production, not a gain to port.

**Why top-k helps code reading.** A question's probe gives any vote to only 20–67 of 6,678 files.
Seven answer files scored exactly 0 under one vote per token, and no normalization lifts a zero.
When an answer does get a vote it ranks 1–11 on 16 of 23 questions. The binding limit is votes
per token, and top-k multiplies them.

**Limits of these numbers.**
- n = 23.
- The lab scores each probe's whole window. Production scores `cap_probe_window` (question head
  plus trailing window) at its reprojection triggers.
- The decoded text was not checked for leaving `</think>`.
- The battery's own results files are in the corpus.

## 3. Tools: the same vote law

R6 is production's chain. R7 runs the same chain with each raw score from the lab's vote law.
The top-1, power-1 row of R7 reproduces R6 exactly, which checks that R7 scores like production.

| Vote law (190 probes, 95 tools) | Top-1 | Top-5 | MRR | Raw question Top-1 | Other tools ≥ 800, a probe |
|---|---:|---:|---:|---:|---:|
| **Top-1, power 1 (production)** | **80.0%** | 97.9% | 0.879 | 77.4% | 0.71 |
| Top-1, power 0.5 | 80.5% | 98.4% | 0.881 | 75.8% | 0.73 |
| Top-2, power 0.5 | 75.8% | 97.4% | 0.847 | 72.6% | 1.37 |
| Top-4, power 0.5 | 67.9% | 95.8% | 0.795 | 70.5% | 2.29 |
| Top-8, power 1 | 67.4% | 90.5% | 0.778 | 70.0% | 2.85 |
| Top-8, power 0.5 (code reading's law) | 63.7% | 89.5% | 0.745 | 67.4% | 2.96 |

**Top-k hurts tool selection, and more votes per token hurt more.** A tool probe has one clean
match, its own tool's past calls, so a runner-up's vote is a vote for a wrong tool. Those votes
lift wrong tools over the selection gate: 0.71 a probe becomes 2.96. Vote power alone is neutral.

Code reading and the calibration competition show the same split.
- **Code-reading content** is vote-starved: the right file is one of many partial matches, and
  spreading the vote finds it.
- **A calibration question** (0.53 → 0.40 under top-k) and **a tool call** are each a clean
  match to one case, and spreading the vote dilutes it.

The vote law is therefore a property of the competition, not of the scorer, and is set per
group.

## 4. Decision

The formula goes to production on the code-reading groups only. Tools keep one vote per token at
power 1.

| Group | Top-k | Vote power | Fold groups | Calibration | Kind z | Chain out |
|---|---:|---:|---|---|---|---|
| `scopes` (code reading) | 8 | 0.5 | `[1,1,1]` | ×1.0 on `[1,1,1]`, one vote per token | yes | yes |
| tools collection | 1 | 1.0 | unchanged | unchanged | no | no |

## 5. The port and what it costs

**What changed.** The scan carries a vote law per segment (`provenance::VoteLaw`: `top_k`,
`power`).
- **Kernels.** Each BDP backend (scalar, b1 BMMA, INT8 IMMA) emits `max_k` rows per (token,
  group, segment). A single-vote segment runs the leader/runner-up loop unchanged. A top-k
  segment keeps its best k+1 in registers, one compare per case that does not place, and
  every vote goes through the shared `bdp_vote`.
- **Host.** The tally reads every vote of a row and raises each kept, weighed vote by the
  segment's power.
- **Configuration.** A group sets the law with `vote_top_k` / `vote_power`, reads its files
  against their kind with `normalize: kind_z`, and leaves its target's lineage out with
  `exclude_lineage`. The calibration competition keeps one vote a token at the group's power.
  Every default is the previous behaviour.

**Cost.** Measured on the RTX PRO 5000 (IMMA), best of five fresh probes
(`bench_wide_turn_group_scan`):

| Shape | Probe tokens | One vote | Top-8 |
|---|---:|---:|---:|
| 9 segments × 116 cases (a turn group), 8.0 M tokens | 66 / 256 / 1,024 | 14.3 / 29.5 / 102.2 ms | — |
| 1 segment × 6,678 cases (code reading), 3.4 M tokens | 66 / 256 / 1,024 | 7.2 / 18.3 / 60.4 ms | 8.2 / 20.1 / 64.8 ms |

Top-8 adds 1.0–4.4 ms (7–14%) to the code-reading scan. Single-vote launches, which is every
other group, run the same work and read back the same bytes as before. The added time is the
finalize: one thread per (token, group, segment) walks every case in series under either law.

## 6. Reproduction

```bash
# Code reading (BlackWorld's zend as the daemon)
cargo run -p candle-conversation --release --example provenance_gates -- \
  <workspace> lab docs/provenance_expectations/dependency_battery.yaml <formula.yaml> \
  --decode 256 --daemon http://192.168.1.2:8081 --node 223ebb209828d84c

# Tools
cargo run -p candle-conversation --release --example selection_experiments -- 6
cargo run -p candle-conversation --release --example selection_experiments -- 7
```

Commits: the lab 162a0f059 … bfa6e723b; the tools comparison 8570fa06e.
