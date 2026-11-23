# The needle in the haystack — the 128-question battery, 2026-11-24

Figure 9.12-7 of `docs/unbounded_agents.md`: where production's provenance formula ranks each
battery question's answer file among every ingested file. The scoring lab of
`docs/results/provenance_scoring_lab_2026-11-20.md`, unchanged, over the whole battery instead of
its 23 file-answered questions, on BlackWorld after battery-128 (build 5482-65c8493ef).

## How it was run

Fresh probes: the 11-20 probe, scan and ranking caches were moved aside
(`target/provenance-*-cache*-2026-11-20`), so every question was probed and scanned again on the
live gallery — 7,943 files, 36,053,822 tokens, 1,177,878 windows; calibration over 7,912 files,
189,914 questions; 132 background probes. 716 s on the daemon, the formula scored in 56 ms.

```bash
cargo run -p candle-conversation --release --example provenance_gates -- \
  D:/prog lab zend/examples/dependency_battery/battery.yaml production.yaml \
  --topk --decode 256 --daemon http://192.168.1.2:8081 --node 223ebb209828d84c
```

`production.yaml` is production's `scopes` scan (`zend/src/prompts/projection.yaml`) as the lab
writes it:

```yaml
budget_files: 10
part: whole
topk: 8
vote_power: 0.5
groups: [1.0, 1.0, 1.0]
calibration: 1.0
calibration_groups: [1.0, 1.0, 1.0]
keep: 0.25
exclude_chain: true
type_norm: question_z
```

## Results

121 of the 128 questions ranked. Seven — d03, d15, d18, d19, d23, d37, t33 — name only answer
files whose modules have since been split, so no loaded file is any of them; 37 more lost some of
their answer files the same way and are ranked on the ones that remain (2.55 a question on
average). `fig_9_12_7_needle_ranks.csv` holds each question's best rank and its answer files
ranked.

| | Questions | hit@1 | hit@5 | hit@10 | hit@100 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Production, whole probe | 121 | 0.49 | 0.69 | 0.74 | 0.88 | 0.578 |
| Direct | 37 | 0.43 | 0.65 | | | |
| Transitive | 42 | 0.52 | 0.74 | | | |
| Architectural | 42 | 0.50 | 0.69 | | | |
| A random ranking | | 0.0003 | 0.0016 | 0.0032 | 0.032 | |

A random ranking of N = 7,943 files puts one of a question's m answer files in its top k with
probability ≈ k·m/N; at the mean m of 2.55, 0.032% for k = 1.

## What it shows

**The whole battery is harder than its 23 file-answered questions**: hit@1 0.49 against 0.65 on
11-20, MRR 0.578 against 0.713. The ranking still clears chance by three orders of magnitude at
its first point.

**The misses are hub files.** On most questions whose answer ranks below ten, the top five are
the same few files that score on every question — `tests_grid.rs`, `file_grep.yaml`, `tasks.rs`,
`AGENTS.md` — rather than a near neighbour of the answer.

The lab's `prod` column reads 0.00 again, as on 11-20; it is not the placement the battery
measured (§9.12's placement figures come from the answers' own projection records).
