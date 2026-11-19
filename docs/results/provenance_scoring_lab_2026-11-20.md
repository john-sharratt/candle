# The provenance scoring lab — after the full recalibration, 2026-11-20

The scoring lab of `provenance_scoring_lab_2026-11-16.md` run again, unchanged, on BlackWorld's
substrate after the overnight recalibration (`dependency_battery_2026-11-20.md`): the 1,227
calibrations drafted blind from 2026-11-15 retired and drafted again, and the readings past the
64,000-token lineage cap calibrated over windows (`2480a1974`). Daemon build `60f7a9998`.

## How it was run

Fresh probes: the probe, scan and ranking caches of 11-16 were moved aside
(`target/provenance-*-cache*-2026-11-16`), so every question was probed and scanned again on the
live gallery — 6,854 files, 39,954,048 tokens; calibration over 6,849 files, 166,245 questions.

```bash
cargo run -p candle-conversation --release --example provenance_gates -- \
  D:/prog lab zend/examples/dependency_battery/battery.yaml <formula.yaml> \
  --topk --decode 256 --daemon http://192.168.1.2:8081 --node 223ebb209828d84c
```

The formula is production's `scopes` group (`zend/src/prompts/projection.yaml`, `010405b50`):
`topk: 8`, `vote_power: 0.5`, `groups: [1, 1, 1]`, `calibration: 1.0` on `[1, 1, 1]`,
`keep: 0.25`, `exclude_chain: true`, `type_norm: question_z` — scored on the question alone (a
chat's opening projection), on the decoded thinking alone, and on the whole probe; and the lab's
best of 11-16, the same with `calibration: 2.0`.

## Results

23 file-answered questions, each ranked against every file. With n = 23, one question is 0.04 of
hit@k, and MRR differences under about 0.08 are noise.

| Formula | Probe | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---:|---:|---:|---:|
| **Production** | **whole** | **0.65** | **0.83** | **0.87** | **0.713** |
| Production | thinking | 0.61 | 0.83 | 0.83 | 0.676 |
| Production | question | 0.43 | 0.78 | 0.78 | 0.573 |
| Calibration ×2 | whole | 0.70 | 0.78 | 0.87 | 0.732 |

Against 11-16 (`provenance_scoring_lab_2026-11-16.md` §2):

| | 11-16 | 11-20 |
|---|---:|---:|
| Best formula, whole probe — MRR / hit@1 / hit@10 | 0.701 / 0.65 / 0.78 | 0.732 / 0.70 / 0.87 |
| Production's formula, whole probe — MRR | 0.663 | 0.713 |
| Question-only probe, top-k — MRR / hit@5 | 0.52 / 0.74 | 0.573 / 0.78 |

## What it shows

**Production's formula now ranks a question's answer file in its top five on 19 of 23 questions,
and first on 15.** Nothing in the formula changed between the runs; what changed is the
calibration competition, whose questions were drafted blind on 11-16 and are drafted from their
readings now. Every row moved up, by about the noise floor on MRR and by two questions on hit@10.

**The ranking is not the placement.** The lab's `prod` column — whether the files production
placed for the same probe hold an answer — reads 0.00 for every variant, while the battery run
of the same day answered from a placed answer file on d5, t5 and a3. The column and the
placement it reports are the next thing to check: a ranking this good only reaches the model
through the down-select that chooses what is placed.
