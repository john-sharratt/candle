---
name: sweep
description: Run every model's batched-forwarding gate (test_parallel_batched_forwarding*) serially, smallest model first, with zend/npcd stopped and restarted afterwards; report per-model throughput/compression/pass tables and a summary. Full sweep on >64 GB VRAM, partial (DeepSeek-V4-Flash skipped) otherwise; Qwen3.8-Flash-Next runs on every machine. Failures are fixed forward.
disable-model-invocation: true
---

# /sweep — model forward-gate sweep

A sweep is one release run of each model's `test_parallel_batched_forwarding*` gate, one
`cargo` process per model, one after the other. **The sweep passes only if every gate
passes.** Anything that fails is fixed forward (step 9) — it does not matter whether this
session introduced the bug or it was already there.

The repository rules in `CLAUDE.md` apply throughout: read logs with Read/Grep (never
`cat`/`Select-String`), never mask an exit status, never commit without permission.

## 1. Size the sweep from VRAM

```powershell
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
```

- `memory.total` **> 65536 MiB (64 GiB) → full sweep**: every gate.
- Otherwise **→ partial sweep**: skip only `deepseek4` (DeepSeek-V4-Flash, 284B), and any
  newly found gate whose checkpoint is that class of size (judge from its `#[ignore]` text
  and model file). Name the skipped gates in the report.
- **Qwen3.8-Flash-Next (`qwen38-fn`) runs on every machine, full sweep or partial** — both
  its gate `quantized_qwen38_moe` and the `qwen38_flash_next` engine probe of step 4b
  (which runs the preset for this card's rung — `Q3_KO` under 64 GiB, `Q4_KO` at 64+).
  Model size is not bounded by VRAM (`CLAUDE.md`): the expert cache streams VRAM → pinned
  RAM → mmap, and on a 16 GB card it runs from the prepared Q3_KO hybrid artifact
  (`qwen4exp::prepare`, recipe-hashed, cached under
  `~/.cache/zend/models/unsloth--Qwen3.8-Flash-Next-GGUF/`). The gate sizes its ladder from
  free VRAM itself — a rung it cannot hold on this card (×16 needs ≥ 24 GiB) is the gate's
  to skip, not the sweep's. If the artifact is not built yet, the first run builds it,
  fetching ~192 GB of pinned `Q8_0` sources and deleting them after the merge (the
  `Q2_KO` artifact was ~11 min and 88 GiB; `Q3_KO`'s experts are 52 B per 128 weights
  against 36, so expect more) — check free disk before starting. A failure here —
  OOM, a fault, a missed story — is a bug to fix forward (step 9), never a reason to skip
  the model on a smaller card.

## 2. Discover the gates — never use a remembered list

```
Grep  pattern: fn test_parallel_batched_forwarding   glob: *.rs   (whole repo, -n, -B 6)
```

Every hit is a gate, including ones added since the last sweep. For each, read its
`#[ignore]` attribute / doc comment (it carries the run command) and derive the full test
path from the file: `candle-transformers/src/models/<module>.rs` →
`models::<module>::tests::<fn>`. As of 2026-09-13 there are 12, all in
`candle-transformers/src/models/`; if a hit lives elsewhere, build its `-p`/path from that
file instead.

## 3. Stop zend and npcd — remember how they were started

```powershell
Get-CimInstance Win32_Process -Filter "Name='zend.exe' OR Name='npcd.exe'" |
  Select-Object ProcessId, Name, ExecutablePath, CommandLine
```

Record each process's `ExecutablePath` and full `CommandLine` (every flag — e.g. zend's
`--host`, `--port`, `--model`) in the conversation before touching it. Then:

```powershell
Stop-Process -Id <pid> -Force -Confirm:$false
```

Re-run `nvidia-smi` and confirm `memory.used` is back to the idle floor (~0.5 GB). The
gates size themselves from free VRAM, and `real_cuda` tests invert with a co-resident
process — **do not start a gate while anything else holds the card.**

## 4. Run the gates serially, smallest first

Order by model size (parameters / checkpoint bytes), smallest first. Slot any new gate in
by its size. The 2026-09-13 order, with that run's wall-clock:

| # | gate (`models::…::tests::…`) | model | ~time |
|---|---|---|---|
| 1 | `quantized_qwen2::…::test_parallel_batched_forwarding` | Qwen2-0.5B | 15 s (+ build) |
| 2 | `quantized_qwen35::…::test_parallel_batched_forwarding_0_8b` | Qwen3.5-0.8B | 50 s |
| 3 | `quantized_llama::…::test_parallel_batched_forwarding_llama3` | Llama-3.2-3B | 90 s |
| 4 | `quantized_llama::…::test_parallel_batched_forwarding_llama2` | Llama-2-7B | 90 s |
| 5 | `quantized_qwen3::…::test_parallel_batched_forwarding` | Qwen3-8B | 80 s |
| 6 | `quantized_qwen35::…::test_parallel_batched_forwarding_9b` | Qwen3.5-9B | 80 s |
| 7 | `quantized_qwen38::…::test_parallel_batched_forwarding_27b` | Qwen3.8-27B | 140 s |
| 8 | `quantized_qwen3_moe::…::test_parallel_batched_forwarding` | Qwen3-30B-A3B | 90 s |
| 9 | `quantized_qwen35_moe::…::test_parallel_batched_forwarding_35b` | Qwen3.5-35B-A3B | 100 s |
| 10 | `quantized_qwen36_moe::…::test_parallel_batched_forwarding_36_35b` | Qwen3.6-35B-A3B | 100 s |
| 11 | `quantized_qwen38_moe::…::test_parallel_batched_forwarding` | Qwen3.8-Flash-Next | 250 s (every machine) |
| 12 | `deepseek4::…::test_parallel_batched_forwarding` | DeepSeek-V4-Flash | 200 s (full only) |

Each gate is its own command, run in the background with its output redirected to a log in
the scratchpad, and **the next one starts only after the previous exits**:

```bash
cargo test --release --features cuda -p candle-transformers --lib \
  models::<module>::tests::<fn> -- --exact --ignored --nocapture --test-threads=1 \
  > <scratchpad>/sweep/<NN>_<fn>.log 2>&1; echo "EXIT=$?"
```

- Never put two gates in one `cargo` invocation and never run two at once — each loads a
  multi-gigabyte checkpoint and sizes itself from the whole card.
- A gate **passed** only if `EXIT=0` **and** the log has
  `test result: ok. 1 passed`. A filter that matches nothing also exits 0 with
  `0 passed` — that is a broken sweep, not a pass.

### 4b. Then the engine probes — the rows the gates above cannot reach

The gates above drive `forward_wave` from a clean slate. They never construct a
`ConversationEngine`, so **nothing in them exercises admission, per-turn projection, the
persistence thread or KV compaction** — their tables print `-` for `Frontier`/`Eff%` and
their logs contain no compaction line at all. A change to any of that machinery can break
every daemon while all twelve gates stay green.

`candle-conversation/tests/kv_fragmentation.rs` is where those rows live. Discover them the
way step 2 discovers gates — `Grep` the file for `#[test]` and read each one's `#[ignore]`
text — rather than trusting this table, which is orientation only. Run them serially, after
the gates and under the same "card to itself" rule. Both run on **every machine**: the 30B
probe holds ~17 GB resident, and the Flash-Next probe, ~56 GB on the big card, runs on a
smaller one through the expert cache's streaming tiers like its gate (step 1):

| # | probe | model | ~time |
|---|---|---|---|
| 13 | `qwen3_30b_a3b_q4` | Qwen3-30B-A3B, engine probe | 185 s |
| 14 | `qwen38_flash_next` | Qwen3.8-Flash-Next, engine probe | 187 s (every machine) |

```bash
cargo test --release -p candle-conversation --features hub --test kv_fragmentation \
  <probe_fn> -- --exact --ignored --nocapture --test-threads=1 \
  > <scratchpad>/sweep/<NN>_<probe_fn>.log 2>&1; echo "EXIT=$?"
```

`--features hub` is not optional: without it the probe is not built and the filter reports
`0 passed` while exiting 0.

**Run these in the foreground, or read the `EXIT=` line.** A `; echo "EXIT=$?"` chain makes
the *shell's* status `echo`'s, which is always 0 — so a backgrounded probe is announced as
"completed (exit code 0)" while `cargo` exited 101 and two gates failed. That happened on
2026-09-27 and was reported as a pass. The `EXIT=` line in the task's own output file is the
real status; the notification's is not.

Each probe runs **three gates** and its panic message names which failed
(`N of 3 probe gates failed:`). Read them separately, because they fail for unrelated
reasons and only one of them is about numerics:

- **the story** (`story N/N`) — correctness. Anything below full is corruption, and on
  Flash-Next it is the recurrent-state case: that model is the only arch here carrying
  per-sequence state outside the paged K/V, so a pass that treads on it shows up as
  non-finite recurrent layers and an all-NaN logits row, which no 30B row can catch.
- **VRAM efficiency** (`worst sustained efficiency N%`) — how much of the ground below the
  arena frontier is actually holding KV.
- **weight uptake** — whether the weight side took the ground the frontier gave up, judged
  against `weight grow: asked=…` in the same block. A model whose every expert already
  fits (the 30B on a 72 GB card) has nothing to take; the probe then prints
  `weights fully resident` instead of a percentage, and that is a pass.

**KV compaction is on** (`compact_backings`, and `compact_recurrent` for the span tenants),
so the efficiency gate is expected to pass: 2026-10-11 read 100% (30B) and 100% (Flash-Next)
on the 16 GB card, and 2026-10-07 read 99% (30B) and 97% (Flash-Next)
worst sustained. A failure is a fragmentation regression to attribute, not a standing cost.

**The uptake gate is a different question.** `reclaim_spare_ground` runs between forwards
from every wave loop (`qwen4exp/wave.rs`, `latent_moe`, and `quantized_qwen3_moe` through
`BatchedModelCore`), and the growth policy's own refusal (`Refusal::Pressure`, when KV demand
is rising or the KV side asked since the last negotiation) is a KV-pressure gate. Flash-Next
reads well under 100% with its zone at its limit (59% on 2026-10-07) — read that as the zone
being full, not as a missing call.

The story gate is the one that is about numerics, and it is the one that must stay green:
`story N/N` with no `non-finite` line and no `!!!!` in the log is what says K/V and the
recurrent state are intact.

## 5. After each gate, report its table

Parse the log's `=== Performance Comparison ===` box (read it with Read/Grep). Columns:
`KvMode | int8 | Batched | Contexts | Valid | t/s (bulk) | t/s (single) | … | Compress`.
`t/s (bulk)` is **prefill**, `t/s (single)` is **decode**.

Pass rates come from the per-config block after `=== All tests completed ===`: each
`--- Config: TestConfig { mode: …, num_contexts: … } ---` is followed by
`✓ P/N sessions …` or `✗ Only P/N sessions passed validation …`, in the same order as
the table rows. (The table's `Valid` column shows `-` for configs whose mode is not
validated for reproduction; the pass-rate line is still printed — use it.) A `✗` line on
session isolation (`different-name pairs`) is a failure too.

Post, per model:

| mode | ctx | prefill t/s | decode t/s | compression | pass |
|---|---:|---:|---:|---:|---|
| BF16 | 1 | … | … | — | 1/1 |
| C10 | 10 | … | … | 4.11× | 10/10 |

Numbers exactly as the log prints them. Mark every failing row.

## 6. At the end, the summary table

| # | model | result | best prefill t/s | best decode t/s | best compression | time |
|---|---|---|---:|---:|---:|---:|

- **result** — PASS / FAIL (and SKIPPED for gates a partial sweep excluded).
- **best prefill / decode** — max over that gate's rows; name the mode × ctx it came from.
- **best compression** — max `Compress` over rows whose validation passed, with its mode.

Then a second table for step 4b's engine probes, because none of those columns describes
them — they run one workload, not a ladder, and what they measure is delivery rather than
ceiling:

| # | probe | story | worst sustained eff% | weight uptake | result | time |
|---|---|---:|---:|---:|---|---:|

- **weight uptake** — the percentage, or `weights fully resident` when the probe says so.
- **result** — PASS only when all three of that probe's gates passed. An efficiency failure
  is a fragmentation regression to attribute and fix forward (step 9), like any other
  failure — KV compaction is on, so it is never a standing cost.
- Then one line: **SWEEP PASS** only if every run gate **and every engine probe** passed;
  otherwise **SWEEP FAIL** and the list of failing gate × mode × ctx and failing probe ×
  gate.

## 7. Restart what step 3 stopped

Restart each process with exactly the recorded executable and arguments, from the repo root,
detached:

```powershell
Start-Process -FilePath "<ExecutablePath>" -ArgumentList '<args after the exe>' `
  -WorkingDirectory "d:\prog\candle"
```

Confirm it is running (`Get-CimInstance` again; for zend, its log/port answers). Restart
**whenever the sweep ends** — green, or stopping to hand back to the user — never leave them
stopped at the end of a turn. If a fix-forward loop (step 9) needs the card again, stop them
again by the same procedure.

## 8. Reading a failure

- **C10 fails** — C10 is the top rung and is calibrated to sit just under the breaking edge,
  so it can drift red. It *may* need its thresholds tuned down, carefully:
  the model's `*_KV_FACTORS` row in
  `candle-nn/src/kv_cache/chunked/sampled_selection/params.rs` (lower factor = tighter
  threshold = less compression). The doc comments on those rows record every
  re-derivation — read them first: V is usually the lever at the top rung, prefer a value a
  previous sweep measured passing over a fresh guess, record the new re-derivation (date,
  old → new, why) in the same comment, and re-run the gate at least twice because the
  ladder is statistical.
  **If passing would cost a significant reduction in compression** (the recorded
  re-derivations each moved the ratio by about 1%), or C10 fails by many sessions rather
  than one, **treat it as a model-accuracy bug**, not a threshold: find and fix the cause.
- **Any other mode fails** (BF16/F16/F32, Q8_0, Q4_*, C0–C9, session isolation) — that is
  a **bug**. These never normally fail. Never loosen a threshold to make them pass.
- A panic, CUDA fault, OOM or `0 passed` is a bug in the gate or the engine, not a
  flaky run — attribute it.

## 9. Fix forward

- Finish the remaining gates first so the sweep's whole picture is known, then fix.
- Fix the root cause in the code, with a unit test at the level the bug lives where one
  can be written (`CLAUDE.md`: TDD, raw expected values, no stubs, no env flags, no
  compatibility shims).
- Re-run the failing gate until it passes; if the fix touched shared code (kernels,
  `candle-nn` KV cache, the batched harness), re-run the whole sweep.
- The sweep is reported PASS only when every gate is green on the final code.
- Show the diff and propose a commit message at the end; **do not commit without the
  user's explicit go-ahead.**
