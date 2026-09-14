---
name: fast-test
description: The quick check — stop zend/npcd (remembering their arguments), run every crate's default (non-ignored) unit and integration tests with GPU tests at one thread and CPU tests at full width, iterate fixes forward until every suite is green, report one line per pass while running and a summary table at the end, and restart zend/npcd.
disable-model-invocation: true
---

# /fast-test — quick checks, fixed forward until green

These are the everyday checks, so they must be **fast and green**. The loop is: run → fix →
re-run, until every pass is green. **A failure is fixed forward — never waved off as
"pre-existing", "flaky", or "not from this change".** If it fails, it is ours to fix now.

The repository rules in `CLAUDE.md` apply: read logs with Read/Grep (never
`cat`/`Select-String`), never mask an exit status, no env-var feature flags, no stubs, no
`TODO`s, never commit without permission.

## 1. Stop zend and npcd — remember how they were started

```powershell
Get-CimInstance Win32_Process -Filter "Name='zend.exe' OR Name='npcd.exe'" |
  Select-Object ProcessId, Name, ExecutablePath, CommandLine
```

Write each process's `ExecutablePath` and full `CommandLine` into the conversation before
touching it, then `Stop-Process -Id <pid> -Force -Confirm:$false`. Confirm with
`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` that the card is back
at its idle floor (~0.5 GB). GPU tests gate on a *present device*, not a feature flag, and
VRAM-headroom assertions invert with a co-tenant — the card must be ours alone.

## 2. Two passes per crate: GPU tests at ONE thread, CPU tests at full width

- **GPU pass — `-- --test-threads=1`. Mandatory.** Every test that can reach the GPU runs
  one at a time. Only a few of them serialise themselves (`candle-core`'s `real_cuda`
  `GPU_LOCK`, `candle-nn`'s `gpu_test_lock`, `tools_integration`'s scenario mutex); the rest
  would allocate device memory side by side and exhaust the card.
- **CPU pass — default threads (32 on this machine).** Only tests that cannot reach the GPU.

### Auditing which tests are GPU tests — every run, before running anything

A test is a **GPU test** if it is compiled only with CUDA (`#[cfg(feature = "cuda")]`, the
`_cuda` variants of `test_device!`), or if it — or a helper it calls — constructs or picks
a CUDA device or boots an engine. Grep the test sources (the crate's `tests/`, and the
`#[cfg(test)]` modules under `src/`) for:

```
Device::new_cuda|cuda_if_available|Device::Cuda\(|new_cuda\(|ZendSession::|gpu_test_lock|GPU_LOCK
```

A match inside production code is not a GPU test by itself — but a test that *calls* that
production path is (e.g. `zend::model_choice`'s ladder calls `Device::new_cuda`). When a
match is in production code, check whether the crate's tests reach it. Classify
**conservatively**: when in doubt, it goes to the GPU pass — a CPU test at one thread costs
seconds, a GPU test at 32 threads costs the card.

Record the classification in the conversation (crate → GPU modules / binaries) before the
first run, and re-derive it every run — tests are added all the time.

### How the passes are built, by how CUDA reaches the crate

Re-read `[workspace] members` in the root `Cargo.toml` for new crates, and check each
crate's `Cargo.toml` for how CUDA reaches it.

**A. CUDA is an opt-in feature — `candle-core`, `candle-nn`, `candle-transformers`.**
A build *without* `--features cuda` cannot touch the GPU, so it is the CPU pass by
construction. The GPU pass is the CUDA build filtered to the GPU tests: diff the two builds'
`-- --list` output (tests only in the CUDA list), plus the tests present in both whose body
picks a device at runtime (the audit grep).

```bash
cargo test -p <crate> --tests                                               # CPU pass
cargo test -p <crate> --features cuda --tests -- --list > <scratchpad>/fast-test/<crate>.cuda.list 2>&1; echo "EXIT=$?"
cargo test -p <crate> --tests -- --list > <scratchpad>/fast-test/<crate>.cpu.list 2>&1; echo "EXIT=$?"
cargo test -p <crate> --features cuda --lib -- <gpu_module> [<gpu_module> …] --test-threads=1
cargo test -p <crate> --features cuda --test <gpu_target> [--test …] -- --test-threads=1
```

(As of 2026-09-13 the audit grep matched 110 files across these three crates — too many to
list here; re-derive them.)

**B. CUDA is always compiled in — `candle-conversation` (`--features hub`), `zend`
(`--features cuda`), `npcd`.** There is no GPU-free build, so the audit decides:

```bash
# CPU pass, full width: the lib minus its GPU modules, plus every CPU integration binary
cargo test -p <crate> [features] --lib -- --skip <gpu_module> [--skip …]
cargo test -p <crate> [features] --test <cpu_target> [--test …]
# GPU pass, one thread: the lib's GPU modules, then every GPU integration binary
cargo test -p <crate> [features] --lib -- <gpu_module> [<gpu_module> …] --test-threads=1
cargo test -p <crate> [features] --test <gpu_target> [--test …] -- --test-threads=1
```

The 2026-09-14 audit, for orientation only — re-derive it:

| crate | GPU lib modules | GPU integration binaries | CPU integration binaries |
|---|---|---|---|
| `candle-conversation` | `persistence::transfer`, `persistence::elevate`, `batched_sampler::tests`, `provenance::gpu`, `provenance::gallery_arena` | `cold_warm_hot_path`, `compression_integration`, `conversation_tests`, `deepseek_rung4`, `narrator_integration_test`, `narrator_window_test`, `prefill_ab`, `projection_identity`, `summarization_tests`, `thinking_span`, `turn_belief_scan` | `integrity_repair`, `literal_text_real_vocab`, `live_turn_gallery`, `narrator_tests`, `selection_replay`, `staged_ingest_events`, `store_tests`, `token_bias_chatml`, `token_bias_real_vocab` |
| `zend` | `model_choice` | `coherence_integration`, `duplication_replay`, `gui_api_harness`, `hybrid_recurrent_state`, `infinite_conversation_deep`, `infinite_conversation_smoke`, `integration`, `memory_continuity`, `persistence_integration` (ignored), `provenance_fold_real_geometry`, `recall_quality`, `reproject_control`, `reproject_wave`, `restart_turn_recovery`, `section_quantize_end_to_end`, `startup_stall_watchdog`, `tools_integration`, `zen_code_phase12_smoke` | `code_read_integration`, `repo_scan_integration`, `stencil_tool_call`, `tokenizer_special_tokens`, `tool_catalog`, `watcher_integration` — plus `--bin zend` (`main.rs`'s `wipe_tests`), which `--lib` does not reach |
| `npcd` | none (its device is constructed only on the model-load path, which no test reaches) | none | `makers`, `tools`, `vault` |

**C. No CUDA in the dependency graph — CPU pass only, full width.**
`zend-tools`, `web`, `npc-map`, `target-prune`, `candle-datasets` (confirm from each
`Cargo.toml`), and `candle-kernels` unless the audit finds device construction in its tests.

```bash
cargo test -p <crate> --tests
```

### Rules for every pass

- **One `cargo test` process at a time.** Passes never run concurrently — and that includes
  the `-- --list` builds, which compile and link exactly as much as a pass does.
- **`-j 6` for the crates that link dozens of large test binaries — `candle-conversation`,
  `zend`, `npcd`.** At cargo's default width `zend`'s ~25 test binaries link side by side,
  each mapping multi-GB rlibs, and on 2026-09-14 that exhausted the whole 95 GB commit limit
  (31.5 GB RAM + a 64 GB page file). The linker reports it as **`os error 1455` — "The paging
  file is too small"** — and the failed maps surface beside it as `unresolved external
  symbol`, `found invalid metadata files` and `crate … required to be available in rlib
  format`, which read like corruption and are not. Before a run, check the commit headroom
  (`(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory`, in KB); under ~40 GB, find
  what is holding it before starting.
- **`--no-fail-fast` on every pass.** Without it cargo stops at the first failing binary and
  the rest of the pass never runs, so one failure hides every result behind it.
- **`--tests`** (or `--lib` / `--test <name>`) — unit and integration tests, no doc tests.
- **Features are not optional.** A feature-gated test that is not built is reported as
  `filtered out`, not as a failure — `candle-conversation` without `hub` silently drops every
  hub test.
- **The passes must cover everything.** For each crate, CPU-pass + GPU-pass test counts must
  equal the crate's total from `-- --list`; a test in neither is a gap, a test in both is
  wasted time.
- **Misclassification shows up as GPU trouble.** A CUDA out-of-memory, a device-allocation
  failure, or a VRAM-headroom assertion in a CPU pass means the audit missed a GPU test:
  move it to the GPU pass and re-run it there before judging it.
- **Never pass `--ignored`.** `#[ignore]`d tests are the long gates (see `/sweep`) and three
  zend tests destroy or copy the live substrate.

Run each command in the background, output redirected, exit status checked separately:

```bash
cargo test … > <scratchpad>/fast-test/<NN>_<crate>_<cpu|gpu>.log 2>&1; echo "EXIT=$?"
```

### One chain, and stopping it for real

- Run the passes as **one** sequential chain — one script, one background task, one line
  out per pass. Never start a second chain, or any other cargo command, while one is alive.
- **Stopping a background chain does not stop its children.** On Windows, `TaskStop` on a
  Git Bash chain kills only the top shell: the script's `cargo`, `rustc`, `link` and test
  binaries carry on, and a new chain started beside them doubles every build. That is what
  exhausted memory on 2026-09-14 — two chains, not one wide build. After stopping a chain,
  kill the rest by command line and confirm nothing is left before starting anything else:

  ```powershell
  Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*fast-test*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }
  Get-Process cargo,rustc,link -ErrorAction SilentlyContinue   # must print nothing
  ```

  Orphaned test executables count too — `<name>-<hash>.exe` under `target\debug\deps`. One
  left behind that day held 4 GB of host memory and the card.
- **Never kill `rustc` mid-compile.** A compile killed half-way can leave an rlib whose
  codegen units disagree with each other, and cargo still counts it as fresh, so it is never
  rebuilt. It shows as `LNK2019`/`LNK2001 unresolved external symbol ….llvm.<digits>` where
  the missing symbol and the function referring to it are both inside the **same**
  `lib<crate>-<hash>.rlib`, and there is no `os error 1455` beside it. With no cargo running,
  delete that `lib<crate>-<hash>.rlib` and its `.rmeta` from `target\debug\deps`, and every
  `target\debug\incremental\<crate>-*`, then rebuild. If the error survives a clean rebuild of
  that crate, it is a real defect.

### Tests leave nothing in the tree

Before the first pass and after the last, record the tree:

```bash
git status --porcelain --ignored --untracked-files=all -- . ':!target' > <scratchpad>/fast-test/tree_<before|after>.txt; echo "EXIT=$?"
```

Compare the two with Read/Grep. **Any path that appears is a test that wrote into the source
tree — a defect, fixed forward like a failure.** A test's workspace is a `tempfile::TempDir`,
or a fixed directory under `env!("CARGO_TARGET_TMPDIR")` when it must persist across runs
(a tool-catalog calibration, a substrate copy — `zend/tests/common/mod.rs`'s
`production_workspace()`). It is never `std::env::current_dir()` or `CARGO_MANIFEST_DIR` —
under `cargo test` both are the crate directory, which is how a 3.2 GB `zend/.substrate` came
to sit in the tree — and never the repo root, whose substrate is the user's live one. A named
directory in `std::env::temp_dir()` that is never removed accumulates just the same; use a
`TempDir`.

## 3. While running — one line per pass

As each pass finishes, post exactly one line:

```
✓ candle-nn · GPU · 1 thread · 214 passed, 0 failed, 3 ignored · 11.8 s
✗ zend · CPU · 32 threads · 598 passed, 2 failed · 3.1 s — session::title_tests::… (assert at session.rs:5561)
```

Read the log to get it: each binary's `test result: ok|FAILED. P passed; F failed; I ignored;
… finished in Ts` line (sum them), and for a failure the `---- <name> stdout ----` block's
`panicked at <file>:<line>` and assertion values. A pass succeeded only if `EXIT=0` **and**
every `test result` line says `ok`. Note any binary whose `finished in` exceeds **20 s** — a
speed defect (step 5).

For per-test times inside a slow binary, run the already-built exe directly (setting
`RUSTC_BOOTSTRAP` on `cargo` itself changes the build fingerprint and rebuilds the world):

```bash
RUSTC_BOOTSTRAP=1 target/debug/deps/<name>-<hash>.exe -Z unstable-options --report-time \
  [filters] [--test-threads=1] > <scratchpad>/fast-test/<name>.times.log 2>&1; echo "EXIT=$?"
```

## 4. Fix forward, and iterate until green

- Fix the **root cause** in the code, not the symptom in the test. Loosening an assertion,
  widening a tolerance, adding `#[ignore]`, or skipping a test to make a pass green is not a
  fix. If the test itself is wrong (a stale expectation, a harness bug), correct the test and
  say why it was wrong.
- Pre-existing or not makes no difference — it is fixed in this run.
- Add or update a unit test at the level the bug lives (`CLAUDE.md`: TDD, raw expected
  values).
- Re-run the failing pass (one line, as above); when it is green, re-run **every** pass once
  more, because a fix in a shared crate (`candle-core`, `candle-nn`, `candle-conversation`)
  reaches the others. Stop only when every pass is green.
- **Speed is part of the job.** A test over 20 s is optimised, not accepted: find where the
  time goes (per-test times above, then the code — host-side fixture loops at `opt-level 0`
  are the usual cause; see the per-package `opt-level` block in the root `Cargo.toml`).
  `#[ignore]` is the last resort, only for a test that genuinely needs a production model or
  the whole card, with the reason in the attribute.
- If a GPU-pass failure looks like contention (headroom assertions, OOM), re-check step 1 —
  nothing else may hold the card — before treating it as a code defect; if the card was idle,
  it *is* a code defect.

## 5. Restart what step 1 stopped

Restart each process with exactly its recorded executable and arguments, from the repo root
(the checkout this session runs in — the three dev machines keep it in different places),
detached:

```powershell
Start-Process -FilePath "<ExecutablePath>" -ArgumentList '<args after the exe>' `
  -WorkingDirectory "<repo root>"
```

Confirm it is running. Restart **whenever the run ends** — green, or stopping to hand back
to the user — never leave them stopped at the end of a turn. If a fix needs the card again,
stop them again by the same procedure.

## 6. At the end — the summary table

Once, after the final iteration:

| # | crate | pass | threads | binaries | passed | failed | ignored | pass rate | test time | wall time | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

- One row per crate × pass (CPU / GPU), from the final iteration.
- **test time** — sum of the `finished in` values; **wall time** — the command's duration,
  build included.
- **status** — PASS, FAIL, or SLOW (passed, but a binary or test over 20 s).
- Below it:
  - **Errors** — for every failure seen during the run, fixed or not: crate › pass › binary ›
    test — one-line cause — `file:line` — and the fix (file changed, test that now covers
    it), or why it is still open.
  - **Slow** — every test or binary over 20 s, with its time and what was done about it.
  - **GREEN** only if every pass of every crate passed on the final code; otherwise **RED**
    with the count.
- Then a proposed commit message for the fixes. **Do not commit without the user's explicit
  go-ahead.**
