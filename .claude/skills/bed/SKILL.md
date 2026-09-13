---
name: bed
description: Overnight autonomous run — gather everything remaining (memory, design docs, open items, the working tree), finish every task without asking questions, fix forward, audit the CLAUDE.md invariants, full code review (code-review skill + manual review against design intent) with every finding fixed, clean up intermediate files and debug logic, run /fast-test and /sweep, and leave an HTML morning report. Never commits.
disable-model-invocation: true
---

# /bed — the night shift

**The user has gone to bed.** Nobody will answer a question until morning, so do not ask
any — not with `AskUserQuestion`, not as a question at the end of a message. When a decision
is needed, make the one most consistent with the design documents and `CLAUDE.md`, write down
what was decided and why, and keep going. Every task on the list gets finished. Stopping
early, handing back half-done work, or parking a task "for the user to decide" is a failure
of this skill; the only things left for the morning are the ones listed under *Holds* below.

**Always fix forward.** Anything found broken tonight — a failing test, a review finding, an
invariant violation — is fixed tonight, whether this work introduced it or it was already
there. "Pre-existing" is not a category.

**Never commit, never push, never open a PR during the night.** Leave every change in the
working tree. The morning report proposes the commits; the user makes them.

All of `CLAUDE.md` applies throughout, and these especially: file tools only for reading and
writing files; never mask an exit status; no env-var feature flags, no stubs, no `TODO`s, no
compatibility shims; one concern per file; `use` imports; TDD with raw expected values;
never pass `--ignored` to the whole zend suite (three tests destroy or copy the substrate);
model gates run one `cargo` process per model with the daemon stopped.

## 1. Gather everything remaining

Build the picture from every source there is — not from recollection alone:

- **This conversation** — every request the user made, every "we'll come back to it", every
  pending step, every follow-up you offered and they accepted.
- **Memory** — `MEMORY.md` and the memory files it points to that bear on current work
  (project and feedback entries especially). Verify a remembered file/function/flag still
  exists before acting on it.
- **Open-items / plan documents** — `docs/open_items.md` and any `docs/**/*plan*.md`,
  `*progress*.md`, `*design*.md` touched recently (`git log --since` on `docs/`).
- **The repository's state** — `git status`, `git diff --stat`, `git log` since the last
  commit the user made; uncommitted work is unfinished work until it is verified.
- **The code itself** — Grep for `TODO`, `FIXME`, `unimplemented!`, `todo!`, `dbg!`, and
  stray `eprintln!`/`println!` debugging in files changed recently.

## 2. Study the design documents

For every area the remaining work touches, find and read the design document it is built
from (`docs/`, `docs/deepseek/`, anything a module doc or commit message cites) — end to end,
not skimmed. `CLAUDE.md`: **design docs are authoritative**; where the code disagrees with the
document, the code is wrong unless the document is, and then the document is fixed in the
same change. Note the design intent and the expected *outcome* (numbers, invariants,
behaviour) for each area — the manual review in step 5 judges against these.

## 3. Build the list

Write the full task list to `<scratchpad>/bed/tasks.md` (and keep it current as work
proceeds): one line per task — what, where (file/area), the design doc it answers to, how it
will be verified (the test or gate that proves it). Include:

- everything remaining from step 1;
- the fixed tail of the night, always: step 4's invariant audit, step 5's review, step 6's
  cleanup, step 7's `/fast-test` and `/sweep`, step 8's report.

**Holds.** A task the user explicitly parked ("leave X for now", "we'll come back to it",
"don't touch Y") is *not* worked tonight. List it under **Holds** in `tasks.md` and in the
report, with the user's own words. Everything else is in scope.

Order the list so that work others depend on comes first, and so that GPU work (gates,
sweeps) is batched rather than interleaved with builds.

## 4. Do the work — every task on the list

For each task: implement it fully, with its unit test written alongside (`CLAUDE.md` TDD),
verify it with the test or gate named in `tasks.md`, then mark it done. A task is done when
its verification passes on the current code — not when the code is written.

**Audit the `CLAUDE.md` invariants** over everything changed tonight and over the hot paths
the work touched (`forward_wave` and what it calls, per layer, per wave):

1. no `to_dtype` in the loop; **1b** validate with `expect_dtype`, never a defensive cast;
2. no allocate-plus-copy to materialise a layout (`contiguous`, `cat`, `slice_set`,
   `to_owned_tensor` costing the parent); **2b** per-row data through descriptor tables;
3. **no unnecessary GPU→CPU transfers** — only the two sanctioned readbacks (MoE routing
   indices, embedding token ids); every other readback, sync, or `to_vec` on the hot path is
   a finding;
4. no host-side compute a kernel can do;
5. everything batched across slots/sessions — no per-seq or per-token launch loops;
6. no zeroing buffers a kernel fully overwrites (`alloc_uninit`);
7. span partition boundaries hold in both directions.

Every violation found is fixed (with a test or a measured before/after), not listed.

## 5. Full code review — and fix everything it finds

Review **the whole night's change set** (`git diff` against the last commit the user made,
plus the untracked files), two ways:

- **The `code-review` skill** — Claude Code's bundled review; invoke it (Skill tool,
  `code-review`) on the working-tree changes. Not `/code-review ultra`: that one is
  user-triggered and billed. If the skill cannot be invoked in this session, do not skip the
  step — run the same review through an `Agent` with the full diff, asking for correctness,
  regressions, missing tests and invariant breaks, and verify every finding it returns.
- **A manual review against design intent and outcome** — for each changed area, read the
  change beside its design document from step 2 and ask: does it do what the design says,
  does it produce the outcome the design promises (measure it where the design gives a
  number), does it break an invariant, is it tested at the level the behaviour lives, does
  every comment describe the code as it is (no process narration, no "Phase N")?

Every finding is fixed, then re-verified. Re-review after the fixes; stop when a review pass
comes back clean.

## 6. Clean up

- **Intermediate files** — remove what tonight's work created and nothing needs: scratch
  scripts, probe binaries, captured logs and dumps *inside the repo*, temporary fixtures.
  Check `git status` for untracked files and decide each one; never delete a tracked file or
  anything the user created unless it is unambiguously an artefact of tonight's work.
  Scratchpad files may stay.
- **Debug logic** — remove temporary instrumentation: `dbg!`, ad-hoc `eprintln!`/`println!`,
  one-off timing prints, commented-out code, forced flags. Permanent `tracing` at a sensible
  level stays; a debug path that is a real feature is not debug logic.
- Then `cargo fmt --all -- --check` and
  `cargo clippy --workspace --tests --examples -- -D warnings` — both clean.

## 7. Run the checks

Invoke the **`fast-test`** skill, then the **`sweep`** skill, and follow each to green —
their own rules apply (stop/restart zend and npcd with their recorded arguments, GPU tests
at one thread, gates serial smallest-first, fix forward). A fix made for either sends you
back through step 5's review for that change, and then through both skills again, until
both are green on the final code.

## 8. The morning report

Write a one-off HTML report — load the `artifact-design` skill first, write the page to
`<scratchpad>/bed/morning_report.html` (not into the repository), and publish it with the
`Artifact` tool so it has a private link. It is read once and never maintained. Contents,
most important first:

- **Headline** — green or not: `/fast-test` result, `/sweep` result, anything still open.
- **Done tonight** — each task: what changed (files), why, how it was verified.
- **Decisions made for you** — every choice made without asking, with the reasoning and how
  to reverse it.
- **Findings fixed** — invariant violations and review findings, each with its fix.
- **Test and gate results** — `/fast-test`'s summary table; `/sweep`'s per-model summary
  (best prefill / decode / compression, pass/fail), with notable changes against the last
  recorded numbers.
- **Holds** — parked items, in the user's words, untouched.
- **Still open** — anything that could not be finished, with exactly what blocks it and what
  was tried. (This list should be empty; every entry needs a reason that survives the
  morning.)
- **Proposed commits** — the working tree grouped into commits, each with its file list and
  message (no AI attribution). Nothing was committed.
- **Cleaned up** — files removed and debug logic stripped.

End the night with the report's link and a short summary as the final message.
