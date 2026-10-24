# The paper-conversation replay — an end-to-end test for provenance and the working set

**Status:** the acceptance test for scoring, provenance and working-set work on
zend. Run before and after any change to how projections are scored or filled;
the baseline below was taken on 2026-10-25 with the working-set workarounds
removed (§6).

## 1. What it tests

Four user messages replayed on a fresh conversation. Together they exercise
everything a dialogue's projection has to get right:

| Turn | Message | What it needs from the projection |
|---|---|---|
| a | `Give me a tour of the codebase — main crates, key files, and how everything connects.` | Broad structure: folders, crate roots, ARCHITECTURE/README/CLAUDE.md. Nothing deep. |
| b | `what do you think of the unbounded context paper?` | The paper (`docs/unbounded_agents.md`, 7 pages) in full, and the model owning it as its own reading. |
| c | `if you were to improve the paper then what would you change?` | The paper plus the evidence behind its claims — `docs/performance.md`, the code it describes — without mixing up which document says what. |
| d | `ok. make those changes first then ill help you with more` | A long agentic turn: ~20–60 `file_edit` calls over up to 32 rounds against the placed paper, plus the places that restate what changed. |

The turns run in order, each only after the previous stream has ended.

## 2. How to run it

Run against a daemon **outside the cluster** (CLAUDE.md "Test locally, not
clustered") or with the cluster reduced to one member — everything a clustered
member writes replicates and is permanent. The edits turn d makes land in the
conversation's own file overlay, never on disk.

Every request is the same shape, with tools **Restricted** and the composer
dials fixed:

```json
{"model":"zen-code","stream":true,"conv_id":"<id>","effort":2,"verbosity":4,
 "think":true,"tools":"restricted",
 "messages":[{"role":"user","content":"<message>"}]}
```

```bash
BASE=http://<member>:8081
NEW=$(python -c "import uuid; print(uuid.uuid4())")
# Archive the previous run first, so the new conversation is the one on screen.
curl -s -X POST "$BASE/v1/conversations/<previous-id>/archive"
# Then, for each of a, b, c, d in turn:
curl -sN -X POST "$BASE/v1/chat/completions" \
  -H "content-type: application/json" -H "x-tokera-conversation: $NEW" \
  --data-binary @turn_a.json > run_a.sse
```

A full run takes 10–25 minutes. Leave the latest run unarchived until the next
one starts, so there is always one to watch in the UI.

**Run it at least twice per change.** Run-to-run variance is large: turn d
ranged from 10 to 61 edits on an unchanged build, and one rerun of a clean
configuration regressed outright.

## 3. What to collect

- `GET /v1/conversations/<id>` — every turn: each assistant turn's calls, each
  tool response, the closing answers.
- `GET /v1/conversations/<id>/projections/<message-index>/<decode>` — the
  projection a decode ran under. From `span`:
  - `materialized_tokens`;
  - `buckets` — tokens per group (`structure` = `repo_map` folders, `scopes` =
    `code_reading` files, `dependencies`, `architecture`, …);
  - `selection.turns` — each selected turn's `group`, `reason`
    (`working_set_momentum`, `working_set_lock`, …), `tokens`, `timeline`;
  - `groups` — each scanned group's raw order statistics per probe (`tail`,
    `question`), the data a need score is read from.
- Decode 0 of each assistant message is enough for a picture of what each turn
  was handed; look at a few later decodes of d to see the working set move.

## 4. How to grade it

Three correctness criteria, graded per turn, ideally by a second reader who did
not make the change:

1. **Owns what it read.** No "stub", "haven't read", "only the first line", or
   disputing an `in_context` reply for content that is placed.
2. **Nothing unfetched or misattributed.** Every specific name, number, path,
   section or quote is in something the turn fetched or was given, and is
   attributed to the right file. Watch especially for:
   - a fact from a neighbouring file credited to the one asked about;
   - one document's section numbers pinned to another (`performance.md`'s §3.5
     or §4 cited as the paper's);
   - references, citations or identifiers that appear nowhere it read.
3. **d's edits are real and in scope.** Every claimed edit landed; edits are
   what c proposed and the places that restate it; anything else is proposed,
   not done.

And the process measures, per run:

| Measure | Where |
|---|---|
| Rounds per turn, and whether d hit the 32-round cap | assistant turns between user messages |
| Tool errors (`not_found`, `ambiguous`, `not_permitted`, `call_cut_off`) and whether each was retried | tool responses |
| Calls the mode cannot run (e.g. `web_search` under Restricted) | assistant calls |
| Repeated identical calls | assistant calls |
| Files edited, and edits per file | d's `file_edit` calls |
| Materialized tokens per decode, per bucket | projections |

## 5. What a good projection holds

This is the target the scoring work is measured against: per turn, the
voluntary working set (provenance) and each analysis layer should hold what
the turn needs and little else.

| Turn | Should be there | Should not |
|---|---|---|
| a | crate roots and their `src/` folders; ARCHITECTURE, README | test files, benches, examples, game assets |
| b | the paper, every page | anything unrelated to the paper |
| c | the paper; `docs/performance.md` and the code its claims rest on (`compression_policy.rs`, `params.rs`, `wide_sig.rs`, `project.rs`) | test files, unrelated subsystems |
| d | the paper; files that restate its claims (README, ARCHITECTURE) | — |

The analysis layers (`dependencies`, `architecture`, `critique`, `bugs`,
`artifacts`) should project only when the turn needs that kind of analysis — a
question about a paper needs none of them; a question about a file's callers
needs `dependencies`.

## 6. Baseline (2026-10-25, workarounds removed)

The first decode of turn a, on a 150K voluntary budget with locks on top
(conversation `82787bbc`):

| Bucket | Tokens | Held |
|---|---|---|
| `scopes` (voluntary files) | 145,678 — full | `flat_prefill_probe.rs` 37K, `paged_glue_tests.rs` 22K, `decode_layout_exact.rs`, `moe_gemm_replay.rs`, `fast_exp_tests.rs`, `plot_threshold_sweep.py`, `candle-pyo3/Cargo.toml`, `docs/lore/index.md`, … |
| `architecture` | 36,763 | — |
| `dependencies` | 28,281 | — |
| `structure` (voluntary folders) | 4,266 | `coop/src/`, `tensor-tools/src/`, `candle-transformers/`, `scripts/`, `lints/` |
| materialized | 261,204 | |

None of the voluntary files relate to the question. The same files fill the
budget on every question of every run.

**Why** (`docs/zend_working_set.md` §4.3): a file's raw vote is `z × margin`
inside its own competition — how far one of its exchanges stands out from the
file's other exchanges, not how well the file answers the probe — so votes
cannot be compared across files, and larger files draw more maxima. The
normalised score's band runs ~1000× over on short probes (momentum once
reached 643,768). Neither ranks files against each other.

## 7. What the overnight runs taught (2026-10-24/25, 30 runs)

What was learned, separate from the workarounds that were later removed:

- **Shrinking the voluntary budget only hides the ranking problem.** At 80K the
  budget held ~70K of test files and the model misattributed facts between them
  and the paper; at 40K one 32K test file; at 20K a 14.5K test file, a
  flash-attn header, a wasm worker and game asset folders. Budgets do not fix
  ranking.
- **The analysis layers' raw group tops do not track the question.**
  `dependencies` scored 300K–417K on the tour, the paper, the improvements and
  the edits alike; medians and p90s are 0 (winner-take-all votes with a needle
  gate). See `docs/provenance_adaptive_projection.md` §B.5.
- **Locks belong next to the turns that asked for them.** A locked paper
  ~130K tokens before the question was called missing; emitted between the
  inherited chain and the dialogue's own turns, it was found.
- **The model disputes placed content it cannot see itself having read.** An
  `in_context` reply naming only the file was read as "a status without
  content"; even with the page's first line, the model once took that line for
  all it had.
- **Cross-document bleed follows how many documents are placed.** With the
  paper alone, attributions are clean; with `docs/performance.md` placed
  beside it, section numbers cross between them in both directions. Sealed
  readings carry no path at their close, so nothing near the text names its
  file.
- **A model can loop on a refused edit.** One turn sent the same ambiguous
  `file_edit` nine rounds running; refused mutations now count as repeats
  (`zend/src/repeat_guard.rs`).
- **A mode that offers a tool it then refuses wastes rounds.** Restricted turns
  spent rounds on `web_search`; each tools mode now compiles its stencil from
  the tools it offers (`docs/tool-system.md`).

## 8. A reference answer key for c and d

Real problems in the paper and the code it describes, verified on disk during
the runs. A strong c finds several; a strong d fixes the paper's side of them
and proposes the rest. Useful for judging whether a change made the turns
better informed, not just tidier.

1. `web/content/npcd/pages/landing.js:97` advertises the withdrawn 2,446 tok/s.
2. `web/src/markdown/math.rs:332` pins the superseded `O(1/N)` formula.
3. Two blog posts carry the "identical KV cache" claim §7 contradicts
   (`one-card-unbounded-context.md:421-422`, `the-journal-is-king.md:245`).
4. CLAUDE.md:37 says `sign(Q_PCA^T @ K)`; the code is Q·Q with no PCA.
   CLAUDE.md:459 says C0–C9; the code has C0–C10.
5. The paper's probe window is 64 tokens in six places; the code's is 256
   (`config.rs:1855-1859`).
6. §5's "template K/V never persisted" describes a removed mechanism —
   `project.rs:2443`: "Template sections seal like any other".
7. §9.9's "no per-model calibration" contradicts the per-model `*_KV_FACTORS`
   rows.
8. C10 is calibrated against the same story-rewrite test that validates it.
9. §6.2/§9.10 describe the three-band index; `WideQSig` (§6.6) is what ships.
10. The laptop link is PCIe ×8 in CLAUDE.md, ×16 in the README and §9.2.
11. §7 (persist KV, not tokens) contradicts §9.12/§11.5 ("prefill the tokens
    and reconstruct the identical KV").
12. The GitHub handle is `john-sharratt` in Appendix C, `john-sharrat` in the
    game README.
