# scripts/

Operational scripts that sit outside the Cargo build graph.

## `nightly_infinite_conversation.sh`

Runs the Tier 3 "cruise" harness for the infinite-conversation system
(unbounded turn history with provenance-selected recall, see
`docs/archived/infinite_conversations.md` §10.4 and §11) against a
**persistent workspace**, so conversation depth accumulates across repeated
invocations rather than resetting each run.

### What it does

1. Resolves `WORKSPACE_DIR` (default `/var/lib/zen_<mode>_workspace`) and
   creates it if missing; writes a CSV header to `RECALL_CSV`
   (default `$WORKSPACE_DIR/recall-curve.csv`) on first run.
2. Runs `cargo test --release -p zend --test infinite_conversation_deep -- --ignored <TEST_NAME> --nocapture`,
   teeing output to a temp log file.
3. Scrapes the most recent `<mode> CSV: depth=... recalled=... pending=... dirty=... selected_count=...`
   line from the log and appends a row (`ts,git_sha,scale,depth,recalled,pending,dirty,selected`) to `RECALL_CSV`.
4. Exits non-zero if the test failed.

### Modes

The script only implements two of the four cadences documented in its own
header comment:

| Mode (arg) | Test name | Cadence documented in the header comment |
|---|---|---|
| `cruise` (default) | `infinite_conversation_cruise` | Nightly, ~30 min |
| `stress` | `infinite_conversation_stress` | Weekly (manual), ~5 h |

`smoke` (per-PR, `infinite_conversation_smoke`, ~5 min) and `marathon`
(quarterly, ~50 h) are mentioned in the header comment as part of the
intended cadence schedule but are not handled by this script's `case`
statement — invoking either name exits with an "Unknown mode" error.
`infinite_conversation_smoke` does exist as its own test file
(`zend/tests/infinite_conversation_smoke.rs`) and can be run directly with
`cargo test` instead.

### Prerequisites

- A CUDA GPU (`CUDA_VISIBLE_DEVICES`, default `"0"`).
- `zend/tests/infinite_conversation_deep.rs` built in release mode (the
  script builds it via `cargo test --release`, so first invocation pays full
  compile time).
- Write access to `WORKSPACE_DIR` — this directory is the persistent
  substrate the depth measurement grows against; deleting it resets the
  cruise/stress depth curve back to zero.

### Usage

```bash
scripts/nightly_infinite_conversation.sh            # cruise (default)
scripts/nightly_infinite_conversation.sh stress      # weekly stress run
WORKSPACE_DIR=/data/zen_cruise scripts/nightly_infinite_conversation.sh cruise
```

## Related docs

`docs/archived/infinite_conversations.md` (design this harness measures
against — archived; see `docs/immutable_summary_forest.md` for the current
summary-tree design), `zend/tests/infinite_conversation_deep.rs`,
`zend/tests/infinite_conversation_smoke.rs`.
