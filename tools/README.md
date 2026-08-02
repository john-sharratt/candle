# tools

Out-of-workspace helper scripts and standalone binaries — listed in the root
`Cargo.toml` `[workspace] exclude` (not built by `cargo build --workspace`;
build/run each directly).

## `parquet2text.rs`

Standalone bin crate (own `Cargo.toml`, depends only on `parquet = "53"`)
that converts a WikiText-style parquet file (single `"text"` column) into a
newline-delimited raw text file, for feeding into `perplexity-eval` or other
text-file-based tooling.

```bash
cargo run --manifest-path tools/Cargo.toml --bin parquet2text -- input.parquet output.txt
```

## `quality_harness.sh`

Bash harness that builds `zend` in release mode, boots the daemon against a
**blank substrate** (`--wipe-substrate`, baseline-parity config: summariser
off, `code_reading` layer disabled, `repo_map` ingested from the repo root),
runs a fixed 4-question conversation battery twice against it, and saves raw
JSON transcripts plus a decode-speed snapshot for manual quality evaluation.
It kills any running `zend.exe` first (the build would otherwise fail to
relink the locked binary), polls `/v1/status` for readiness (up to 20 min),
and tears down only the daemon it spawned.

```bash
tools/quality_harness.sh <step-label> [extra zend args...]
```

Output lands in `walk/<step-label>/`: `build.log`, `daemon.out`,
`battery{1,2}_q{1..4}.json` (one JSON response per question per battery run),
and `decode_ms.txt` (forward-pass timing grepped from the daemon log). The
four questions probe: codebase orientation, real-time awareness, arithmetic
outside the model's native ability, and conversation self-recall.
