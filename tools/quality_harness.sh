#!/bin/bash
# Quality-walk harness: build zend, bring the daemon up on a BLANK substrate,
# run the standard 4-question conversation battery twice, and save raw
# transcripts for manual evaluation.
#
# Usage:  tools/quality_harness.sh <step-label> [extra zend args...]
#
# Each invocation is hermetic: a per-step working dir (`walk/<step-label>`)
# plus `--wipe-substrate` guarantees a blank substrate without any shell-side
# deletion of substrate paths. The daemon is started with the baseline-parity
# config (summariser off, code_reading layer not ingested, repo_map on) and
# stopped when the battery completes. Transcripts land in
# `walk/<step-label>/battery<N>_q<M>.json` — read and judge them manually.
set -u

LABEL="${1:?usage: quality_harness.sh <step-label> [extra zend args...]}"
shift || true
PORT=8123
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STEP_DIR="$ROOT/walk/$LABEL"
mkdir -p "$STEP_DIR"

# Stop any running daemon FIRST — it locks target/release/zend.exe, which
# would fail the relink (and silently leave a stale binary if unchecked).
# This is the ONE image-wide kill (the relink needs the exe unlocked, whoever
# holds it); it is loud so a production daemon death is never silent.
if tasklist //FI "IMAGENAME eq zend.exe" 2>/dev/null | grep -qi zend; then
    echo "[harness:$LABEL] WARNING: killing ALL running zend.exe to unlock the build target"
    taskkill //F //IM zend.exe >/dev/null 2>&1
fi
sleep 3

echo "[harness:$LABEL] building zend (release)…"
if ! (cd "$ROOT" && cargo build -p zend --release >"$STEP_DIR/build.log" 2>&1); then
    echo "[harness:$LABEL] BUILD FAILED — see $STEP_DIR/build.log"
    tail -5 "$STEP_DIR/build.log"
    exit 1
fi

echo "[harness:$LABEL] starting daemon (blank substrate, code_reading off)…"
cd "$ROOT"
nohup ./target/release/zend.exe \
    -v --disable-summariser --wipe-substrate \
    --working-dir "$STEP_DIR/mind" \
    --disable-layer code_reading \
    --ingest-dir "repo_map=$ROOT" \
    --port "$PORT" "$ROOT" >"$STEP_DIR/daemon.out" 2>&1 &
DAEMON_PID=$!

ready=0
dead_strikes=0
for _ in $(seq 1 120); do
    sleep 10
    if curl -s -m5 "http://127.0.0.1:$PORT/v1/status" 2>/dev/null | grep -q '"state":"ready"'; then
        ready=1
        break
    fi
    # `tasklist` can transiently return nothing under load — require three
    # consecutive misses before declaring the daemon dead.
    if tasklist //FI "IMAGENAME eq zend.exe" 2>/dev/null | grep -qi zend; then
        dead_strikes=0
    else
        dead_strikes=$((dead_strikes + 1))
        if [ "$dead_strikes" -ge 3 ]; then
            echo "[harness:$LABEL] DAEMON DIED — see $STEP_DIR/daemon.out"
            tail -8 "$STEP_DIR/daemon.out"
            exit 1
        fi
    fi
done
if [ "$ready" != "1" ]; then
    echo "[harness:$LABEL] daemon not ready after 20min — aborting"
    exit 1
fi
echo "[harness:$LABEL] ready — running battery"

# Questions are sent as ASCII-only JSON (`—` escapes, not raw UTF-8):
# inline `-d "$str"` payloads mangle multi-byte characters in this shell.
ask() { # $1=conv $2=out $3=question (ASCII / \uXXXX escapes only)
    printf '{"model":"zen-code","conv_id":"%s","messages":[{"role":"user","content":"%s"}],"stream":false,"max_tokens":1200}' \
        "$1" "$3" >"$STEP_DIR/.payload.json"
    curl -s -m600 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@$STEP_DIR/.payload.json" >"$2" 2>/dev/null
}

for n in 1 2; do
    CID="walk-$LABEL-$n"
    ask "$CID" "$STEP_DIR/battery${n}_q1.json" 'Give me a tour of the codebase — main crates, key files, and how everything connects.'
    ask "$CID" "$STEP_DIR/battery${n}_q2.json" 'what time is it?'
    ask "$CID" "$STEP_DIR/battery${n}_q3.json" 'what is the sqrt of 3475923874598723459872345987?'
    ask "$CID" "$STEP_DIR/battery${n}_q4.json" 'what questions did i ask during this conversation?'
    echo "[harness:$LABEL] battery $n complete"
done

# Decode-speed snapshot for the step (raw forward avg from the wave log).
grep -aoE "kv/fwd avg=[0-9]+ fwd avg=[0-9]+ms" "$STEP_DIR/mind/.substrate/zend.log" 2>/dev/null | tail -5 >"$STEP_DIR/decode_ms.txt"

# Teardown is scoped to the daemon THIS harness spawned — never other zends.
taskkill //F //PID "$DAEMON_PID" >/dev/null 2>&1
echo "[harness:$LABEL] done — transcripts in $STEP_DIR"
