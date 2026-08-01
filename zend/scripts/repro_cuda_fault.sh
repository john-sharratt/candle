#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# repro_cuda_fault.sh — black-box reproduction of the "new conversation crashes
# with a CUDA fault" bug. Starts the daemon, waits for it to be ready, opens ONE
# conversation, and reports PASS (tokens streamed, daemon alive) or FAIL (daemon
# crashed / poison-exited / streamed an error / no tokens). Exit 0 = PASS, 1 =
# FAIL — so it drops straight into a bisect / iterate loop.
#
# Usage:
#   zend/scripts/repro_cuda_fault.sh <workspace> [extra zend args…]
#
# Env knobs:
#   ZEND_BIN=target/release/zend.exe   the daemon binary
#   PORT=80                            API port
#   READY_TIMEOUT=1800                 secs to wait for load + ingest to settle
#   PROMPT="…"                         the conversation opener
#   BLOCKING=1   run with CUDA_LAUNCH_BLOCKING=1 → the breadcrumb #0 names the
#                EXACT faulting kernel/thread (≈2-5× slower; the fast pinpoint)
#   SANITIZE=1   run under compute-sanitizer memcheck → exact kernel + .cu line
#                of the out-of-bounds access (10-100× slower; the ground truth).
#                Kernels are already built with --generate-line-info.
#
# NOTE: binds PORT (80) — no other zend may be running. Reads .substrate/zend.log
# for the fault/breadcrumb dump on failure.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

WS="${1:?usage: repro_cuda_fault.sh <workspace> [extra zend args…]}"; shift || true
ZEND_BIN="${ZEND_BIN:-target/release/zend.exe}"
PORT="${PORT:-80}"
BASE="http://127.0.0.1:${PORT}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
PROMPT="${PROMPT:-Give me a tour of the codebase: main crates, key files, and how everything connects.}"
OUT="$(mktemp -t zend_harness.XXXX.log)"
LOG="${WS}/.substrate/zend.log"

launch=("$ZEND_BIN" "$WS" "$@")
[ "${BLOCKING:-0}" = "1" ] && export CUDA_LAUNCH_BLOCKING=1
if [ "${SANITIZE:-0}" = "1" ]; then
  launch=(compute-sanitizer --tool memcheck --error-exitcode 99 --launch-timeout 0 --print-limit 40 "${launch[@]}")
fi

echo "[harness] launching: ${launch[*]}"
echo "[harness]   (BLOCKING=${BLOCKING:-0} SANITIZE=${SANITIZE:-0}, daemon stdout→ $OUT)"
"${launch[@]}" >"$OUT" 2>&1 &
ZPID=$!
cleanup(){ kill "$ZPID" 2>/dev/null; wait "$ZPID" 2>/dev/null; }
trap cleanup EXIT

# 1) Wait for readiness (dies-before-ready is itself a FAIL).
echo "[harness] waiting up to ${READY_TIMEOUT}s for /v1/status = ready…"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
until curl -s -m5 "${BASE}/v1/status" 2>/dev/null | grep -q '"state":"ready"'; do
  if ! kill -0 "$ZPID" 2>/dev/null; then
    echo "[harness] FAIL: daemon exited before it became ready"; tail -40 "$OUT"; exit 1
  fi
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[harness] FAIL: timed out waiting for ready"; exit 1; }
  sleep 3
done
echo "[harness] ready — opening the conversation."

# 2) One streaming conversation.
resp=$(curl -s -N -m 180 -X POST "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"zen-code\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}]}" 2>&1)

# 3) Verdict.
sleep 1
result=0
if ! kill -0 "$ZPID" 2>/dev/null; then
  echo "[harness] FAIL: daemon crashed / poison-exited DURING the conversation."
  result=1
elif printf '%s' "$resp" | grep -qiE "ILLEGAL_ADDRESS|poisoned|internal server error"; then
  echo "[harness] FAIL: the stream carried a CUDA/error signal."
  result=1
elif ! printf '%s' "$resp" | grep -q 'data:'; then
  echo "[harness] FAIL: no tokens streamed (empty/aborted response)."
  result=1
else
  echo "[harness] PASS: streamed $(printf '%s' "$resp" | grep -c 'data:') SSE events, daemon alive."
fi

# 4) On failure, surface the root fault + breadcrumb ring (and sanitizer report).
if [ "$result" = "1" ]; then
  echo "──────── root fault / recent kernel breadcrumb (zend.log) ────────"
  grep -aiE "poisoned|ILLEGAL_ADDRESS|recent CUDA kernel|#[0-9]+ '|panic|PANIC" "$LOG" 2>/dev/null | tail -30
  if [ "${SANITIZE:-0}" = "1" ]; then
    echo "──────── compute-sanitizer report (exact kernel + .cu line) ────────"
    grep -aiE "Invalid|out-of-bounds|=========|memcheck|at 0x|\.cu:[0-9]+|by thread" "$OUT" | tail -50
  fi
fi
exit "$result"
