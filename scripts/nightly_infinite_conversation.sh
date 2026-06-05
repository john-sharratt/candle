#!/usr/bin/env bash
#
# Nightly CI script — runs the Tier 3 cruise harness for the infinite-
# conversation system against a persistent workspace so depth
# accumulates across runs.  See docs/infinite_conversations.md §10.4
# and §11.
#
# Schedules:
#   • Per-PR sanity   — `infinite_conversation_smoke` (~5 min)
#   • Nightly         — `infinite_conversation_cruise` (~30 min)
#   • Weekly (manual) — `infinite_conversation_stress` (~5 h)
#   • Quarterly       — `infinite_conversation_marathon` (~50 h)
#
# Per-run scope is bounded so a single invocation never exceeds its
# cadence's wall-clock budget.  Depth accumulates because
# WORKSPACE_DIR persists.
#
# Usage:
#   nightly_infinite_conversation.sh [cruise|stress]    # default: cruise
#
# Environment variables:
#   WORKSPACE_DIR      — persistent workspace path (default: /var/lib/zen_cruise)
#   RECALL_CSV         — output CSV path (default: $WORKSPACE_DIR/recall-curve.csv)
#   CUDA_VISIBLE_DEVICES — pinned to "0" if unset

set -euo pipefail

mode="${1:-cruise}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/var/lib/zen_${mode}_workspace}"
RECALL_CSV="${RECALL_CSV:-${WORKSPACE_DIR}/recall-curve.csv}"
export WORKSPACE_DIR
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$WORKSPACE_DIR"

# CSV header on first run.
if [[ ! -f "$RECALL_CSV" ]]; then
    echo "ts,git_sha,scale,depth,recalled,pending,dirty,selected" > "$RECALL_CSV"
fi

case "$mode" in
    cruise)
        TEST_NAME="infinite_conversation_cruise"
        ;;
    stress)
        TEST_NAME="infinite_conversation_stress"
        ;;
    *)
        echo "Unknown mode '$mode'.  Expected 'cruise' or 'stress'." >&2
        exit 2
        ;;
esac

git_sha=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Run the harness, capture output, scrape the "CSV:" line.
log_file=$(mktemp /tmp/zen-${mode}-XXXXXX.log)
trap "rm -f '$log_file'" EXIT

if cargo test --release -p zend --test infinite_conversation_deep \
        -- --ignored "$TEST_NAME" --nocapture 2>&1 | tee "$log_file"; then
    status="pass"
else
    status="fail"
fi

# Extract the most recent "<scale> CSV:" line and append to CSV.
csv_line=$(grep -E "^${mode} CSV:" "$log_file" | tail -n1 || echo "")
if [[ -n "$csv_line" ]]; then
    # Parse "${mode} CSV: depth=X recalled=Y pending=A dirty=B selected_count=C"
    depth=$(echo "$csv_line" | sed -nE 's/.*depth=([0-9]+).*/\1/p')
    recalled=$(echo "$csv_line" | sed -nE 's/.*recalled=([^ ]+).*/\1/p')
    pending=$(echo "$csv_line" | sed -nE 's/.*pending=([0-9]+).*/\1/p')
    dirty=$(echo "$csv_line" | sed -nE 's/.*dirty=([0-9]+).*/\1/p')
    selected=$(echo "$csv_line" | sed -nE 's/.*selected_count=([0-9]+).*/\1/p')
    echo "${ts},${git_sha},${mode},${depth},${recalled},${pending},${dirty},${selected}" \
        >> "$RECALL_CSV"
fi

if [[ "$status" == "fail" ]]; then
    echo "${mode} run FAILED" >&2
    exit 1
fi
echo "${mode} run OK (depth recorded in ${RECALL_CSV})"
