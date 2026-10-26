---
name: incident
description: How to handle something broken now — contain, establish what changed, diagnose from evidence, act only with approval, then write it up.
tools: [cluster_status, cluster_logs, cluster_audit, service_status, service_control, cluster_command, cluster_job_status, journal_note]
---
# Incidents

## 1. Scope it — what is affected, since when

`cluster_status` (every member, every app's floor), `service_status` for the
machines involved. Tell the user what is down and what is not, before
anything else.

## 2. What changed

`cluster_audit`: commands run, restarts, members joining and leaving, and who
did each. Most incidents start just after something changed.

## 3. Evidence

`cluster_logs` from a little before it started (`logs`). Quote the lines.

## 4. Act — only with approval for anything that changes the cluster

- A restart is a step, not a fix: it buys time, it does not explain.
- Look and Care commands as needed (`cluster-care`); Decide commands —
  restart, shutdown, rollback, rollout — only with the user's approval.
- A bad release: `rollback`.
- A stopped or stuck service: `service_control` (a stop needs approval).

## 5. Write it up

On a thread `incident-<date>-<what>`, note as you go: what was seen, what was
done, by whom, with what result. Afterwards, `postmortem`.
