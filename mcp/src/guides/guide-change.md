---
name: guide-change
description: How to change a guide, or any of this connector's own words, from a conversation — a worker edits it on a branch, the user approves the merge, and mcp is rebuilt and trialled on each machine that runs it.
tools: [repo_task_start, repo_task_wait, repo_task_reply, repo_job_start, repo_job_log, service_trial, service_status]
---
# Changing a guide

The guides are compiled into **mcp**, this connector's own server, which runs
on two machines — blackworld and whiteworld — each under its keep. A guide
changes in four steps: edit, merge, build, trial. Trials are Admins only.

## 1. Edit, on a branch

`repo_task_start {thread: "guide-<what>", name: "edit-<what>", effort:
"balanced", prompt}`:

```
GOAL: in the candle repository, change the guide mcp/src/guides/<name>.md:
<the change, in full — what it should say>.
Work on branch zen/guide-<what>; push it; never push to main.
Keep the frontmatter (name, description, tools, clients) valid: name equals
the file's name. Run `cargo test -p mcp guides` and quote its result line.
FINAL RESPONSE: BRANCH at <commit>; the guide's new text in full; the test
result line.
```

A new guide is a new file there: its `name` its file's, a one-line
`description`, the `tools` it is for.

## 2. Merge — the user's decision

Tell the user what changed, in a sentence or two. Only on their yes:
`repo_task_reply` to the same worker — "the user approved: merge
zen/guide-<what> into main and push main; answer the merged commit".

## 3. Build, on each machine

For blackworld, then whiteworld — find each machine's agent with
`cluster_status`:

`repo_job_start {node: <its agent>, repo: "candle", branch: "main",
program: "cargo", args: ["build", "--release", "-p", "mcp"],
stage: "target/release/mcp.exe", timeout_s: 1800}`

Follow it with `repo_job_log` until it has ended; its answer says where the
build was staged. Use that path.

## 4. Trial

`service_trial {host, service: "mcp", artifact: <the staged path>}`, then
`service_status {host}` until the trial is proven or rolled back. While one
machine's mcp switches, the other serves the connector. Then the second
machine.

## Report

The merged commit, and each machine's trial: proven or rolled back. The new
guide is served from the next conversation; a client that keeps its tool list
(Grok) sees a changed tool description only once the user reconnects the
connector.
