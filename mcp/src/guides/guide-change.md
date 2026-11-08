---
name: guide-change
description: How to change a guide, or any of this connector's own words, from a conversation — a worker edits it on a branch, a build agent tests it, the user approves the merge, and mcp is rebuilt and trialled on each machine that runs it.
tools: [repo_task_start, job_status, repo_task_reply, repo_job_start, job_read, job_grep, service_deploy]
---
# Changing a guide

The guides are compiled into **mcp**, this connector's own server, which runs
on two machines — blackworld and whiteworld — each under its keep. A guide
changes in five steps: edit, test, merge, build, trial. Tests, builds and
trials run on the build agents; Admins only.

## 1. Edit, on a branch

`repo_task_start {thread: "guide-<what>", name: "edit-<what>", role: "coder",
effort: "balanced", prompt}`:

```
GOAL: in the candle repository, change the guide mcp/src/guides/<name>.md:
<the change, in full — what it should say>.
Work on branch zen/guide-<what>; commit and push it; never push to main.
Keep the frontmatter (name, description, tools, clients) valid: name equals
the file's name. Change nothing else.
FINAL RESPONSE: BRANCH at <commit>; the guide's new text in full.
```

A new guide is a new file there: its `name` its file's, a one-line
`description`, the `tools` it is for.

## 2. Test the branch, on an agent

Read the new text in the worker's answer against the change asked for. Then
`repo_job_start {repo: "candle", branch: "zen/guide-<what>", program:
"cargo", args: ["test", "-p", "mcp", "--bin", "mcp", "guides"], node:
"WhiteWorld", timeout_s: 1800}`, and `job_status {jobs: [<the handle it
answered>], wait_s: 240}` until it ends: its `test result: ok` line is the
proof every guide still reads. A failure names the guide and what is wrong
with it — the ended job's `job_status` answer quotes it, `job_read {job,
page}` reads on — send it back to the worker.

## 3. Merge — the user's decision

Tell the user what changed, in a sentence or two. Only on their yes: if the
worker can still be answered, `repo_task_reply` to it — "the user approved:
merge zen/guide-<what> into main and push main; answer the merged commit".
If it cannot (it ended, or its member restarted), start one task for it:
`repo_task_start {thread: "guide-<what>", name: "merge-<what>", role: "coder",
effort: "balanced", prompt: "In the candle repository, merge branch zen/guide-<what>
into main and push main; change nothing else. FINAL RESPONSE: the merged
commit on main."}`.

## 4. Deploy, on each machine

For blackworld, then whiteworld: `service_deploy {host, service: "mcp"}`
builds main on that machine's agent and has keep try it; it answers with the
deploy's handle. `job_status {jobs: ["deploy:76c3a0f4d39bd9db:XeV_I_QPzpg"], wait_s: 240}`
waits until it is proven or rolled back (the `deploy` guide). While one machine's mcp
switches, the other serves the connector. Then the second machine.

## Report

The merged commit, and each machine's trial: proven or rolled back. The new
guide is served from the next conversation; a client that keeps its tool list
(Grok) sees a changed tool description only once the user reconnects the
connector.
