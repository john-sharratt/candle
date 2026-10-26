---
name: branch
description: How a worker's change lands — always on a branch, pushed for review, merged only with the user's approval.
tools: [task_start, task_wait, agent_job, agent_job_log]
---
# Branches and merges

- **Workers push branches, never a default branch.** Every code prompt says
  so: "work on branch `zen/<what>`; push it; never push to main".
- **Name branches for what they do**: `zen/implement-<what>`,
  `zen/fix-<what>`, `zen/refactor-<what>`.
- **Before review**: the branch builds and its tests pass on an agent
  (`test`), and its final response names the branch and commit.
- **Review**: read the diff yourself or delegate it (`review`).
- **Merging is the user's decision.** Ask, with what was built, tested and
  reviewed. Merging or committing to the default branch yourself is only done
  when the user says so for that change.
- **A local or branch build never joins the cluster**: it would serve real
  traffic and everything it writes replicates permanently. Only a commit on
  origin/main is deployed (`deploy`).
