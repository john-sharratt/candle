---
name: ab-trial
description: How to compare a new build against the running one — trial it on one machine while the others serve the old build, compare what matters, then decide whether to roll out.
tools: [service_trial, service_status, cluster_status, cluster_logs, repo_job_start, repo_job_log]
---
# A/B trial

## Do

1. **Decide what to compare** before starting: error rate, latency of a
   named operation, an answer's quality on a fixed set of prompts. Write it
   down on the thread.
2. **Baseline**: measure it on the machines still on the old build.
3. **Trial the new build on one machine** (`deploy`, steps 1–5).
4. **Compare** under the same load: `cluster_logs` for errors on each member
   (`node` narrows to one), `cluster_status` for state, the operation timed on
   each. Same inputs, same window.
5. **Decide with the user**: roll out (`release`), keep testing, or roll back
   the trial machine (`service_trial` with the previous build, or
   `rollback`).

## Notes

- Traffic splits by the gateway's routing, not evenly: say how much each
  build actually served when comparing.
- A trial machine that rolled itself back has answered the question: the
  build did not serve.
