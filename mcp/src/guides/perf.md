---
name: perf
description: How to investigate a performance problem — measure before changing, compare like with like, one change per measurement, numbers with the machine they came from.
tools: [task_start, task_wait, agent_job, agent_job_log, cluster_status]
---
# Performance work

## Prompt

`task_start {thread, name: "perf-<what>", effort: "exhaustive", prompt}`:

```
GOAL: find why <operation> is slower than <expected>, and make it faster, in
<repository>.

CONTEXT: <the measurement that shows it: numbers, machine, conditions; the
target>

METHOD:
1. Measure the baseline yourself, on a named machine, and record the
   conditions (build profile, inputs, load).
2. Profile or instrument to find where the time goes; read the code there.
3. One change per measurement; measure again under the same conditions.
4. A change that does not move the number is reverted.

CONSTRAINTS:
- Compare like with like: same machine, same build, same inputs.
- A measurement on a machine serving traffic is noisy — say so.
- Branch `zen/perf-<what>`; never the default branch.

FINAL RESPONSE:
BASELINE: the number, machine, conditions
WHERE: where the time goes, with evidence
CHANGES: each change and the number after it
RESULT: before vs after
COULD NOT DO: anything not done, and why
```

## Read the result

Numbers without a machine and conditions are not results. Re-measure the
final result yourself before reporting it.
