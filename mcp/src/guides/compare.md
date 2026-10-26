---
name: compare
description: How to have workers compare libraries, services or approaches — fixed criteria for every option, evidence for each cell, a recommendation with its reason.
tools: [task_start, task_wait, task_reply, journal_grep, timeline_read]
---
# Comparing options

Read `research` first: the gather → verify → check method applies to every
cell of the comparison.

## Delegate

One task per option when they are many, or one task for a short list, each
`effort: "exhaustive"`:

```
GOAL: compare <options> for <purpose>.

CRITERIA — apply every one to every option:
- <e.g. license; maintenance (last release, open issues); performance on
  <workload>; API fit for <use>; cost>

METHOD:
1. For each option and criterion, find the answer in the option's own
   documentation or source, then independent evidence.
2. A cell you could not establish is UNVERIFIED, not guessed.
3. Check yourself: the same criterion measured the same way for every option.

FINAL RESPONSE:
TABLE: options × criteria, each cell with its URL or file:line
RECOMMENDATION: the option, and the deciding reasons
WOULD CHANGE IF: what would make another option the better choice
UNVERIFIED: cells you could not establish
SOURCES: every URL opened
COULD NOT DO: anything not done, and why
```

## Read the result

Check the cells that decide the recommendation yourself. Present the table
and the recommendation; the choice is the user's.
