---
name: literature
description: How to have workers survey papers on a technique — the papers read, not their abstracts; methods and numbers side by side; where results conflict and why.
tools: [repo_task_start, repo_task_wait, repo_task_reply, journal_grep, timeline_read]
---
# Literature survey

Read `research` first; papers add their own traps: an abstract overstates,
numbers are measured under different conditions, and a table is often an
image.

## Delegate

`effort: "exhaustive"`:

```
GOAL: survey the work on <technique> for <purpose>.

METHOD:
1. Find the papers: the original, the main follow-ups, recent surveys.
2. Read each paper — the method and the results sections, not only the
   abstract. A number taken from an abstract alone is UNVERIFIED.
3. For each result, record what was measured, on what model or data, under
   what conditions.
4. Where papers disagree, find why: different conditions, different
   baselines, a reproduction that failed.
5. A results table shown as an image is "not readable in the text" — find
   the numbers in the text, the code release, or a reproduction.
6. Check yourself: numbers compared only when measured comparably.

FINAL RESPONSE:
PAPERS: each — title, authors, year, URL, the claim, the method in a sentence
RESULTS: a table of comparable results, each with its conditions and source
CONFLICTS: where results disagree, and why
GAPS: what has not been measured
UNVERIFIED / SOURCES / COULD NOT DO: as in research
```
