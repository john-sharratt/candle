---
name: fact-check
description: How to have a worker check a set of claims — someone's, or another worker's — each confirmed, contradicted or unverified by the source that decides it.
tools: [repo_task_start, repo_task_wait, journal_grep, timeline_read]
---
# Fact-checking

**Use when** claims need checking before they are relied on: a document, an
article, a worker's answer, a vendor's announcement. Claims about the user's
code go to a worker; claims checked against the web go to one only when the
user asks — otherwise check them yourself, by the same method.

## Delegate

`effort: "exhaustive"`:

```
GOAL: check each claim below.

CLAIMS:
1. <claim>
2. <claim>

METHOD:
1. For each claim, find the source that decides it — the primary source
   first, then independent evidence. Fetch and read it.
2. Decide: CONFIRMED, CONTRADICTED (with what the source says instead), or
   UNVERIFIED (with what was tried).
3. A claim can be partly right: say which part.
4. Check yourself: a verdict that rests on a snippet is UNVERIFIED.

FINAL RESPONSE:
VERDICTS: each claim — CONFIRMED | CONTRADICTED | PARTLY | UNVERIFIED — the
deciding source's URL, and the quoted text that decides it
SOURCES: every URL opened
COULD NOT DO: anything not done, and why
```

## Read the result

A CONTRADICTED verdict is the one to check yourself before telling anyone —
`timeline_read {turn}` on the fetch that decided it.
