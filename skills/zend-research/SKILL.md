---
name: zend-research
description: Delegate web research to zend — a question answered from the web with every claim cited by the URL it came from, the sources' raw results kept in the worker's journal for checking. Use for comparing libraries, reading documentation or specifications, or surveying an approach, where the reading is the cost. Follows zend-thread.
---

# Research on zend

## When

A question whose answer is out on the web and takes many pages to establish:
which crate does X, what a specification says about Y, how others solved Z.

Not for questions about this codebase — that is zend-ask.

## Thread

The feature's thread, or `research-<topic>`.

## Delegate

Split a broad question into independent ones and start them together, each
with `effort: "exhaustive"` — research weighs many results against each other,
and thinking is where a contradiction between two findings gets noticed. Each:

```
Research: <the question, in full, with what the answer is for>

Work in three phases.

1. Gather. Search widely for the candidates: what exists, what was released, who says what.
2. Verify, for each item you will report:
   - Fetch its primary source and read it — the announcement, the specification, the
     documentation, the source code — not the search snippet. A figure that only a snippet
     gives is UNVERIFIED.
   - Search for independent evidence: third-party evaluations, benchmarks run by others,
     criticism, reports that contradict the primary source. Report what they found, in both
     directions — where the item does better than claimed as well as worse.
   - Where sources give different figures, settle on one by source quality — the primary
     source, then independent measurement, then reporting — and say why. Do not list both.
   - Not finding something in a page's text is not evidence it was not published: a fetched
     page that says it held images may have shown its tables or charts as pictures. Say what
     you could not read, rather than that it does not exist.
3. Check yourself. Before answering, read your draft against itself: a claim that
   contradicts another of your claims, a figure stated two ways, a "none" beside a number
   taken from that same thing. Resolve each, or mark it.

- Prefer primary sources: specifications, official documentation, source code, the authors.
- For a comparison, give each option the same criteria.
- Cite every claim with the URL it came from. Mark anything you could not confirm UNVERIFIED.
- Keep vendor claims apart from verified facts: say "the vendor reports" for what only the
  vendor says.
- Say nothing about this codebase you did not read in this task: a file, symbol or setting
  named in your answer cites the file:line you read it at, or is marked UNVERIFIED. What you
  know from your context is not a reading.
- Before you answer, start a part named verify-sources with sub_run that re-opens each cited
  URL you rely on and checks the claim is there — and checks each conclusion you drew from it
  (what a method needs, what it changes) against what the source says.

Final response, in this shape:
ANSWER: the answer, in a few sentences
DETAIL: the supporting points, each with its URL, independent evidence beside the vendor's
CONFLICTS: figures the sources disagreed on, the one you settled on, and why
UNVERIFIED: what could not be confirmed, and what you could not read
SOURCES: every URL read
```

## Read the result

Spot-check the claims you will rely on: `journal_grep` the worker's journal for
the URL, and read the turn that fetched it — `timeline_read` with `turn` shows
a turn whole, the search queries and fetched text included, and its metadata
the effort the worker ran at. Check the CONFLICTS against each other and the
DETAIL against itself before passing anything on.

## If zend is down

Research it yourself.
