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

Split a broad question into independent ones and start them together. Each:

```
Research: <the question, in full, with what the answer is for>

- Search the web; read the sources themselves, not only the search snippets.
- Prefer primary sources: specifications, official documentation, source code, the authors.
- For a comparison, give each option the same criteria.
- Cite every claim with the URL it came from. Mark anything you could not confirm UNVERIFIED.
- Say nothing about this codebase you did not read in this task: a file, symbol or setting
  named in your answer cites the file:line you read it at, or is marked UNVERIFIED. What you
  know from your context is not a reading.
- Before you answer, start a part named verify-sources with sub_run that re-opens each cited
  URL you rely on and checks the claim is there — and checks each conclusion you drew from it
  (what a method needs, what it changes) against what the source says.

Final response, in this shape:
ANSWER: the answer, in a few sentences
DETAIL: the supporting points, each with its URL
UNVERIFIED: what could not be confirmed
SOURCES: every URL read
```

## Read the result

Spot-check the claims you will rely on: `journal_grep` the worker's journal for
the URL, and read the turn that fetched it.

## If zend is down

Research it yourself.
