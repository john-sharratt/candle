---
name: zend-ask
description: Ask zend a question about the codebase — answered from its analysis layers (repo map, code reading, dependency analysis, code review, security review) and a read of the files, every claim cited by file and line. Use instead of reading many files yourself when you need to understand how something works, where something is, or what depends on what. Follows zend-thread.
---

# Asking zend about the code

## When

You need to understand the code before deciding something — how a subsystem
works, where a value is read, what breaks if X changes — and answering means
reading more than a few files.

## Thread

The feature's thread the question serves.

## Delegate

```
Answer this question about <repository>: <the question, with what the answer is for>

- Start from what your analysis layers retrieve, then read the files to confirm.
- Cite every claim with file:line, and quote the line for anything the answer turns on.
- Grep a name before you state it exists. Say plainly what you could not find.

Final response, in this shape:
ANSWER: a few sentences
EVIDENCE: each claim with file:line and the quoted line
NOT FOUND: what you looked for and could not find
```

Several independent questions: one task each, started together.

## Read the result

Open the cited lines for anything you will act on. zend can attribute a fact
to the neighbouring file; the citation is how you catch it.

## If zend is down

Read the code yourself.
