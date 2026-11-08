---
name: docs-write
description: How to have a worker write any document to keep — a README from the code, a design proposal, a plan — landed on a branch, every statement about the code cited, what is planned marked as planned.
tools: [repo_task_start, job_status, journal_grep]
---
# Writing a document to keep

Any document meant to outlast the conversation — a README or reference from
the code, a **design proposal**, a plan, a write-up of findings — is written
by a `coder` worker and **landed**: committed and pushed to its branch, not
returned as text. A proposal names what it proposes, the alternatives
weighed, and what is decided versus open; a document from the code cites the
code. For a task that will run with no one to answer it, add its landing,
done-when and never to the brief (`drive`).

## Prompt

`repo_task_start {thread, name: "docs-<subject>", role: "coder", effort: "deep", prompt}`:

```
GOAL: write <the document> for <subject> in <repository>, at <path>, on
branch `zen/docs-<subject>`.

AUDIENCE: <who reads it, and what they need to do with it>

CONTEXT: <what exists already; decisions to record; the shape wanted>

METHOD:
0. Commit and push the document to the branch: the answer points at the
   commit, it does not carry the document instead.
1. Read the code the document describes, end to end.
2. Write what is, in the present tense. Anything planned is marked planned.
3. Cite the file:line behind every statement about the code.
4. Check yourself: re-read each citation; a statement you cannot cite is
   removed or marked UNVERIFIED.

FINAL RESPONSE:
DOCUMENT: path, branch and commit
OUTLINE: its sections
CITATIONS: each statement about the code, with the file:line behind it
UNVERIFIED: statements you could not pin to the code
COULD NOT DO: anything not done, and why
```

**For a draft to review before anything is written** — a paragraph, a
section — make it draft only: drop the path and branch from GOAL, add
"read only — change nothing, push nothing; return the text in the answer",
and replace DOCUMENT with DRAFT: the text itself. Ask for plain language:
the statements a reader needs, not phrasing.

## Read the result

Read the document yourself; spot-check its citations. A design document is
the user's to approve.
