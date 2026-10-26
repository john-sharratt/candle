---
name: knowledge
description: How to make a lesson outlive the conversation — project rules every task starts with, decisions on the thread, and what a bad run taught.
tools: [journal_note, journal_read, journal_grep]
---
# Keeping what was learned

Your context ends; journals do not. What should change future work goes where
future work will read it:

| Kind | Where | Read by |
|---|---|---|
| A rule of the whole repository — a convention, a "never do X", a lesson from a failure | `journal_note {journal: "P:<repository>"}` | every task in every thread, at its start |
| A decision about this work, and why | `journal_note {journal: "self"}` | every worker on this thread; you after compaction |
| A method that should change — how research or review is done | a proposed guide edit, to the user | everyone using the connector |

Write a rule as something to do, with its reason:
"Fetch every URL you cite before citing it — a run once cited pages it never
opened." Not "be careful with sources".

Before adding a rule, `journal_grep` the project journal for one that already
says it; refine that one rather than adding a second.
