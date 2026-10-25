---
name: zend-thread
description: How to keep a zend thread while working with the zend MCP tools — attach to the feature's thread before delegating, note each decision and anything the user states, pass the thread to every task, and re-read its journal after context is compacted. Use whenever you call any zend_* or journal_* tool, and before following any other zend skill.
---

# Keeping a zend thread

zend is the workhorse; you orchestrate and review. Every piece of work you
delegate runs on a **thread** — zend's record of one feature or area — and every
task is a **worker** under it. The thread's journal is where the feature's
anchors live: what you decided and why, what the user said, what each task
found. Every worker starts with it in its prompt, so what you note there is
what every later task knows.

## The loop

1. **Attach first.** `zend_attach {thread}` with a kebab-case name for the
   feature (`review-layers`, `bc-wearables`, `release-5739c544`). Two
   unrelated pieces of work never share a thread: a shared thread answers from
   the wrong history. Attach again after your context is compacted.
2. **Read where it stands.** The attach answer is the thread's journal. Read
   it before you plan; it holds decisions you may no longer remember.
3. **Note as you decide.** `journal_note {journal: "self", text}` — one note
   per decision, with its reason, and one for anything the user states that
   later tasks must keep. A standing rule of the whole repository goes on the
   project journal instead: `journal_note {journal: "P:<repository>"}`. Every
   task in every thread starts with that.
4. **Delegate with `zend_task`**, passing the thread and a name saying what the
   worker does. State the task in full: the worker shares the thread's
   journal, not your conversation.
5. **Wait on everything at once.** Start every independent task first, then
   `zend_wait {workers: [...]}` on all of them; it answers as soon as one
   settles. Handle that one, then wait again on the rest.
6. **Answer what asks.** A worker that is blocked ends its turn with a
   `QUESTION:` block — options, what it tried, pointers. Check the pointers,
   decide, and `zend_answer`. A **stuck** worker (its loop was closed) is
   answered the same way: say how to go on.
7. **Steer without waiting** with `zend_nudge` while a worker runs.
8. **Archive** the thread with `zend_archive` once the feature is done.

## Reading results

A finished task answers with its **journal entries**: a line, then claims, each
labelled with how it was known — `Saw:` (a tool result), `Did:`, `Claude said:`
(from your own prompt), and `Concluded, not seen:`. An entry is evidence, not
truth:

- Act on a claim — merge, deploy, report to the user — only after checking it
  against the turns it cites: `journal_read {journal: "<worker path>", entry}`
  pages them; `journal_grep` finds a claim's evidence across a worker, or a
  whole thread with `T:<thread>/*`.
- Never act on `Concluded, not seen` unchecked.
- A `(re-check)` claim may have changed since.

## What you must never do

- Never delegate a decision that needs the user's approval: merging to main,
  committing under the repository's rules, any Decide operation. Those stay
  with you and the user.
- Never paste a worker's claims to the user as fact without the check above.
- Never let zend stall you: if `zend_wait` keeps answering `running` past
  the skill's timeout, or zend is down, do the work yourself.
