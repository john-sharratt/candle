---
name: journal
description: How to record and find what a thread or a project knows — notes for decisions and rules, and searching every worker's evidence.
tools: [journal_note, journal_read, journal_grep]
---
# Journals

| Journal | Holds | Written by |
|---|---|---|
| `T:<thread>` (or `"self"`) | every task started and how it ended, and your notes | the server and you |
| `T:<thread>/<name>` | what the worker found, each entry citing the turns behind it | the worker |
| `P:<repository>` | standing rules every task in every thread starts with | you |

## Note

- On the thread: a decision and why; what the user said or asked for; a
  constraint every later task must keep. One note per decision, a sentence or
  two, as you make it.
- On the project: a rule of the whole repository — a convention from its
  CLAUDE.md, something the user says always holds, a lesson from a bad run
  that every future task should start with.

## Read

- `journal_read {journal: "self"}` — the thread's journal; first thing after
  compaction.
- `journal_read {journal: "T:<thread>/<name>"}` — a worker's entries.
- With `entry` (or an address `T:<thread>/<name>#12`): one entry — for a
  worker, the turns behind it, the evidence for what it claims.

## Search

`journal_grep {journal, pattern}` — case-insensitive. Search one worker, or
every worker on a thread with `T:<thread>/*` when you do not know which one
learned something. Each hit names its journal, entry, turn and line; read
around it with `journal_read` and the entry.
