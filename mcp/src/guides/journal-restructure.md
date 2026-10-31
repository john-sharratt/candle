---
name: journal-restructure
description: How to restructure Zen's memory — move, merge, rename or split journals, or clean up duplicate lines — as a coding task for Zen on the `journals` checkout, previewed for the user and pushed straight to main.
tools: [repo_task_start, repo_task_wait, repo_task_reply, journal_read, journal_grep]
---
# Restructuring the memory

The memory is a git repository of YAML files: `john-sharratt/journals`, held
by zend's agent as its `journals` checkout. It is a different repository from
`candle`, which holds the code that reads it (the `journals` crate). The tools
only append. Anything that changes the tree's shape is a coding task for a
Zen worker on that checkout. That covers moving, merging, renaming or
splitting journals, removing duplicate lines, and moving lines from one
journal to another.

Correcting one line is not a restructure: use `journal_edit` on it, or
`journal_forget` for a line that should not be there.

## The layout a worker must keep

- `persons/<p>/person.yaml` holds identity only. `groups/<g>/group.yaml` and
  the group's tree sit beside it.
- Each journal is a folder: `journal.yaml`, then one `<id>.yaml` per line,
  named for its id (`2026-10-31-0131-4a96`), then `archive/<period>/`, then
  the folders of its sub-journals.
- The master's five sections (`people`, `conversations`, `topics`,
  `threads`, `bots`) are fixed: never renamed, never given a cue.
- Every named journal has a `cue`. A conversation is named
  `<yyyy-mm-dd-hhmm>-<channel>`.
- A journal holds at most 10 live sub-journals.
- A reference (`refs:` in `journal.yaml`) leads to a journal in the same
  group that exists, never to another reference.

## Briefing the worker

Start `repo_task_start` on `repo: "journals"`. The worker shares the
thread's journal, not this conversation, so state in full:

1. **What moves where, and why.** Name the lines by id and the journals by
   path.
2. **Keep every line's file and id.** Move a line with `journal_move`. Never
   rewrite one into a new file. To merge two duplicate lines, keep the most
   complete one, edit its text if it needs words from the other, and delete
   the other.
3. **Leave a reference at each old path** that readers or other journals
   used (`refs:` on the old parent), so a path through the old name still
   resolves. This also means a write the live service makes to the old name
   lands at the new one.
4. **Never touch a summary.** A summary rebuilds by itself once its lines
   change.
5. **Check before committing.** Run `journal_check`, which reads this
   session's copy, changes included, as the service would. Fix every problem
   it lists. A file that does not parse is refused by the service until it is
   fixed.
6. **Preview first.** Before pushing, end the turn with a `QUESTION:` block
   holding the `git diff --stat` and a table of what moved, merged and
   changed. Show every changed line's text before and after, in full. Under
   a heading `Meaning changed`, list each edit that does more than tidy
   words: who said it (from "colleagues describe him" to "he describes
   himself"), a claim made stronger or weaker ("contributed" to "wrote"), or a
   fact dropped. Push only after the reply says yes.
7. **Push to main.** Never branch the journals repository: a branch there is
   never merged, and the service reads `main` alone. The live service commits
   to `main` all the time, so pull with rebase first. If the push is refused,
   rebase again and retry. The service writes one file per line, so a rebase
   conflicts only on a `journal.yaml`. Take the service's summary and keep
   your other fields.

## After it lands

The service fetches the new `main` within one sync. Check the result:
`journal_read` the journals that changed, and `journal_grep` for a moved
line by its text. Tell the person what moved, in a sentence. A line the
service wrote to an old path during the push is found through its
reference.
