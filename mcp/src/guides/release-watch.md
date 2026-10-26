---
name: release-watch
description: How to run a periodic survey — model releases, a dependency's changes, a vendor's announcements — over a date window, every date checked, vendor claims kept apart, what moved since last time.
tools: [repo_task_start, repo_task_wait, repo_task_reply, journal_read, journal_grep, timeline_read]
---
# Watching for releases

Read `research` first. A survey adds three things: a date window, a list kept
across runs, and the difference from last time.

## Thread

One thread per subject, kept across surveys (`model-releases`), so each run
starts from the last one's findings. Between surveys it archives itself after
a day idle; `thread_attach` reopens it, journal and all — but the last
survey's worker can no longer be continued, so the next survey is a new task
that starts from the notes.

## Delegate

`effort: "exhaustive"`:

```
GOAL: survey <subject> released between <from> and <to>.

CONTEXT: the previous survey on this thread found <items>; check what changed
for them, and find what is new.

METHOD: research's three phases, plus:
- Every release date checked against the primary source (the announcement,
  the release page, the repository's tag) — not a roundup's date.
- Anything outside the window is left out, even if notable.
- Promised but not yet shipped (weights "due by", "coming soon") is reported
  as promised, with the date promised, and checked: has it shipped?
- For each release: what it is, what the vendor claims, what independent
  evaluation found (both directions), price, license, availability.

FINAL RESPONSE:
RELEASES: each item, date (with its primary-source URL), what it is, vendor
claims, independent evidence, price, license
CHANGED SINCE LAST TIME: items from the previous survey whose status moved
PROMISED: announced but not shipped, with the promised date
NOT CHECKED: candidates found but not verified
CONFLICTS / UNVERIFIED / SOURCES / COULD NOT DO: as in research
```

## After

Verify (`verify`), then note on the thread what this survey settled, so the
next one starts from it: the items and their status in a line each, citing
the worker's entry (`T:<thread>/<name>#<n>`) for the detail — a note holds
2,000 characters, not a release table. Give the next survey's prompt this
note in CONTEXT, with the research prompt in full: a worker reads its prompt
and the thread's notes, not the guides.
