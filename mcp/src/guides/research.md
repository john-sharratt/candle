---
name: research
description: When the user asks for web research to run on Zen's workers — how to have them answer with every claim checked against the page it came from; gather, verify, check themselves, then you verify them. Otherwise research with your own tools.
tools: [repo_task_start, repo_task_wait, repo_task_reply, journal_grep, timeline_read]
---
# Research

**Use when** the user asks for research to run on Zen — a long or exhaustive
question, run while they are away, or one whose evidence should stay on a
thread. Otherwise research with your own web tools, by the same method. Not
for questions about a codebase — that is `ask`.

What went wrong before, and what this guards against: a worker searched ~40
times and fetched 5 pages; it cited "primary sources" it never opened; its
standout figures came from one aggregator; it reported "no benchmark table
published" for a table that was an image; it contradicted itself; and it said
a figure was "unresolved" when several sources agreed. The method below
answers each of those.

## Delegate

Split a broad question into independent ones and start them together, each
with `effort: "exhaustive"`:

```
GOAL: <the question, in full, and what the answer is for>

CONTEXT: <what is already known; date window if any; earlier findings on this
thread; what not to redo>

CONSTRAINTS:
- If you have no web access, say so in your first line and stop.
- <date window>: report only what falls inside it, and check each date
  against the primary source.

METHOD — three phases:
1. GATHER. Search widely for the candidates: what exists, what was released,
   who says what.
2. VERIFY, for each item you will report:
   - Fetch its primary source and read it — the announcement, the
     specification, the documentation, the model card, the source. A figure
     you saw only in a search snippet is UNVERIFIED.
   - Search for independent evidence: evaluations run by others, criticism,
     reports that contradict the primary source. Report both directions —
     where it does better than claimed as well as worse.
   - Where sources disagree, settle on one figure by source quality — the
     primary source, then independent measurement, then reporting — and say
     why. Do not list both.
   - A figure seen only on an aggregator or roundup is marked
     "aggregator-only".
   - A page that reports `images` may have shown its table or chart as a
     picture: say "not readable in the page text", never "not published".
3. CHECK YOURSELF before answering: read your draft against itself — two
   claims that contradict, a figure stated two ways, a "none" beside a number
   taken from that same thing. Resolve each.

RULES:
- Cite every claim with the URL you fetched it from. Cite only pages you
  opened: your answer is audited against what web_fetch returned you, and
  anything cited that you did not open sends you back to open it.
- Keep vendor claims apart: "the vendor reports …" for what only the vendor
  says.
- Before answering, start a part named verify-sources with sub_run: it
  re-opens each URL you rely on and checks the claim is there.

FINAL RESPONSE, in this shape:
ANSWER: the answer, in a few sentences
DETAIL: each item, its facts with the URL each came from; independent
evidence beside the vendor's; aggregator-only figures marked
CONFLICTS: figures the sources disagreed on, the one you settled on, and why
UNVERIFIED: what you could not confirm, and what you could not read
SOURCES: every URL you opened
COULD NOT DO: anything asked that you did not do, and why
```

## Verify the result (read `verify`)

1. Read the citation audit at the end of the answer. The worker has already
   been sent back once to open anything it cited unopened, so what is still
   `seen` or `never seen` is unverified — say so, or fix it in step 2.
2. Continue the same worker with `repo_task_reply` — it keeps everything it read —
   to open what remains, settle a conflict, or go deeper on one item. It can
   be continued until its thread is archived or it sits a day untouched.
3. Spot-check the facts you will pass on: the audit proves a page was opened,
   not that it says what the answer claims. `journal_grep` finds the turn a
   URL or figure is in; `timeline_read {id, turn}` shows the page as the
   worker got it.
4. Tell the user what was checked, and keep "the worker reports" apart from
   what you confirmed.
