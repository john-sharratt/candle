# Zen coding sessions: why one failed, and the fixes

**Status:** built (2026-11-05), except as noted under each cause. `file_edit`
was made mandatory first (Cause 3, commit `934225224`). Where the build
departs from the design below, the departure is recorded beside it.

**Source:** web-UI conversation `451502ff-4526-4407-a68d-69225467477a`
(code.tokera.com): 64 turns, 226,566 tokens, 156 tool calls. At turn #41 it was
asked to make the journal-note bubble change, push to main and deploy; it did
none of the three. Turn numbers below are that conversation's; the evidence is
its call listing (`timeline_read`, view `calls`) and its projection record
(view `projections`).

## Summary

| Cause | What went wrong | Decision |
| --- | --- | --- |
| 1a Decoding | A parameter value holding `</parameter>`, `</function>` or `</tool_call>` ended the call early, so every write of the bubble's code failed (#48–#60) | A value ends only at a `</parameter>` that structure follows: `Until` takes a fixed set of markers, and the grammar, the XML-to-JSON conversion, the splitter and npcd share one rule |
| 1b Visual | The GUI finds calls by searching the streamed text for tags, so it has 1a's blind spot, pairs results by position, and drew failed rounds as progress | The GUI draws from call events on the live stream and from call records in the history; it parses no text |
| 2 `in_context` | A read answered with a pointer, not the page; the model could not tie it to the placed content and looped (#23–#62) | The `in_context` reply and the locks go; every read runs for real; provenance skips files the turn's loop has read |
| 3 Delegation | "Launch a research task" went to `sub_run`, which cannot run in a dialogue, so the research ran inline (#4–#10) | `sub_run` is removed; dialogues, chats and tasks start tasks, which nest one level |
| 4 Forced note | The forced `journal_note` was the only call that landed in failed rounds, and was forced with nothing to say | A forced note is followed by a forced call; failed calls answer with an error; a note is forced only after 200 tokens of reasoning |
| S1–S5 | Smaller issues: sandbox scope, repository-wide grep, a stale tool schema, a cut-short view, a context block on every connector call | All in scope |

Two steps come before building: the live capture in 1b, and the turn #1
projection check in Cause 3.

## Cause 1a — Decoding: markup in a value ends the call

Qwen3.8 (`host::MODEL`) writes calls as function blocks: XML-shaped, with each
parameter's value raw and unescaped. The only thing that ends a value is the
text `</parameter>`. So a value that contains that text, or `</function>` or
`</tool_call>`, is cut short. The bubble's code and tests had to contain all
three, and every write of them failed (#48–#60). The model blamed JSON escaping
(#50–#52), then payload size (#55–#56); the user named the cause at #60.

**Where it is cut**

| String in the value | What ends early | Where |
| --- | --- | --- |
| `</parameter>` | The grammar's value span, which then forces the next parameter or `</function>`; the rest of the content is never decoded as content | `Terminator::Until`, `candle-conversation/src/stencil/terminator.rs:94` |
| `</function>` | XML-to-JSON conversion, `after.find(FN_CLOSE)` | `candle-conversation/src/stencil/function_block.rs:109` |
| `</tool_call>` | The splitter's `find(CLOSE)` | `zend/src/tool_call_split.rs:51` |

npcd's `engine/act.rs` reads the same format the same way.

**The rule.** The work happens in string space: `</parameter>` is not a special
token in any vocabulary we target (`terminator.rs:91–92`), and the model may
spell it in one token or several. A value ends at a `</parameter>` that
structure follows. Any other `</parameter>` is content.

**Grammar: `Until` takes several markers.** `Until { marker }` becomes
`Until { markers }`, each marker being `</parameter>` + a separator + a tag:

| Separator | Why |
| --- | --- |
| `\n` | Qwen3.8's template; nearly every call |
| `\r\n` | Windows line endings |
| `\n\n` | a blank line between parameters |
| ` \n` | a trailing space before the newline |
| none | a tag hard against the close |

The tag is `<parameter=` or `</function>`, so there are at most 10 markers. All
share the 12-byte prefix `</parameter>`, so matching costs nothing until that
prefix appears, then checks a short tail against a small trie built once per
span.

1. **The whole close is consumed, and the span forks.** Pushing the separator
   and tag back to the next node cannot work: the stencil commits each token as
   it decodes, and lookahead push-back holds only within one token, while the
   tail arrives over several. So a function block's value is a
   `NodeSpec::Element` that consumes `</parameter>`, the separator and the tag,
   and `TerminatorState::fired` says which tail it was; the span goes to that
   tail's fork (`FreeTextSpan::forks`), the continuation with the tag already
   written (`ToolTreeBuilder::after_written`).
2. **No marker completes means content.** `</parameter>` followed by anything
   else is written into the value, and matching starts again.
3. **The tree supplies the markers.** If a required parameter is still missing,
   `</function>` is not offered; if every parameter is written, `<parameter=` is
   not. A branch the tree would refuse must never fire, or a real close would be
   read as content.
4. **`completion()` writes no second close.** A value cut short by EOS or the
   hard limit is finished with `</parameter>`, unless it already ends on it with
   layout that, before the successor's newline, is a separator; any other
   layout is the value's, and `</parameter>` closes it again.

**Extraction uses the same rule.** `function_blocks_to_json` finds each value's
end by the same marker set, not `find("</parameter>")` or `find("</function>")`.
It takes the text raw and lets `serde_json` escape it into the JSON string.
**Which markers end a value is the tree's, so a reader asks the catalog.** One
statement of the field order (`stencil/field_order.rs`) decides what may follow
each argument — the next element, the close, or either — for the grammar's
gates and for `element_close::follows_in`, which the readers holding the
catalog the grammar was compiled from use: the conversion, the passthrough
splitter (the client's specs) and the GUI's part splitter (the daemon's
catalog). A last value holding `</parameter>\n<parameter=` then reads back
whole, as it decoded.

**The splitter, in the same change.** `zend/src/tool_call_split.rs` counts a
`</tool_call>` only after a real `</function>`, by the shared rule.

**npcd, in the same change.** `npcd/src/engine/act.rs` moves onto the shared
reader, and it reads with the catalog its grammar was compiled from: `parse`
and `raw_calls` take `&[ToolSpec]`, and every caller passes the list its own
decode was built from — the acting turn and the probe the list
`specs_within` built the turn's grammar from (`mind::act_specs`), a question
its one `answer` spec, a composition sitting `draft` plus `mend` with the
piece's anchors, the authority's forwarded call the whole catalog (the
sender's `Within` never crosses the wire). A character's last value holding
another element's tags therefore reads back whole, as it decoded
(`a_last_value_holding_another_elements_tags_reads_whole`).

**Tests, byte for byte**

- A value holding `</parameter>`, `</function>` and `</tool_call>` mid-text
  decodes and extracts intact.
- Each separator closes a value; a close hard against prose still closes.
- A `</parameter>` followed by a branch the tree refuses is content.
- Content holding a full marker the tree offers there, such as
  `</parameter>\n<parameter=` before a later element, ends the value; after the
  last argument it is content, for the grammar and the catalog's readers alike
  (`a_block_reads_back_as_its_grammar_closed_it`).
- Every copy of the go-again choice after a forced first call is held, one per
  way its last value can close (`every_copy_of_the_choice_after_a_function_block_is_held`).

**Design doc:** `docs/stencil_tree.md`, in the same change.

## Cause 1b — Visual: the GUI parses markup to draw calls

The GUI finds calls by searching the raw text for tags. `renderContent`
(`zend/web/index.html:1200–1251`) re-parses the whole message on every token.
Four problems follow, all from that one root.

| # | Problem | Where | Effect |
| --- | --- | --- | --- |
| V1 | Complete calls split by a lazy `/<tool_call>([\s\S]*?)<\/tool_call>/`; the streaming call found by `lastIndexOf('<tool_call>')`; `<think>` blocks pulled out first over the whole text | `index.html:1208–1231` | A value holding `</tool_call>`, `<tool_call>` or `<think>` draws broken cards, even after 1a |
| V2 | Results pair with cards by position (`toolIdx`) | `index.html:1245` | One wrong split moves every later result onto the wrong call |
| V3 | The journal-note bubble does not show live. `noteOf` handles JSON, partial JSON and `<function=journal_note>`, and the Playwright test passes, but only against the mock's seeded call | `index.html:1117–1137` | The live text differs from the mock. Candidates: the forced note's layout; a bare call with no wrapper; markup changed between decode and history |
| V4 | A call that failed to parse is still drawn as a card | `toolCard` | Failed rounds look like progress |

**Decision: the GUI draws from call events, never from text.**

1. **Capture first.** One live capture of a turn's raw content, stream and
   stored history side by side, to pin V3. It shows whether anything changes the
   markup between decode and the history, which the event path could otherwise
   hide.
2. **Events on the existing live stream.** The daemon sends call events on
   `/v1/conversations/:id/live` and the chat stream, beside the text deltas and
   ordered with them.
3. **History returns records only.** The history API returns call records for
   every turn, so a reload draws what the stream drew. Turns stored before this
   change are rebuilt once from their text on the server. The GUI's text parsers
   (`noteOf`, the `<tool_call>` and `<think>` regexes) are deleted, with no
   second path.
4. **The raw text keeps its markup.** The model needs it when a turn is
   replayed; the GUI stops reading it.
5. **Reasoning moves to events in the same change,** so the GUI keeps no text
   parser at all.

**Event schema**

Today the stream sends the reply's text as OpenAI `delta.content` chunks, markup
included, beside named events: `tool` (`phase`, `tools`, and `results` /
`tokens` matched by position; `ToolStatusOut`, `zend/src/session.rs:213`),
`think` (`tokens`, `done`), `prefill`, `status`, `projection`, `turn_error`.

After the change, `delta.content` carries **prose only**: the reply's visible
text, with no call or reasoning markup. Calls and reasoning arrive as one new
named event, `part`. It replaces `tool` and `think`. `prefill`, `status`,
`projection` and `turn_error` stay as they are. Everything is on one stream, so
the GUI draws segments in the order they arrive.

| `op` | Fields | Meaning |
| --- | --- | --- |
| `open` | `id`, `kind` (`call` or `think`), `tool` (calls) | A call or a reasoning block starts |
| `arg` | `id`, `name`, `text` | The next chunk of a call's parameter `name`, as decoded |
| `text` | `id`, `text`, `tokens` | The next chunk of reasoning; `tokens` is the running count |
| `close` | `id`, `ok`, `error` (on failure); for `think`: `tokens`, `exact` | The call parsed, or failed with its reason; the reasoning finished at its length |
| `run` | `ids` | These calls have started executing (was `tool` `phase: running`) |
| `result` | `id`, `result`, `tokens` | A call's result and its length in tokens in the context (was `tool` `done`, by position) |

An `id` is `p<n>`, the n-th part of the reply, the same in the stream and in
the history. So a reload draws what the stream drew, and a result always lands
on its own call.

**As built**, the events come from a splitter on the server
(`zend/src/parts.rs`) that reads the streamed text with the rule the grammar
closes values on (`element_close`), not from inside the stencil — the same
events, with no engine plumbing. A history message's call `args` is a list of
`[name, value]` pairs, in the order written. The job panel's task transcripts
are still drawn from their text (`renderContent`): they come from a worker's
stored turns, which have no parts yet.

**History shape.** Each message in `GET /v1/conversations/:id` carries `parts`,
in order:

- `{"kind":"prose","text"}`
- `{"kind":"think","id","text","tokens","exact"}`
- `{"kind":"call","id","tool","args":{name: value},"ok","error","result","tokens"}`

`toolOutputs` and `toolTokens` go: a call carries its own result and length. The
bubble is a `call` part whose `tool` is `journal_note`, drawn from `args.text`,
or from its `arg` chunks while it streams.

**What each problem becomes**

- V1: no splitting; a card is a call record.
- V2: results pair by call id.
- V3: the bubble is the `text` chunks of a `journal_note` call, drawn as they
  arrive.
- V4: a failed call is drawn as failed with its reason, from Cause 4's error
  result. A note-only round cannot occur (Cause 4, decision 1).

**Tests**

- Daemon: a decoded call emits `open`, `arg` chunks and `close` in order, and
  the chunks join to the parsed value byte for byte, including a value holding
  `</tool_call>` and `<think>`.
- Daemon: the history's records for a turn equal the events its stream sent.
- GUI (Playwright, against the mock fed the same events): a note streams into
  the bubble; a failed call draws as failed; results land on their own calls
  when a value holds `</tool_call>`.

**Design doc:** `docs/zend_ui_redesign.md`, in the same change.

## Cause 2 — Reads answered in_context

**Decision: remove the `in_context` reply. Every `file_read` runs for real, and
provenance skips files the current turn's loop has read.** The placed content
was there, as earlier tests showed, but in the short term the model did not tie
the pointer to it. Fixing that is a bigger problem than the saving was worth.
Provenance already reduces reads, and improving its hit rate is the long-term
lever.

**What happened**

The fast path answered a read with `{"status":"in_context","anchor":…}` when a
finished `code_reading` conversation held the file whole, and locked that
reading into the working set (`zend/src/fast_path.rs:225–239`). The prompt rule
says to re-fetch content that is not there, but a re-fetch hit the same fast
path, so the model looped:

| Turn | File | What the model said |
| --- | --- | --- |
| #23–#24 | `zend/web-tests/zend.spec.js` | "keeps returning in_context with no body" |
| #44 | `zend/web/index.html` p1, p7, p8 | "never in a tool result I have" |
| #45–#46 | `zend/src/tools.rs` p1 | retried with `line=68`, the one read that always runs (`fast_path.rs:219`) |
| #61–#62 | `zend/web/partial-call.js` | "never actually seen its contents"; refused to edit blind |

**Why removal is sound**

- The page arrives where the model trusts it most, as a tool result in its own
  dialogue, and an edit gets the exact text.
- No pointer means nothing to resolve and no loop.
- The lock tier exists only for served reads (`docs/zend_working_set.md` §1),
  so its whole release machinery goes with it.

**What changes**

| Piece | Where | Fate |
| --- | --- | --- |
| Serving a read: `screen`, `hit`, `read_hit`, `served_response`, `Served` | `zend/src/fast_path.rs`; call at `zend/src/session.rs:5176` | Removed |
| Locks and their release: `departed`, `round_release`, `holding`, promise marks | `zend/src/working_set/promises.rs`; `session.rs:5054`, `session.rs:5412` | Removed |
| `file_conversation`, `Coverage` | `fast_path.rs`, `fast_path/coverage` | Kept (used by seeds and audit); moved into the working set |
| The `in_context` prompt rule | `zend/src/prompts/projection.yaml:482–492` | Anchor and fade guidance kept for provenance content; the sentence about a tool answering in_context removed; section renamed |
| Design doc | `docs/zend_working_set.md` §1 (locks), §4.6 (release) | Rewritten to provenance and seeds, in the same change |

**Loop exclusion: no duplicates**

A file read in the current turn's tool rounds is already verbatim in the
dialogue, so provenance must not place it again.

1. **The set.** The session collects the files read (`file_read`, by workspace
   path) in the current turn's rounds and hands them to provenance's admission
   step.
2. **Block entry; never evict.** A file in the set cannot enter provenance. A
   file already placed when the model reads it stays where it is: evicting it
   closes the gap and re-prefills every later member and the whole dialogue, for
   one file's tokens.
3. **Whole file.** Provenance places whole ingest conversations, so a read of
   any page excludes the file for the loop.
4. **Lifetime.** The set ends with the turn's loop. At the next user turn
   provenance may bring the file back, since its read may be summarised away by
   then.

**Tests**

- A `file_read` of a file a finished `code_reading` holds whole returns the page
  body, never `in_context`.
- A file read in a round does not enter provenance for the rest of the loop, and
  may enter again at the next user turn.
- A file already in provenance when it is read keeps its place.

**Afterwards: watch the repeated-read rate** (the same page read twice in a
conversation). It is what the fast path saved, and a direct measure for the
provenance hit-rate work.

## Cause 3 — The research ask did not reach a worker

The first request was "launch a research task on rate.rs". The right tool is
`repo_task_start` with `role: researcher`, whose first trigger question is
"start a research task on …" (`zend/src/prompts/tools/repo_task_start.yaml`).
The model chose `sub_run` instead, which starts parts of a task delegated
through the MCP server. It was refused (#4), and the model then did the research
inline: all 30 pages of `rate.rs` over #5–#10.

1. **`sub_run` is removed.** Parts duplicated what tasks with roles already
   do: a task that wants a piece done apart starts a task of its own. The tool,
   its runner (`zend/src/mcp/parts.rs`), the `subagent` tool in zend-tools, the
   lineage of parts below a worker and their journals, and the guides' and
   skills' verifier parts are gone; a verifier is a reviewer or researcher task.
2. **Tasks nest one level.** A dialogue, a chat, and a task Claude or a person
   started each have the `repo_task_*` tools. A task a task started
   (`Worker::by_task`) has them withheld from its signatures and its grammar
   alike (`zen_tasks::level_of`, `withhold_by_level`,
   `SelectionState::withhold_members`), and `repo_task_start` from it is
   refused. The one level keeps the task that started the others in the loop:
   no chain of tasks can feed its own output back as the next one's brief.
3. **The `repo_task_*` tools stay scored, not mandatory.** `repo_task_start`'s
   trigger questions already include "start a research task on …". Check the
   `projections` view of turn #1 for whether it was shown; if it was not, tune
   its trigger questions.

**Tests**

- A nested task holds out the task tools, and only those its mode offers; a
  dialogue and a task at the top hold out nothing
  (`zen_tasks::a_nested_task_holds_out_the_task_tools`).
- A conversation's level is read from its worker record: a dialogue and a chat
  are conversations, a task started from one is a task, a task a task started
  is nested, and a record that no longer reads is nested
  (`a_level_is_read_from_the_record`).
- A withheld member is never projected
  (`projection::tests::a_withheld_member_is_never_projected`).
- A task a chat started on its thread neither blocks the chat's next message
  nor answers it (`chats::talk::a_send_while_zen_answers_is_refused`).

**Design doc:** `docs/zen_tool_reach.md` §6.3, in the same change.

**The large context is by design.** At #61 each decode carried about 250k
tokens: `scopes` 133k–145k (the working set's placed files), the dialogue 96.8k,
tools 5k–8k. Holding long context as the work runs forward is intended, so the
model does not lose track. It is not a fault to fix.

**Tool projection: `file_edit` is mandatory (done, `934225224`).** Across #61's
39 projection points the tools shown changed about every 64 tokens. `file_read`
and `file_grep` are mandatory and stayed shown throughout; `file_edit` came and
went. It now projects on every turn (`zend/src/prompts/tools/file_edit.yaml`,
the pinned set in `zend/src/tool_def.rs`).

## Cause 4 — The forced note hid failed rounds

In a conversation with a journal, `journal_note` is mandatory and the stencil
forces it as the first call of any round that opens a tool call
(`docs/zend_journal.md` §5.2; `zend/src/session.rs:627–670`). The note keeps the
thinking's conclusions as visible text, so they outlive the `<think>` window.
The design stands; three things change.

**What happened**

| | What happened | Turns |
| --- | --- | --- |
| 4a | The real call failed to parse (1a) and the note was the only call that landed, so the round looked like progress while the notes narrated each retry | #48–#52 |
| 4b | A forced note with nothing to say: its text was "…" | #38 |
| 4c | Back-to-back notes nearly repeating each other | #1, #2 |
| 4d | About 60 of the 156 calls were notes, each answered `noted` | throughout |

**Decisions**

1. **The stencil forces a second call after the note.** A round that opens with
   the forced `journal_note` goes on to at least one real call; the grammar does
   not offer the round's close straight after the note. A note-only round can no
   longer be decoded, which removes 4a at its source.
2. **A failed call gets an error result.** A call that does not parse, or that
   the grammar ended early, is answered with an error naming the reason: the
   call, the parameter, and where it ended (for example, the call ended at
   column 556 inside parameter `content`). The model never reads a note's
   `noted` plus silence as success. The same failure reaches the GUI as the
   call's `close` with `ok: false` (1b).
3. **A note is forced only after enough reasoning.** The stencil forces the note
   only when both hold: the round opens a tool-call block, and the round's own
   reasoning (the `<think>` span that opened it) ran to at least **200 tokens**.
   A final answer, or a round with less reasoning, is never forced. This removes
   4b and much of 4d. `journal_note` stays mandatory, so the model can still
   note by choice.

**Tests**

- After a forced note the grammar refuses the round's close and accepts only a
  further call.
- A round whose reasoning is under 200 tokens is not forced to note; one at 200
  or more that opens a call is.
- A call cut short, or one that fails to parse, is answered with an error naming
  the call, the parameter and the position, and the stream's `close` for it
  carries `ok: false`.

**Design doc:** `docs/zend_journal.md` §5.2 is rewritten in the same change:
forced only on both conditions, always followed by a call, failed calls
answered with an error.

## Smaller issues — all in scope

| # | Issue | Evidence | Fix |
| --- | --- | --- | --- |
| S1 | `code_session_exec` keeps earlier declarations, so a second `const src` collided | #57 | Its description says the session persists and every earlier name stays bound, so reuse or reassign rather than redeclare |
| S2 | A `file_grep` with no `prefix` searched the whole repository and wasted a turn | #62 | Dropped: a required prefix would break `repo: "*"` searches and force a guess for "where is X defined" |
| S3 | The `timeline_read` schema the client sees lacks `view` and `turn`, though the server accepts them, so a client following its schema cannot read a turn | this review | Cause found: the server advertises `tools.listChanged: false` (`mcp-server/src/protocol.rs`), so a client keeps the listing it took when it connected — this one predates `view` (`bf1fd4d2f`, 2026-11-04). The fix is to send `notifications/tools/list_changed` when a member's tool set changes, which needs a server-to-client channel; open |
| S4 | The `calls` view cuts each argument short, so the failed writes' raw text was not visible | #50, #55 | Not a defect: `view: turn` reads a turn whole. The `calls` view says so in its last line, naming the turn to read in full |
| S5 | Every connector call appends the ~2k-token user-context block, and the conversation id changes between calls (`…-0159-claude-code`, then `…-1109-claude-code`) | every Zen call in this review | The block is appended to the first call of a connector session only. Fix why a session's calls each open a new conversation. Test: two calls in one session, the block on the first alone, one conversation id |

S5 costs the most: in a long session it adds tens of thousands of tokens. S3 is
next: it hides tools' arguments from every client.

## The work

| # | Work | Test | Design doc |
| --- | --- | --- | --- |
| 1a | `Until` with a marker set; one shared rule for the grammar, XML-to-JSON, the splitter and npcd | a value holding every closing tag decodes and extracts byte for byte | `docs/stencil_tree.md` |
| 1b | `part` events on the live stream; `parts` in the history; the GUI parses no text | stream events equal the history's records; the bubble streams from `arg` chunks | `docs/zend_ui_redesign.md` |
| 2 | Remove the `in_context` reply and the locks; every read runs for real; provenance skips files read in the turn's loop | a held file's read returns its body; a file read in a round does not enter provenance for the loop | `docs/zend_working_set.md` §1, §4.6 + the `projection.yaml` rule |
| 3 | `sub_run` removed; tasks nest one level | no `repo_task_*` in a nested task's tools, grammar or prompt; a dialogue's, a chat's and a top task's hold them | `docs/zen_tool_reach.md` §6.3 |
| 4 | A forced note is followed by a forced call; failed calls answer with an error; notes forced only after 200 tokens of reasoning on a round that opens a call | no round closes after its note; a cut call gets an error naming where; under 200 tokens no forced note | `docs/zend_journal.md` §5.2 |
| S1–S5 | The smaller issues | S2's refusal; S3's schema equality; S5's once-per-session block | `docs/tool-system.md` for S1, S2 |

**Before building**

- [ ] 1b: one live capture of a turn's raw content, stream and stored history
  side by side.
- [ ] Cause 3: the `projections` view of turn #1, for whether `repo_task_start`
  was shown.
