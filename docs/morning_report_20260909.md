# Morning report — 9 September 2026

**The tools are integrated, the messaging system is real, and you can talk to a
character from the console.** The daemon is running the new build with zero
errors and zero warnings in its log.

The one thing worth reading first is the transcript at the bottom: a character
answered a message sent from outside the world, on its own schedule, using the
same act it would use to answer anybody.

---

## What was asked, and where it stands

| | |
|---|---|
| Messaging state kept in the world module, apart from the lore | done — it already was; the caps and delivery are new |
| A cap on messages per direct thread and per group | done — 64 and 128, with the index shifting that trimming needs |
| NPCs sending **real** messages to other NPCs | done — this was the gap, and it was total |
| Chatting to a character through the GUI | done — API, console tab, and verified live |
| Unit tests for every tool call | done — plus two audits that found six inert acts |
| Well described, so they inject properly | done — 25 descriptions were too thin to choose on |

**1,301 tests passing**, clippy clean at `-D warnings`, suite runs in 4.4 s.

---

## The messaging system

### Messages went nowhere

`sim/phone.rs` held threads and members correctly, and `message` wrote into
them. Nothing read them. `messages_waiting` and `waiting_names_for` were
consumed at **no call site in the daemon** — so the sender was told *"they will
see it when they next look"* and the recipient was never told there was
anything to look at.

The cause is structural rather than an oversight: what a body perceives comes
off the map through `npc_map::Attention`, which keeps a cursor per reader. A
phone reaches somebody who may be on the other side of the world, so the map has
nothing to say about it and there was no second path.

There is now. `environment::deliver_messages` runs in the same sweep as
perception — one batched pass per moment, no route by which anything arrives
outside it. Each thread keeps a `delivered` cursor per member, exactly as the
map keeps one per reader, so a message is handed over **once**: without a cursor
the sweep either re-delivers the whole thread every moment or delivers nothing,
and it was delivering nothing.

A message arrives as its own `EventKind::Message`, not as `Speech`. Speech is
delivered by place and overhearing is normal; a message reaches one thread and
nobody else, so it carries the thread rather than a room — and the thread is
named the way *that* character calls it, which is the argument it needs to
answer. A direct message is `URGENT` and a group is `NORMAL`: somebody
addressing you alone has the same claim on you as being spoken to in a room, and
a group of six should not preempt whatever you were doing six times.

### Caps

`KEEP_DIRECT = 64`, `KEEP_GROUP = 128`. A group holds more because six voices
land in one record, not because its history is worth more.

The part that needed care is that **two indices point into the message list** —
where each member joined, and how much each member's mind has been handed.
Trimming the front without shifting both leaves valid numbers aimed at the wrong
messages: a newcomer would be shown the argument it joined after, or somebody
would be re-handed a backlog. That is the kind of wrong nothing reports, so
`Thread::say` shifts both and a test pins it.

---

## Talking to a character from the console

**A person messaging a character is a party on a thread, not a side channel.**
What the console posts lands in `sim::phone`, the character is told by the
ordinary sweep, and it answers with the ordinary `message` act from wherever it
is standing. A private pipe between a console and a mind would be a different
thing wearing the same word: the reply would not be the character speaking from
inside the world, and nothing else in the world could see it had happened.

- `POST /v1/npc/:nid/message` — say something. Answers with `can_reply`, which
  is false for a character carrying no handset, because a console that looks
  like it is working and never gets a reply is worse than a plain no.
- `GET /v1/npc/:nid/message` — the conversation, oldest first.
- `Runtime::message_npc` / `messages_with` hold the behaviour, so the HTTP
  handlers are thin and the logic is tested through the real engine.

The console has a **Messages** tab on the character page, above Interactions.
It polls rather than streams, and that shape follows from the design: a reply
arrives when the character's next turn comes round, because it is deciding to
answer rather than being queried. There is no event to stream — the thread
simply has one more line on it.

`/v1/interaction/*` is still stubbed. I read "the chat function" as the
messaging system you had just asked for rather than the substrate-forking
interaction system, because that is coherent with the rest and reuses all of it.
Say if you meant the other thing.

---

## The tool audit

Two new tests, and both found real defects.

**`every_act_in_the_catalog_reaches_an_implementation`** drives all 102 acts
with their own calibration examples. `is_of_the_body` says a name is *claimed*
by a dispatcher; it does not say the dispatcher does anything. All 102 reach
code — no stubs.

**`every_act_that_reports_success_leaves_the_world_changed`** is the sharper
one. It compares the serialised sim before and after every act that returns
`Did`. Six were reporting success and changing nothing:

- `structure_lay_out_scenes`, `structure_test_the_want`,
  `structure_find_the_slack` — a character did the reading, said so, and the
  next one to pick the piece up found no sign of it, so the work was done again
  and again with nothing accumulating. Each now lands a verdict on the ledger.
- `record_tidy_index` — checked the thing existed and returned prose. The index
  it claimed to correct was exactly as wrong afterwards. It now writes the entry.
- `stores_put_back` — returned `Did` on *every* failure, so a character could
  rack the same imaginary object every turn and be told it worked each time.
- `recall` is exempt and correct: it moves a body, which is the map's business.

**Descriptions.** A one-line description is not a style problem — the name is
four tokens and the grammar admits every act equally, so the description is the
whole of what the model chooses on. A floor now enforces 60 characters for an
act and 20 for a parameter, which caught **25** too thin to choose on. All
rewritten.

---

## Also fixed, earlier in the same run

**The grammar-killing duplicate.** You saw *"That did not come out as a call"*
in the GUI. A character had opened two groups it had both called `none`;
`names_for` did not deduplicate; two identical arms tokenize to one common
prefix with nothing left over, which the stencil refuses as `EmptyArm` — and
that fails the **whole** grammar, so every turn free-decoded and emitted
malformed JSON. Forever, because nothing it could do removed the duplicate.
`live_set` now deduplicates at the single chokepoint every world-enumerated
branch passes through, and `open_group` refuses a name already in use.

**A successful read threw away what it read.** `record_act` returned
`act.summary()` for every `Did`, so `file_read` put `file_read — path` into the
character's window and discarded the document. `body::ANSWERS` now keeps the
world's line for the ten acts whose product *is* what they say — including
`read` and `scan`, which had the same defect before I touched anything.

**Worlds hosted directly had no documents.** Only `host_authored` set a bench
root, so the same world reached two ways could or could not edit the mind.
It now comes from the `Mind` handle `Runtime::new` is already given.

---

## Live, on the running daemon

```
layer eras:    0 written, 13 unchanged, 0 failed (documents)
layer stories: 0 written,  7 unchanged, 0 failed (documents)
turn grammar armed over 102 acts, 2 per turn
cast: 3 character(s) awake, 3 standing in a world
engine ready in 12.0s
```

`/v1/tools` serves **102 acts**. Log has **0 errors, 0 warnings**. Logs now go
to `npcd.log` / `npcd.err.log` — previously the daemon was launched detached
with no redirect and its output went nowhere, which is why the first thing I
could not do was read the logs you asked about.

### The transcript

```
POST /v1/npc/7875766493935720493/message
  {"text": "Are you at a terminal right now, and what are you working on?"}
  → {"can_reply": true, "to": "Wailen Wylde", "waiting_for_them": 1}

GET  /v1/npc/7875766493935720493/message
  Johnathan Sharratt: Are you at a terminal right now, and what are you working on?
  Wailen Wylde:       I am in the stacks. I have no station claimed.
```

That reply went the whole way round: posted from outside, onto a thread in the
world, delivered to the character's mind by the perception sweep, decoded under
the 102-act grammar, chosen as a `message` act, and written back onto the same
thread.

---

## What I did not do

- **Nothing is committed.** Every change is in the working tree.
- `/v1/interaction/*` remains stubbed — see above.
- The console's mind browser still cannot see `layers/eras` or `layers/stories`.
  Its `Section` enum in `mind/address.rs` is a hardcoded list, and it still
  carries `agency` and `beliefs` pointing at directories that no longer exist.
  NPCs can edit the new layers; you cannot see them in the console. Two lines
  each to add, and a decision on the two dead ones.
- The character that was stuck in the `none` loop cleared on restart, because
  `Sim` is not persisted. The fix means it cannot recur, but no migration was
  needed and none was written.
