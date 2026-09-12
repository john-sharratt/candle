# Follow-up report — 9 September 2026

The three open items are done. **1,313 tests, clippy clean, daemon healthy with
zero errors and zero warnings**, and every piece verified against the running
daemon rather than only in tests.

---

## 1. The console's mind browser

`layers/eras` and `layers/stories` were invisible to the console because
`mind::address::Section` is a fixed enum and did not name them. Added.

`Agency` and `Beliefs` are **removed**. They pointed at `layers/agency` and
`layers/beliefs`, which stopped existing when those layers came out of
`projection.yaml` — so the console listed two sections that could never hold
anything and reported `count: 0` for both. Putting them back is the enum, the
`SECTIONS` list and four `match` arms the compiler will demand, so this is a
cheap decision to reverse if those layers return.

Verified live:

```
GET /v1/mind/list       → canon eras stories memory responses moods
                          characters worlds settings
GET /v1/mind/list?id=eras     → 13 documents, The awakening … The zenling plague
GET /v1/mind/list?id=stories  →  7 documents, The attrition finding … What vasko knew
```

---

## 2. The interaction system

`/v1/interaction/*` was six stubs returning `no_engine`. All six are real, and
the console's existing operator page drives them unchanged.

**An interaction is a session, not a fork.** The old doc comment said opening
one "forks the character's substrate and starts a decode loop", and it does not.
A forked substrate would give an operator a private copy of a mind to talk at:
nothing said in it would have happened, the character would not remember it, and
the rest of the world could not see it took place. What is said here goes
through the same door as everything else — parsed by `slash`, delivered by the
scheduler, perceived on the character's next turn.

So the session holds only **who is present, in what mode, and since when**. The
conversation lives where everything else that happened lives.

- `POST /v1/npc/:nid/interaction` — open, or continue the one already open with
  this person in this mode. Two live sessions with the same person in the same
  mode is not a thing that can be true of a conversation.
- `GET /v1/npc/:nid/interaction`, `GET|DELETE /v1/interaction/:ix`
- `POST /v1/interaction/:ix/inject` — say something
- `GET /v1/interaction/:ix/stream` — SSE

**Mode is reach, not decoration.** `Mode` already drives `specs_within`, so
somebody on a voice call cannot be handed an object and a character in the room
with you can. Physical mode refuses a character with no body — there is nothing
to stand beside — and the messaging modes do not, because reaching somebody who
is nowhere near is the whole point of them.

**Idle is computed, not reaped.** `idle_remaining` is derived from the last
thing said, so a session nobody has touched is already over by the time anyone
asks and a daemon that was asleep does not wake owing anybody a sweep. Timeouts
follow the mode: five minutes standing in a room, fifteen on a call, a day on a
thread — a message thread left overnight is not a conversation that ended.

**The stream polls the scheduler's own ring** rather than taking a broadcast out
of `record_act`. A second path by which an act becomes observable is a second
path that can drift; the Pulse view reads this ring, so a session watching the
same rows is watching the same truth. There is no `narration` frame and its
absence is honest — this daemon renders an act *as* its line rather than
producing a separate account, so a narration frame would be prose nothing wrote.

### Verified live

```
POST .../interaction {"mode":"physical"}
  → interaction_id 17888990768710001, idle_timeout_secs 300, state live

POST .../inject {"line":"Wailen, come to the chronicle terminal and
                         tell me what the third era says."}
  → "Johnathan Sharratt says to you: Wailen, come to the chronicle…"

GET .../stream
  event: open
  event: act   {"tick":36,"tool":"move_to","intent":"the reading room"}
  event: tick  {"tick":36,"acts":1}
  event: act   {"tick":39,"tool":"move_to","intent":"the concordance table"}
  event: tick  {"tick":39,"acts":1}
  event: act   {"tick":42,"tool":"wait_for","intent":"someone_arrives"}
```

It was asked to go to the chronicle and it walked there, two rooms, then waited.

### A bug found on the way

The first inject rendered **`"you says to you: …"`**. `slash::parse` writes the
placeholder `you` for a speaker it cannot know, and nothing substituted it — so
the character was told an ungrammatical sentence that was also wrong about who
spoke, and that went to the model exactly like that.

`Parsed::spoken_by` now names the speaker, and both injects call it. **This was
live in `/v1/npc/:nid/pulse` before I touched anything** — every operator line
sent through the Pulse view had it.

---

## 3. Committed

Both repositories.

- **candle** — `ebf23d8a`, 106 files, +18,417/−424. On branch `npc-engine`,
  **not pushed**: the branch was already 8 commits ahead of its remote before
  mine, and resolving that is yours to decide.
- **battle-mind** — `e333d7b`, 21 files. Pushed to `origin/main`, since that
  remote exists for exactly this.

I removed a stray `bash.exe.stackdump` from the repo root rather than commit it.
`npcd*.log` is already gitignored.

---

## Where things stand

| | |
|---|---|
| Acts in the catalogue | 102, all reaching a real implementation |
| Acts that report success and change nothing | 0 (was 6) |
| Descriptions too thin to choose on | 0 (was 25) |
| Tests | 1,313, 4.5 s |
| Clippy `-D warnings` | clean |
| Daemon | running, 0 errors, 0 warnings |
| Ways to reach a character | messaging (own schedule) and interactions (present, in a mode) |

Nothing is outstanding from the morning list. The next thing you named was
high-level brain functions.
