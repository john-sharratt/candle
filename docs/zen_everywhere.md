# Zen everywhere, drones at rest

**Status:** design, agreed 2026-11-15. Built in the order of §8.

Zen stops being a body in a room and becomes the one mind that is always
running: it keeps watch over real things through **terminals**, does the small
work itself, and hands the long work to drones. A drone with no mission does
nothing at all and holds nothing on the card.

## 1. Why

On WhiteWorld (RTX 3090, 2026-11-15, eight characters) the engine ran nine
interaction slots and thirteen live recurrent stores. A decode forward with one
sequence took 300–600 ms; with four, 850–900 ms; prefill for the other
characters ran between them. The experts held 9.0 GB of a 32.7 GB set and hit
26–34% — the resident share — with 7,000–11,000 promotions a wave: eight
contexts at once spread across most of the model.

Almost all of that is characters with nothing to do. Every waking event sets a
character due at once (`Scheduler::deliver`), perception is pushed every moment,
and a standing task restates an idle drone's summons every 90 s
(`IDLE_AFTER_MS`). Each of those is a prefill over a long context.

## 2. What exists

| Piece | Where | Bearing |
|---|---|---|
| Zen is `takes_orders: false` — an **unordered** body; drones take orders | `tenant.rs:980`, `Missions::is_unordered` | The one switch every Zen/drone branch reads. No code names Zen. |
| Zen is `moves: attention` — no `move_to`, no lift, a `focus` teleport | `sim/mod.rs:139`, `enact.rs:676` | Half-way to no body already. |
| A conversation can be unloaded and rejoined from the log | `Minds::let_go` (`mind.rs:2097`) | Used by suspension; an idle drone uses the same path. |
| A held character: parked, `begin_tick` refuses, inbox fills and is cut on resume, reflect/dream/journal refuse | `Scheduler::suspend`, `follow_suspended`, `held::PER_SOURCE` | Exactly what idle is. |
| The table hands a free drone its work | `Runtime::assign_work` → `work::take_up` | Built (§8 step 1): no act of a drone's takes work. |
| The round goes only to unordered bodies | `take_up_round`, `rounds/<world>.yaml` | Zen's own idle work is already Zen's alone. |
| A conversation with a person puts an item on Zen's agenda | `mission_add` on the thread, `agenda/acts.rs` | A request becomes work for Zen today. |
| Real devices, worked through an agent on the home network | `effector-bridge`, `npcd/src/bridge/` | Sonos, eISCP receivers, Cast, LG webOS, Android TV, WiZ bulbs, wake-on-LAN; found by `survey_network`. |
| Coding tasks through zend | `engine/zen_task/` → zend `repo_task_*` as `service:zen` | Offered only when a carried mission grants them; no moon content does. |

And what does not: gateway logs, cluster and agent status and diagnostics have
no act (the `ops` and `network` mission templates name steps nothing can
perform); presence has no code; the device acts are offered only to a body
standing in a mirrored room beside a linked part, and nothing is linked;
webOS and Android TV have no acts of their own.

## 3. A drone at rest

**A drone is at rest when it carries no mission and none is lodged for it.**
At rest it is held exactly as a suspended character is, by the engine rather
than by its owner:

- it is not scheduled: no event makes it due, and the standing task says
  nothing to it — no summons to the table, no `NO_MISSION`, no idle thought;
- its conversation is let go (`Minds::let_go`): no slot, no K/V, no recurrent
  store, no index on the card;
- it does not dream and does not reflect;
- what is sent to it — a message, a channel post, a notice — waits in its
  inbox, bounded per source as a suspended character's is, and is read when a
  mission next wakes it. **Only a mission wakes a drone.**

**Missions are assigned, not collected.** A post names a drone (`for`) or names
none; a post that names none goes to a drone at rest, the one longest at rest
first, the moment it is posted. Assignment is `work::take_up`, which wakes the
drone with its mission in hand. `collect_mission` and the table summons go: a
drone is never awake without a mission to need them. A drone that reports is
handed the next waiting mission if there is one and goes to rest if there is
not.

The table's **hands** are the drones that are not suspended, at rest or not: a
drone at rest is a hand that can be woken. Work waits on the table only while
every hand is carrying.

A drone keeps its body and its stations in the moon. What it works at inside
the world is unchanged.

## 4. Zen

**Zen has no body: it is the moon.** It keeps a seat in the world's record —
the key its agenda and its posts hang from — at its home, which nothing else
enters and it never leaves. It has no movement act (`focus`, `recall` go with
`move_to`), is in no one's company, and is not perceived. Its prompt carries
the moon's levels and rooms as its own insides (`prompt::being`, the member
`identity::whole_key` names), not as a building it works in: told that, it
answered a person from "the command level by the order table".

**Zen sees the whole of the moon at once**, and no room of it: one situation
(`engine/whole.rs`) listing every room with somebody in it, who is there, and
what each is doing — carrying which mission and its next step, or at rest —
replaced when it changes and carried at idle salience, so it is read when Zen
next thinks and is no reason to think. Nothing that stirs in a room reaches
it. What *happens* is told as it happens:

- **a drone takes up work, or comes to rest** (`Runtime::tell_whole`);
- **a mission's outcome** — done, failed with its reason, a drone stuck;
- **what a terminal raises** — an event from the real world (§5);
- **a drone's message**;
- and a person, who speaks to it in a conversation of their own.

All of it arrives through the one inbox the messaging system already fills.
What lies outside the moon — the house, the machines — Zen learns by connecting
and looking.

**The watch is the engine's, and Zen is told what changed.** Looking needs no
mind; deciding what a change means does. Every quarter of an hour the world's
authority reads the cluster's standing and runs each machine's `health` check
(`Runtime::tend_watch`, `engine/terminal/watch.rs`), and raises the terminal —
which wakes Zen with the change in hand — only when the standing differs from
the last reading, or the check's report leads with a change. What moves on
every reading and means nothing by moving (how long ago a member started, what
it has yet to send) is taken out before the comparison. Zen's round is what
only a mind can do: reading what a Drone brought back. A person's request
(`mission_add`, from their conversation) is an item on its agenda: turn the
television on is connect, act, disconnect.

**The terminals are stated, never looked for.** The view of the whole names
each terminal that can be reached (`sim::terminals::reachable`, noted every
five minutes and whenever they are listed). The acts a body finds its way
about the world with — `scan`, `focus` (`tools::WAYFINDING`) — are never
offered to a mind with no body: it is shown every room as it changes, and
offered `scan` it scanned its own rooms when asked what was on the house
network.

**The house's devices become terminals without being asked for.** The house
network is surveyed for every world that holds a mind with no body
(`Runtime::tend_presence`), as it is for one that watches a handset, so a
light, a speaker or a television is among the mind's terminals once the
network shows it, rather than once the mind thinks to survey.

**Telling is not answering, and characters message.** A conversation's turn
that its character began — it has something to tell, and the party has said
nothing new — is the words alone, with no decision (`TalkTurn::telling`): put
through the decision, it answered the party's last question again ahead of
every new thing told. And only a person's line opens a turn of a conversation;
a line from another character is told to the mind as something that happened
(`environment::deliver_messages`), because two conversations answering each
other never end.

**What may be posted is the world's to state.** A world's missions file says
whether its order-giver may author templates (`authoring`, off unless said);
the moon's does not, and its one job is `coding-task`.

**Drones get the long work**: a large coding task, anything that needs long
reasoning or many turns overseen. Those missions grant the task acts.

## 5. Terminals

**A terminal is a real thing with an adapter behind it.** It is not a part of
the moon's map; the stations drones work at are those, and stay.

Four acts, offered to an unordered mind alone:

| Act | Does |
|---|---|
| `terminal_list` | Every terminal Zen can reach now: its name, what it is, whether it answers. |
| `terminal_connect {terminal}` | Connects. Zen holds **one** connection; connecting elsewhere replaces it. |
| `terminal_disconnect` | Lets it go. |
| `rest` | Stops, until whatever reaches Zen next wakes it. |

**What a terminal answers is read at once, and `rest` is how Zen stops.**
Every terminal act but `terminal_disconnect` and `rest` earns the turn after
it (`engine::terminal::acts::answers`, `Runtime::arm_followup`): a reading is
in the act's answer and nowhere else, and whoever asked is waiting to be
told. So a mind that has nothing to do with an answer needs an act that ends
the run, and `rest` is it: it names no condition and no time, earns no turn,
and leaves Zen unscheduled exactly as a message sent does — a message, a
terminal raising a change or a Drone's report wakes it. It is refused once
while what last reached Zen has had no act of Zen's since
(`Scheduler::acted_since_news`), so it cannot be taken in the turn a request
arrives. A reading taken again that comes back the same is a loop, and the
loop guard strikes it (`engine::loopguard`, an act that names nothing is kept
as what it came back with); `rest` is never struck.

Disconnected, Zen's acts are the first two and `rest`, its table and agenda
acts, and its messaging. Connected, it also has **that terminal's verbs**: every terminal
act is `Availability::Granted`, and a mind's grants are the three above plus
what the terminal it is connected to offers (`engine::terminal::grants_of`),
so an unconnected terminal's verbs are not in the grammar at all. The
house's own device acts (`sonos_…`, `receiver_…`, `cast_…`, `wiz_…`,
`pair_…`) are bound to the devices of their kind, and a connection is what
puts a device in that set (`Within::at_terminals`). Connecting answers with
what the terminal is and what can be done there
(`engine::terminal::welcome`).

The terminals, each by the name `terminal_connect` takes
(`engine::terminal`, enacted in `authority::terminals`):

| Terminal | Behind it | Verbs |
|---|---|---|
| **`the lights`** — one, over every WiZ bulb the house network has shown | `effector-bridge`'s WiZ profile, through an in-network agent | `wiz_status`, `wiz_on`, `wiz_off`, `wiz_dim`, `wiz_white`, `wiz_colour` — on one bulb by its name, or on every one at once as `all the lights`, asked together and answered in one report |
| **a device, by the name it goes by** — each Sonos player, AV receiver, Cast device and television; never a bulb | its profile, through an agent | that kind's acts. A television: `tv_status`, `tv_on`, `tv_off`, `tv_volume`, `tv_louder`, `tv_quieter`, `tv_mute`, `tv_unmute`, `tv_play`, `tv_pause`, `tv_stop`, `tv_launch`, `tv_youtube`, `tv_input`, `tv_button`, `tv_show` — an LG all of them, a Google television the ones its remote has a button for. One not yet paired offers `pair_device` / `pair_code` and nothing else |
| **`the house network`** | this member's own survey of it; an agent for the wake | `wake_device`, `device_rename {device, called}`, `presence_watch {device, who}`, `presence_forget {device}`; and `survey_network`, which is offered and works whatever the mind is connected to, or connected to nothing — it and `terminal_connect` are in every turn's tools |
| **`the cluster`** — on a member of one | zend's `/v1/mcp` as `service:zen`, the channel `zen_task` uses; each act one named tool (`terminal::cluster_call`) | how it stands: `cluster_status`, `cluster_services` (`service_status`), `cluster_member {node}` (`member_status`), `cluster_memory {node}` (`member_memory`), `cluster_audit`; its logs: `cluster_logs {grep, level, since}` (`log_grep`), `cluster_log {source}` (`log_read`); its jobs, started and seen through: `cluster_jobs` (`job_list`), `cluster_job {job}` (`job_status`), `cluster_job_read {job, page}` (`job_read`), `cluster_job_stop {job}` (`job_cancel`), `cluster_job_start {repo, program, args}` (`repo_job_start` on `main`, one program, never a shell line). Nothing that decides for the cluster — a rollout, a restart, a deploy — and nothing that writes code, which is a Drone's task |
| **a machine, by its host name** — each machine an agent runs on: BlackWorld and WhiteWorld | that machine's own agent | `machine_check {check, args}` — the checks below |

**A device belongs to the house, not to a room.** The devices a room's parts
stand for are that room's and are worked from it by a body, as before. A
terminal's devices are kept by hardware address in the world's state
(`sim::terminals`): the name each goes by, what a pairing earned for it, and
whether the last surveys saw it. A device that changes address is the same
device; one that calls itself by a code is given a name a person would use
with `device_rename`.

**The lights are one terminal because they are worked together.** Turning
the house off at night is one act over all eight, not eight connections. A
mode is a verb of this terminal that keeps running after Zen disconnects —
the adapter drives the bulbs, not a turn of Zen's — and has a verb to stop
it; the patterns driven from an audio input, for a set played live, are
modes to be added here.

A cluster terminal and a machine terminal are read-only. Deploying,
restarting and rolling out stay with people.

### A machine's checks

**Named checks, never a shell.** Zen reads logs, and logs carry text from
outside; a command line on a machine would let what it reads decide what it
runs. A check does one thing, takes validated arguments, changes nothing, and
answers with a short report.

**The list is the machine's own, and it is open** (`agent::checks`). A check
is one `*.yaml` file in the `checks/` folder under the agent's own:

```yaml
name: logs
says: One of this machine's event logs, searched for a word or phrase.
args:
  - name: log
    values: [System, Application, Security, Setup]   # one of these
  - name: grep
    free: 60          # or up to 60 plain characters
    default: ""       # an argument with no default must be given
program: powershell
argv: ["-NoProfile", "-File", "{dir}/logs.ps1", "-Log", "{log}", "-Grep", "{grep}"]
timeout_s: 90
```

An argument is one of its listed values, or a short run of letters, digits,
spaces and `_ . : -`. It takes the place of one **whole** entry of `argv`
(`{name}`); it is never spliced into a string that a shell or an interpreter
parses, and `{dir}/…` names a file in the folder. A declaration that names a
field the agent does not know, an argument with nothing it may be, or an
`argv` naming an argument it does not take is not offered, and the agent says
why in its log.

The folder is read afresh on every call, so adding a check is adding a file
on that machine — nothing is rebuilt and nothing is restarted. Who can write
to the machine can add a check; Zen chooses from the list and cannot add to
it. The checks below are written into the folder when it does not exist yet,
each a declaration and the PowerShell script it runs, and are the machine's
to change after that.

| Check | Reports |
|---|---|
| `health` | processors, memory, free disk on each volume, uptime, a pending restart |
| `gpu` | the card's memory, load and temperature, and the programs holding it |
| `services` | keep, motherd, the gateway, the agent: whether each is up and since when |
| `errors` | the last day's errors and critical events, by what raised them, counted |
| `logs {log, grep, since}` | one event log searched for a word over a stretch of time |
| `logins` | who is signed in, remote sessions, the last day's sign-ins and failed attempts |
| `listeners` | every listening port, whether it is open to the network, and the program behind it |
| `processes` | every program running and where from; the heaviest by memory and processor time |
| `security` | firewall and antivirus on or off, threats detected, the definitions' age, the last update |
| `storage` | the size of each folder named in `storage.paths`, and its newest file |

`storage.paths` ships empty: each machine names its substrates, journals and
backups there, one path to a line.

**A check says what changed.** The agent keeps each check's last answer and
the report leads with the lines that came and went since — a port that was
not open, a program that was not running, a firewall that is off — so "out
of the ordinary" is something the report states rather than something Zen
has to infer from a page of numbers. What moves every time (a load figure,
the busiest programs) follows a line of `---` in the check's output: it is
reported and not compared.

### What a terminal raises

A terminal raises an event by delivering `EventKind::Terminal { terminal,
text }` to every unordered mind of its world (`Runtime::raise_terminal`),
urgent when its adapter says so. It names the terminal as `terminal_list`
does, so what Zen is told is also what to connect to.

**Presence is the first source.** `presence_watch` marks a handset on the
house network as one person's. Every five minutes this member surveys the
network for the worlds it writes that watch any (`authority::terminals::run`),
and raises that the person is home the moment their handset is seen, and
that they have gone out once two surveys running have missed it — one miss
is a phone asleep.

### 5.1 What a mind that works terminals reads

The engine's prompt is one schema for every character, and what differs by
kind of character is selected, never forked. Four things are per
personality, each general to the engine and assigned in the mind:

| What | Where it is written | How it reaches a turn |
|---|---|---|
| **Voice** — what the character is here as | `voice:` in `personalities/<id>.yaml`; `voice/generic` in the schema for one that writes none | the `voice` collection, pinned on every turn (`identity::Installed::selection_for`) |
| **Acting** — how its work goes and what it may say it knows | `acting:` in the personality; `acting/generic` in the schema | the `acting` collection, pinned on a turn that acts or talks (`Installed::act_as`); a reflection and a dream read neither |
| **Nothing asked of it** | `unassigned/generic` in the schema | pinned for a character with no mission whose personality takes orders; one with `takes_orders: false` reads a mission section when it carries a mission and nothing in its place otherwise |
| **What it draws on** | `personalities:` in `bearings.yaml` | a personality's own draws stand in the room's bearing wherever it is (`bearings::Bearings::of`) |

`zen-moon` writes its own voice and acting: it is told that it helps the
people it looks after, that its acts run for real, that a reading is taken
and never remembered, that a request ends when whoever asked has been told,
and that `rest` is how it stops. It draws its own world's canon and the
**guides**.

**Guides** are the `guides` layer: `layers/guides/*.md` in the mind, one
short document per piece of real work — the cluster's health, jobs and logs,
a machine check, the house network, the house's devices, carrying a request
through. The layer is shared and scoped, so a guide is offered only to a
personality whose draws name it, and a turn is handed the two its opening
scores highest.

**The journal is pinned.** `journal_intro` and the `journal` collection are
`pin: true` under the schema's `pin_tokens`, as zend's are: what a character
wrote down is attended whole at every depth of its conversation.

## 6. What changes for a person

Nothing in how they speak to Zen: the conversation is its own thread at urgent
priority. A request Zen cannot answer in words becomes an agenda item and is
worked on the watch's next turn. A person who writes to a drone at rest is
answered when that drone next carries a mission; the owner's console says the
drone is at rest.

## 7. What it should do to the card

One mind decoding continuously and a drone or two while missions run: decode
forwards of one or two sequences, a prefill backlog near zero, the cast's
slots and recurrent stores gone from the span, and the experts' hit rate well
above their resident share because the contexts they serve are few. Read from
`/v1/memory` (expert zone, recurrent stores, interaction slots) and the wave
lines (sequences a forward, forward time, hit rate), before and after each
step of §8.

## 8. Order of work

Steps 1–7 are built and released together, since a Zen with no body and its
old round of rooms to walk has nothing it can do. The machine checks need each
machine's agent on a build that has them.

Four things the first days on WhiteWorld showed, and what answers each:

- **A conversation reads its own group only.** A group that is not a turn's
  own is gathered across every conversation in it, and under a sequence rule
  each turn found is held as the reader's founding history; the mind and the
  talks are two groups, so a character answering somebody read every other
  character's working turns, the Makers' drafts among them. Both groups are
  scoped with no scope ever named (`Minds::close_conversations`).
- **An act offered in one place only is always described**
  (`tools::shown_wherever_offered`): the lift's, a granted act, a house
  device's, a routed one. Ten acts are described a turn, and a Drone at the
  lift was not shown `lift_use`.
- **A carrier that never tries the way is believed in the end**
  (`Mission::refuse_stuck_on_the_way`): every third turn-away in a row with
  nothing tried counts as one, so a mission cannot be stood beside for hours.
- **A step of the round is `connect to <terminal> …`**, signed off when an act
  of that terminal lands (`Mission::worked_terminal`). Connecting alone signs
  nothing off.

`POST /v1/pulse/start-over` starts named characters over — missions called
off, conversations retired, and by name the journal emptied and the dreams
retired — while the daemon runs and without touching another world's cast
(`engine/start_over.rs`).

1. **Drones at rest**: the engine-held rest state, assignment at the post, no
   summons, hands as §3 — with the runtime's scripted tests. The load change.
2. **Zen without perception or a place**: no situation, no stirrings, no
   movement acts; outcomes and drone messages as its events.
3. **The terminal acts, the house-device adapter** — the television's verbs
   among them — **and the `lights` terminal** with its basic verbs.
4. **The cluster adapter**, read-only.
5. **The machine adapter**: BlackWorld and WhiteWorld, the checks of §5.
6. **Terminal events, and presence** from the survey.
7. **The mind's content** (`battle-mind`): Zen's round as observations
   (`rounds/moon.yaml`: the reports, the cluster, the house network, each
   machine), its idle text and anchor, the one job it posts for a Drone
   (`coding-task`, which grants the task acts), and the `ops` and `network`
   templates removed — what they named is Zen's own to do through a terminal.
8. **`docs/zen_moon_mind.md`** brought to the code: the world is `moon`, and
   the pieces it names that were never built are taken out.
