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

**Zen has no body.** It keeps a seat in the world's record — the key its
agenda and its posts hang from — at its home, which nothing else enters and it
never leaves. It has no movement act (`focus`, `recall` go with `move_to`), is
in no one's company, and is not perceived.

**Zen does not perceive the vault.** No situation is pushed to it and no
stirring reaches it. It is told only:

- **what a terminal raises** — an event from the real world (§5);
- **a mission's outcome** — done, failed with its reason, a drone stuck;
- **a drone's message**;
- and a person, who speaks to it in a conversation of their own.

All of it arrives through the one inbox the messaging system already fills.
Anything Zen should notice unprompted is one of these or goes unseen; it learns
the rest by connecting and looking.

**Zen's own work is the watch.** Its round stops sending it to rooms to read
panels. A round item is an observation: connect to a terminal, read what it
shows, and act on what is out of the ordinary — itself when it is a switch to
throw, by posting a mission when it is long. A person's request
(`mission_add`, from their conversation) is an item on the same agenda: turn
the television on is connect, act, disconnect, and back to the watch.

**Drones get the long work**: a large coding task, anything that needs long
reasoning or many turns overseen. Those missions grant the task acts.

## 5. Terminals

**A terminal is a real thing with an adapter behind it.** It is not a part of
the moon's map; the stations drones work at are those, and stay.

Three acts, offered to an unordered mind alone:

| Act | Does |
|---|---|
| `terminal_list` | Every terminal Zen can reach now: its name, what it is, whether it answers. |
| `terminal_connect {terminal}` | Connects. Zen holds **one** connection; connecting elsewhere replaces it. |
| `terminal_disconnect` | Lets it go. |

Disconnected, Zen's acts are these three, its table and agenda acts, and its
messaging. Connected, it also has **that terminal's verbs** — offered through
the live set the catalogue already filters by, so an unconnected terminal's
verbs are not in the grammar at all. The frame's state says what Zen is
connected to.

A terminal is an implementation of one trait — its id, its line for the list,
its verbs as catalogue specs, and `perform` — registered by an adapter:

| Adapter | Terminals | Verbs | State |
|---|---|---|---|
| **House devices** (`effector-bridge`, through the house's agent) | one per workable device the survey has found: `device-<mac>` | the device profile's actions — including webOS and Android TV, which gain their verbs here | built; today gated on a room and a linked part, which a terminal is not |
| **House lights** (the WiZ profile of `effector-bridge`) | **one** terminal, `lights`, over every bulb the survey has found — eight WiZ E27 today | list the bulbs and their state; on, off, brightness, colour and white temperature, a built-in scene — for one bulb by name, several, or all of them in one act | the bulb profile is built; the one terminal over all of them is to build. A bulb is not also a `device-<mac>` terminal |
| **Home network** | the house link | `survey`, `wake` | built as `survey_network`, `wake_device` |
| **Cluster** (zend's MCP, as `service:zen`, the channel `zen_task` uses) | the gateway's logs; the cluster; each machine's services | read-only: list, search and read logs; cluster, member and service status | to build — zend has the tools, npcd has the channel |
| **Machines** (each machine's own agent) | one per machine: BlackWorld and WhiteWorld first | the checks below | to build — the agent already takes the house's device work from npcd |
| **Presence** | the home network | raises an event when a known handset joins or leaves | to build — the survey sees handsets; nothing watches for them |

**The lights are one terminal because they are worked together.** Turning
the house off at night is one act over all eight, not eight connections. A
mode is a verb of this terminal that keeps running after Zen disconnects —
the adapter drives the bulbs, not a turn of Zen's — and has a verb to stop
it; the patterns driven from an audio input, for a set played live, are
modes added here.

A cluster terminal and a machine terminal are read-only. Deploying,
restarting and rolling out stay with people.

### A machine's checks

**Named checks, never a shell.** Zen reads logs, and logs carry text from
outside; a command line on a machine would let what it reads decide what it
runs. A check does one thing, takes validated arguments, changes nothing, and
answers with a short report.

**The list is the machine's own, and it is open.** A check is a declaration
the agent loads from its `checks/` folder, a file each: its name, the line
Zen reads in the terminal's verbs, its arguments with the values each may
take, and what it runs — one of the agent's built-in probes, or a program
with a fixed argument list into which only validated arguments are placed.
Adding a check is adding a file on that machine; the agent reads the folder
when a terminal is listed, so there is nothing to rebuild. Who can write to
the machine can add a check. Zen chooses from the list and cannot add to it.
The checks below ship as the first files.

| Check | Reports |
|---|---|
| `health` | CPU, memory, free disk a volume, uptime, a pending reboot |
| `gpu` | the card's memory, load and temperature, and what holds it |
| `services` | each service keep supervises: its build, whether it runs, how its last trial ended |
| `errors` | the daemons' recent errors and warnings, grouped and counted |
| `logs {log, grep, since}` | a named log searched for a pattern over a window |
| `logins` | recent sign-ins, failed attempts, remote sessions open |
| `listeners` | ports open to the network and the process behind each |
| `processes` | what runs now that did not at the last check; the heaviest by CPU and memory |
| `security` | the firewall and antivirus state, the definitions' age, pending system updates |
| `storage` | the substrate's and the journals' size, their growth since the last check, the newest backup's age |

**A check says what changed.** The agent keeps each check's last answer and
the report leads with the difference — a port that was not open, a process
that was not running, a volume that lost ten gigabytes — so "out of the
ordinary" is something the report states rather than something Zen has to
infer from a page of numbers.

A terminal raises an event by delivering `EventKind::Terminal { terminal,
text }` to every unordered mind of its world, urgent when its adapter says so.

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

1. **Drones at rest**: the engine-held rest state, assignment at the post, no
   summons, hands as §3 — with the runtime's scripted tests. The load change.
2. **Zen without perception or a place**: no situation, no stirrings, no
   movement acts; outcomes and drone messages as its events.
3. **The terminal acts, the house-device adapter** — the television's verbs
   among them — **and the `lights` terminal** with its basic verbs.
4. **The cluster adapter**, read-only.
5. **The machine adapter**: BlackWorld and WhiteWorld, the checks of §5.
6. **Terminal events, and presence** from the survey.
7. **The mind's content** (`battle-mind`): Zen's round as observations, its
   idle text, mission templates that grant the task acts, the `ops` and
   `network` templates rewritten to what a terminal can do or removed.
8. **`docs/zen_moon_mind.md`** brought to the code: the world is `moon`, and
   the pieces it names that were never built are taken out.
