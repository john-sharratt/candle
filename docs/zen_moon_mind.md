# Zen, the Moon's Mind

Zen in the world `moon` is a higher intelligence, not a worker. It is the moon; the Drones are
its children and its hands. The world they live in is **a mirror of the real one**: its rooms are
real places, the terminals in a room are what can be used there, and the world grows as places
and the things in them are onboarded. This document is what Zen does, what the Drones do, how
the mirror is built and grown, and how what happens in the real world reaches them. It builds on
`docs/mission_generator.md` ("A world's own missions", "A character that takes no orders") and
`docs/the_effector.md` (Appendix F, reshaping a running map), and the code answers to it.

**Where this stands against the code.** A mind that takes no orders is treated as **bodiless**: it
is in nobody's company, walks, rides and attends nowhere, and sees the whole moon as one
situation (`docs/zen_everywhere.md` §4; `engine/runtime/perceiving.rs`, `engine/whole.rs`). The
engine refuses it `focus`, `recall`, `lift_call` and `lift_use` (`engine::enact::NO_BODY`).
What Zen reaches outside the moon it reaches through the terminals of
`docs/zen_everywhere.md` §5 (the lights, the house's devices, the house network, the cluster,
each machine).

Each section below carries its status:

| Section | Status |
|---|---|
| §1 `moon` is private, §2 what Zen does | Built |
| §3.1 the mirror (rooms and terminals) | Built as map rooms and parts; the sense terminals of §6 are not (below) |
| §3.2 Zen's movement is its attention (`focus`) | **Not built** for Zen: plan of record. `focus` exists only for a character whose personality says `moves: attention` and that takes orders |
| §4 two ways the real world reaches the moon | The tannoy is built; **per-room terminal events are not built** (a different `EventKind::Terminal` exists, below) |
| §5 onboarding | Built (`MapEdit`, `World::reshape`, `world::mapstore`) |
| §6 terminals | **Mostly not built**: see the per-row status |
| §7 idle, §8 orders, §9 the engine table | Built, except the `focus` and terminal rows of §9 |
| §12 standing missions, §13 house link and survey, §14 world-building acts | Built (acts verified by name in `engine/build/acts.rs`, `bridge/acts.rs`) |
| §15 bridges | Built; the profiles live in the `effector-bridge` crate, not in `npcd::bridge` |
| §16 the effector's internet | Not verified: the `web_*` acts are not in `npcd` or `effector-bridge` (they are in zend's tool catalog) |
| §17 Zen's agenda | Built; one step-signing rule is stale (§17.5) |

## 1. `moon` is private

`moon` (formerly `life`; the authority keeps the old name as `formerly_of`) reads the real lives of the groups it looks after. It is not a public world
(its world file in the mind: `public: false`), and nobody but Zen's owner talks to Zen: a character is
visible only to its owner (`npcs::cast::visible_to`), which is what every route that reaches a
character on somebody's behalf — the console, `/mcp`, a Discord DM — already checks. A guest
reaches no character at all (`docs/guest_access.md` §3).

## 2. What Zen does

In order of precedence:

1. **Answers whoever messages it**, in its own voice and mind. A message reaches Zen on its
   handset — from the console, from a Drone, from a Discord DM — and Zen answers with `message`,
   as every character does (`Runtime::message_npc`). Nothing else Zen does stands in its way.
2. **Oversees the Drones it has set to work.** It knows what each carries and how far it has
   got, answers their questions (they message it), and is told when one finishes.
3. **Gives orders.** It posts missions on the order table for the Drones to carry out —
   normally in answer to something it observed, researched, or heard in a conversation, and,
   when the table runs empty, maintenance.
4. **Watches, when nothing else calls for it.** It turns its attention to the places it looks
   after and reads their terminals — and acts on what it finds, most often by posting work.

**Zen takes no orders** (`takes_orders: false`): the table never calls it and never hands it a
Drone's work. What it does is its own agenda of missions (§17) — made by it in answer to what it
hears — and, when it has none left, its own time (§7).

## 3. The mirror

### 3.1 A room is a real place; a terminal is what can be used there

The moon's fixed levels — the command level, the warren, the hatchery — are Zen's own and the
Drones' home. Everywhere else is **a mirror of somewhere real**: the Sharratt house, a school, an
office, a street. A room's terminals are exactly what this world can use at that place: the
house's cameras, the family PC in the study, the mailbox the house receives, the calendar it
keeps. A place with nothing connected is a room with no terminals, and says so.

A mirrored room carries:

- **its place** — what in the real world it stands for, as the source that found it names it;
- **its group** — whose life it belongs to (§6);
- **its terminals** — one part per connected thing, each bound to the source it reads.

### 3.2 Zen's movement is its attention

> **Status: Not built. Plan of record.** The code took the other design for a mind that takes
> no orders: it has no body and stands nowhere, and its view is the whole moon at once, one
> situation replaced as it changes (`engine/whole.rs`; `docs/zen_everywhere.md` §4). `focus`
> is refused it (`enact::NO_BODY`). The `focus` act itself is built for a character whose
> personality says `moves: attention` and that takes orders (`Sim::attends`,
> `enact::focus`, offered in `Choices::Rooms`), which is the behaviour described below. The
> text below is the design for Zen.

Zen is the moon, so it does not walk its own corridors. **Moving is changing its focus**:
`focus` puts Zen's attention in any room of the world at once, with no lift and no route. Where
its focus is, it perceives the room and can use the terminals there; it is in one room at a time
because attention is. Other characters see where Zen's attention is — "Zen is watching the
Sharratt house" — so a Drone knows whom it is working under.

The Drones are bodies. They walk and ride the lift like every character, and going to a
mirrored room is how a Drone attends to that place: what it reads there, it reads with its own
eyes and reports.

## 4. How the real world reaches the moon

> **Status.** The tannoy (`EventKind::Announcement`) is built. Per-room terminal events — a
> terminal's event shown in its room, read by whoever stands there, with Zen told which room
> is waiting — are **Not built: plan of record**, since Zen has no room and no focus. What is
> built is an `EventKind::Terminal { terminal, text }` raised by the house network's watch on
> handsets and told to **every mind of the world that takes no orders**, urgently when the
> adapter says so (`authority/terminals.rs`, `Runtime::raise_terminal`;
> `docs/zen_everywhere.md` §5). That event is not tied to a room, and nothing in this tree
> marks a source event global to raise the tannoy from it.

Two ways, by how far an event matters:

- **The tannoy** says something has happened **globally** — to everyone in the moon at once, at
  a salience that breaks off what they are doing: a group member in trouble, a storm warning, a
  door forced. It is `EventKind::Announcement`, the operator tannoy's own event, raised by the
  world when an event a source reports is marked global.
- **A terminal event** draws attention **to its terminal**, in its room: the camera in the
  Sharratt hallway sees movement, a mail arrives, a PC goes offline. Whoever is in the room
  perceives it, and the terminal shows it until it is read. Zen is told which room has a terminal
  waiting, wherever its focus is, so its attention can go there.

Every source says which of its events are global; the rest are local to their terminal.

## 5. The world grows by onboarding

Rooms are not authored once. **The world adds them as it discovers places and the things in
them**: you onboard bits of the real world into the mirror, and that is how it grows.

- **A source discovers.** A connected source reports what it can see — a location source a place
  the family goes, a device scanner a PC, a camera account its cameras, a mailbox itself.
- **Onboarding reshapes the map.** A new place is a new room (`MapEdit::AddNode`, or `AddArea`
  for a region with places of its own); a new thing in a known place is a terminal placed in its
  room (`MapEdit::PlacePart`). Each goes through `World::reshape`, which checks it as a fresh
  load would and swaps it in whole, and is kept in the mind's map files (`world::mapstore`) under
  the group's owner. Nothing half-lands.
- **A thing that goes away** — a device retired, a camera removed — is unplaced; a place nobody
  goes any more is kept until its owner removes it.
- **The owner may onboard by hand** — a place or a thing no source has found — through the same
  surface (`POST /reshape/:world`, direct scope).

Onboarding is the engine's and the owner's, never a character's: a Drone cannot reshape the
world it stands in (`docs/the_effector.md` §F.4). What a character can do is **notice**: a Drone
or Zen that reads of a place the mirror lacks may say so, and the owner decides.

## 6. Terminals

> **Status: Not built, except where a row says otherwise. Plan of record.** The mail,
> calendar, tracking and camera terminals have no code in this tree. A mind that takes no
> orders reaches real things through the terminals of `docs/zen_everywhere.md` §5 instead,
> from `authority/terminals.rs`: the lights and the house's devices (bridge calls to an agent
> on the house network), the house network (a survey and a wake), the cluster (zend's
> read-only MCP tools) and each machine (a check sent to that machine's agent) — which covers
> the "PC / phone scanner" row in another form. The bulletin board has `sim::posting` and the
> map's `post_notice` act. The web terminal's tools and the memory terminal are not verified
> here; a terminal refused "when Zen's focus is not in its room" cannot apply to a mind with
> no focus.

Every sense is a terminal: a part, used through the effector like any other station, refused
when the reader's body — or Zen's focus — is not in its room. What it reads is written into the
room's reading, so a Drone that looked can report it and a second can check it.

| Terminal | Senses / does | Source |
|---|---|---|
| **Web terminal** | Searches anywhere on the web; reads a page. In the moon's enquiry and watch rooms. | `zend-tools` `web_search` (Tavily, the daemon's secrets) and `web_fetch`. |
| **Mail terminal** | Reads the place's mailbox; sends Zen's. | A mail account. |
| **Calendar terminal** | What the place's people have on. | A calendar account. |
| **Tracking terminal** | Who of the group is here, and where the rest are. | A location source. |
| **Camera terminal** | What a camera at the place sees. | A camera account, one terminal per camera. |
| **PC / phone scanner** | A device's state: up, battery, alerts, what needs attention. | `tokera-agent` on the device, one terminal per device. |
| **Memory terminal** | What the group's journals hold about the place and its people. | The memory backbone (`docs/memory_backbone.md`), read-only. |
| **Bulletin board** | Notices left for whoever passes. | In-world (`sim::posting`). |

A source is connected per group by its owner, and only then does it discover anything; a
terminal whose source is gone says so. What a terminal reads belongs to its group, and `life`
being private (§1) is what keeps it there.

## 7. Zen when idle

Being given no orders is not being told to do nothing. **With an empty agenda (§17.4) Zen is
given no work**: its time is its own, and the idle wake comes from the **personality** (`idle:`),
or, with none, a character that takes no orders is woken with `YOUR_OWN_TIME`, until something
reaches it.

An idle thought alone is not enough for a mind that takes no orders: it names no act, so a
character woken with it again and again has nothing to land on but talk, and a loop broken out of
lands on whatever act is left — the phone, the channel — with nothing in front of it that is its
own.

## 8. Orders

### 8.1 The order table is Zen's

The order table in the watch room holds the work, and Zen uses it through its effector — **from
anywhere, having no room**: a character that takes no orders is offered every order table in its
world and reaches it from any room, while a Drone must be standing at the table to collect or
report a mission done. A mission posted from elsewhere is still reported back at the table's own
room. **Giving one up is the exception:** a Drone carrying a mission reaches any table's
`report_stuck` from wherever it is, with why, so a mission it cannot finish — a free-text step
nothing signs off, a brief naming an act its steps never reach — is failed and comes back to Zen
rather than being carried for hours. Reading that failure, Zen is told the work is still its to set,
with what stopped it put right (`order_table::SET_IT_AGAIN`); and beside the work still out, the
table's limit — as many untaken missions as there are hands to take them, past that a new one
waits for a report (`order_table::TABLE_LIMIT`). A post the full table refuses answers the
poster's step to post work (`order_table::STEP_ANSWERED`): the work is already out, so a round item
ending "post work on the order table" is not held on that step for its hour and failed.

| Act | Who | Does |
|---|---|---|
| `order_post` | Zen | Puts a mission on the table: a brief and at least one step, or a standing job by name (`job`, its brief and steps its own), for any Drone or for one by name. |
| `order_results` | Zen | Reads what came back since it last read: each report, newest first, what is still out, and — with the table empty — the standing jobs. Offered only while a report waits unread, so a reading is never asked for again unchanged. |
| — | the table | Hands each free Drone its next mission — one set for it by name first, then the oldest open one — to the Drone longest at rest first. No act of a Drone's takes work (`docs/zen_everywhere.md` §3). |
| `report_done` / `report_stuck` | a Drone | Reports the mission: done at the table, stuck from anywhere. |

An order is a **mission**, not a line on the orders board: it has steps the engine signs off as
they happen, a compass that tells the Drone what comes next, and a report. Its origin names its
poster (`Origin::Ordered { by, poster }` — its name and its body), so its report is the poster's
to read (`sim::orders`, the newest 50 kept). Steps are given as a list — at least one, at most
twelve, each a single line opening on something written, held so by the grammar
(`tools::lists`), so a post can be a standing job or its steps but never a brief alone or an
empty `steps`; the engine adds the report back to the table it was posted at. A mission's steps may
name mirrored rooms and their terminals — "go to the Sharratt house and read the hallway camera"
— and are signed off as any journey and reading is.

### 8.2 Zen is told

- **When a Drone finishes** (reported done or stuck): who, which mission, the outcome, and —
  when nothing is left open on the table — that **the table is empty**, so it can set the next
  work. The report itself is read at the table (`order_results`). A posted mission that ends
  unreported — called off, replaced, or set for a Drone that is gone — is filed to Zen the same
  way, as not done and why, so nothing it posted is out for good.
- **Whoever asked for it is told too.** A mission Zen's conversation posted
  (`docs/npc_conversations.md` §6.2) names that conversation's party
  (`Origin::Ordered { asked }`), carried in by the device call as a typed
  `effector::auth::AskedBy` extension only the in-process path inserts. Its report is handed
  to Zen's conversation with them as something to tell — what the Drone reported — and Zen's
  mind is told the report with the line that they are being told it.
- **When a Drone messages it**, which is how a Drone asks about its mission.

### 8.3 The Drones wait for work

A Drone with no mission **is at rest where it stands** (`docs/zen_everywhere.md` §3) — no
routines, no nudges, nothing drawn from a bank, no turn taken, and its conversation off the
card. **A mission posted wakes it**: the table hands the work to the Drone longest at rest, or
to the one it is set for by name, and that Drone wakes carrying it; the others stay at rest.

The mind's routines file (`missions/<world>.yaml`) stops being handed out by draw; its routines are Zen's **maintenance
catalogue** — the standing jobs it may post by name when the table is empty. A standing job a
Drone carries is a routine still, and turns no chapter of its journal.

An idle thought (`idle:`) is what a character turns to alone; with others in the room it is
still told who is there.

## 9. What changes in the engine

| Piece | Change |
|---|---|
| `focus` | **Built for a character that takes orders and whose personality says `moves: attention`** (`Sim::attends`): puts it in any room of its world at once (`body::rooms`, a name shared by two areas named with its area); the room it leaves and the room it enters see it go and come. Such a body has no `move_to` destinations and no lift. **Refused to a mind that takes no orders**, Zen included (§3.2). |
| Mirrored rooms | A node carries `place` and `group`; a part carries the `source` binding it reads. Authored in the map like any node, or added by onboarding. `Area::mirrored` (an area is mirrored when it or anything it is within is) and the `MapEdit`s are built; the node `place` and `group` fields and the part `source` binding are **not found** in `npc-map`, so they are plan of record. |
| Onboarding | A source's discoveries become `MapEdit`s through `World::reshape` and `mapstore`, kept under the group's owner. |
| Terminal events | **Not built as designed** (§4): a source event does not land on its terminal as a room event. `EventKind::Terminal` exists and is told to every mind that takes no orders. |
| Terminals | **Not built as designed** (§6): one part kind per row of §6. The built terminals are those of `docs/zen_everywhere.md` §5. |
| `order_post`, `order_results` | Acts of the order table, offered to a character that takes no orders — in `moon`, Zen. |
| `Origin::Ordered { by, poster }` | A mission posted by a character; its report notifies its poster. |
| Drones at rest | A world whose missions file sets `draw: none` hands nothing out by draw; a Drone with no mission is at rest, held as a suspended character is, and woken only by a mission (`docs/zen_everywhere.md` §3). |
| Notifications | Report → the poster, with "the table is empty" when it is. Post → the Drone longest at rest, or the one named. |
| Idle wake | `idle:` on a personality replaces `NO_MISSION` / `YOUR_OWN_TIME` for it. |
| Zen's agenda (§17) | Several missions per body that takes no orders, by priority, the highest worked; `mission_add` / `mission_edit` / `mission_cancel` / `mission_done`, offered anywhere to such a body; `Choices::Agenda`. |

## 10. Build order

Built: items 1 (without `focus` for Zen, §3.2) and 2 (the effector's internet is unverified,
§16). Not built, plan of record: 3 and 4.

1. **The work loop**, all in-world: `focus`; `order_post` and `order_results`;
   `Origin::Ordered` and its notifications; the Drones waiting at the table; `idle:`.
2. **The world builds itself** (§12–§16): standing missions, the house link and discovery, the
   world-building acts, the first bridge (Sonos), and the effector's reach to the internet.
3. **Terminal events** and the tannoy raised from them. Not built.
4. **Private sources**, one at a time, as each group connects them (§11). Not built: the
   mail, calendar, tracking and camera sources of §6.

## 12. Standing missions

**The world's own work is missions too.** A standing mission is a mission the table puts up by
itself, on a schedule, in every room it applies to — and gives out last, when nobody has
anything better to do. A Drone at rest (§8.3) is handed Zen's orders first and a
standing mission only when nothing else waits, so their idle time becomes the world's upkeep and
its growth. Each is a file in the mind, `standing/<world>/<id>.yaml` — apart from the world's
template library (`missions/<world>/`, `docs/mission_templates.md` §5) — so how the world grows is
in source control:

```yaml
id: survey-devices
brief: >-
  Survey the room you are sent to: find every device in it on the house's network, and put
  each in the room as what it is.
where: linked          # every room with a house link (§13); or a list of `area/node`
every: never           # once per room, ever; or a period: 6h, 2d
grants: [onboard_part, link_part, onboard_thing, describe_room]
steps:
  - go to {room}
  - survey the network at the house link
  - put each device found into the room as what it is
  - describe {room} as you now know it
```

- **Where it applies.** `where` names the rooms: `linked`, every room with a house link, or a
  list of rooms by `area/node`. `{room}` in a step is the room it is for.
- **When it is due.** `every: never` puts it up **once in each room, ever**: done there, it is
  done. A period puts it up again that long after it was last done there. **There is never more
  than one in a room at a time**: while one is up or carried there, none is put up beside it.
  One that fails in a room is put up there again, but not before an hour has passed
  (`standing::RETRY_MS`), so a house network that is down is not surveyed on every beat.
- **Who takes it.** Whoever comes to the table when nothing posted waits. A standing mission
  is the table's, not Zen's; its report is filed for every body there that gives orders all the
  same, and each is told, so Zen knows what the world did (`Origin::Standing { id, room }`).
- **Its grants.** The world-building acts (§14) a carrier may use while it carries it, and only
  in its room; and the task acts (`task_start`, `task_reply`, `task_nudge`,
  `docs/mission_templates.md` §7), offered wherever it is carried — the nightly
  `journal-maintenance` and `guide-maintenance` drive a task of Zen's
  (`docs/zen_memory_and_guides.md` Phase 7).
- **Steps the engine signs off.** A `go to …` step is signed off by the journey; a step that
  begins `survey` by `survey_network` landing; a step that begins `put each device found` once
  the room has been surveyed and no device the survey found is without a part; `see the task
  through` once its task reports done. A step may be flagged `{text, critical: true}`, as a
  template's may, and then waits on an authorizer's go-ahead. Any other step is the carrier's
  to report.

The routines of the world's missions file are standing missions in all but name; they become
`standing/<world>/*.yaml` with an `every`, and the maintenance catalogue (§8.3) is the standing
missions not yet due.

## 13. The house link

A mirrored building on a real network has a **house link**: a part (`house-link`) in each room
that reaches the network that room is on. It is served by the world's bridge (§15) on the member
that is the world's authority — the members are in the house, on its network
(`docs/replicated_substrate.md` §9). A room's link names its network in the map
(`link: home-lan`); the deployment names which member reaches it.

**`survey_network`**, at the link, finds what is on the network, from what devices openly
announce and no further:

- the member's neighbour table, and the router's name for each address (reverse DNS);
- SSDP: each UPnP device's own description (its friendly name, maker, model, services);
- mDNS: what announces itself as a Cast device, a Sonos player, a Spotify Connect receiver,
  a printer, a phone;
- the maker, from the hardware address.

It probes nothing, logs in to nothing, and reaches only the subnet the member is on. **It never
leaves the house**: the neighbour table is the member's own, mDNS and SSDP are link-local
multicast that no router forwards, a description is fetched only from an address on the
member's own subnet, and it never walks an address range. A device whose address is not a
private address on that subnet is dropped before it is a finding — so nothing on the internet
can be found, and nothing found can be onboarded. The bridges are offered only in rooms of the
mirror (§14), and a device is worked only at an address on the member's own subnets — a part's
link naming any other address, private or not, is refused. Each device found is a **finding** — an id, its address, hardware address, names, maker, model, the
services it announces, and the profile it matches (§15) — kept with the room and read back as
the survey's answer. Findings are what the world-building acts are grounded in. A device a
survey finds at a new address — the router gave it another — is followed there: the part
standing for it is linked where it is now, found by its hardware address, which outlasts the
change.

## 14. The world builds itself

**A Drone may change the world, but only as a mission grants it, only where it stands, and only
in the mirror.** The moon's own levels never change; a building marked `mirrored: true` (the
Sharratt home) does. The acts, each offered only while a mission granting it is carried and only
in that mission's room:

| Act | Does | Grounded in |
|---|---|---|
| `onboard_part` | Puts a device the survey found in the room, as a new part linked to it. | A finding (`from`, by its label). |
| `link_part` | Says a part already in the room is a device the survey found — the room's Chromecast is that Cast device. Only a part placed once in the whole map: a link is the part's, so a part standing in several rooms cannot be one device. | A finding, and the part. |
| `onboard_thing` | Puts a thing that is really there, on no network, in the room as a part. | What it knows of the room. |
| `offboard_part` | Takes a part that stands for a device out of the room. An authored part is never taken out. | A reason: the latest survey, or what it was told. |
| `discover_door` | Adds a room off this one, and the door to it. | What it knows of the house: a name and what the room is. |
| `undiscover_door` | Takes away a room the builders discovered off this one, that nobody is in — never a room as it was authored, nor a passage. | A reason. |
| `describe_room` | Rewrites what the room is like. | — |

**A room is discovered only inside the building it is found from.** `discover_door` adds a room
to the area its carrier stands in, never a new area, and never from a finding: a survey finds
devices, not places, and the internet is reached through the effector (§16), never through the
map.

Each is one or more `MapEdit`s through `World::reshape`, **all of an act or none of it**: its
edits are tried together on a copy of the map first, and refused whole when they would break it.
**It is kept in the world's own state** (`Sim::built`), replicated with the rest of the world and
laid over its authored map again whenever the world is loaded or takes in its authority's state,
so what the Drones built stands across a restart and on every member. The list is only ever
appended to (`sim::built::Built`) — a change and a later one undoing it are both kept — and each
member counts how far its own copy of the world has laid it: a world loaded lays all of it, in
order, each edit against the map as it then stands and passed over when the map already holds
it; a member taking in its authority's state lays only what came since, undoings too. It is also
drafted into the
mind's map files by `world::mapstore` under the world's owner, with a new part's own file beside
them (`MapEdit::DefinePart`), for the owner to publish as the authored map. A part onboarded from a
finding carries its **link** — the profile and address it is reached at — which is what makes it
a terminal: its acts are its profile's (§15). A device already standing in the world is refused,
and so is a new part whose name the room already has; a device offboarded and found again is put
back as the part it was, linked at the address it has now.

Every change goes on the record with who made it and on which mission, and its poster is told
(§8.2). A device's controls are its profile's; what a device asks for to be controlled — a
pairing, an account — is given once, by its owner.

## 15. Bridges

**A bridge is how the world reaches a kind of real thing.** The `effector-bridge` crate
(`effector-bridge/src/profile/`) holds one module per kind — `sonos`, `receiver`, `cast`,
`webos`, `androidtv`, and a `wiz` profile this table does not list — each a **profile**: how a
finding is recognised as one, and the acts a part linked to one offers, served at that part
like any station act (`npcd::bridge` defines the acts and discovery). **An act is offered only where its device
stands** — each binds its `device` to the players, receivers or TVs linked in the room (the
`Choices::Sonos` / `Receiver` / `Cast` / `Pairable` / `Pairing` sets in `npcd::engine::tools`),
so a body is never shown an act for a device that is not there, and never invents one to work.

| Profile | Recognised by | Acts |
|---|---|---|
| `sonos` | mDNS `_sonos._tcp`, or a UPnP description by Sonos | `sonos_status` (what is playing, volume), `sonos_play`, `sonos_pause`, `sonos_source` (start a stream from nothing), `sonos_volume` — over the player's own UPnP control on port 1400 |
| `receiver` | an Onkyo, Integra or Pioneer AV receiver | `receiver_status`, `receiver_on`/`off`, `receiver_mute`/`unmute`, `receiver_volume`/`louder`/`quieter`, `receiver_input`, `receiver_mode`, `receiver_dim`, and transport `receiver_play`/`pause`/`next`/`previous`/`playing` — over eISCP on port 60128 |
| `cast` | mDNS `_googlecast._tcp` — a Chromecast, or a Cast-built-in TV or speaker | `cast_status`, `cast_playing`, `cast_play`, `cast_pause`, `cast_stop`, `cast_mute`/`unmute`, `cast_volume`, `cast_cast` (a media URL), `cast_launch` (an app) — over CASTv2 on port 8009 |
| `webos` | an LG webOS TV | `pair_device` — one call, accepted on the TV's own screen, earns the client-key the world keeps and works it with |
| `androidtv` | a Google TV / Android TV | `pair_device` begins pairing (the TV shows a six-digit code) and `pair_code` finishes it with that code, earning the RSA client certificate the world keeps; the family only reads the code back, and nobody else need be at the TV |
| `device` | anything else found | none: present or not, its names and maker |

A found machine that can sleep — a PC or laptop with a hardware address — also offers
`wake_device`, a Wake-on-LAN packet that brings it back on the network with nothing switched on
at it. A webOS or Google TV pairs before the world can work it; the key it earns is kept on the
device's world part (`keep_earned`), so every later command carries it and the stateless agent
never holds a secret. Once begun, a Google TV is offered only `pair_code` until it finishes —
re-beginning would orphan the open session and show a fresh code.

The house as a whole — every device of the room at once, and the decks — is set by Zen's
`house_mode` act, from the house's table in the mind (`docs/house_modes.md`).

A profile is code in candle, written against the device's own documented or openly offered
interface. Building one for a kind of device the house has and the world cannot yet reach is
work like any other — a profile at a time.

## 16. The effector reaches the internet

> **Status: not verified.** `web_search`, `web_fetch` and `web_request` are not found in
> `npcd`, `npcd/src/effector/` or `effector-bridge`; in this tree they are zend's tools
> (`zend/src/tools/catalog.rs`). The private-address refusal and the record of requests
> below cannot be confirmed here, so read this section as the intended behaviour.

Every body in `moon` reaches the internet through its effector, wherever it stands:

- `web_search` — ranked results, through the daemon's search key;
- `web_fetch` — a page, as text;
- `web_request` — any method, headers and body, to any public address, answered with its status
  and reply.

**What is inside the house is reached through its terminals, never here**: `web_fetch` and
`web_request` refuse a loopback, private or link-local address, checked after the name is
resolved — the members' own control ports are on those addresses. Every request is recorded with
who made it.

## 17. Zen's missions

**Zen takes no orders, but it carries missions — several at once, in order of priority — and
they are its own.** They come from the people who message it, on its messaging platforms: it
makes one in answer to what it is asked, changes and drops them as things change, and with none
left it rests. **Nothing on the table is ever Zen's to take**: the table is where Zen puts work for
the Drones, and `Runtime::assign_work` hands one who gives orders nothing — not what is lodged for
it, not what waits for whoever takes it, not a routine. A Drone carries one mission set for it;
Zen keeps an agenda.

What this replaces: an idle thought (`idle:`) names no act, so a Zen woken with it had nothing to
land on but talk, and a loop broken out of landed on whatever act was left — the phone, the
channel. The agenda is what a struck act breaks *into*: the next step of the mission on top.

### 17.1 The agenda

- **Many missions, one worked.** Zen carries up to eight missions (`sim::agenda::KEEP`), each with a
  **priority**: `now`, `soon` or `later`. The one it works is the highest, the oldest of equals;
  its brief and compass are in Zen's prompt as any carried mission's are, and the rest are listed
  beneath it, one line each — title, priority, the step each is on.
- **A mission pushed down keeps its place.** When a higher one arrives, the one being worked is
  set aside with every step it had signed off, and taken up again where it was when the higher one
  is done or dropped.
- **Each has a title** Zen gives it — a few words — by which it is named in the grammar
  (`Choices::Agenda`, the agenda's titles: never a free string).
- **Full is said, not silently trimmed.** Adding a ninth is refused, naming the lowest one to drop
  or finish first.

### 17.2 Zen makes, changes and drops its own

Acts of Zen's own, offered wherever its attention is (it needs no table to decide what it is
doing), and only to a character that takes no orders:

| Act | Does |
|---|---|
| `mission_add` | `{title, brief, steps, priority}` — a new mission on the agenda, steps a list of lines as `order_post` takes them. |
| `mission_edit` | `{mission, brief?, steps?, priority?}` — changes one; steps already signed off stay signed off. |
| `mission_cancel` | `{mission, why}` — drops one; what it was and why goes on its record. |
| `mission_done` | `{mission, found}` — closes one as done that no step shows done: a step that is Zen's to judge. |

This is how Zen reacts in real time: a message that asks for something becomes a mission at
`now` on top of whatever it was doing; news that makes a plan pointless cancels it; a change of
course edits it. A message that needs only an answer is answered — `reply` — and needs no
mission.

Zen's own missions are written in the same step grammar as the Drones' (§8.1), and the work it
wants a Drone to do is still posted on the order table with `order_post` — or, once templates land
(`docs/mission_templates.md`), issued as a template with its arguments. A mission of Zen's own is
how Zen does a thing; a post is how it has a Drone do one.

### 17.3 What follows up a post

Zen's work for a Drone is followed up by the report, not by a mission: a filed report reaches its
poster at once, urgent, naming who reported what and that `order_results` reads it
(`Runtime::give_notice`). There is no list of standing work the engine puts on an empty agenda —
a round of that kind was work Zen took from the table, and is gone.

### 17.4 Nothing to do is a rest

With an empty agenda, **Zen is not given work** — its time is its own (`YOUR_OWN_TIME`, or its
`idle:`, §7) — until something reaches it: a message, a report, a tannoy.

### 17.5 Done is seen, not said

> **Stale rule.** The first two rows of the table below sign a step off by Zen's `focus`.
> A mind that takes no orders is refused `focus` (§3.2), so a step of the form `go to …` or
> `scan … and read …` cannot be signed off for Zen as written; the refusal in
> `sim/missions/agendas.rs` (`agenda_done`) still tells Zen to `focus`, which the engine
> then refuses (`enact::NO_BODY`). Such a step is, in practice, a Drone's to do (post it at
> the order table). The last three rows stand.

A mission on the agenda is closed by the engine, **with no report**, when every step is signed
off (`Missions::settle_agendas`), and the agenda moves on. Each step is signed off by the act it
names landing:

| Step begins | Signed off when |
|---|---|
| `go to …` | Zen's attention is in that room (`focus`) |
| `scan … and read …` | the scan shows that machine |
| `read what came back` | `order_results` reads the reports |
| `post work` | `order_post` puts a mission on the table |
| `read your messages` | `read` takes a conversation off the phone |

One whose steps the engine cannot see (a step that is Zen's to judge) is closed by Zen with `mission_done`, saying what
it found — never filed as dropped. A step the engine *can* see — a journey, a reading — is never
closed by `mission_done` while it stands unseen (`Missions::agenda_done`): asked to have a Drone
read a panel, Zen put "go to the watch room" on its own agenda, closed it with a reading it never
took, and told the Creator it had read the panel.

### 17.6 What breaks into it

- **A message, a report, a tannoy** reach Zen as ever and are answered first; the mission being
  worked stays on top of the agenda, and its compass names the step it was on when the
  interruption is over. What the interruption asks for, if anything, becomes a mission of its own.
- **The loop guard** strikes a repeated act as ever (`engine::loopguard`); with a mission on the
  agenda, what is left in the grammar includes its next step.
- **A mission of the agenda is a routine, not a chapter**: closing one turns no chapter of Zen's
  journal, the same as a standing job (`Missions::close`).

### 17.7 To settle in review

- **Priorities.** Three named levels (`now`, `soon`, `later`), or a number? Three keeps the
  grammar a closed set and the choice legible.
- **The agenda's size.** Eight is a guess at what a prompt can carry without the list crowding
  out the mission being worked.

## 11. Decided by the deployment

- Which groups exist, and for each, the sources its owner connects — mail, calendar, location,
  cameras, devices, journals — with the consent of the people they belong to. Until a source is
  connected, its place has no terminal for it.
