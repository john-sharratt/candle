# Zen, the Moon's Mind

Zen in the world `life` is a higher intelligence, not a worker. It is the moon; the Drones are
its children and its hands. The world they live in is **a mirror of the real one**: its rooms are
real places, the terminals in a room are what can be used there, and the world grows as places
and the things in them are onboarded. This document is what Zen does, what the Drones do, how
the mirror is built and grown, and how what happens in the real world reaches them. It builds on
`docs/mission_generator.md` ("A world's own missions", "A character that takes no orders") and
`docs/the_effector.md` (Appendix F, reshaping a running map), and the code answers to it.

## 1. `life` is private

`life` reads the real lives of the groups it looks after. It is not a public world
(`worlds/life.yaml`: `public: false`), and nobody but Zen's owner talks to Zen: a character is
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

**Zen takes no orders** (`takes_orders: false`): the table never calls it and never hands it
work. What sets it going when idle is its own nature (§7), not the world.

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

Zen is the moon, so it does not walk its own corridors. **Moving is changing its focus**:
`focus` puts Zen's attention in any room of the world at once, with no lift and no route. Where
its focus is, it perceives the room and can use the terminals there; it is in one room at a time
because attention is. Other characters see where Zen's attention is — "Zen is watching the
Sharratt house" — so a Drone knows whom it is working under.

The Drones are bodies. They walk and ride the lift like every character, and going to a
mirrored room is how a Drone attends to that place: what it reads there, it reads with its own
eyes and reports.

## 4. How the real world reaches the moon

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

Being given no orders is not being told to do nothing. A character's idle wake comes from its
**personality** when the personality has one (`idle:`); Zen's says what Zen is — when nothing
calls for it, it turns its attention to the places it looks after, reads what their terminals
hold, and acts on what it finds. A personality with no `idle:` that takes no orders is woken with
`YOUR_OWN_TIME`.

## 8. Orders

### 8.1 The order table is Zen's

The order table in the watch room holds the work, and Zen uses it through its effector:

| Act | Who | Does |
|---|---|---|
| `order_post` | Zen | Puts a mission on the table: a brief and at least one step, or a standing job by name (`job`, its brief and steps its own), for any Drone or for one by name. |
| `order_results` | Zen | Reads what came back since it last read: each report, newest first, what is still out, and — with the table empty — the standing jobs. Offered only while a report waits unread, so a reading is never asked for again unchanged. |
| `collect_mission` | a Drone | Takes the next mission on the table — one set for it by name first, then the oldest open one. |
| `report_done` / `report_stuck` | a Drone | Reports the mission at the table. |

An order is a **mission**, not a line on the orders board: it has steps the engine signs off as
they happen, a compass that tells the Drone what comes next, and a report. Its origin names its
poster (`Origin::Ordered { by, poster }` — its name and its body), so its report is the poster's
to read (`sim::orders`, the newest 50 kept). Steps are given one to a line; the engine adds the
report back to the table it was posted at. A mission's steps may
name mirrored rooms and their terminals — "go to the Sharratt house and read the hallway camera"
— and are signed off as any journey and reading is.

### 8.2 Zen is told

- **When a Drone finishes** (reported done or stuck): who, which mission, the outcome, and —
  when nothing is left open on the table — that **the table is empty**, so it can set the next
  work. The report itself is read at the table (`order_results`). A posted mission that ends
  unreported — called off, replaced, or set for a Drone that is gone — is filed to Zen the same
  way, as not done and why, so nothing it posted is out for good.
- **When a Drone messages it**, which is how a Drone asks about its mission.

### 8.3 The Drones wait for work

A Drone with no mission **goes to the order table and waits there, idle** — no routines, no
nudges, nothing drawn from a bank. **A mission posted wakes it**: every Drone waiting at the
table is told the moment work arrives, and the table hands it to the first of them to take a
turn there; a mission set for one by name wakes only that one.

`missions/life.yaml`'s routines stop being handed out by draw; they are Zen's **maintenance
catalogue** — the standing jobs it may post by name when the table is empty. A standing job a
Drone carries is a routine still, and turns no chapter of its journal.

An idle thought (`idle:`) is what a character turns to alone; with others in the room it is
still told who is there.

## 9. What changes in the engine

| Piece | Change |
|---|---|
| `focus` | A body act for a character whose personality says `moves: attention` (`Sim::attends`): puts it in any room of its world at once (`body::rooms`, a name shared by two areas named with its area); the room it leaves and the room it enters see it go and come. Such a body has no `move_to` destinations and no lift. |
| Mirrored rooms | A node carries `place` and `group`; a part carries the `source` binding it reads. Authored in the map like any node, or added by onboarding. |
| Onboarding | A source's discoveries become `MapEdit`s through `World::reshape` and `mapstore`, kept under the group's owner. |
| Terminal events | A source event lands on its terminal as a room event (`EventKind::Terminal`), shown on the terminal until read; Zen is told which room is waiting. A global one is raised on the tannoy. |
| Terminals | One part kind per row of §6, each verb a station verb served by the effector, reading through its source. |
| `order_post`, `order_results` | Acts of the order table, offered to a character that takes no orders — in `life`, Zen. |
| `Origin::Ordered { by, poster }` | A mission posted by a character; its report notifies its poster. |
| Waiting at the table | A world whose missions file sets `draw: none` hands nothing out by draw; a Drone with no mission is sent to the table to wait (`TO_WAIT`) and waits there told nothing, woken by a post. |
| Notifications | Report → the poster, with "the table is empty" when it is. Post → every Drone waiting at the table, or the one named. |
| Idle wake | `idle:` on a personality replaces `NO_MISSION` / `YOUR_OWN_TIME` for it. |

## 10. Build order

1. **The work loop**, all in-world: `focus`; `order_post` and `order_results`;
   `Origin::Ordered` and its notifications; the Drones waiting at the table; `idle:`.
2. **The world builds itself** (§12–§16): standing missions, the house link and discovery, the
   world-building acts, the first bridge (Sonos), and the effector's reach to the internet.
3. **Terminal events** and the tannoy raised from them.
4. **Private sources**, one at a time, as each group connects them (§11).

## 12. Standing missions

**The world's own work is missions too.** A standing mission is a mission the table puts up by
itself, on a schedule, in every room it applies to — and gives out last, when nobody has
anything better to do. Drones waiting at the table for work (§8.3) take Zen's orders first and a
standing mission only when nothing else waits, so their idle time becomes the world's upkeep and
its growth. Each is a file in the mind, `missions/<world>/<id>.yaml`, so how the world grows is
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
  in its room.
- **Steps the engine signs off.** A `go to …` step is signed off by the journey; a step that
  begins `survey` by `survey_network` landing; a step that begins `put each device found` once
  the room has been surveyed and no device the survey found is without a part. Any other step is
  the carrier's to report.

The routines of `missions/life.yaml` are standing missions in all but name; they become
`missions/life/*.yaml` with an `every`, and the maintenance catalogue (§8.3) is the standing
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

**A bridge is how the world reaches a kind of real thing.** `npcd::bridge` holds one module per
kind, each a **profile**: how a finding is recognised as one, and the acts a part linked to one
offers, served at that part like any station act. **An act is offered only where its device
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

A profile is code in candle, written against the device's own documented or openly offered
interface. Building one for a kind of device the house has and the world cannot yet reach is
work like any other — a profile at a time.

## 16. The effector reaches the internet

Every body in `life` reaches the internet through its effector, wherever it stands:

- `web_search` — ranked results, through the daemon's search key;
- `web_fetch` — a page, as text;
- `web_request` — any method, headers and body, to any public address, answered with its status
  and reply.

**What is inside the house is reached through its terminals, never here**: `web_fetch` and
`web_request` refuse a loopback, private or link-local address, checked after the name is
resolved — the members' own control ports are on those addresses. Every request is recorded with
who made it.

## 11. Decided by the deployment

- Which groups exist, and for each, the sources its owner connects — mail, calendar, location,
  cameras, devices, journals — with the consent of the people they belong to. Until a source is
  connected, its place has no terminal for it.
