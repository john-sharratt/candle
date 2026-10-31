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
2. **The mirror's machinery**: mirrored rooms (`place`, `group`, `source`), terminal events and
   the tannoy raised from them, onboarding through `World::reshape`.
3. **The web terminal**, the first source, which touches nothing private.
4. **Private sources**, one at a time, as each group connects them (§11).

## 11. Decided by the deployment

- Which groups exist, and for each, the sources its owner connects — mail, calendar, location,
  cameras, devices, journals — with the consent of the people they belong to. Until a source is
  connected, its place has no terminal for it.
