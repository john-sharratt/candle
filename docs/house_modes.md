# The house's modes

**Status: built (2026-11-09).** The family live in the front room, so the house has **one mode at a
time**, set by Zen. A mode says what the room's devices are doing, whether rekordbox is open, and so
whether BlackWorld is held for the decks. ZenJ (`docs/zenj.md`) is the Party mode's DJ; the TV and
the stereo are the moon world's bridges (`docs/zen_moon_mind.md` §13–§15).

## 1. Mode, not mood

A **mode** is what the house is set up for: which devices are on and showing what, whether
rekordbox is open, whether BlackWorld is reserved. How loud or how lively it is within that is not
a mode — dinner with friends is the stereo on its music at a quieter volume, which is a step of
*listening*, not a mode of its own. The test: two set-ups that differ only in volume or energy are
one mode. Within a party, how lively it is over the night is ZenJ's energy curve.

"Mood" is already the word for a character's feeling in npcd (`mind/moods/`), and ZenJ's own
"who has the decks" is its **hands** (Human, Assist, ZenJ — `zenj/src/hands.rs`), so in the code
*mode* means only the house's.

## 2. The modes

| Mode | For | rekordbox | BlackWorld |
|---|---|---|---|
| **away** | nobody home | closed | free |
| **asleep** | home, asleep: nothing makes a sound | closed | free |
| **home** | somebody in, nothing on | closed | free |
| **listening** | music in the background | closed | free |
| **watching** | the television: a film, a show, a game | closed | free |
| **party** | Zen at the decks | **open** | **reserved** (§4) |

What each mode does to the room's devices is the house's table (§3.1). Only a party opens
rekordbox; every other mode closes it (`npcd/src/house/mode.rs`).

## 3. Setting a mode

**One act, `house_mode {mode}`** (`npcd/src/house/acts.rs`). It is offered to a body that takes no
orders — Zen — standing (its attention) in the room the house's devices stand in, with `mode`
bound to the six modes; anywhere else, and to anybody else, it is not in the grammar
(`Choices::HouseModes`, `engine/build/within.rs`). Zen sets it when the family ask for it or
plainly want it.

A switch (`npcd/src/house/switch.rs`), in order:

1. **The decks.** rekordbox is asked to open (party) or close (anything else) over the gateway's
   bus (§3.2).
   - **Leaving a party while a deck plays is refused, before anything is touched.** The music stops
     at the decks, never because the stereo went to standby under it.
   - **Nothing listening is said, not waited for.** With `zenj` not running, nothing is subscribed,
     and the gateway answers at once. The switch goes on and the answer says the DJ was not there.
2. **Each device step** of the mode, through the device's own bridge act, exactly as Zen's act on
   that part would run (`bridge::enact`). A device that does not answer, or a part not linked to a
   device, is a line in the answer. The rest still happen.
3. **The mode**, kept with the world (`sim/house.rs`: the mode, when, by whom) and replicated with
   it.

The answer is one line for the mode, then one line per part: what the decks and each device said.

### 3.1 The table

One file of the mind, `house/<world>.yaml` (`npcd/src/house/table.rs`), read afresh by every
member on the mission generator's tick:

```yaml
room: sharratt-apartment/front-room
modes:
  away:
    - {device: stereo, act: off}
    - {device: Sonos speaker, act: pause}
    - {device: Chromecast, act: stop}
  watching:
    - {device: stereo, act: off}
  party:
    - {device: stereo, act: on}
```

- **A step names its device and its act.** The device is a part by the name the room calls it. The
  act is the bridge's own (`on`, `off`, `input`, `volume` for a receiver; `pause` for a Sonos;
  `stop` for a Cast device), with that act's arguments beside it (`source: tv`, `volume: 30`).
- **A mode the file leaves out has no device steps.** It still opens or closes the decks.
- **The file is refused whole for one fault:**
  - a room that is not `area/node`
  - a mode that does not exist
  - a step with no device or no act

### 3.2 The decks, over the bus

- **The ask.** The switch asks `dj/rekordbox` with `{id, act}` and waits on
  `dj/rekordbox/result/<id>` for `{ok, line}` (`host/src/cluster/bus.rs`, the member's ask that
  zend's repository gate also uses).
- **Who answers.** `zenj` subscribes to the topic as the service `zenj`, with its service key, over
  the gateway's `/gateway/events` socket (`zenj/src/bus.rs`; the key's topics are in
  `web/web.yaml`).
- **What the answer runs.** It goes to the same call as Zen's `zenj_rekordbox` tool
  (`zenj/src/rekordbox/control.rs`).
- **How rekordbox opens and closes.**
  - Opening starts rekordbox's executable (`zenj.yaml` `rekordbox.program`) in the console session
    `zenj` runs in, the only one with the desktop.
  - Closing asks its windows to close, so it saves its library first. It is never killed
    (`zenj/src/rekordbox/process.rs`).
  - Each wait ends inside the bus's 30 s.

## 4. The reservation: BlackWorld while the decks are open

BlackWorld is also a build agent and a model member. A build there commits more memory than the
machine has, and its motherd's inference shares the card with the screens. So while the decks are
open, the machine is the decks'.

**The signal.**
- `zenj`'s status says `refuse` while rekordbox is **open**, playing or between tracks
  (`zenj/src/http.rs`). A build begun in a pause is still running when the music starts again.
- Whether rekordbox runs is read by the deck reader as it opens rekordbox (`deck/reader.rs`).

**What keep does with it** (`keep/src/reserve.rs`). keep reads its guards (`keep.yaml` `guards:`)
every second. While one refuses, the machine is **reserved**:
- **Every switch, accept, restart, stop and trial is refused** unless forced (`X-Keep-Force`). That
  includes a start of a service that yields, and the cluster's own trials.
- **Every service marked `yields: true` is held down** — BlackWorld's motherd. Holding it down
  takes away the machine's:
  - agent: no builds, no tests, and no `service_deploy` builds
  - inference
  - ingest and calibration
  - claim on the card and on the memory

  The gateway places all of that work on WhiteWorld and AlienWorld. Zen's choices for the night
  are served there, and the journals stay up, because WhiteWorld's agent serves them too.
- **A forced start of a held service is left running** until the reservation ends: a person asked
  for exactly that.

**It ends ten minutes after the guard last refused** (`RELEASE`). A `zenj` that crashes and is
started again by `zenj watch`, or a rekordbox restarted by hand, does not let a motherd the size of
the machine start in the gap. A refusal inside the ten minutes starts the count again. Once
released, keep starts motherd again.

## 5. Nothing waits on a house that is not there

Every piece fails open. A machine with no `zenj`, or a `zenj` that is not running, is exactly a
machine with no guard:

| Absent | What happens |
|---|---|
| `zenj` not running | keep's guard reads nothing and reserves nothing; a status that does not answer within 3 s refuses nothing |
| `zenj` not running, a switch | the bus has no subscriber and says so at once; the devices are still set |
| `zenj` not running, its tools | the gateway's `/v1/zenj` route falls back to `127.0.0.1:8097`, which refuses at once; the aggregator lists the `dj` backend's tools as last seen |
| the zenj service key's file, on a machine | that gateway admits no zenj, warns, and starts (`web/src/cluster/service_key.rs` `load_present`); a file present and malformed still stops it |
| no `house/<world>.yaml` | the world has no house: `house_mode` is in nobody's grammar |
| a device not linked, or not answering | its line says so; the switch goes on |

The cluster sees `zenj` as a service app (`cluster/src/gateway/hub.rs` `SERVICES`): never certified,
rolled out or floored, so a missing `zenj` member blocks no rollout.

## 6. Limits, as built

- **The house's table is the house's facts.** Which input of the stereo the FLX4's master out is
  cabled to, and which input of the television BlackWorld's fourth output feeds, are steps of
  `party` that only the house can say. Until they are written into `house/moon.yaml`, a party opens
  the decks and switches the stereo on, and the inputs are set by hand.
- **A mode is what Zen last set.** A device changed by hand — the television turned on with its
  remote — does not move the mode: nothing on the network reports it.
- **Zen knows the mode from its own act.** The answer is on its record. The room's reading does not
  repeat it each turn.
