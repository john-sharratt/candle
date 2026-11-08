# ZenJ — Zen as DJ

**Status: design, not built.** First use: New Year's Eve 2026, at home, on BlackWorld. Supersedes
the listener design on branch `zen/docs-midi-dj-set` (`docs/midi_dj_set.md`), whose findings about
MIDI are kept where they still apply (§6.1) and whose architecture — a passive listener inside
`npcd` — is replaced by this one.

ZenJ is Zen running the music: choosing the next track from what is playing, the time, the
conversation and the playlists; carrying out the transition on the beat; handing the decks to a
person and taking them back; and showing all of it on four screens. rekordbox is the engine that
plays the audio. ZenJ is the brain above it and the hands on it.

---

## 0. Decisions already made

These were settled in conversation on 2026-11-08 and are not reopened here.

| Decision | Choice | Why |
|---|---|---|
| Playback engine | **rekordbox 7.2.19**, with the DDJ-FLX4 attached natively | It already plays streaming audio, analyses tracks, beat-syncs, and runs the FLX4 with its full feature set. Writing our own audio engine buys nothing for this event. |
| Audio source | **TIDAL** (Individual + DJ Extension), logged in inside rekordbox | Spotify inside rekordbox was tried and is not good enough. TIDAL tracks can be analysed and stored offline. |
| Taste and requests | **Spotify** — both accounts' history and likes, plus a shared request playlist | That is where John's and Yen's music lives and where requests are made from a phone. |
| Where ZenJ runs | **Its own process, `zenj`, on BlackWorld**, joining the cluster as its own app | Not inside the agent: the agent restarts with every motherd deploy, and the music must not. |
| Licences | Not a constraint: a private party | Only technical risk is weighed below. |

## 1. What ZenJ does that Automix does not

rekordbox's Automix plays a playlist in order with a generic blend. ZenJ does what a DJ does:

1. **Plays to an arc** — warm-up, build, peak, the midnight moment, wind-down — planned against the
   clock, and anchored backwards from midnight so the right drop lands on 00:00:00.
2. **Chooses the next track for this moment**: what is playing and how it is landing, the time,
   who asked for what, and what is being said.
3. **Chooses how to mix each pair**: where to leave and enter, on phrase boundaries; long blend with
   a bass swap, quick cut on a drop, echo-out, filter sweep, loop roll.
4. **Makes moments**: the midnight countdown, a dedication, a sound effect on the right beat.
5. **Reacts**: "more energy", "play one of Yen's", a request arriving, a person taking the decks.

Automix stays, as the safety net below ZenJ (§7.4), not as ZenJ.

## 2. The shape of it

```
 Spotify (phones)          Zen chat / Discord          the clock
      │ requests, taste         │ conversation              │
      ▼                         ▼                           ▼
 ┌──────────────────────────── zenj (BlackWorld) ─────────────────────────────┐
 │  perception (§4) ──► the brain (§5): set plan · next track · transition plan │
 │                                   │                                           │
 │                         the conductor (§6) — beat-accurate, no model           │
 │                                   │ virtual MIDI          ▲ deck state        │
 │  visuals (§8) ◄── beat + audio     ▼                       │ (memory reader)   │
 └────────────────────────────── rekordbox 7.2.19 ────────────┘────────────────┘
                                   │ ▲
                         DDJ-FLX4  │ │  John's / Yen's hands
                     (sound card)  ▼ │
                                 speakers
```

Three loops, each at the speed its job needs:

| Loop | Period | Runs | Never |
|---|---|---|---|
| Conductor | per beat, ~10 ms precision | executes the transition plan through MIDI; verifies what it did | calls a model, waits on the network |
| Brain | per track, minutes; seconds when the conversation asks | set plan, next-track choice, transition plan | touches MIDI directly |
| Zen | seconds | chooses among the shortlist, reacts to the conversation, writes the set up | runs the beat |

The model is never on the beat. Zen has a whole track — three to five minutes — to make each
choice, and the conductor makes it happen to the beat. That is the "super fast loop": instant
reflexes, unhurried taste.

## 3. The process

### 3.1 A cluster member of its own

`zenj` is a new crate and binary. It joins the cluster exactly as the agent does
(`agent/src/member.rs`: same handshake and group key, no model, no substrate, its node id kept in a
`NODE` file beside its logs) but as **its own app, `zenj`**, not as `agent` — so the gateway never
routes it a build job. That needs one change in `web`: accept `/v1/cluster/zenj`. Requests reach it
only on routes it declares (MCP invariant 5, `CLAUDE.md`).

It is supervised by keep as its own service. A motherd deploy never touches it. keep refuses a trial
of `zenj` while a set is live (`zenj` reports `live` in its `CAPACITY`).

rekordbox is a desktop application and needs a logged-in Windows session with the display attached,
so `zenj` runs in John's session on BlackWorld, started by keep there.

### 3.2 Threads

Each thread does one thing; they talk through lock-free single-producer queues (`rtrb`), so nothing
waits on anything else.

| Thread | Priority | Does |
|---|---|---|
| MIDI in | high | decodes the FLX4's messages (observed, not proxied — §6.3) |
| Conductor | high | runs transition plans at beat boundaries; owns the virtual MIDI out port |
| Deck reader | high | samples rekordbox's live deck state (§4.2) every ~5 ms |
| Audio capture | high | captures the FLX4's output for loudness and the visuals (§4.4) |
| Brain | normal | set plan, shortlist, transition plans; calls Zen |
| Library | low | Spotify/TIDAL sync, matching, track profiles |
| Render, one per screen | normal | the visuals (§8) |
| Cluster | normal | membership, tools in, events out |

## 4. Perception — what ZenJ knows

### 4.1 The library: Spotify for taste, TIDAL for audio

1. **Taste.** John and Yen each request Spotify's privacy export ("Download your data": account data
   and extended streaming history). It holds every liked track, every playlist and every play, with
   track ids. It is their personal data; it is the richest taste signal there is. The Spotify Web API
   fills in what the export lacks: current playlists, and each track's ISRC, duration and album.
2. **Matching.** Each Spotify track's **ISRC** is looked up on TIDAL (`GET /tracks?filter[isrc]=…`,
   up to 20 per call, TIDAL API v2). A match is accepted only when it is unique and the artist, title
   and duration agree; otherwise TIDAL search, under the same check. Unmatched tracks are listed, not
   guessed.
3. **Crates.** Matched tracks go into TIDAL playlists ZenJ owns — *Warm-up*, *Build*, *Peak*,
   *Classics*, *Midnight*, *Liked by both* — which rekordbox shows in its TIDAL tree.
4. **Offline.** Every crate is stored offline in rekordbox. TIDAL offline tracks play for 30 days
   from storage, and a login between day 27 and 30 resets it, so storage happens after 2026-12-10
   and a login is done on 2026-12-30.
5. **Analysis.** rekordbox analyses the TIDAL tracks: BPM, key, beat grid, phrases. ZenJ reads that
   analysis from rekordbox's library, read-only, through the `rbox` crate (`master.db` and the ANLZ
   files: `PQTZ` beat grid, `PCOB` cues, phrase tags).

TIDAL is driven through its official API v2 (`openapi.tidal.com/v2`, OAuth with PKCE, scopes
`playlists.read`, `playlists.write`, `search.read`, `collection.read`). There is no maintained Rust
client, so `zenj` has its own small one, written against TIDAL's published OpenAPI description.

### 4.2 Live deck state

rekordbox publishes no deck state; its Ableton Link only receives. ZenJ reads it from rekordbox's
memory, as `rkbx_link` does: for each deck the loaded track, BPM, play state, beat and bar position,
phrase, and which deck is on air. `rkbx_link` itself targets rekordbox 6.7.3 and is not maintained,
so `zenj` has its own reader for 7.2.19.

**rekordbox is pinned to 7.2.19 until after New Year's Eve**: automatic updates off. An update moves
the memory layout, and the reader would read nothing — or worse, the wrong thing. The reader
therefore validates what it reads every sample (a track id that exists in the library, a BPM in
range, a beat position that advances with time) and reports itself `blind` rather than guess. A
blind reader drops ZenJ to Automix (§7.4).

### 4.3 Track profiles

Computed ahead of time, never live, for every track in the crates:

- from rekordbox's analysis: BPM, key (as Camelot), beat grid, phrase map (intro, up, chorus, down,
  outro);
- from the audio: an energy curve over the track, and a "sounds like" embedding from an audio model
  run offline on BlackWorld (§9);
- from Zen: what it knows of the song — a singalong, a classic, a slow burner, the year, the scene.
  This is written once per track and kept, so it is a lookup at decision time, not a model call.

### 4.4 The room

The FLX4's output is captured from the system (WASAPI loopback of the FLX4 render device), giving
loudness and spectral energy over the last minutes. If rekordbox takes the device exclusively and
loopback captures nothing, the fallback is physical: the FLX4's master out into a USB audio input.
This is the first thing tested (§11).

Requests are a signal too: their rate, and who is making them. A "more energy" / "chill" pair of
pads (on the FLX4 in a spare pad mode, or on the phones through the Zen chat) gives the room a
direct voice. A camera is not in scope for New Year's Eve.

### 4.5 The conversation, the clock, and the hands

- **Conversation**: the Zen chat and Discord. "More energy", "something for Yen's mum", "save that
  one for midnight" — each becomes a change to the set plan or the shortlist (§5).
- **Requests**: the collaborative Spotify playlist *ZenJ Requests*, which John and Yen add to from
  the Spotify app. ZenJ reads it every few seconds through the Web API.
- **Clock**: time to midnight and time into the set.
- **Hands**: every message the FLX4 sends, observed (§6.3). From it ZenJ knows whether a person is
  DJing and where every physical control sits.

## 5. The brain

### 5.1 The set plan — minutes to hours

An **energy curve** over the night, from the start time to the end, with fixed points:

- **Midnight.** The midnight track and its drop are chosen ahead (Zen proposes, John confirms). If
  its drop is bar *n*, it must start at 00:00:00 − *n* bars at its tempo, so the plan works backwards:
  the track before it must reach its outro then, and so on for the last four or five tracks. From
  23:40 the run-in is locked and only a person can change it.
- **Moments** John or Yen ask for ("Yen's song at 22:30").

A **budget** keeps the night balanced: John's taste, Yen's, both, requests, and genre spread, each
with a share. Zen redraws the curve and the budget when the conversation or the room says so.

### 5.2 The next track — every few minutes, always one ahead

The main decision, split so that hard rules can never pick a clash and Zen supplies the taste:

1. **Shortlist (deterministic, milliseconds).** Every candidate in the crates is scored on:
   tempo fit (within ±6%, or half/double), key fit (same or adjacent on the Camelot wheel), distance
   from the curve's target energy at the moment it will play, not played tonight, no same artist
   within N tracks, request priority, taste weight (liked by both above one), and similarity to what
   is landing well now. The top eight, with their scores, go to Zen.
2. **Zen chooses.** It reads the shortlist with what only it has — the conversation, the moment, the
   note on a request, who has not heard one of theirs in a while — and picks one, or reaches outside
   the list when someone asked for something specific. It says why in one line. The line goes on the
   screens and into the journal.

The next track is always decided before the current one is half way through, and the one after it
tentatively, so a slow or failed model call is never felt. If Zen does not answer in time, the
shortlist's top entry is taken.

A request is placed, not obeyed blindly: ZenJ puts it where it fits (usually within two tracks) and
says where. "Play it now" from the requester skips the fit and mixes at the next phrase.

### 5.3 The transition plan — as soon as the next track is chosen

From both tracks' phrase maps and the energy step between them:

- **out point** in the playing track and **in point** in the next, on phrase boundaries;
- **style**: blend with a bass swap (the common case), cut on a drop, echo-out, filter sweep, loop
  roll into a key change;
- **length** in bars, and the moves within it: when B's fader rises, when the bass swaps, when A's
  fader falls, which effect and when;
- any **sample** fired, and on which beat.

The plan is data — a list of `(bar, beat, action)` — written before it runs. So it can be shown,
logged, overridden, and replayed in tests with no party and no rekordbox.

## 6. Hands

### 6.1 The virtual controller

`zenj` opens a virtual MIDI output port — a Windows MIDI Services loopback endpoint if BlackWorld's
build has them, otherwise loopMIDI (spike 3) — that rekordbox maps with MIDI LEARN as a second
controller. The map covers, per deck: load selected, play, cue,
sync, tempo, channel fader, trim, EQ high/mid/low, filter, beat FX on/off and level, loop in/out,
hot cues 1–8; and globally: crossfader, browser up/down, the sampler pads, Automix on/off.

The map lives in `zenj/maps/rekordbox-7.2.19.yaml` and is loaded with the round's rule: a control
named twice, or a message used twice, is refused at load, naming the file and the fault.

What the earlier design learned about MIDI still applies: running status, SysEx termination,
real-time messages interleaved; the decoder is pure and tested against raw bytes. Where a control
sends a 14-bit value as a pair of CCs (MSB and LSB), the decoder pairs them; which FLX4 controls do
is read from the message list, not assumed.
The FLX4's message map is taken from Pioneer's published MIDI message list and checked on the wire.

### 6.2 Closed-loop loading

The next track sits at the top of a short TIDAL playlist, *NYE Next*, that ZenJ keeps. To load it,
the conductor moves the browser to that playlist's first row and sends "load to deck *n*", then
**reads the deck state** (§4.2) to confirm the loaded track is the one intended, and retries — at
most twice — if not. A load that cannot be confirmed falls back to Automix for that transition.

How quickly rekordbox shows a change to a TIDAL playlist made through the API is the largest
unknown in this design (§11, spike 1). If it does not refresh on its own, the map gets rekordbox's
refresh action, or *NYE Next* is filled further ahead so the delay never matters.

### 6.3 A person and ZenJ on the same mixer

rekordbox owns the FLX4 directly. `zenj` only **observes** it — Windows MIDI Services lets any number
of applications open the same port — and never sits between the hands and rekordbox. Proxying would
show rekordbox a virtual port instead of the FLX4, losing its native support, and a crash of `zenj`
would kill the controller mid-set.

Modes, each visible on the screens:

| Mode | Entered | ZenJ does |
|---|---|---|
| **Human** | any touch on the FLX4 | keeps *NYE Next* stocked and the next track profiled; never touches a deck that is on air |
| **Assist** | explicitly: a pad combination or "help me" to Zen | also loads the next track on the idle deck and cues its in point |
| **ZenJ** | explicitly: a pad combination or "take over" to Zen | mixes: the brain chooses, the conductor plays the plan |

Moving toward more ZenJ is always explicit, never a timeout. Moving back is immediate: any touch on a
mixer control returns that control to the person, and a touch on a deck's play or jog returns the
whole mixer.

**No jump on hand-back.** The FLX4's faders and knobs are not motorised. If ZenJ moved a control in
software and a person then moves the physical one, rekordbox takes the physical value and the mix
lurches. ZenJ knows every physical position (it observes them), so before handing back it walks each
control it moved, over a beat or two, to where the hardware sits. The first touch is then seamless.

## 7. Failure

### 7.1 The rule

The floor is never silent. Every failure degrades a level and says so on the screens; none stops the
music.

### 7.2 What fails and what happens

| Fault | What happens |
|---|---|
| Internet drops | Crates are stored offline (§4.1); requests stop arriving and the screens say so |
| A request is not on TIDAL | Answered, not dropped: "not on TIDAL — closest: …" on the screen and in the chat |
| Deck reader is blind (§4.2) | ZenJ drops to Automix on *NYE Live*; Zen keeps curating it |
| A load cannot be confirmed (§6.2) | Automix takes that transition |
| Zen does not answer in time | The shortlist's top entry is taken (§5.2) |
| `zenj` crashes | keep restarts it; rekordbox keeps playing the loaded track and Automix continues |
| rekordbox crashes | Restarted by hand; the crates live in TIDAL and the offline store, so nothing is lost |
| motherd is busy or restarting | No effect: `zenj` needs Zen only for choices, and falls back to the shortlist |

### 7.3 Watchdog

The conductor checks, every bar: the on-air deck has more than 32 bars left or the next track is
loaded and its plan is armed. If not, it turns Automix on and reports why.

### 7.4 The ladder

**ZenJ** (brain + conductor) → **Automix on *NYE Live*, Zen curating** → **a person**. Each rung
works without the one above it.

## 8. Screens

Four outputs: three monitors on BlackWorld's card and the TV on a fourth, wired output (an adapter
if the card's outputs are DisplayPort) — not cast, which adds seconds of delay. One borderless window per output, rendered with `wgpu`.

- **Scenes**: audio-reactive visuals driven by the beat position (§4.2) and the captured audio
  (§4.4), so a scene change lands on a phrase. `lactea` runs the MilkDrop preset library in Rust, so
  there are thousands of scenes from the first day.
- **The layer we own**: now playing and next, a request arriving ("Yen's request is up next"), Zen's
  one-line reasons, the mode (§6.3), and the midnight countdown, which takes over every screen from
  23:59:00.
- The visuals are driven from the predicted beat grid, so they can lead the sound by the TV's
  measured latency, and land on the beat on every screen.

The visuals share BlackWorld's GPU with motherd. For the night, BlackWorld runs in **event mode**
(§10), so motherd's inference yields to the render threads.

## 9. Models

| Model | Used for | When |
|---|---|---|
| Zen (motherd's model) | choosing among the shortlist, the set plan, reacting to the conversation, the write-up | live, per track |
| An audio embedding model | "sounds like" similarity between tracks (§4.3) | offline, once per track |
| Beat This! (MIT) | beat and downbeat check where rekordbox's grid is doubtful | offline |
| HTDemucs | stems, for the visuals to follow vocals and drums separately | offline; optional for New Year's Eve |
| Stable Audio Open | generating sound effects for the sampler | before the night; the effects are files |

Offline models run in motherd's candle engine where a port exists (candle already runs audio codecs:
`encodec`, `dac`, `snac`), otherwise once through ONNX Runtime. Nothing heavier than Zen's own call
runs during the set.

## 10. The night

- **Event mode on BlackWorld**, 2026-12-31 18:00 to 2027-01-01 06:00: no deploys of anything (keep
  refuses trials), ingest and calibration paused, the GPU's share for inference reduced.
- **Freeze**: rekordbox 7.2.19 pinned; `zenj` frozen on 2026-12-20; the map file frozen with it.
- **Offline store** refreshed after 2026-12-10, with a TIDAL login in rekordbox on 2026-12-30.
- **Two full rehearsals**, each at least four hours with a midnight run-in, before 2026-12-20.

## 11. Spikes, in order

Each is a day or less and decides something.

1. **TIDAL playlist → rekordbox.** Add a track to a TIDAL playlist through the API; time how long
   rekordbox takes to show it, during Automix. Then removal and reordering through the API. Confirm
   that rekordbox analyses TIDAL tracks (§4.1) and that Automix plays them, online and from the
   offline store.
2. **Audio capture.** Capture the FLX4's output while rekordbox plays (WASAPI loopback), in both of
   rekordbox's audio driver modes.
3. **Virtual controller.** A virtual port learned in rekordbox alongside the FLX4: does each learned
   control act, and does a physical touch after a software move behave as §6.3 assumes?
4. **Deck reader.** Locate deck state in rekordbox 7.2.19's memory; validate across load, play,
   sync, and both decks.
5. **Closed-loop load.** Browser to *NYE Next*, load, confirm through the reader.

## 12. Build order

| Dates | Work |
|---|---|
| 2026-11-09 → 11-15 | Spikes 1–5. Spotify exports requested. |
| 11-16 → 11-29 | `zenj` crate: cluster membership, MIDI decoder and observer, conductor, deck reader, TIDAL client, Spotify request watcher, library matching and crates |
| 11-30 → 12-06 | The brain: set plan, shortlist, Zen's choice, transition plans; Zen's tools and event stream |
| 12-07 → 12-13 | Screens: scenes, the overlay, the countdown, event mode |
| 12-14 → 12-20 | Rehearsals, fixes, freeze |
| 12-30 | Offline refresh login |
| 12-31 | The night |

Every module lands with its unit tests: the decoder against raw bytes, the shortlist scorer against
fixed track profiles, the transition planner against fixed phrase maps, the midnight back-scheduler
against a fixed run-in. A transition plan is replayable against a recorded deck-state trace with no
rekordbox at all.

## 13. Zen's surface

Tools on the `/mcp` surface, routed to `zenj`, each answering from memory within the 30 ms rule:

| Tool | Does |
|---|---|
| `zenj_now` | what is playing on each deck, where in it, the mode, and the next track with its reason |
| `zenj_queue` | *NYE Next* and the set plan; reorder or pin |
| `zenj_request` | add a request, with an optional note and urgency |
| `zenj_mode` | Human / Assist / ZenJ (moving toward ZenJ needs a person's confirmation in the chat) |
| `zenj_moment` | schedule a moment: a track at a time, a dedication, an effect |
| `zenj_sfx` | fire a sampler slot now or on the next beat or bar |
| `zenj_events` | the event stream since a cursor: track changed, request arrived, mode changed, fault |

## 14. The journal

Every pick and its reason, every transition and its plan, every request and where it went, every
mode change and fault: written to the `@zen` journal as the set runs, `by` the writer that made the
decision (`zenj` for the conductor, Zen for its choices). The next morning Zen writes the set up from
that record: what the night was, what landed, what to do differently. That record is also how the
next party's ZenJ starts out knowing what worked.

## 15. Open questions

1. Which pad combinations mean "help me" and "take over" (§6.3).
2. The midnight track (§5.1).
3. Whether stems (HTDemucs) are in scope for New Year's Eve, or after.
4. Whether guests beyond John and Yen can add to *ZenJ Requests* (a QR code to the collaborative
   playlist), or only the two of them.
