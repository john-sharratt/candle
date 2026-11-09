# ZenJ — Zen as DJ

**Status: built (`zenj/`, 2026-11-08; the lights, §9, 2026-11-09, waiting on their bulbs); the spikes of §12 run on BlackWorld against it.** First
use: New Year's Eve 2026, at home, on BlackWorld. Supersedes
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
| Playback engine | **rekordbox 7** (7.2.19 installed 2026-11-08), with the DDJ-FLX4 attached natively | It already plays streaming audio, analyses tracks, beat-syncs, and runs the FLX4 with its full feature set. Writing our own audio engine buys nothing for this event. |
| Audio source | **TIDAL** (Individual + DJ Extension), logged in inside rekordbox | Spotify inside rekordbox was tried and is not good enough. TIDAL tracks can be analysed and stored offline. |
| Taste and requests | **Spotify** — both accounts' history and likes, plus a shared request playlist | That is where John's and Yen's music lives and where requests are made from a phone. |
| Where the party is | **John's home, where BlackWorld is** | The FLX4, the monitors and the TV are cabled straight to BlackWorld; nothing of the set crosses the network but Zen's choices. |
| Where ZenJ runs | **Its own process, `zenj`, on BlackWorld**, joining the cluster as its own app | Not inside the agent: the agent restarts with every motherd deploy, and the music must not. |
| Licences | Not a constraint: a private party | Only technical risk is weighed below. |

| Lights | **WiZ Colour A60 E27 Gen 2** (Signify 929003601001 / WIZ601001): 806 lm, RGB + tunable white, dimmable over WiFi, 2.4 GHz WiFi + BLE. Twenty-two bought — twenty fitted on the ceiling track, the rest spares | The track is E27 with A60 globes in deep black cowls, aimed in fixed directions. The wall circuit stays at full power: all dimming is over WiFi (§9). |
| How the lights are driven | **WiZ local UDP JSON on port 38899**: `setPilot` (`r,g,b`, `dimming` 10–100, `temp`, `state`, and `sceneId`, `speed`), `getPilot` for status, heartbeats on 38900 after `registration`, no authentication. Needs "Allow local communication" ON in the WiZ app (Settings → Security) | Community-measured ~20–50 ms. Fire-and-forget UDP: there is no ack, which is why a key moment is sent twice (§9.2) and why a bulb is judged by its `getPilot` answers, not by its behaviour (§7.2). |
| Where the bulbs live | Their own 2.4 GHz SSID, each with a DHCP reservation; ZenJ addresses them by IP from a fixture map, never by broadcast discovery on the night | Discovery is a thing that can fail in front of an audience. A reservation is a line in a file. |
| Alternatives weighed and rejected | Athom WLED E27, LIFX, Philips Hue, TP-Link Tapo L530E, Tuya-based bulbs (Connect SmartHome, Mirabella Genio, Arlec), a Tuya relay socket adapter | Athom WLED has the best protocol of the lot — DDP and Art-Net — but ships from China, and the schedule is the risk. LIFX has a good LAN protocol with on-bulb `SetWaveform` at ~20 msg/s per bulb, for about four times the cost. Hue's Entertainment API caps a stream at 10 lights on one bridge and runs at ~25 Hz, for more money again. Tapo's local access needs cloud credentials and a KLAP/TPAP handshake, and TP-Link has broken local control by firmware before. The Tuya bulbs speak an encrypted local protocol that needs a key extracted from the cloud, and are slow. A relay socket adapter dims nothing and colours nothing, goes through the cloud, and wears out. |

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
 │  lights (§9)  ◄── beat + energy   UDP, no ack               │                   │
 └────────────────────────────── rekordbox 7 ─────────────────┘────────────────┘
                                    │ ▲
                          DDJ-FLX4  │ │  John's / Yen's hands
                      (sound card)  ▼ │
                                  speakers

 lights (§9):           UDP JSON :38899 ──► the ~20 bulbs on their own 2.4 GHz SSID
                        (never through the cloud; never through rekordbox)
```

Four loops, each at the speed its job needs:

| Loop | Period | Runs | Never |
|---|---|---|---|
| Conductor | per beat, ~10 ms precision | executes the transition plan through MIDI; verifies what it did | calls a model, waits on the network |
| Brain | per track, minutes; seconds when the conversation asks | set plan, next-track choice, transition plan | touches MIDI directly |
| Zen | seconds | chooses among the shortlist, reacts to the conversation, writes the set up | runs the beat |
| Lights | on beats and phrase boundaries, ≤ ~10 msg/s per bulb | sends state changes to the bulbs over UDP | calls a model, waits on the network |

The model is never on the beat. Zen has a whole track — three to five minutes — to make each
choice, and the conductor makes it happen to the beat. That is the "super fast loop": instant
reflexes, unhurried taste.

## 3. The process

### 3.1 A cluster member of its own

`zenj` is a new crate and binary. It joins the cluster with the generic member machinery in the
`cluster` and `cluster-dial` crates (`AppMember`, `Identity`, `Capacity`, `kept_node`, `Dial`,
`RouteTable`) — no model, no substrate, its node id kept in a `NODE` file beside its logs. The
template is the `mcp` service's presence (`mcp/src/presence.rs`, about 60 lines), not the agent:
one route table, a capacity, `INFO serving` printed once admitted.

It joins as **its own app, `zenj`**, never as `agent`, so the gateway never routes it a build job.
Apps are a fixed list in the `cluster` crate, and that is where the change is:

- `zenj` is added to `APPS` (`cluster/src/gateway/hub.rs:109`); `web` takes the list from there
  (`web/src/cluster/endpoint.rs:60-66`) and needs no change of its own for the socket.
- `zenj` is added to `SERVICES` (`hub.rs:122`). An app outside `SERVICES` and not `agent` is
  *locked* and must name a model, so a `zenj` with `model: None` would be closed at the handshake
  (`cluster/src/gateway/admission.rs:26-28`, `:148-152`). As a service it has no floor, no model
  lock and no rollout: each machine's is released on its own — which is exactly right for a process
  motherd deploys must never touch.

Requests reach it only on routes it declares (`Registry::declaring`,
`cluster/src/gateway/registry/pick.rs:205-217`; MCP invariant 5, `CLAUDE.md`).

**The reservation.** `MemberState` has no "live" and `Capacity` no field to carry one
(`cluster/src/frame/types.rs:77-91`, `cluster/src/frame/membership.rs:30-100`). `zenj` says so on
its listener's `/v1/status` instead, with `refuse: "<why>"` **while rekordbox is open** — playing
or between tracks, since a build begun in a pause is still running when the music starts again
(`zenj/src/http.rs`). keep and motherd read it as one of the machine's **guards** (`guards:` in
`keep.yaml` and in motherd's `host:`) and reserve BlackWorld: no switch, accept, restart, stop or
cluster trial goes ahead unless forced; the agent runs no builds or tests; the tenants read and
calibrate nothing; and the member drains while another member serves every app it does. Zen is
never taken down for it — a party is run by Zen. The reservation ends ten minutes after the guard
last refused (`cluster/src/reserve/`). The whole of it is `docs/house_modes.md` §4.

### 3.2 The machine and the session

rekordbox, its audio through the FLX4, and full-screen output on local monitors all need **a local,
logged-in Windows desktop — the console session**. BlackWorld today is run as a server
(2026-11-08, read on the machine):

- keep, motherd and web run in **session 0**, from a boot task as an S4U logon with no desktop
  (`keep/src/owner_access.rs:3-5`, `docs/deployment.md:264-278`). keep starts children with
  `CREATE_NO_WINDOW | DETACHED_PROCESS` (`keep/src/trial.rs:87-104`), and nothing in the repository
  launches into a user's session (no `WTSQueryUserToken` / `CreateProcessAsUser`).
- John reaches it over **Remote Desktop** (session 1). An RDP session plays audio either at the
  client or at the remote machine, not both, and its redirection hides the host's own audio
  devices; its display is virtual, not the monitors on the card.
- rekordbox is **not installed**, the DDJ-FLX4 is **not connected**, and the card drives **one**
  physical monitor (a Lenovo on DisplayPort).

So for the party BlackWorld runs a console session: John's account signed in at the machine, with
automatic sign-in at boot, and no RDP connection during the set (an RDP connection would take the
session from the console). The logon task — "at log on", interactive, from the repository's
root — runs `zenj watch`, which runs `zenj` and starts it again whenever it stops. keep keeps
supervising motherd and web in session 0 as today. Teaching keep to launch a service into the
console session (`WTSQueryUserToken` + `CreateProcessAsUser`) is the cleaner end state and is not
needed for New Year's Eve.

`zenj` is its own Cargo workspace (`zenj/Cargo.toml`, excluded from the repository's): its screens
need wgpu 30, whose `web-sys` the wasm examples' pinned `=0.3.70` cannot meet. Its cluster crates
are path dependencies. Build and test with `--manifest-path zenj/Cargo.toml`.

### 3.3 Threads

Each thread does one thing. Samples and touches travel through lock-free single-producer queues
(`rtrb`), and snapshots and state are published whole (`arc-swap`), so no one waits on anyone.

| Thread | Does |
|---|---|
| MIDI in (midir's callback) | decodes the FLX4's messages (observed, not proxied — §6.3) |
| Conductor | runs transition plans at beat boundaries every 2 ms; owns the virtual MIDI out port |
| Deck reader | samples rekordbox's live deck state (§4.2) every 5 ms |
| Audio capture (cpal's callback) | captures the FLX4's output for the room and every screen (§4.4) |
| Room | levels and landing from the captured audio |
| Runtime (tokio) | the brain (per 500 ms), the request watcher, the tools, the cluster socket |
| Main | the screens: every window, each frame (§8) |
| Lights | reads the deck reader's snapshot, the live view and the conductor's cues, sends state changes to the bulbs over UDP (§9.3) |

The library is built ahead of the night by commands, not by a thread: `zenj library sync` (taste →
TIDAL → crates) and `zenj library analyse` (rekordbox's analysis → profiles).

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

TIDAL is driven through its official API v2 (`openapi.tidal.com/v2`, JSON:API, user login with
OAuth; scopes `playlists.read`, `playlists.write`, `search.read`, `collection.read`). There is no
maintained Rust client, so `zenj` has its own small one, written against TIDAL's published OpenAPI
description. **What is verified and what is not** (2026-11-08): reading user playlists has been
public since August 2024; adding items is reported working by users since 2025
(`POST /playlists/{id}/relationships/items`); removing and reordering items, the ISRC filter's exact
form, the login flows offered (authorization code with PKCE; device code is unconfirmed) and the
rate limits could not be read from TIDAL's reference, which renders only in a browser. Spike 1
settles them against the live API. Responses page at 20 items (`links.next`).

Spotify is read in **development mode** (rules of 2026-02-11/03-09): the app owner needs Premium,
at most five users (two are needed), and playlist contents come from `GET /playlists/{id}/items`
— the old `/tracks` paths are gone — and only for playlists the user owns or collaborates on, which
*ZenJ Requests* is. `external_ids` (the ISRC) stays on track objects. Polling compares each
playlist's `snapshot_id` and fetches items only when it changed. The privacy export is one JSON
record per play (`ts`, `spotify_track_uri`, `ms_played`, `skipped`, …); `spotify_track_uri` can be
null and such records are skipped.

The OAuth client in the repository is the gateway's Google sign-in (`web/src/auth/oidc.rs`: PKCE,
no refresh, no device flow). `zenj` has its own small client for both services: authorization code
with PKCE through a loopback redirect on BlackWorld, refresh tokens kept in its data folder.

### 4.2 Live deck state

rekordbox publishes no deck state; its Ableton Link only receives. ZenJ reads it from rekordbox's
memory, as `rkbx_link` does: for each deck the loaded track, BPM, play state, beat and bar position,
phrase, and which deck is on air. Upstream `rkbx_link` ships offsets for 7.2.2 and sells updates;
an unofficial fork (`BA-24/rkbx_link`, 2026-11-02) publishes Windows offsets for **7.2.19**, checked
on decks 1–2 for beat, BPM and master across restarts, with track titles still empty and decks 3–4
extrapolated. Upstream issue #63 reports wrong phrase data for local files on 7.2.19. `zenj` has its
own reader, starting from those offsets and the fork's offset-finding method, and reads the track
id rather than the title (the title then comes from the library, §4.1).

**rekordbox takes its updates until the freeze, and is held from then.** An update moves the memory
layout, so after each one the reader's offsets are found again with the offset-finding method above,
about an hour a release. From 2026-12-20 the version that was rehearsed on is held, with rekordbox's
update check off, so nothing moves under the reader before the night. The reader validates what it
reads every sample (a track id that exists in the library, a BPM in
range, a beat position that advances with time) and reports itself `blind` rather than guess. A
blind reader drops ZenJ to Automix (§7.4).

### 4.3 Track profiles

Computed ahead of time, never live, for every track in the crates:

- from rekordbox's analysis: BPM, key (as Camelot), beat grid, phrase map (intro, up, chorus, down,
  outro);
- from the audio: an energy curve over the track, and a "sounds like" embedding from an audio model
  run offline on BlackWorld (§10);
- from Zen: what it knows of the song — a singalong, a classic, a slow burner, the year, the scene.
  This is written once per track and kept, so it is a lookup at decision time, not a model call.

### 4.4 The room

The FLX4's output is captured from the system (WASAPI loopback of the FLX4 render device), giving
loudness and spectral energy over the last minutes. The FLX4 has no ASIO driver on Windows: it is
class compliant and rekordbox drives it through WASAPI (`DDJ-FLX4 WASAPI: MASTER` and `PHONES`),
which is also how Pioneer's own streaming guide captures it in OBS — so loopback has a device to
capture. If it captures nothing in practice, the fallback is physical: the FLX4's master out into a
USB audio input (spike 2).

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

The next track is decided as soon as the current one is on air, so a slow or failed model call is
never felt. If Zen does not answer in time (20 s), the shortlist's top entry is taken.

Zen is asked through the gateway's `/v1/chat/completions`, as the service `zenj` — a service key,
as yenbot acts as itself (`web.yaml` `cluster.service_keys`, `host/roles.yaml`) — with no tools and
no thinking, on one conversation for the night (`zenj-2026-12-31`). Every choice and its reason
therefore stays in Zen's memory of the night. The key is made once on BlackWorld, **before the
gateway next starts** (a missing key file stops it): `target\release\web.exe --new-ops-key
web/secrets/zenj.key`.

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

`zenj` opens a virtual MIDI output port that rekordbox maps with MIDI LEARN as a second
controller. BlackWorld runs Windows 11 25H2 (build 26200) with the Windows MIDI Services service
installed (`MidiSrv`, manual start); its built-in Basic Loopback endpoints reach most machines
between November 2026 and January 2027, so the port is a Windows MIDI Services loopback if present
and loopMIDI otherwise (spike 3). MIDI LEARN needs a paid rekordbox plan **or** a connected
hardware-unlock device, which the FLX4 is; the 7.2.18 manual lists the Automix window among the
functions it drives. The full list of learnable functions — in particular browser movement and
"load to deck" — is read from the MIDI LEARN screen on BlackWorld in spike 3. The map covers, per deck: load selected, play, cue,
sync, tempo, channel fader, trim, EQ high/mid/low, filter, beat FX on/off and level, loop in/out,
hot cues 1–8; and globally: crossfader, browser up/down, the sampler pads, Automix on/off.

The map lives in `zenj/maps/rekordbox-7.yaml` and is loaded with the round's rule: a control
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
unknown in this design (§12, spike 1). If it does not refresh on its own, the map gets rekordbox's
refresh action, or *NYE Next* is filled further ahead so the delay never matters.

### 6.3 A person and ZenJ on the same mixer

rekordbox owns the FLX4 directly. `zenj` only **observes** it — Windows MIDI Services lets any number
of applications open the same port — and never sits between the hands and rekordbox. Proxying would
show rekordbox a virtual port instead of the FLX4, losing its native support, and a crash of `zenj`
would kill the controller mid-set.

Who has the decks — the **hands** (`zenj/src/hands.rs`, `Holder`), named apart from the house's
mode (`docs/house_modes.md`), of which the decks are one part — each visible on the screens:

| Hands | Entered | ZenJ does |
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
| `zenj` crashes | `zenj watch` starts it again; rekordbox keeps playing the loaded track and Automix continues |
| rekordbox crashes | Opened again by Zen (`zenj_rekordbox`) or by hand; the crates live in TIDAL and the offline store, so nothing is lost. The reservation outlasts it by ten minutes, so motherd is not started under the gap |
| motherd is busy or restarting | No effect: `zenj` needs Zen only for choices, and falls back to the shortlist |
| A bulb stops answering `getPilot` | marked dead and excluded; the rest of the track carries on (§9.7) |
| Most bulbs are dead | the warm house scene; `zenj_now` says which are not answering; the audio is untouched |
| The lights thread dies | the bulbs hold what they were last sent; nothing else notices, since nothing reads a light |

### 7.3 Watchdog

The conductor checks, every bar: the on-air deck has more than 32 bars left or the next track is
loaded and its plan is armed. If not, it turns Automix on and reports why.

### 7.4 The ladder

**ZenJ** (brain + conductor) → **Automix on *NYE Live*, Zen curating** → **a person**. Each rung
works without the one above it.

The lights are not on this ladder (§9.7). They hang off it: whatever rung the decks are on, the
lights follow the beat, and a light failure changes nothing above.

## 8. Screens

Four outputs: three monitors on BlackWorld's card and the TV on a fourth, wired output (an adapter
if the card's outputs are DisplayPort) — not cast, which adds seconds of delay. One borderless window per output, rendered with `wgpu`.

- **Scenes**: audio-reactive visuals driven by the beat position (§4.2) and the captured audio
  (§4.4), so a scene change lands on a phrase. `lactea` runs the MilkDrop preset library in Rust, so
  there are thousands of scenes from the first day.
- **The layer we own**: now playing and next, a request arriving ("Yen's request is up next"), Zen's
  one-line reasons, who has the decks (§6.3), and the midnight countdown, which takes over every
  screen from 23:59:00.
- The visuals are driven from the predicted beat grid, so they can lead the sound by the TV's
  measured latency, and land on the beat on every screen.

While rekordbox is open BlackWorld is reserved (§3.1): no builds, no ingest or calibration, and its
member drained while another serves every app, so the cores and the card's work are the decks' and
the screens'. Zen's choices are served by the other members while they are up, and by BlackWorld's
own member when they are not.

## 9. Lights

Built in `zenj/src/lights/` against WiZ's protocol and its tests. The bulbs arrive and are fitted
after it, and spikes 6–10 run against the real ones. Until `zenj/lights.yaml` names them there is no
lights thread. ZenJ drives the bulbs **only while rekordbox is open** — the house in Party; the rest
of the time they are the house's (§9.8).

The party room has about twenty adjustable spotlights on a ceiling track: E27 sockets, A60 globes
in deep black cowls, aimed in fixed directions. They are being refitted with **WiZ Colour A60 E27
Gen 2** bulbs (Signify 929003601001 / WIZ601001): 806 lm, RGB plus tunable white, dimmable over
WiFi, 2.4 GHz WiFi and BLE. Twenty-two were bought — twenty fitted, the rest spares. The wall
circuit stays at full power: every dim is a message.

Control is WiZ local UDP JSON on port 38899: `setPilot` with `r,g,b`, `dimming` (10–100), `temp`
and `state`, and `sceneId` and `speed` where a scene wants them; `getPilot` for status; heartbeats
on 38900 after `registration`. There is no authentication, and the WiZ app must have "Allow local
communication" ON (Settings → Security). Community-measured latency is ~20–50 ms, which spike 8
measures here rather than trusting.

### 9.1 The fixture map

`zenj/lights.yaml` lists each bulb once (`zenj/src/lights/fixtures.rs`), and is refused at load
when an id or an address is given twice:

| Field | What it is |
|---|---|
| `id` | what the tools and the cues name it by |
| `ip` | its reserved address on the bulbs' SSID |
| `mac` | what the reservation is for, and what a health poll proves |
| `pos` | its index along the track, from one end |
| `run` | which run of track, when there is more than one |
| `aim` | what it points at: `floor`, `wall`, `decks`, `wash` |
| `tags` | the groups a scene addresses it through |

Position drives chases: a chase walks `pos` in order, within a run. Aim decides which bulbs take
which role: a hit lands on the floor bulbs, a wall wash stays off the dancers' faces, the decks
light the hands. ZenJ addresses every bulb by the `ip` in this file — never by broadcast discovery
on the night (§0). Spike 7 pairs the bulbs, turns on local communication, makes the reservations
and writes this file.

### 9.2 The rate budget

WiZ is not a streaming protocol. Budget ~10 msg/s per bulb sustained, ~20 in a burst, and treat
that as the limit the renderer is designed against rather than the bulb's spec sheet.

The renderer sends **state changes**, not a frame stream: on beats, half-beats at most, and at
phrase boundaries. A per-bulb token bucket (`zenj/src/lights/gate.rs`: 10 a second, a burst of 20)
enforces the rate and coalesces — the latest state wins, so a bulb that missed four updates arrives
at the fifth. Key moments (a drop, midnight) are sent at once whatever the bucket says, and again
15 ms later: UDP has no ack, and one lost datagram is one dark bulb at the moment the room was
waiting for.

The load is twenty bulbs at ≤10 msg/s of ~100-byte datagrams — about 20 KB/s. The network is not
the limit; the bulb is.

### 9.3 Where it runs

The `zenj-lights` thread (`zenj/src/lights/run.rs`), every 10 ms, reads the deck reader's
snapshot — the same beat grid the screens use (§4.2, §8) — and the live view, and takes the
conductor's cues and the tools' asks from queues. It leads the sound by the bulbs' latency,
`lights.lead_ms` in `zenj.yaml`: 35 ms, the community figure, until spike 8 measures the room's.

It never calls a model, and it never waits on the network. Its one UDP socket is non-blocking
(`wire.rs`); a send the socket cannot take is dropped and the gate sends that bulb's state again: a
light that is late is a light that is late, not a beat that is late. The conductor hands it a cue
without waiting either. Nothing in the audio path waits on a bulb.

### 9.4 What drives the look

Energy — seven parts the set plan's target (§5.1), three the room's landing (§4.4) — chooses the
palette and the brightness (`palette.rs`): deep and cool while it warms up, red and white at the
peak. A phrase boundary changes the scene, on the same 32 beats the screens change on: a wash below
0.4, a split and a chase in turn through the middle, a pulse from 0.75 (`scene.rs`).

The transition plan (§5.3) carries lighting cues as `Move::Light` (`zenj/src/plan.rs`), so a cue is
armed, logged and replayed like any other move, and runs only while ZenJ mixes
(`brain/transition.rs`):

| Style | Cues |
|---|---|
| cut | a blackout on the bar before the drop, the hit on it |
| blend | a wash across the blend |
| echo out | a wash for its 8 beats |
| filter sweep | a wash across its 16 beats |

The scene vocabulary is small and named:

| Scene | What the room looks like |
|---|---|
| `wash` | all bulbs one colour, slow |
| `split` | the groups in two colours |
| `chase` | a position-ordered step, one per beat |
| `pulse` | dim and bright, on the beat |
| `hit` | one full-bright flash — real strobe is out of reach for this hardware and is not attempted |
| `blackout` | everything dark |
| `house` | warm white, for hand-back and the end of the night |

### 9.5 Midnight

Midnight is driven from the clock, like the screens' countdown (`director.rs`): the last ten
seconds are a gold wash dimming from full to the bulbs' floor; at 00:00:00 every bulb gets the hit,
sent twice; then a minute of gold pulse, and the set's scenes again. The midnight track's drop lands
on that second (§5.1), so the hit and the drop are one. Both full rehearsals rehearse it.

### 9.6 Hands

When a person has the decks (§6.3, `zenj/src/hands.rs`), the lights keep following the beat but
Zen's scene choices pause to a plain energy-following default — a wash below 0.5, a pulse above —
and no cue runs: nothing Zen imagined is imposed on someone who is mixing. The house lights are on
call whoever has the decks (`lights_house`, §14).

### 9.7 Lights never touch the audio

Lights are the one subsystem that can fail and leave the night untouched. A bulb that stops
answering `getPilot` — each asked every 10 s, staggered so twenty bulbs do not answer at once, dead
after three unanswered (`health.rs`) — is excluded and the rest carry on; one that answers again is
sent its state afresh. With most of the bulbs dead the room falls to the warm house scene, and
`zenj_now` names the ones not answering. If the thread itself dies the bulbs hold what they were
last sent. Nothing in the audio path reads a light.

### 9.8 Outside the party

The bulbs are parts of the moon world's front room like the stereo and the TV
(`docs/house_modes.md`): the survey knows a WiZ bulb by its maker's hardware prefix or the `wiz_`
name it gives the router, and the world works one with `wiz_on`, `wiz_off`, `wiz_dim`,
`wiz_colour`, `wiz_white` and `wiz_status` (`effector-bridge/src/profile/wiz.rs`). The house's
table sets them all at once with `every: wiz`. ZenJ and the house share one wire,
`effector_bridge::proto::wiz`, and never both drive the bulbs: ZenJ only while rekordbox is open,
the house's table the rest of the time.

## 10. Models

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

## 11. The night

- **The house in Party** (`docs/house_modes.md`): rekordbox open, so BlackWorld is reserved for
  as long as it is — nothing deployed there, no builds, no ingest or calibration, and its member
  drained while another serves every app. Zen runs on the other members for the night, and on
  BlackWorld's when no other is ready.
- **Freeze, 2026-12-20**: the rekordbox version rehearsed on is held with its update check off; `zenj`
  and the map file are frozen with it, and so are `zenj/lights.yaml` and the bulbs themselves —
  §16.5 asks whether their internet access is cut at the same moment.
- **Offline store** refreshed after 2026-12-10, with a TIDAL login in rekordbox on 2026-12-30.
- **Two full rehearsals**, each at least four hours with a midnight run-in, before 2026-12-20, both
  with the lights on (§9.5).

## 12. Spikes, in order

Each is a day or less and decides something.

0. **The machine.** rekordbox installed (7.2.19, done 2026-11-08), the FLX4 on USB, the monitors and TV cabled, John
   signed in at the console with automatic sign-in, and a test program started "at log on" playing
   through the FLX4 with no RDP connection.
1. **TIDAL playlist → rekordbox.** Add a track to a TIDAL playlist through the API; time how long
   rekordbox takes to show it, during Automix. Then removal and reordering through the API. Confirm
   that rekordbox analyses TIDAL tracks (§4.1) and that Automix plays them, online and from the
   offline store.
2. **Audio capture.** Capture the FLX4's output while rekordbox plays (WASAPI loopback), in both of
   rekordbox's audio driver modes.
3. **Virtual controller.** A virtual port learned in rekordbox alongside the FLX4: does each learned
   control act, and does a physical touch after a software move behave as §6.3 assumes?
4. **Deck reader.** Locate deck state in the installed rekordbox's memory; validate across load, play,
   sync, and both decks.
5. **Closed-loop load.** Browser to *NYE Next*, load, confirm through the reader.

6. **Lights: fit and dimmer on one fitting.** One bulb in one cowl, on the track, dimmed through
   its whole range. Decides that the cowl does not trap the heat, that the bulb reaches the dim floor
   the map assumes, and that the socket is really E27 in that fitting.
7. **Lights: the fleet.** Pair all twenty-two, turn on local communication, make the DHCP
   reservations, write `zenj/lights.yaml`. Decides whether every bulb is addressable by IP and
   whether the fixture map is real rather than hoped for.
8. **Lights: rate and latency.** The highest sustained `setPilot` rate per bulb before drops and
   lag, and the end-to-end latency from a sent datagram to light in the room. Decides the token
   bucket's rate in §9.2 and the lead the lights thread sends by in §9.3.
9. **Lights: the beat test.** Pulse on every beat of a test track, four bulbs, then all twenty.
   Decides whether the bulbs land together and whether one bulb's lag differs from its neighbours —
   which is what a chase looks like when it is wrong.
10. **Lights: a four-hour soak.** The renderer and the health polls running through a rehearsal.
    Decides whether bulbs drop off the network over a night, and whether the thread leaks.

## 13. Build order

| Dates | Work |
|---|---|
| 2026-11-09 → 11-15 | Spikes 1–5. Spotify exports requested. |
| 11-16 → 11-29 | `zenj` crate: cluster membership, MIDI decoder and observer, conductor, deck reader, TIDAL client, Spotify request watcher, library matching and crates |
| 11-14 → 11-22 | Lights: the bulbs arrive and are fitted; spikes 6–10 — fit, fleet, rate and latency, beat test, soak — write `zenj/lights.yaml` and replace the rate and the lead with the room's numbers (the lights are built, 2026-11-09) |
| 11-30 → 12-06 | The brain: set plan, shortlist, Zen's choice, transition plans; Zen's tools and event stream |
| 12-07 → 12-13 | Screens: scenes, the overlay, the countdown |
| 12-14 → 12-20 | Rehearsals, fixes, freeze — both rehearsals with the lights on |
| 12-30 | Offline refresh login |
| 12-31 | The night |

Every module lands with its unit tests: the decoder against raw bytes, the shortlist scorer against
fixed track profiles, the transition planner against fixed phrase maps, the midnight back-scheduler
against a fixed run-in. A transition plan is replayable against a recorded deck-state trace with no
rekordbox at all.

## 14. Zen's surface

Tools on the `/mcp` surface, served by `zenj` itself, each answering from memory within the 30 ms
rule. `zenj` mounts `mcp_server` at `/v1/zenj/mcp` and declares it, with `/v1/zenj/status` and
`/v1/zenj/events`, in its route table (`zenj/src/presence.rs`). `web/web.yaml`'s `code.tokera.com`
site routes the `/v1/zenj` prefix to `cluster: zenj`, falling back to `127.0.0.1:8097`, and
`mcp/served.yaml` names the `dj` backend, so the aggregator lists and routes its tools. Only the
night's people may call them (`zenj.yaml` `people[].email`). No change to zend.

| Tool | Does |
|---|---|
| `zenj_now` | what is playing on each deck, where in it, who has the decks, and the next track with its reason; and the lights — the scene held, and which bulbs are dead |
| `zenj_queue` | *NYE Next* and the set plan; reorder or pin |
| `zenj_request` | add a request, with an optional note and urgency |
| `zenj_hands` | Human / Assist / ZenJ (moving toward ZenJ needs a person's confirmation in the chat) |
| `zenj_rekordbox` | open or close rekordbox; closing with a deck playing needs `force` (`docs/house_modes.md` §3) |
| `zenj_moment` | schedule a moment: a track at a time, a dedication, an effect |
| `zenj_sfx` | fire a sampler slot now or on the next beat or bar |
| `zenj_events` | the event stream since a cursor: track changed, request arrived, the decks changed hands, rekordbox opened or closed, fault |
| `zenj_said` | a line the room said, which Zen hears at the next choice |
| `zenj_midnight` | the midnight track, whose drop lands on 00:00:00 |
| `zenj_energy` | more or less energy for a while: the curve's target moved from now |
| `lights_scene` | hold a scene from §9.4 — `wash`, `split`, `chase`, `pulse`, `hit`, `blackout`, `house` — on a group (a tag or an aim) or the whole track; `auto` lets every held scene go |
| `lights_house` | house lights on, or back to whatever the set wanted |

Both light tools are served by `zenj` on the same surface, gated by the same `people[].email` list,
and answer from memory within the 30 ms rule; they hand the request to the lights thread and do not
wait for the bulb (`zenj/src/tools/lights.rs`). With no bulbs mapped they say so.

## 15. The journal

Every pick and its reason, every transition and its plan, every request and where it went, every
change of hands, every open and close of rekordbox, and every fault is an event (`zenj/src/events.rs`), and every event is appended, one JSON line
each, to the night's record in ZenJ's data folder (`D:/zenj/nights/2026-12-31.jsonl`). The record
outlives the process, and `zenj_events` reads it back from any cursor. Zen's own choices are also in
its conversation for the night (§5.2).

The next morning Zen writes the set up from that record, as John's AI, through the ordinary journal
tools: what the night was, what landed, what to do differently. That entry is also how the next
party's ZenJ starts out knowing what worked. `zenj` writes nothing to the journal itself: a writer
of its own would need a `zenj` `Writer` and `Actor` in the journals crate and a service-authenticated
write route on zend, and the night's record makes neither necessary.

## 16. Open questions

1. Which pad combinations mean "help me" and "take over" (§6.3).
2. The midnight track (§5.1).
3. Whether stems (HTDemucs) are in scope for New Year's Eve, or after.
4. Whether guests beyond John and Yen can add to *ZenJ Requests* (a QR code to the collaborative
   playlist), or only the two of them.
5. Whether the bulbs' internet access is cut at the freeze, so firmware cannot change under ZenJ
   (§9). **Recommend yes**, mirroring the rekordbox freeze in §11: the bulbs are addressed by IP
   and never reach the cloud during the set, so the only thing the connection can do on the night
   is change them.
6. Whether the bulbs' built-in transition smoothing softens a hit, and if so whether to work with
   it — a `speed` on the `setPilot` — or around it, by sending the hit as two hard states.
7. The exact safe message rate per bulb (§9.2). The budget of ~10/s sustained, ~20 burst, is
   community-measured and assumed; spike 8 replaces it with a number measured here.
