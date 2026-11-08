# Zen and the DJ set

**Status: design, not built.** Nothing in this workspace speaks MIDI today — §0.2 says what was
searched and what came back empty. This document is the design basis for hooking a physical MIDI
controller on BlackWorld into Zen: what the device is, how Zen finds it and claims it, how a knob
turn becomes something Zen can read, what Zen does with it, and what happens when the cable comes
out.

Zen on BlackWorld opens the controller as a **source**, not as a tool the model drives. A listener
lives on the machine the cable is in, decodes bytes into named events, and hands them to the world
as terminal events and to the journal as observations — so a set played at two in the morning is
something Zen can read back next week and say what happened. The controller is never driven by the
model: the hands are John's. And the model never sees raw MIDI, only what the bridge has decided a
message means.

The one hard decision, and the reason this is not simply another bridge profile, is that MIDI is
not an IP device. The house bridge's whole safety model is an address test — `in_house` (`effector-bridge/src/scope.rs:49-53`)
— and a USB device has no address to test. §2 replaces that test with the one that does apply here:
the agent that owns the port is the machine the cable is in.

---

## 0. Sources, and what was not found

Everything this document asserts about the estate comes from something in it. Where a claim is
about the MIDI specification itself it is cited to the web; where it is about the device, the
source is named and the gap is stated.

**Read, and used:**

- `docs/zen_moon_mind.md` — §13 the house link and `survey_network`, §14 the world building itself,
  §15 bridges and profiles, §16 the effector reaching the internet; §4 how the real world reaches
  the moon (the tannoy and the terminal event); §17 Zen's agenda, the round, and what breaks into
  it.
- `docs/the_effector.md` — the one device tool, verbs as resources, prescriptive errors, and
  Appendix F on reshaping a running map.
- `effector-bridge/` — `Cargo.toml`, `src/lib.rs`, `src/scope.rs`, `src/finding.rs`, `src/work.rs`.
  The profiles that exist are `sonos`, `receiver`, `cast`, `webos`, `androidtv`
  (`effector-bridge/src/profile/mod.rs:1-11`); the profile constants are `finding.rs:12-34`.
- `agent/src/wire.rs:379-425` — `BridgeCall::{Work, Wake, Pair}` and `Bridged {ok, line, creds}`.
- `agent/src/routes.rs:227-254` — `POST /v1/bridge`, where an agent runs `effector_bridge` for a
  call; `agent/src/client.rs:778-797` for the in-process form.
- `npcd/src/bridge/mod.rs` — `enact` (lines 244-394): the survey, `wake_device`, pairing, and the
  profile acts; `parts_here` (44-68) which refuses a bridge anywhere but a mirrored room; `submit`
  (377-393) which refuses rather than silently doing nothing.
- `docs/npcd_discord.md` — the one inbound source in the estate that is pushed rather than polled:
  §5 "everything runs on the world's authority", the shard handing events to an inbound queue
  drained on the bots' beat (`npcd/src/discord/gateway/`, `docs/npcd_discord.md:725-735`), and the
  reconnect ladder of 1 s doubling to 60 s. It is the shape this document copies.
- `docs/journal.md` — the entry shape and the four kinds; `journal-core/src/entry.rs:15-28`, where
  `Observed` is "a tool result, or an event the writer itself received".
- `docs/deployment.md` — the machines, `keep`, and the release path; `host/machines.yaml:29-35` for
  BlackWorld's own entry (RTX PRO 5000, 72 GB, PCIe 5.0; conversations rank 1, min 4 max 8;
  pictures 2; characters 3; builds 1). **The three sources that state its host RAM disagree** —
  `docs/hardware_machine_consumer.md:35` says 192 GB DDR5-5600 (4×48 GB, 1 DPC), `README.md:198`
  says 189 GB, and `host/machines.yaml:29` says 194 GB. I believe the build document's 192 GB
  nominal (4×48 GB is exactly that, and 189 is what Windows reports as usable), and the machine
  table's 194 is the outlier; nothing in this design depends on the number, so it is noted here
  rather than resolved.
- `docs/zen_journal.md` and `journals/README.md` — the journal as the record, and the entry format
  `yyyy-mm-dd-hhmm-xxxx.yaml` with `at`/`kind`/`by`/`account`/`text`.
- `docs/mcp_user_context.md`, `docs/mcp_app.md`, `zend/src/mcp/tools/` — the tool surface Zen
  actually has, and the 30 ms p95 rule for every `/mcp` tool that runs no model
  (`CLAUDE.md:271-310`).

**Not found, and therefore not claimed:**

- No MIDI code, dependency, or design anywhere in the workspace. `midi` searched across every
  repository returned only false positives — `humidity` in weather tooling and the wiki corpus, and
  `Midian`/`Emidia` in `crates/llm/src/mind/name_gender_dataset.csv`. `midir`, `winmm`, `coremidi`
  and `usb-midi` appear nowhere; the only USB in the tree is the AV receiver's USB *source* input
  (`effector-bridge/src/profile/receiver.rs:86,142`) and the board's USB4 ports, which are switched
  off in BIOS to feed the second NVMe (`docs/hardware_machine_consumer.md:63,104`).
- **Which controller John owns.** No journal entry, doc, or config names it. Every byte-level
  statement in §1 about pads, faders and LEDs is therefore generic MIDI, not this device; §1.3
  lists exactly what has to be read off the hardware before §3's mapping table can be written.
- **That the controller is connected to BlackWorld today.** Nothing in the estate says so; the
  thread this design came out of (`blackworld-dj-set`) is recorded in the journals only as a
  thread attach and a repo task start (`journals/groups/zen/conversations/2026-11-07-2126-grok/`),
  with no device named.
- **BlackWorld's operating system is not stated in the deployment doc.** It is inferred from
  `keep/keep.yaml` and `keep/src/owner_access.rs` (Win32 security tokens, `D:\prog\candle` paths in
  `docs/deployment.md`), which is Windows. §2.3 and §5.3 depend on that, and say so.
- No journal entry in `journals/` mentions MIDI, a DJ set, or audio at all.

## 1. What the DJ set is, and what it exposes over MIDI

### 1.1 The thing itself

A DJ controller is a USB surface of faders, knobs, pads, transport buttons, a crossfader and a
pitch jog, which speaks MIDI over the cable. It is not storage, it is not a display, and it holds
no state Zen can read back: what it is, is a stream of small messages. The estate's other devices
are asked a question and answer it — `sonos_status`, `receiver_status`, `cast_status`
(`docs/zen_moon_mind.md:339-346`) — and a controller answers nothing. It talks, and the only way to
know what it said is to be listening when it said it. That is why it is a **source** and not a
profile of the kind `effector-bridge` already has, and why this design is mostly about the listener.

### 1.2 What MIDI carries, and what it does not

MIDI 1.0 messages are a status byte with its high bit set (0x80–0xFF) followed by zero to two data
bytes in 0x00–0x7F, System Exclusive being the exception: an arbitrary number of data bytes between
0xF0 and 0xF7 (https://midi.org/summary-of-midi-1-0-messages, https://studiocode.dev/kb/MIDI/midi).
What matters for a controller:

- **Note On / Note Off** (`0x9n` / `0x8n`, note number and velocity). Pads and buttons on most
  controllers are notes; a Note On with velocity 0 acts as a Note Off, which is why running status
  can work at all (http://www.somascape.org/midi/tech/spec.html).
- **Control Change** (`0xBn`, controller 0–119, value 0–127 — midi.org's table). Faders, knobs and
  the crossfader are CCs. A fader at rest emits nothing; a fader moving emits a stream, and a
  jittery contact emits a burst of near-identical values.
- **Program Change** (`0xCn`) — a bank or preset button.
- **System Real-Time**: Timing Clock (`0xF8`) is sent 24 times per quarter note when the device is
  the clock master, with Start / Continue / Stop around it (midi.org). A controller that sends a
  clock is the only thing in the estate that hands Zen a *tempo* rather than a fact, and it does so
  at a rate that must never reach a model.
- **System Exclusive** (`0xF0 … 0xF7`) — manufacturer-specific: device dumps, and on some
  controllers the LED and RGB-pad writes that are the only way to light them. SysEx cancels running
  status, so any message after one must carry its own status byte (somascape.org).
- **Running status** — a device may omit a repeated status byte, so a byte stream is not a message
  stream: the decoder must carry the last status forward, and drop it on SysEx and on real-time
  messages (somascape.org).

MIDI carries **no names, no semantics, and no timestamps of its own** beyond what the host stamps.
CC #1 is "modulation" and CC #7 "volume" only by convention (midi.org); which physical knob is
CC #41 on one controller is a different knob on another. Nothing in the byte stream says which
controller sent it either — the port does.

### 1.3 What must be read off the hardware before the mapping is written

The estate has no record of which controller this is (§0.2), and the mapping table in §3.3 is
unwritable without it. These are the facts to gather, and how:

1. **The port list, with each port's name and id.** `midir::MidiInput::ports()` gives one entry per
   port with metadata, and `port_name` errors when the port is no longer valid — e.g. the device has
   been disconnected (https://docs.rs/midir/latest/midir/struct.MidiInput.html). A throwaway example
   that prints the ports with the device plugged in and then unplugged names it and proves the id.
2. **Whether it is class-compliant or SysEx-bound.** Plug it in, watch the stream, and see whether a
   knob move yields `0xBn` or `0xF0`. Class-compliant means the mapping below is enough; SysEx means
   the manufacturer's MIDI implementation document is needed, and that is a research task with a
   known answer, not a guess.
3. **Its control map.** Which CC number each knob and fader sends, which note each pad sends, and
   what channel. This is either in the manufacturer's manual or read off the wire: move each control
   once, in order, and read the log. The log of §3.1 is built for exactly this and is the cheapest
   way to get it.
4. **Whether it sends a clock, and whether it can be told not to.** §1.2's `0xF8` at 24 per beat is
   48 messages a second at 120 BPM, and 96 at 240 BPM with a fast jog.
5. **Whether it is bus-powered or self-powered, and whether it enumerates before the daemon starts.**
   This decides whether §2.3's claim is a startup step or a hot-plug step.

## 2. Discovery, and claiming the device

### 2.1 Why the survey does not find it

The house survey finds what announces itself: the neighbour table and the router's name for each
address, SSDP for UPnP devices, mDNS for Cast, Sonos, Spotify Connect, printers and phones
(`docs/zen_moon_mind.md:264-271`). A USB MIDI controller announces nothing. It has no IP address,
it is not in the neighbour table, and it does not answer a query. So `survey_network` will never
find it, and `onboard_part` — which takes a finding, `from` a label (`docs/zen_moon_mind.md:294-302`)
— has nothing to onboard from.

This is not a defect to fix in the survey. The survey is a network survey, and the reason it is
safe is that it is one: it probes nothing, never walks an address range, and drops any device whose
address is not private on the member's own subnet (`docs/zen_moon_mind.md:273-277`,
`effector-bridge/src/scope.rs:49-53`). Adding a USB device to it would mean inventing an address
for something that has none, and then applying an address test to it that would always pass.

### 2.2 The claim, stated plainly

**The device is claimed by the machine the cable is in, and that machine is the only place it can
be claimed.** USB is the scope. The house-scope test is replaced by the port test: the listener
opens a port that exists on this machine, and nothing anywhere else can open it at the same time.

This is the whole of the security argument, and it is stronger than the one it replaces: nothing
remote can reach the controller, and nothing Zen does can make it reach anything. The controller is
an input. It has no network, no credentials, and no route to the internet, so §16 of
`docs/zen_moon_mind.md` (the effector reaching the internet, refusing private addresses) is not
engaged at all.

The one thing the claim must still decide is *which member*, because BlackWorld is in a cluster
(`host/machines.yaml:29-35`) and a world's authority can move (`docs/npcd_discord.md:534-538`,
`npcd/src/authority/`). The cable does not move. So the listener is **not** bound to the world's
authority the way a Discord shard is: it is bound to the machine, and it forwards to whichever
member is the authority, holding events when none is.

### 2.3 Claiming it

1. **Config, not discovery.** The device is named in `motherd/motherd.yaml` — the machine's own
   config, like `host/machines.yaml` is the machine table — by port id, or by a name substring with
   the rule that exactly one port must match. A name that matches nothing is a startup warning, not
   a failure: the daemon must start with the controller unplugged (§5.2).
2. **One opener.** `midir` opens an input port exclusively enough for this purpose on Windows; the
   claim is that the daemon that opened the port is the one that owns it, and a second daemon
   starting on the same machine either fails to open or gets an empty stream. The open is done once
   at startup and on replug, never per event. **This is the one claim in §2 that has not been
   verified** — it is what the spike in §6.1 is for.
3. **The part in the world.** The controller is a part in the room it stands in, placed by the
   owner, not by a Drone: `onboard_thing` is the act for a thing that is really there and on no
   network (`docs/zen_moon_mind.md:294-302`), and that is exactly this. Its link is not an address
   but a port id, and the part's `link` field is where the estate already keeps how a part is
   reached (`npcd/src/bridge/mod.rs:34-42`, `Here { link }`).
4. **The journal records the claim.** One entry, `kind: note`, `by: claude-code`, in the group's
   journal, naming the device, the port id, the machine and the mapping's version — the same shape
   as `journals/groups/battle-cities/2026-10-31-0039-46c9.yaml`, which is the precedent for a note
   that says what a thing is, which machine it is on, and where its design lives.

## 3. From a knob to a memory

### 3.1 The path

```
knob, pad, fader            USB
      ↓
midir callback (midir's own thread)          decode.rs: bytes → MidiEvent, stamped
      ↓  ring buffer, one per port, no allocation per event
coalescer (midi/mod.rs)                      last value per control per window; clock counted
      ↓  inbound queue, drained on a beat     the shape of npcd's bots' inbound queue
   ┌──────────────────────┬────────────────────────────┐
   ↓                      ↓                            ↓
terminal event          journal note            the live surface
EventKind::Terminal     kind: note, Observed    midi_read / midi_state
on the part, in the     (docs/journal.md §1)    over the coalescer, not the log
room, shown until read
```

The listener is a thread of the daemon on BlackWorld, and it owns the port: one opener, one
reader, one decoder. It does not call the engine and it does not write the world — the Discord
shard's rule, "it never writes the world itself" (`docs/npcd_discord.md:725-735`), is the rule here
for the same reason: the thread that owns the socket, or the port, is not the thread that owns the
world, and the world's authority can move out from under it.

The decoder is a pure function from bytes to events, and it is the only part of the bridge that
knows MIDI at all: running status carried forward, SysEx terminated at `0xF7` or by a later status
byte, real-time messages interleaved without breaking the running status (somascape.org, §1.2).
It is unit-testable against raw bytes with no device at all, which is what the repo's rule wants
("for serialization / quantization / codec code, assert against raw expected bytes", `CLAUDE.md:120`).

### 3.2 Coalescing, and why it is not optional

A controller's raw stream is not a record of what happened; it is a record of contact. A fader
swept once sends dozens of CCs; a clock at 120 BPM sends 48 messages a second, and a jog wheel
more. Zen's context is the scarce thing in this estate — the whole of `docs/unbounded_agents.md`
is about that — and a raw MIDI stream would spend it on nothing. So:

- **Control changes coalesce to last-value-wins per control**, with the first and last value and the
  count, so a sweep becomes "crossfader moved from 8 to 119, 43 steps" rather than 43 lines.
- **Notes and pads do not coalesce** — each press is a fact, and a pad is a button.
- **The clock is never forwarded as events.** It is folded into a tempo estimate: beat times, and
  the estimate derived from them. The raw ticks are not kept.
- **A window is closed by quiet, not by a timer alone.** A second of silence on a control is the
  end of a gesture; the coalescer emits at the boundary, so what Zen reads is gestures, not bytes.

The numbers are the ones to measure, not to guess: §6.2 is the spike that counts what a real set
puts through the port, and the coalescer's ratios are tuned against that count.

### 3.3 The mapping is data, not code

The mapping from control to meaning lives in `midi/<controller>.yaml` in the mind, beside the
standing missions and the round (`docs/zen_moon_mind.md:212-218`, `docs/zen_moon_mind.md:430-431`),
so it is in source control and readable, and so a controller swap is a file change and not a build.
It is loaded with the same rule as the round: a file that names a control twice, or names a control
that is not in the control map read off the wire (§1.3), is refused whole at load, naming the file
and the fault.

What a mapping gives is *names*: `deck_a_crossfader`, `cue_1`, `fx_knob_2`, `sync`. A name is what
makes the coalesced event worth reading and what makes the journal entry about it worth keeping.
Without a name, a journal entry is `CC 41 → 96`, which is not a memory.

### 3.4 The two tools, and the 30 ms they must answer in

Two tools, both on the `/mcp` surface, both reading the coalescer in memory and nothing else:

- **`midi_state`** — what the controller is now: each named control's last value, when it last
  moved, whether the device is present, and the tempo estimate if the device sends a clock.
- **`midi_read`** — what has happened since the caller's cursor: the gestures the coalescer
  emitted, and the cursor to read with next time.

Both are subject to the MCP request-path rules and must pass the same gate: every `/mcp` tool that
runs no model answers within 30 ms at the 95th percentile, gateway hops included, and the gate is
`cargo test --release -p zend --lib latency -- --ignored --test-threads=1` (`CLAUDE.md:271-310`,
`zend/src/mcp/latency_tests.rs`). That is why the tools read a coalescer and never the port: a tool
that waited on the device would be one waiting on something, which the estate's rule says is the
only way a tool goes over budget (`CLAUDE.md:273-276`).

They are registered as ordinary tools in `zend/src/mcp/tools/` and declared in the projection
schema, so provenance selection decides when they are in the window — the same way tools are
selected into context today, by the shipped per-token `WideQSig` folding and late-fusion scorer
(`docs/tool_selection_provenance_results.md`, §23 onward, as the docs index describes it) and the
scoring scopes of `docs/provenance_score_normalization.md`. A tool that is never in the window is a
tool that is never called, and the DJ set is not a thing Zen should be thinking about while it is
writing code.

## 4. What Zen does with it

### 4.1 The default: it notices, and it remembers

A gesture is a **terminal event** on the controller's part: the same shape as the hallway camera
seeing movement (`docs/zen_moon_mind.md:71-78`), shown on the terminal until read, with Zen told
which room has a terminal waiting wherever its focus is. It is not a tannoy. `EventKind::Announcement`
is for what happened to everyone at once at a salience that breaks off what they are doing
(`docs/zen_moon_mind.md:71-74`), and a knob is not that. Two things on a DJ set could be: the set
ending, and the deck stopping when it should not have. Those are the only two this design raises
above terminal events, and each is raised by the bridge, not by the model.

What Zen does with a terminal event is Zen's, and the estate already has the machinery for it: the
agenda (`docs/zen_moon_mind.md:385-397`), where a message that asks for something becomes a mission
at `now` and a message that needs only an answer is answered and needs no mission
(`docs/zen_moon_mind.md:411-414`). A gesture is neither a message nor a mission until Zen makes it
one. The default is the cheapest and the right one: **read it, and write it down**.

### 4.2 The memory is the point

The reason this is in the estate and not in a Max patch is that the set is worth remembering. A
journal entry per stretch, `kind: note`, `by: claude-code`, `account: john` — the shape of
`journals/groups/battle-cities/2026-10-31-0039-46c9.yaml` — whose text is what the set was: which
track was on which deck when, where the mix went, what was cued and never taken, where the energy
went and what moved it. The claim kind is `Observed` — "a tool result, or an event the writer
itself received" (`journal-core/src/entry.rs:15-20`) — and the entry cites the stretch it came
from, which is the citable-turn rule of `docs/journal.md:71-81`.

Two kinds of entry, and the difference is the one that matters:

- **During the set**, a note is cheap and short: what changed, when, and by how much. It is written
  by the bridge, not by the model, and it costs nothing.
- **After the set**, a note is what the set meant. That is a model turn, and it is a round item
  (`docs/zen_moon_mind.md:430-454`) with `when: set-ended` — a new `when`, like `reports` and
  `table-empty`, holding that something is waiting to be read — and `every:` absent, because a set
  that has been written up has been written up.

The engine's own rule is the one that keeps this honest: **an item is due only when doing it can
change something** (`docs/zen_moon_mind.md:477-478`), and a round mission is closed by the engine
with no report when its steps are signed off (`docs/zen_moon_mind.md:491-492`). The step "write the
set up" is signed off by the journal entry landing.

### 4.3 What Zen must not do

- **It must not answer every gesture.** A controller in a set is a firehose; a model woken per
  gesture is a model that talks over the music and spends the context the engine exists to save.
  Zen is woken by the room, at most once per beat, and by the end of the set.
- **It must not drive the controller.** There is no `midi_write` in this design, and no LED, no
  fader, no transport. Output is a different design with a different safety argument, and the one
  thing that makes this one safe is that the device cannot be made to do anything.
- **It must not loop.** The loop guard strikes a repeated act as ever (`engine::loopguard`,
  `docs/zen_moon_mind.md:513-515`); a source that keeps firing while Zen keeps reading it is the
  exact shape of that bug, and the read cursor of §3.1 is what keeps it from happening — a read
  advances it, and a read of nothing is not a reason to be woken again.
- **It must not pretend the device is there.** The estate's rule for a source whose source is gone
  is that the terminal says so (`docs/zen_moon_mind.md:120-122`), and `npcd/src/bridge/mod.rs:377-393`
  refuses rather than silently doing nothing. The MIDI listener follows both.

## 5. Failure, and the cable coming out

### 5.1 The rule

**A device that is not there is a fact about the world, not an error in the daemon.** The estate
already has the right shape for this twice over: a bridge's refusal is a refusal and not a failed
job (`Bridged { ok: false, line }`, `agent/src/wire.rs:414-425`, and `npcd/src/bridge/mod.rs:377-393`
which refuses rather than silently doing nothing), and the fault path in the engine is built to end
a wait and leave the context usable rather than to trap (`candle-core/tests/pinned_ceiling_probe.rs`
proves the same discipline for the expert path: the launch ends, the synchronise succeeds, the host
fails the forward from the fault word).

So: the listener never panics on a device fault, and the daemon never fails to start because a
controller is unplugged.

### 5.2 Unplug

The port's callback stops. That is the whole of the detection — there is no error to catch, no
exception, no disconnect event that midir promises (§6.1 must establish what the library does
offer, and `port_name` on a stale port is documented to error:
https://docs.rs/midir/latest/midir/struct.MidiInput.html).

What happens, in order:

1. The coalescer closes its open windows as it is, so the last gesture is not lost, and marks the
   stretch **ended by disconnect** rather than by quiet.
2. The part's terminal shows the device as gone, and the room says so — the rule for a terminal
   whose source is gone (`docs/zen_moon_mind.md:120-122`).
3. A journal note: the set was interrupted at HH:MM, with the last gesture before it. This is the
   entry that matters most, because it is the one that explains why the record stops.
4. The listener stops trying to read and starts trying to **find**, at the reconnect ladder's pace
   — 1 s, doubling to 60 s, which is the Discord shard's ladder (`docs/npcd_discord.md:725-735`)
   and is right here for the same reason: a thing that is not coming back must not be asked every
   500 ms, and a thing that has come back must not be missed for a minute.
5. Nothing else stops. The world, the daemon, the model and the other members are untouched. The
   device is an input; losing it is like losing a camera.

### 5.3 Replug

The listener finds the port by id, and if the id is gone, by name (§2.3). The estate's rule for a
device that turns up at a new address is to follow it by its hardware address, which outlasts the
change (`docs/zen_moon_mind.md:283-285`) — for MIDI the port id is the analogue, and it is weaker:
Windows is known to re-enumerate USB ids across replugs and reboots, which is why the name is the
fallback and why the mapping of §3.3 is keyed by name, not by id.

On a successful reopen:

- **The state the device does not have is not restored.** The controller has no memory of where
  its faders were, and its physical position may not match what the last values say. So the first
  thing after a reopen is **not** a gesture: the bridge waits for each control to be moved and
  takes its first value as the new truth, rather than assuming the old one. This is the MIDI
  "pick-up" problem, and it is a real one — a fader that reads 12 in the last known state and sits
  at 60 physically will jump the mix the moment it is touched if the bridge assumes the old value.
- A journal note records the reopen, and how long the gap was.
- If the gap was longer than the set, the round item for the set is already due, and the reopen
  does not open a new one.

### 5.4 The faults that are not the cable

| Fault | What it looks like | What happens |
|---|---|---|
| Port claimed by another program | The open fails at startup | The daemon starts, the terminal says the device is there and not ours, and the log says which program holds it when the OS says |
| No port matches the config | Nothing at startup | A warning naming the config key and the ports that were there; the daemon runs; the terminal says so |
| More than one port matches | Ambiguous at startup | Refused whole, naming both ports, and the config is edited to name one — the same rule the round file is held to: a file that names a thing twice is refused at load, naming the file and the fault (`docs/zen_moon_mind.md:466-467`) |
| Malformed bytes / running status gone wrong | A decode that cannot be completed | The bytes are dropped and counted, never guessed at; the count is on the terminal, and a count that grows is a bug to file, not a thing to fix by guessing |
| A SysEx dump mid-stream | A long run of bytes with no `0xF7` | Dropped at the next status byte, per the spec (somascape.org); counted, not forwarded |
| The listener thread dies | The queue stops draining | The terminal shows the device as gone after one beat's silence; keep's supervision is the backstop (`docs/deployment.md`, `keep/keep.yaml`) |
| BlackWorld is not the world's authority | Events with nowhere to go | The queue holds them; the authority takes them when it is. The queue is bounded, and when it is full the oldest go — the gateway bus's own rule for a subscriber that falls behind (`web/src/bus/mod.rs:1-5`, `QUEUE = 256` at `web/src/bus/mod.rs:30-31`) |

## 6. Open questions, and how to settle them

### 6.1 Open, and answerable by a spike

1. **Which controller, and what it sends.** Nothing in the estate says (§0.2). Spike: a
   `cargo run --example midi_ports` in `effector-bridge` that lists ports with
   `midir::MidiInput::ports()`, opens the one named in the environment, and prints every packet
   with a monotonic timestamp. Plug it in, move everything once, unplug it, replug it. That one
   program answers the port id, the class-compliant question, the control map, the clock question,
   and what a disconnect looks like in this library on this OS. It is a day's work and it is the
   first thing.
2. **Does midir report a disconnect, and how fast?** The docs say `port_name` errors on a port that
   is no longer valid (https://docs.rs/midir/latest/midir/struct.MidiInput.html) and that hot-plug
   is a live question on the library itself (https://github.com/Boddlnagg/midir/issues/86). Until
   the spike answers it, §5.2's detection is "the callback stops" and the find-again ladder is the
   only mechanism, which is correct but slow.
3. **Does Windows let a second opener have the port, and what does it do to the first?** §2.3's
   claim 2 is unverified. The spike answers it by opening the port twice.
4. **Where does the listener live: `zend`, `npcd`, or `effector-bridge`?** The argument for
   `effector-bridge` is that it is the crate for devices and has no engine dependency
   (`effector-bridge/src/lib.rs:10-13`); the argument against is that its whole API is
   `work(profile, name, address, action, args, nets)` and a MIDI listener has no address, no
   action, and no `nets` — it is a subscription, not a call. The argument for `npcd` is that it is
   where the world, the authority and the inbound-queue pattern already are
   (`docs/npcd_discord.md:534-538`). The argument for `zend` is that Zen is the one who wants it.
   **This is a design decision, not a discovery, and it is the one this document cannot settle.**
   The recommendation is `effector-bridge` for the decoder and the mapping (pure, testable, no
   engine dependency) and `npcd` for the listener thread and the queue, with the same split the
   Sonos profile already has between the library and the bridge act.
5. **Which world, and which group.** The device is in the house, so `life`; the set is a thing
   John does, and the `@zen` group is the household's shared memory
   (`journals/groups/zen/group.yaml`). Whether the DJ set is a `life` room or a `battle-cities`
   thing is a question about what the set is for, and only John can answer it.

### 6.2 Open, and answerable only by measurement

- **How many events is a set?** Everything in §3.2 about coalescing is a shape, not a number. The
  spike logs a real set and counts: messages per second at rest, during a mix, during a drop; how
  many CCs per fader sweep; whether the clock is on. The estate's habit is to measure before sizing
  (`docs/hardware_machine_consumer.md` sizes the whole machine from measured working sets; the
  sweep records in `docs/results/` are the precedent), and the coalescer's window is the knob that
  the count sets.
- **Does the coalesced gesture survive into a useful journal entry?** That is a model question, and
  the answer is read out of the entry, not argued. The test is the one the estate already uses for
  memory: does a later session, reading only the entry, say something true about the set that it
  could not have said without it?

### 6.3 Build order

1. **The spike** (§6.1.1): ports, control map, clock, disconnect, double-open. No engine changes.
   Its output is a journal note and the mapping file.
2. **The decoder**, pure, in `effector-bridge`, tested against raw byte vectors — running status,
   SysEx, real-time interleaving, the pick-up rule (§5.3) — with no device and no daemon in the
   test.
3. **The listener and the coalescer**, on BlackWorld, behind a config key; the log of gestures it
   writes is the thing John reads first, before any of Zen is wired to it. This is the piece that
   earns its keep on its own: a set logged is a set remembered, even with no model in the loop.
4. **The part in the world**, and the terminal event on it, with the `midi_state` and `midi_read`
   tools and the latency gate (`CLAUDE.md:271-310`) passing for both.
5. **The journal entry and the round item** — `when: set-ended`, the write-up, and the entry that
   cites the stretch it came from.
6. **Nothing else.** No LED output, no driving the device, no audio, no analysis of the audio
   itself. Each is its own design, and none of them is this one's first step.

Each step is built, tested, formatted and committed before the next, and each is released through
keep (`docs/deployment.md`, `CLAUDE.md:408-443`); a build with local changes is never tried, and a
binary that does not print `INFO serving` is rolled back.
