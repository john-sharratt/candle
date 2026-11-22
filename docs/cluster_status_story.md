# A failed install: what the member does, and what the status says

**Status: partly built** (2026-11-19).

**Built:**

- the attempts record and the bounded retry of §2.1–§2.2
  (`cluster/src/upgrade/runtime/attempts.rs`), replacing `refused.json`;
- the retries reported apart from the refusals in `COORD`, so the gateway's
  close over a refused floor build (`gateway::upgrades::left_behind`) applies
  only once the tries are spent;
- the status of §3–§4 (`zend/src/mcp/tools/cluster_status.rs`):
  - `behind` for a below-floor member installing nothing;
  - each retry with its kind and when it is due;
  - the rollout line's counts;
  - a machine's keep trial as a timed event, and its connected members said
    as such.

**Not built:**

- §2.3, serving the old build between attempts. It needs the restored member
  to say so at admission, which is a handshake change.
- §2.5, the connector following its daemon out of rotation, and the
  `serving` signal for the daemon's loops.

The kind of a keep rollback is read as `slow`, because keep's local port
serves no state a restored process could read. A watchdog's failure record
names `died` and `cancelled`.

This document is authoritative over:

- what a member does after an install of a new build fails and keep restores
  the old one;
- what may serve from a machine whose member is below the floor;
- how `cluster_status` describes a member and a machine
  (`zend/src/mcp/tools/cluster_status.rs`).

It changes `docs/replicated_substrate.md` §3.5 and §3.9 where they differ, and
those sections take its summary when it is built.

## 1. What went wrong

On 2026-11-18, about 13:20 UTC, WhiteWorld's keep tried a new `motherd` and
rolled it back: *"no proof line within the limit"*, no `INFO serving` inside the
10-minute window. keep restored `5207-e74ade355` (pid 22508). Around 13:30 UTC
BlackWorld's `5211-8088a2457` became the floor for zend and npcd. Six hours
later:

- **WhiteWorld served nothing, and nothing was going to change that.** Its
  zend and npcd were connected below the floor, routed no traffic, and no
  install was under way.
- **Its connector served on.** keep's `mcp` on WhiteWorld, still
  `5207-e74ade355`, was `ready` and routed connector traffic beside
  BlackWorld's `5211`. Two connectors of different builds answered one client,
  which §3.9 of `docs/replicated_substrate.md` calls worse than either.
- **`cluster_status` described it in three contradictory lines**, each true of
  one part of the present and none saying what had happened:

  ```
  - WhiteWorld: not answering — keep runs motherd 5207-e74ade355 as pid 22508, so it is
    loading or stuck
    zend WhiteWorld … build 5207-e74ade355: upgrading — below the floor, with no upgrade
    under way: it is left behind
  upgrade zend: floor raised 6.0 h ago, members below it upgrading
  ```

  | Line | What it said | What was true |
  |---|---|---|
  | machine (`absent_line`) | loading or stuck | its members were connected. Below the floor they are routed nothing, so `/v1/status` cannot reach them through the gateway |
  | member state | `upgrading` | nothing was upgrading. The gateway stands a member admitted below the floor as `Upgrading` for its whole session (`cluster/src/gateway/hub/sockets.rs`) |
  | member note (`upgrade_line`) | no upgrade under way, left behind | true, with no time and no reason |
  | rollout (`render_gateway`) | members below it upgrading | none was |

### 1.1 Why a rolled-back member sits there for good

The path as built:

1. The new process does not prove itself. keep kills it, puts the stable file
   back and starts it with its usual flags.
2. The restored process finds that its install failed. The install plan the
   old process wrote names another build (`upgrade/runtime/failed_install.rs`).
   It adds that build to `<data>/upgrade/refused.json` **for good**
   (`upgrade/runtime/attempts.rs`).
3. It reconnects. Its HELLO is below the floor, so the gateway admits it
   **upgrading**: no traffic, no events applied or sent, no claim, no lease,
   no vote (`docs/replicated_substrate.md` §3.5, `cluster/src/member/app.rs`).
4. `decide::target` takes the floor's build unless it is refused, and a newer
   certified build unless that is refused too. When the build that failed is
   the one it would take, it takes nothing (`decide.rs`, the `refused.contains`
   checks).
5. §3.5 says the gateway then closes it 4410 with the reason. It never does.
   The refusal is the member's alone, never told to the group (`decide.rs`'s
   own module doc says so), so the gateway holds a socket it routes nothing to,
   labelled `upgrading`, until the socket happens to close.

So a single failure is permanent. A trial that missed its window once, on a
machine short of memory while a person was using it, refuses that build for
the life of the data directory. The member neither retries nor goes back to
serving the build that was serving fine a minute before the install.

The exact path on WhiteWorld is still to be read there (§6), because its log
cannot be read through the gateway while it is below the floor. The design
below does not depend on which build it was.

## 2. What a failed install leads to

**A rollback puts the member back where it was before the install: serving on
its old build. It then tries again, a bounded number of times, while the
cluster still lets that build serve. When it may no longer serve, it leaves the
cluster, saying why. It never sits connected and idle.**

### 2.1 The failure is recorded with its cause and counted

`refused.json` becomes `<data>/upgrade/failed.json`: for each build an install
of which failed here, the attempts, and for each attempt when it ended and how:

| Kind | From keep's trial end | Meaning |
|---|---|---|
| `slow` | `rolled back / no proof line within the limit` | the process ran but did not prove itself in the window. Measured causes: catch-up that stalled, a card or host short of memory, a slow prefill |
| `died` | the trial process exited before its proof line | a crash or a refused start: the new build cannot run here as it is |
| `cancelled` | cancelled by hand | not a fault of the build; no attempt is counted |

The restored process reads keep's board for the trial it was restored from,
and its own plan for the build. It records the kind and keep's detail, the
trial log's last lines included.

### 2.2 Retry, bounded

A failed build is tried again **three times**, `RETRY_AFTER` = 10, 30 and 90
minutes after each failure. After that it is refused.

- `slow` retries. The same build on the same machine at a quieter moment is the
  measured fix. WhiteWorld's earlier rollbacks (2026-10-20, and 2026-11-18) were
  both a busy machine, not a bad build.
- `died` retries once. If the second attempt dies with the same last trial
  line, the build is refused at once: the crash is the build's, and three more
  starts would only repeat it.
- A retry is an ordinary install, under the same lease and the same proof
  window, so it is still one member at a time.
- A build named by a forced rollout (§3.11 there) is tried as the rollout says,
  as today. A build the member runs is forgiven, as today (`refused::forgive`).
- Each attempt's end is in the member's upgrade report, so the status can show
  the history (§4).

### 2.3 Between attempts: serve the old build if the cluster allows it

The member was serving on its old build when it began the install. Restored, it
asks to be admitted as it was:

- Its HELLO carries **`restored_from: <failed build>`**. The gateway admits a
  restored member below the floor as an **older member waiting to upgrade**,
  the state every connected older member is in after the floor rises (§3.9
  step 1 there). It is routed traffic, applies and sends events, and holds
  claims, as before the install.
- This holds **while older builds may serve**, which is exactly as long as an
  older member that never tried to install would still be serving beside the
  floor:
  - not past the app's deadline for older builds (`deadline_ms`, "older builds
    served until the grace period ends");
  - not across a protocol bump (§3.5 there: a member behind the protocol cannot
    take the group's frames);
  - not when the floor's build changed the log format or a record type, the
    checks `upgrade/compat.rs` already makes.
- Otherwise it is admitted upgrading, as now, and serves nothing until the next
  attempt.

A member serving on its old build between attempts is still a member below the
floor. The rollout does not count it settled, and its account says it serves an
older build with an install pending (§4).

### 2.4 When it cannot install and may not serve: out, with the reason

When the retries are spent, or the build is refused, and the member may not
serve its old build (§2.3), it **leaves the cluster** rather than sitting
connected and idle:

- It reports itself **`stranded: install of <build> failed <n> times — <kind>,
  <last detail>`** in `COORD`. The gateway closes it 4410 at once with that
  reason, as §3.5 there already says for a stranded member. Now the refusal
  reaches the group.
- It does not redial until something changes: a newer certified build it never
  failed, a forced rollout naming it, or a restart by hand. A 4410 member
  that redials at once only repeats the close.
- keep keeps it running on the old build. A person on the machine still has the
  process, its local GUI and its log. The cluster simply does not route to it.

### 2.5 Nothing serves from a member that is not serving

A member admitted upgrading, or closed 4410, serves nothing through the
gateway, and its machine's other parts follow it:

- **The connector follows the daemon.** The gateway routes `/mcp` only to a
  connector whose machine's daemon is serving, or to one of the floor's build.
  WhiteWorld's `5207` connector would have been out of rotation from the moment
  its daemon was admitted upgrading. A connector of the floor's build on a
  machine whose daemon is behind may stay. It serves through the gateway to
  whichever members are serving.
- **The daemon's own background work waits for admission.** Every loop in zend
  and npcd that writes is gated on one host signal, `serving`: set when the
  member is admitted as a serving member, cleared when it is admitted upgrading,
  closed or drained. That covers resume sweeps, worker revive, journal sync,
  ingest, calibration and scheduled tasks. The 10 s handed-turn sweep
  (`zend/src/session/unfinished_turns.rs` `resume_handed_turns`) is gated today only on
  `leaving`. It relies on the claim to refuse, and the claim is the second
  guard, not the first.
- **The local listener does not stand in for the cluster.** §5.8 there serves
  from the member's own listener only when it is *cut off*, not when it is
  admitted below the floor. A below-floor member answers its local listener
  with what it is: below the floor, and why.

## 3. Gateway state words

- **`upgrading` means an install is under way.** A member admitted below the
  floor and not installing is reported as **`behind`**: a new `MemberState`
  the gateway gives a below-floor admission in place of `Upgrading`. The
  member's own `Upgrading`, sent while `install_tick` runs, keeps the word.
- **An older member serving between attempts** (§2.3) is `ready` with a note,
  `on <build>, install of <floor> pending (attempt <n> of 3 at <time>)`.
- **The rollout line counts:** `floor raised 6.0 h ago; 2 members below it: 0
  installing, 1 serving an older build, 1 behind`, each named in its machine's
  account.

## 4. What the status says: one account per machine

`cluster_status` writes one account per machine, built from every source that
knows something about it. It reads oldest first, then the present:

| Source | What it gives |
|---|---|
| keep's board | what keep runs and since when; the trial in progress; each trial's build, end, proven or rolled back, and keep's detail |
| the gateway's status | each member on the machine: app, build, when admitted, below which floor, its state (§3), `silent`, `lagging`, `syncing`; the connector's routing (§2.5) |
| the member's upgrade report | an install under way and its step; a fetch's progress; `failed.json`'s attempts with their kinds and the next attempt's time; `stranded` and why |
| the app's floor | which build, and since when |

```
- WhiteWorld — motherd 5207-e74ade355, below the floor 5211-8088a2457 (raised 6.0 h ago)
  6.5 h ago  keep tried motherd <build>: rolled back — slow: no proof line within 10 min
             (last line: INFO syncing …)
  6.5 h ago  keep restored 5207-e74ade355 (pid 22508); serving on 5207, install pending
  6.0 h ago  floor raised to 5211; older builds may serve until 30 min from now
  now        attempt 2 of 3 of 5211 at 20:10 UTC; connector 5207 out of rotation
```

- **Every event carries its time**, from the record that holds it.
- **The last line is the present**, worded from the same facts. It never
  guesses between causes ("loading or stuck"). When there is no member session
  and keep runs the process, it says *no member connected since <time>*.
- **A reason on record is quoted verbatim**: keep's trial detail, the trial
  log's last line, a `stranded` reason.

## 5. Tests

**Member** (`cluster/src/upgrade/runtime/`, raw expected values):

- A restored process records the failed build in `failed.json` with its kind
  from keep's trial end: `slow`, `died` or `cancelled`. Cancelled counts no
  attempt.
- `decide::target` takes a failed build again at 10, 30 and 90 minutes after
  each failure, and not before. After three it is refused. A `died` failure
  repeated with the same last line is refused at once.
- A restored process's HELLO carries `restored_from`.
- With the retries spent and no serving allowed, the member reports
  `stranded` with the attempts in its reason.
- The `serving` signal clears on a below-floor admission and on close, and
  every writing loop it gates does nothing while it is clear.

**Gateway** (`cluster/src/gateway/`):

- A restored member below the floor is admitted serving while older builds may
  serve, and upgrading once they may not.
- A below-floor admission that is not installing is `MemberState::Behind`. A
  member's own `Upgrading` passes through.
- A stranded member is closed 4410 with its reason.
- `/mcp` is routed to no connector whose machine's daemon is behind or closed,
  unless it is of the floor's build.

**Status** (`zend/src/mcp/tools/cluster_status.rs`, rendered text exactly):

- The WhiteWorld case renders the account of §4, times and reasons included.
- A trial in progress is the newest event, with how far into its window it is.
- The rollout line counts installing, serving-older and behind members.
- Keep running motherd with no member session renders *no member connected
  since <time>*, never "loading or stuck".

## 6. Open

- **WhiteWorld's exact path.** Which build its rolled-back trial was, whether
  `refused.json` names the floor's build, and what its upgrade step says. All
  of it is on WhiteWorld (`<data>/upgrade/refused.json`, `watch.json`, keep's
  trial log under `keep/state/`), and none of it can be read through the
  gateway while its members are below the floor. Read it there before building
  this, so §2's first test is that machine's case exactly.
- **BlackWorld's unreceipted events.** Over the same hours, BlackWorld's
  replication warned every ten minutes, from 14:36 UTC, that 964 of its own
  events in flight had no receipt (`cluster::replicate::tick`, oldest
  `oseq` 41461282 throughout). It cleared briefly at 19:12, 19:34 and 19:49 and
  stalled again. Whether a two-member cluster with its other member behind can
  receipt at all is to be traced. It is not assumed to be this fault.
