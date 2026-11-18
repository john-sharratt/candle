# What happened to a machine, not only what it is now

**Status: designed, not built.** Settled with the user on 2026-11-19. This
document is authoritative over how `cluster_status` describes a member and a
machine (`zend/src/mcp/tools/cluster_status.rs`), and over the state word the
gateway reports for a member admitted below its app's floor. When it is built,
`docs/claude_zend_gateway.md`'s tool table takes its summary.

## 1. What went wrong

On 2026-11-18, about 13:20 UTC, WhiteWorld's keep tried a new `motherd` and
rolled it back: *"no proof line within the limit"*, no `INFO serving` inside the
10-minute window. keep restored `5207-e74ade355` (pid 22508). Around 13:30 UTC
BlackWorld's `5211-8088a2457` became the floor for zend and npcd. Six hours
later, `cluster_status` described WhiteWorld in three lines:

```
- WhiteWorld: not answering — keep runs motherd 5207-e74ade355 as pid 22508, so it is
  loading or stuck: read its lines with `log_grep`
  zend WhiteWorld 13deb3b8062294a0 build 5207-e74ade355: upgrading — below the floor,
  with no upgrade under way: it is left behind — read its log for why
upgrade zend: floor raised 6.0 h ago, members below it upgrading
```

Each line reads only the present, and each reads a different part of it.
Together they contradict one another:

| Line | What it reads | What it said | What was true |
|---|---|---|---|
| machine (`absent_line`) | keep's `running` pid, and no answer to `/v1/status` through the gateway | loading or stuck | its members were connected. A member below the floor is routed nothing, so `/v1/status` cannot reach it. Not answering was the floor's doing, not the process's |
| member state | the gateway's `MemberState` | `upgrading` | nothing was upgrading. The gateway stands a member admitted below the floor as `Upgrading` for its whole session (`cluster/src/gateway/hub.rs`, `capacity` of a `behind` admission) |
| member note (`upgrade_line`) | the member's last upgrade report | no upgrade under way, left behind | true, but it gave no time and no reason, and it sat beside the word `upgrading` |
| upgrade (`render_gateway`) | the rollout's start | members below it upgrading | none was |

Nothing said what had happened: that an install had been tried and rolled back,
when, and why, and that the machine had been below the floor since. That was
all on record, on keep's board (the last trial, its build, its end, its
reason) and in the gateway (when each member was admitted, and below which
floor).

## 2. The rule

**A machine's status says what happened to it, in order, with times, and then
what it is now.** No line reports a state word that a cause, an event or a
measurement does not stand behind. When the cause is not on record, the line
says what is known and that the rest is not.

### 2.1 One account per machine

`cluster_status` writes one account per machine, built from every source that
knows something about it:

| Source | What it gives |
|---|---|
| keep's board (`GET` keep state) | what keep runs and since when; the trial in progress; the last trial: its build, when it ended, proven or rolled back, and keep's reason |
| the gateway's status | each member on the machine: its app, its build, when its session was admitted, whether it was admitted below the floor and which floor, `silent`, `lagging`, `syncing` |
| the member's own report (`/coord/upgrade`) | an install under way and its step; a fetch's progress; `stranded` and why; the last failed install, restored, with its reason and last line; a paused rollout |
| the app's floor | which build is the floor, and since when |

It reads as a short history, oldest first, then the present:

```
- WhiteWorld — motherd 5207-e74ade355, below the floor 5211-8088a2457 (raised 6.0 h ago)
  6.5 h ago  keep tried motherd <build>: rolled back — no proof line within 10 min
  6.5 h ago  keep restored 5207-e74ade355 (pid 22508)
  6.4 h ago  zend and npcd connected below the floor; routed nothing since
  now        no install of 5211 has started on it since the floor was raised
             (its upgrade report says nothing under way); its log cannot be
             read through the gateway while it is below the floor
```

- **Every event carries its time**, from the record that holds it: keep's
  `started_ms` / `ended_ms`, the gateway's admission time, the floor's
  `rollout_since_ms`.
- **The last line is the present**, worded from the same facts. It is never a
  guess between causes ("loading or stuck"). When a member cannot be reached
  at all — no session in the gateway and keep running the process — that is
  what it says, with how long since it was last admitted.
- **A reason the record holds is quoted.** keep's trial detail and the
  member's install-failure reason and last line are shown verbatim. A machine
  stuck below the floor with no failure on record says that no install has
  started since the floor was raised, and since when.

### 2.2 No state word that nothing stands behind

- **`upgrading` means an install is under way.** A member admitted below the
  floor and not installing is reported as **`behind`**: a new `MemberState`
  the gateway gives a `behind` admission in place of `Upgrading`, shown as
  `behind the floor <build> since <time>`. The member's own `Upgrading`, sent
  while `install_tick` runs, keeps the word.
- **The rollout line counts.** `floor raised 6.0 h ago, members below it
  upgrading` becomes `floor raised 6.0 h ago; 2 members below it: 0
  installing, 2 behind`, each named in its machine's account.
- **"not answering" is said only of what was asked and got no answer.** A
  machine whose members are connected below the floor is not "not answering".
  Its `/v1/status` is not routed, and the account says so.

## 3. Tests

Each against fixture JSON of keep's state, the gateway's status and the
members' reports, asserting the rendered text exactly:

- The WhiteWorld case of §1 renders the account of §2.1, times and reasons
  included.
- A trial in progress renders as the newest event, with how far into its window
  it is.
- A member installing renders `upgrading` with its step and its fetch. One
  admitted below the floor and not installing renders `behind` with its floor
  and since when.
- The rollout line counts installing and behind members.
- A machine with keep running motherd and no member admitted renders "no member
  connected since <time>" with keep's pid. It is not "loading or stuck".
- A last trial proven renders as a proven event. One rolled back quotes keep's
  detail.
- The gateway gives a `behind` admission `MemberState::Behind`, and the
  member's own `Upgrading` while installing is passed through.

## 4. Open: why WhiteWorld did not install 5211

This design makes the status say that no install started. It does not explain
why one did not. WhiteWorld's own log is not readable through the gateway while
its members are below the floor, and its agent runs only `cargo`, so the reason
has to be read on WhiteWorld. A member below a raised floor with a certified
build to fetch is meant to install it (`docs/replicated_substrate.md` §3.9). One
that does not, with no `stranded` reason and no failure reported, is a fault of
its own. It is to be traced on WhiteWorld before this design is built, so the
account can name that cause too.
