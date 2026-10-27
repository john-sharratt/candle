# Performance series

How the performance readings — `/v1/telemetry`'s engine half and `/v1/phases` —
are answered, and how a cluster's are added up.

Code: `host/src/endpoints/slots.rs` (the grid), `host/src/endpoints/waves.rs`
(each series' fold and its `fuse` declaration), `cluster/src/dispatch/fuse.rs`
(the cluster's fold), `web/src/cluster/aggregate.rs` (where the gateway applies
it), `zend/web/perf.html` (the page).

## 1. Why

The engine records a row per wave (about one every two seconds) and a phase
measurement per phase per wave. Answered raw, a page drew an hour as thousands
of points, and a cluster's page drew every member's interleaved, so the more
members answered the finer it got. Each page also added the members up itself.

## 2. One grid

Every row is one **5-second slot of the wall clock**, stamped `at` with the
slot's start in unix milliseconds. The slot still filling is not answered, so
a row once answered never changes, and every member's rows for a moment carry
the same `at`.

Within a member, a slot's samples fold as follows:

| Series | Field | Fold |
|---|---|---|
| `waves` | `ws`, `tok`, `waves`, projection times (`pdrain` … `layout`) | sum |
| | `fwd` | longest |
| | `budget`, `used`, `backlog`, `vram`, `slots`, `dec`, `pre`, `sec` | the slot's last wave |
| | `wave_ms`, `quanta_ms` | from the summed microseconds over all the slot's waves |
| `migrate` | every field | sum |
| `mix` | tokens by format | the slot's last census, keyed by format name |
| `points` | `dur`, `tok`, `by`, `n` | sum, per phase |
| | `win` | the slot's waves' windows, each counted once |
| | `seq` | the most one wave carried |
| | `peak` | the best wave's phase tokens over that wave's window (tok/s) |

`peak` is the delivered rate the phase-timing tooltip quotes for a single
wave. It is not tokens over the phase's own time, which is the scheduler
thread's time to issue work rather than the card's to run it, and not an
average over the slot, which smooths a two-second burst into the quieter waves
beside it.

## 3. A cluster's

Each answer declares how it adds up across members, in a `fuse` block:

```json
"fuse": { "series": { "waves": ["ws", "fwd", "wave_ms", "quanta_ms"],
                      "migrate": ["migrate", "quant", "copy", "total"],
                      "mix": [] },
          "sum": ["host"],
          "union": ["mix_formats"] }
```

An `Aggregate(Envelope)` answer (`docs/replicated_substrate.md` §5.5) whose
members declare `fuse` also carries `cluster`, the members fused by the first
declaration:

- **`series`**: rows with the same `at` and the same string fields (a phase's
  `kind`) are one row. Numbers add, except the fields the series names, which
  take the largest, because the cards ran side by side and a slot's wave
  lasted as long as the longest member's. Tokens over `ws` therefore reads as
  the cluster's rate, and a phase's `peak` as the members' peaks added, which
  is what the cards delivered together. A `null` is a reading that row lacks.
- **`sum`**: numbers add, and a reading one member lacks makes the cluster's
  absent rather than low.
- **`union`**: every member's entries, in first-seen order.

The fused body carries only the declared fields. The page draws `cluster` for
the cluster as a whole and a member's body for one member, through the same
code, and adds nothing up itself.
