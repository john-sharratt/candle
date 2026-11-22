# Projection swap cap

A mid-decode reprojection may bring only a few new conversations into each
selected group. The rest wait for the next reprojection.

## Why

A reprojection whose selection changes rebuilds the slot from the first piece
that changed. Every piece placed after it is re-injected, and every newly
selected conversation is elevated into VRAM.

Belief hysteresis delays an eviction but does not bound how many newcomers
arrive at once. A group whose fresh scores reshuffle a handful of near-equal
files on each cadence reprojection paid the full rebuild and elevation every
time. The decode it interrupted also read a context that never settled.

## The rule

A mid-decode reprojection (decode position > 0) applies one cap:

| Where | Setting | Default | Unit |
|---|---|---|---|
| Belief-driven turn groups (`repo_map`, `code_reading`, memory tiers) | group `max_swaps` | 2 | conversations (a file, a folder) |

For a **belief-driven group** (`projection/swap_cap.rs`):

- Newcomers are admitted best score first, up to `max_swaps`.
- Each one held back keeps one of the conversations the selection would have
  dropped. The kept one is the leaver with the best current belief, and only
  the turns it already had stay.
- The unit is the conversation, not the turn, because a pick brings its
  exchanges whole.

A held-back newcomer keeps its fresh score. Unselected turns are not carried
(`PriorBelief::from_selection`), so it competes again on the next reprojection
and enters within the cap. New content therefore still arrives, a couple of
conversations per reprojection, without the selection churning.

The **working set** is not capped. `WorkingSet::observe` chooses its pool again
at every observation and a piece that ranks in enters at once
(`docs/zend_working_set.md` §4.3); its size is the card's (`working_set::PoolSize`).

**A cap is never zero.** `max_swaps` is a `NonZeroUsize`, and the YAML loader refuses a
`0`: a cap of zero admits nothing, ever, and the selection freezes on what the
turn opened with. The smallest cap still brings one conversation in, whole.

**What is admitted arrives whole.** The cap bounds how many conversations enter
at once, never how much of one. After the elevate, a conversation (other than
the slot's own dialogue) any of whose projected turns did not come up hot is
withdrawn from that projection entirely and arrives whole on a later one
(`scheduler/whole_conversations.rs`) — a partial lift would otherwise place a
call without its response, or half an exchange, around the gap. A rebuild that
withdrew one does not record its selection's identity, so the next reprojection
of the same selection still brings it.

## Not capped

- **The opening projection of a turn** (position 0). A new question reselects
  freely.
- **Section collections** (the tool catalog). Their selection is small and is
  held across a tool call by the scheduler.
- **The dialogue's working set and its seeds.** A seed is placed when the
  dialogue opens.
- **Ingest conversations reading themselves.** They select every own turn.

## Configuration

```yaml
groups:
  - id: structure
    max_swaps: 2        # per mid-decode reprojection
```
