//! Refuses to seal a per-layer KV snapshot whose layers describe different token
//! windows, since a sealed skew cannot be repaired once it is in the substrate.
//!
//! A skew in a live slot can still be repaired by the rollback and tail-healing
//! paths. Once it is sealed, every projection that borrows the turn carries it
//! into a fresh slot on every layer. The check compares only each chunk's offset
//! and token count, since everything else is legitimately per layer, and it runs
//! once per turn. The two markers name the faults so the conversation layer can
//! recognise a seal failure, and a parked recurrent state is refused until it is
//! restored.

use super::*;

/// The phrase [`assert_sealed_layers_aligned`]'s `bail!` names this fault
/// with — `candle-conversation`'s reactive section-corruption repair
/// (`scheduler::Scheduler::repair_section_if_window_divergence_confirmed`)
/// matches on it to decide whether a seal failure is THIS fault before
/// re-reading the substrate to confirm. A shared constant, not a duplicated
/// literal on each side, so the two can't silently drift apart if this
/// message is ever reworded.
pub const WINDOW_DIVERGENCE_MARKER: &str = "different token windows";

/// The refusal of a parked sequence's state before it is restored
/// ([`ManagedBatchedModel::park_recurrent`]).
pub const RECURRENT_PARKED_MARKER: &str = "recurrent state is parked in its record";

/// Refuse a per-layer snapshot whose layers describe different token windows.
///
/// **This is the boundary between a recoverable skew and a permanent one.** A
/// wave that dies mid-sweep leaves the early layers ahead of the rest, and the
/// engine has repairs for that: `rollback_wave_kv` at the failure,
/// `heal_tail_divergence` at the next forward entry. Every one of them works on
/// a *live slot*. Once the skew is sealed it stops being slot state and becomes
/// substrate state — written per layer, with no cross-layer record — and from
/// then on every projection that borrows the turn injects the skew into a fresh
/// slot, on every layer independently. That is why the same divergence appears
/// at the same chunk index in unrelated sequences: they borrowed the same turn.
///
/// It is also past the reach of the repairs. `heal_tail_divergence` covers a
/// spread of at most one chunk at the tail and clamps at the sealed boundary
/// besides, so a skew in sealed history reports as unhealable — the forward
/// fails, the sequence is dropped, and the conversation dies mid-turn with no
/// way back short of deleting the turn.
///
/// Checking here costs one walk of a list the seal has just built, once per
/// turn, off the hot path. Refusing the seal leaves the live slot intact and
/// repairable; letting it through does not.
///
/// Compares `(offset, token_count)` per chunk — the window geometry, which is
/// what has to agree. Everything else in a `SealedChunk` (gids, palettes,
/// scales, formats) is legitimately per-layer.
///
/// **Every layer is checked, the MTP head's included.** On this stack the head's
/// KV layer is a `stream_only` layer, not a draft one — its `head_wave_pass`
/// steps it over the same rows at the same positions in the same wave, so it is
/// designed to stand at the same length as its siblings and turn-sealing depends
/// on that (see `qwen35::engine::kv_layers`). A divergence on it is therefore a
/// real skew to refuse, not an expected lag.
pub(super) fn assert_sealed_layers_aligned(
    per_layer: &[candle_nn::kv_cache::SealedSequence],
    idx: usize,
    site: &str,
) -> Result<()> {
    /// One chunk's window geometry: `(offset within the physical chunk, valid
    /// tokens from it)`. The part of a `SealedChunk` that must agree across
    /// layers.
    type Window = (u16, u16);

    let windows = |s: &candle_nn::kv_cache::SealedSequence| -> Vec<Window> {
        s.chunks.iter().map(|c| (c.offset, c.token_count)).collect()
    };
    let Some(first) = per_layer.first().map(windows) else {
        return Ok(());
    };
    let Some(bad) = per_layer.iter().position(|s| windows(s) != first) else {
        return Ok(());
    };
    // Name the chunk and group the layers by what they hold there — a
    // contiguous prefix means a per-layer loop stopped early, a lone layer
    // means something special-cases that index, a scatter means an operation
    // applied per layer under its own predicate. The same legend
    // `ChunkedKvBacking::first_window_divergence` reports with, because this is
    // the same fault caught one stage earlier.
    let other = windows(&per_layer[bad]);
    let at = (0..first.len().min(other.len()))
        .find(|&i| first[i] != other[i])
        .unwrap_or_else(|| first.len().min(other.len()));
    let mut groups: Vec<(Option<Window>, Vec<usize>)> = Vec::new();
    for (li, s) in per_layer.iter().enumerate() {
        let v = windows(s).get(at).copied();
        match groups.iter_mut().find(|(k, _)| *k == v) {
            Some((_, ls)) => ls.push(li),
            None => groups.push((v, vec![li])),
        }
    }
    let split = groups
        .iter()
        .map(|(v, ls)| {
            let held = match v {
                Some((off, n)) => format!("(offset {off}, {n} tokens)"),
                None => "no such chunk".to_string(),
            };
            format!("{held} on {} layer(s) {:?}", ls.len(), ls)
        })
        .collect::<Vec<_>>()
        .join("; ");
    candle::bail!(
        "{site}: refusing to seal sequence {idx} — the layers describe {WINDOW_DIVERGENCE_MARKER}, \
         so sealing would persist the skew into the substrate where no repair \
         can reach it. First difference at chunk {at}: {split}. The live slot is still \
         intact and repairable; a wave that died mid-sweep is the usual producer, and \
         its rollback is what should have undone this."
    )
}
