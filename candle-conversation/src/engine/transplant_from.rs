//! What a raw decode takes from a described slot before its user half: the
//! captured K/V of its prompt, the captured index pages, or both.

/// A described slot's capture ([`crate::SlotLayout::capture`]) and what of it
/// to take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransplantFrom {
    pub capture: u64,
    /// Take the captured slot's K/V for the prompt in place of the slot's own.
    pub kv: bool,
    /// Take the captured slot's index pages in place of the slot's own.
    pub index: bool,
    /// The prompt's last pieces left out of the transplant: prefilled after
    /// it, live over the K/V and index it put in place, as production
    /// prefills what follows the sealed sections it places.
    pub live_tail: usize,
    /// With `kv`, the prompt's chunks, by index, that keep the slot's own K/V
    /// instead of taking the capture's: the anchors of a section — its first
    /// and last tokens — computed in the prompt's context, the rest of it as
    /// sealed. Empty takes every chunk.
    pub own_chunks: Vec<usize>,
}
