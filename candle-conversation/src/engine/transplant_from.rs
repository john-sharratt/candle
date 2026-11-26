//! What a raw decode takes from a described slot before its user half: the
//! captured K/V of its prompt, the captured index pages, or both.

/// A described slot's capture ([`crate::SlotLayout::capture`]) and what of it
/// to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransplantFrom {
    pub capture: u64,
    /// Take the captured slot's K/V for the prompt in place of the slot's own.
    pub kv: bool,
    /// Take the captured slot's index pages in place of the slot's own.
    pub index: bool,
}
