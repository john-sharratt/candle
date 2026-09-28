//! The order rows are offered to a wave.
//!
//! Pure policy: no device, no allocator, no measurement. It answers one
//! question — *whose turn is it next* — and nothing about whether the answer
//! fits. What fits is [`super::cost`] and [`super::gate`].

use crate::projection::DecodePriority;

/// Which queue an item came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Decode,
    Prefill,
    /// A section ingest: context-independent K/V prefilled on a scratch slot
    /// and sealed into the substrate — a tool catalog, a repository summary.
    /// No decode follows it, and it belongs to no conversation's priority.
    Section,
}

/// The order rows are offered, as `(priority, kind)` bands taken in sequence:
/// each is drained before the next is offered anything.
///
/// **The order encodes who is waiting.** At `High` a person is blocked on the
/// next token, so that layer's decode takes rows before anything else — a
/// dialogue token stuck behind a bulk-ingest batch is the one latency this
/// engine cannot spend. Below that nobody is waiting on output, and the
/// question inverts: a background decode only finishes a turn already under
/// way, while a background prefill *starts* one, and a chain that never starts
/// is a directory that never completes. So prefill goes first for `Normal` and
/// `Low`, and their decodes take what is left.
///
/// **Sections sit ahead of background prefill.** A section is substrate work
/// with no conversation behind it, so it carries no priority of its own; it
/// takes the `Low` band. It goes before that band's prefills because the
/// caller inserting it is blocked on its seal before it can submit the turn
/// that attends over it — a section that waits behind the turns queued ahead
/// of it is a directory whose next turn cannot be built. Nothing a person is
/// waiting on sits behind it.
///
/// The middle two bands are unexercised by the shipped configuration — only the
/// dialogue layer sets a priority (`High`) and every other layer takes the
/// `Low` default — so today this reduces to High decode, High prefill, section,
/// Low prefill, Low decode. The table is written in full because it is the
/// rule, not because the rule needs seven rows to express.
pub(crate) const BANDS: [(DecodePriority, Kind); 7] = [
    (DecodePriority::High, Kind::Decode),
    (DecodePriority::High, Kind::Prefill),
    (DecodePriority::Normal, Kind::Prefill),
    (DecodePriority::Normal, Kind::Decode),
    (DecodePriority::Low, Kind::Section),
    (DecodePriority::Low, Kind::Prefill),
    (DecodePriority::Low, Kind::Decode),
];

/// A cursor over [`BANDS`].
///
/// Within a band the caller takes items **strictly FIFO**. Order matters more
/// than fit: passing over an item that does not fit for one that does starves
/// the expensive work indefinitely, and the expensive work is never the
/// cheapest — measured, a 2,855-token head held 57 items with the prefill side
/// at zero for as long as the run lasted.
#[derive(Debug, Default)]
pub(crate) struct Order {
    at: usize,
}

impl Order {
    pub(crate) fn new() -> Self {
        Self { at: 0 }
    }

    /// The next band to offer, or `None` when the order is exhausted.
    pub(crate) fn next(&mut self) -> Option<(DecodePriority, Kind)> {
        let band = BANDS.get(self.at).copied();
        if band.is_some() {
            self.at += 1;
        }
        band
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_order_is_the_seven_bands_once_each() {
        let mut o = Order::new();
        let mut seen = Vec::new();
        while let Some(b) = o.next() {
            seen.push(b);
        }
        assert_eq!(seen.as_slice(), &BANDS);
        assert_eq!(o.next(), None, "exhausted stays exhausted");
    }

    /// A person waiting outranks everything; below that, starting work
    /// outranks finishing it.
    #[test]
    fn interactive_decode_leads_and_background_prefill_precedes_its_decode() {
        assert_eq!(BANDS[0], (DecodePriority::High, Kind::Decode));
        let pos = |p, k| BANDS.iter().position(|b| *b == (p, k)).unwrap();
        assert!(pos(DecodePriority::High, Kind::Decode) < pos(DecodePriority::High, Kind::Prefill));
        assert!(
            pos(DecodePriority::Normal, Kind::Prefill) < pos(DecodePriority::Normal, Kind::Decode)
        );
        assert!(pos(DecodePriority::Low, Kind::Prefill) < pos(DecodePriority::Low, Kind::Decode));
        // Every High band precedes every Low band.
        assert!(pos(DecodePriority::High, Kind::Prefill) < pos(DecodePriority::Low, Kind::Prefill));
    }

    /// Sections take the Low band, ahead of its prefills and behind everything
    /// a person could be waiting on.
    #[test]
    fn sections_precede_background_prefill_and_follow_every_interactive_band() {
        let pos = |p, k| BANDS.iter().position(|b| *b == (p, k)).unwrap();
        let section = pos(DecodePriority::Low, Kind::Section);
        assert!(section < pos(DecodePriority::Low, Kind::Prefill));
        assert!(section < pos(DecodePriority::Low, Kind::Decode));
        assert!(pos(DecodePriority::High, Kind::Decode) < section);
        assert!(pos(DecodePriority::High, Kind::Prefill) < section);
        assert!(pos(DecodePriority::Normal, Kind::Decode) < section);
        assert_eq!(
            BANDS.iter().filter(|(_, k)| *k == Kind::Section).count(),
            1,
            "sections have no priority of their own: one band"
        );
    }
}
