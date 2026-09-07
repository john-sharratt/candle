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
/// The middle two bands are unexercised by the shipped configuration — only the
/// dialogue layer sets a priority (`High`) and every other layer takes the
/// `Low` default — so today this reduces to High decode, High prefill, Low
/// prefill, Low decode. The table is written in full because it is the rule,
/// not because the rule needs six rows to express.
pub(crate) const BANDS: [(DecodePriority, Kind); 6] = [
    (DecodePriority::High, Kind::Decode),
    (DecodePriority::High, Kind::Prefill),
    (DecodePriority::Normal, Kind::Prefill),
    (DecodePriority::Normal, Kind::Decode),
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
    fn the_order_is_the_six_bands_once_each() {
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
}
