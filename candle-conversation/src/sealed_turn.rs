//! One turn as the substrate holds it once sealed — its ids and the QSA index
//! pages sealed with it — what
//! [`Sequence::sealed_turns`](crate::Sequence::sealed_turns) returns.
//!
//! This is the state a projection hands the model when it borrows the turn, so
//! two runs of the same conversation that differ here are not the same context,
//! whatever their text says.

use crate::index_pages::{self, MalformedPages};

/// A turn's stored index pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SealedPages {
    /// The turn has no page: a projection borrowing its K/V holds tokens no
    /// index accounts for.
    Absent,
    /// The stored payload does not decode, which a projection treats as no
    /// page at all.
    Malformed(MalformedPages),
    /// Each page's token width, in the order the seal produced them.
    Widths(Vec<usize>),
}

impl SealedPages {
    /// Classify a turn's stored page payload.
    pub(crate) fn of(blob: Option<&[u8]>) -> Self {
        match blob {
            None => Self::Absent,
            Some(blob) => match index_pages::decode(blob) {
                Ok(pages) => Self::Widths(pages.iter().map(|(width, _)| *width).collect()),
                Err(e) => Self::Malformed(e),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedTurn {
    /// The turn's index on its timeline.
    pub index: u32,
    /// Every id the turn sealed: its user half, the glue, and its assistant
    /// half.
    pub token_ids: Vec<u32>,
    pub pages: SealedPages,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_turn_without_a_payload_has_no_pages() {
        assert_eq!(SealedPages::of(None), SealedPages::Absent);
    }

    #[test]
    fn a_payload_reads_back_as_its_page_widths_in_order() {
        let blob = index_pages::encode(&[(95, vec![1, 2, 3]), (2, vec![]), (41, vec![9])]);
        assert_eq!(
            SealedPages::of(Some(&blob)),
            SealedPages::Widths(vec![95, 2, 41])
        );
    }

    #[test]
    fn a_payload_that_does_not_decode_is_malformed() {
        // Claims one page and carries none of it.
        let blob = [1u8, 0, 0, 0];
        assert!(matches!(
            SealedPages::of(Some(&blob)),
            SealedPages::Malformed(_)
        ));
    }
}
