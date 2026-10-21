//! The CPU double's per-position state: every page and unindexed span placed
//! ahead of a sequence's live tail, in order, each as wide as the tokens it
//! covers — kept so a test can compare what two slots built.
//!
//! Pushed, cut and spliced the way the real index is — a reset drops the
//! sequence's ledger whole — and refusing the same things: a cut or a splice
//! that lands inside a page. Pages carry their width ([`toy_page`]), since the
//! double has no rows to measure.

use candle_transformers::models::positional_piece::PositionalPiece;

use super::TOY_PAGE;

/// A toy page covering `tokens` tokens, told apart by `label`.
pub(super) fn toy_page(tokens: usize, label: &[u8]) -> Vec<u8> {
    let mut page = TOY_PAGE.to_vec();
    page.push(b'W');
    page.extend_from_slice(&(tokens as u32).to_le_bytes());
    page.extend_from_slice(label);
    page
}

/// The tokens a toy page covers; `0` for one that does not say.
fn toy_page_width(page: &[u8]) -> usize {
    let at = TOY_PAGE.len();
    match page.get(at..at + 5) {
        Some([b'W', w @ ..]) => u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize,
        _ => 0,
    }
}

/// One thing placed ahead of the live tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Placed {
    Page { bytes: Vec<u8>, tokens: usize },
    Gap(usize),
}

impl Placed {
    fn tokens(&self) -> usize {
        match self {
            Placed::Page { tokens, .. } => *tokens,
            Placed::Gap(tokens) => *tokens,
        }
    }

    fn of(piece: &PositionalPiece) -> Self {
        match piece {
            PositionalPiece::Page { bytes, .. } => Placed::Page {
                bytes: bytes.to_vec(),
                tokens: toy_page_width(bytes),
            },
            PositionalPiece::Gap(tokens) => Placed::Gap(*tokens),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct PositionalLedger {
    items: Vec<Placed>,
}

impl PositionalLedger {
    pub(super) fn push_page(&mut self, bytes: &[u8]) {
        self.items.push(Placed::Page {
            bytes: bytes.to_vec(),
            tokens: toy_page_width(bytes),
        });
    }

    pub(super) fn push_gap(&mut self, tokens: usize) {
        if tokens > 0 {
            self.items.push(Placed::Gap(tokens));
        }
    }

    /// Keep everything before `pos`, refusing a cut inside a page.
    pub(super) fn truncate(&mut self, pos: usize) -> candle::Result<()> {
        let (at, rest) = self.split_at(pos)?;
        self.items.truncate(at);
        if let Some(gap) = rest {
            self.items.push(gap);
        }
        Ok(())
    }

    /// Replace `[from, to)` with `pieces`, moving everything after.
    pub(super) fn splice(
        &mut self,
        from: usize,
        to: usize,
        pieces: &[PositionalPiece],
    ) -> candle::Result<()> {
        if from > to {
            candle::bail!("toy ledger: a splice of {from}..{to} runs backwards");
        }
        // What stands at or after `to`: the items from there on, the part of a
        // gap straddling it that lies past it included.
        let (at, before_to) = self.split_at(to)?;
        let mut after: Vec<Placed> = Vec::new();
        match before_to {
            Some(Placed::Gap(kept)) => {
                after.push(Placed::Gap(self.items[at].tokens() - kept));
                after.extend(self.items[at + 1..].iter().cloned());
            }
            _ => after.extend(self.items[at..].iter().cloned()),
        }
        self.truncate(from)?;
        self.items.extend(pieces.iter().map(Placed::of));
        self.items.extend(after);
        Ok(())
    }

    /// The items with neighbouring gaps merged and empty ones dropped — the
    /// form two ledgers placing the same state agree on.
    pub(super) fn canonical(&self) -> Vec<Placed> {
        let mut out: Vec<Placed> = Vec::new();
        for item in &self.items {
            match (out.last_mut(), item) {
                (_, Placed::Gap(0)) => {}
                (Some(Placed::Gap(a)), Placed::Gap(b)) => *a += b,
                _ => out.push(item.clone()),
            }
        }
        out
    }

    #[cfg(test)]
    fn of(items: &[Placed]) -> Self {
        Self {
            items: items.to_vec(),
        }
    }

    /// The index of the first item ending past `pos`, and — when a gap
    /// straddles `pos` — the part of it before `pos`. A page straddling it is
    /// refused.
    fn split_at(&self, pos: usize) -> candle::Result<(usize, Option<Placed>)> {
        let mut start = 0usize;
        for (i, item) in self.items.iter().enumerate() {
            let end = start + item.tokens();
            if end > pos {
                return match item {
                    _ if start == pos => Ok((i, None)),
                    Placed::Gap(_) => Ok((i, Some(Placed::Gap(pos - start)))),
                    Placed::Page { .. } => candle::bail!(
                        "toy ledger: a cut at {pos} falls inside the page at {start}..{end}"
                    ),
                };
            }
            start = end;
        }
        Ok((self.items.len(), None))
    }
}

#[test]
fn a_page_carries_its_width() {
    assert_eq!(toy_page_width(&toy_page(37, b"t")), 37);
    assert_eq!(
        toy_page_width(b"TOYPAGE-good"),
        0,
        "a page that does not say"
    );
}

/// Replacing, evicting and inserting each leave the ledger a build of the
/// result would — and a cut or splice inside a page is refused.
#[test]
fn a_splice_leaves_what_a_build_of_the_result_would() {
    let page = |w: usize, l: u8| Placed::Page {
        bytes: toy_page(w, &[l]),
        tokens: w,
    };
    let piece = |w: usize, l: u8| PositionalPiece::page(&toy_page(w, &[l]));
    let mut l = PositionalLedger::of(&[page(10, 1), page(5, 2), Placed::Gap(4), page(7, 3)]);

    l.splice(10, 15, &[piece(8, 9)]).expect("replace page 2");
    assert_eq!(
        l.canonical(),
        [page(10, 1), page(8, 9), Placed::Gap(4), page(7, 3)]
    );

    l.splice(18, 22, &[]).expect("evict the gap");
    assert_eq!(l.canonical(), [page(10, 1), page(8, 9), page(7, 3)]);

    l.splice(0, 0, &[PositionalPiece::Gap(3)])
        .expect("insert at the front");
    assert_eq!(
        l.canonical(),
        [Placed::Gap(3), page(10, 1), page(8, 9), page(7, 3)]
    );

    assert!(
        l.splice(5, 13, &[]).is_err(),
        "5 is inside the page at 3..13"
    );
    assert!(l.truncate(20).is_err(), "20 is inside the page at 13..21");
    l.truncate(21).expect("cut at a page boundary");
    assert_eq!(l.canonical(), [Placed::Gap(3), page(10, 1), page(8, 9)]);
}
