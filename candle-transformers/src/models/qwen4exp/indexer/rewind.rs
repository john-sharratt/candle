//! Undoes a failed wave, and cuts a cache back to a position.
//!
//! A snapshot copies the open block into one of the cache's own buffers, so
//! taking one allocates nothing. Restore leaves that copy in place, because a
//! partial accept can restore the same state twice. A cut keeps only the whole
//! pages below its position and never splits a page. A splice replaces the pages
//! that cover a range by changing their bases, and it is refused while the
//! sequence holds live rows.

use super::*;

impl IndexCache {
    /// A rewind point: the counters, and a copy of the open block's live rows in
    /// one of the cache's own rewind buffers — no allocation, which is what lets a
    /// forward take one after its span has opened.
    pub fn snapshot(&self) -> Result<IndexSnapshot> {
        let raw = if self.n_open > 0 {
            let buf = self.snaps.take()?;
            buf.tensor().slice_set(&self.open_rows()?, 0, 0)?;
            Some(buf)
        } else {
            None
        };
        Ok(IndexSnapshot {
            n_blocks: self.n_blocks,
            raw,
            n_open: self.n_open,
        })
    }

    /// Undo everything a failed wave appended. The keys above `n_blocks` are
    /// dead rows, so nothing has to be erased.
    ///
    /// Borrows the snapshot and copies its open rows back, because a rewind may
    /// restore the same entering state more than once — a partial accept
    /// restores, re-appends the accepted rows, and can be asked again on the
    /// next block. Handing the buffer over would leave the second restore with
    /// rows the first one's re-append had already overwritten.
    pub fn restore(&mut self, snap: &IndexSnapshot) -> Result<()> {
        if let Some(buf) = &snap.raw {
            self.raw
                .slice_set(&buf.tensor().narrow(0, 0, snap.n_open)?, 0, 0)?;
        }
        self.n_blocks = snap.n_blocks;
        self.n_open = snap.n_open;
        Ok(())
    }

    /// The largest position at or before `pos` that [`Self::truncate_to`]
    /// accepts: the start of the page `pos` falls inside, or `pos` itself when
    /// no page straddles it.
    ///
    /// A page can span several pieces — a slot's whole injected prefix is
    /// closed into one page at a unit boundary — so a rebuild that wants to keep
    /// a prefix ending inside it must keep less, and re-inject from the page's
    /// start.
    pub fn cut_floor(&self, pos: usize) -> usize {
        self.pages
            .iter()
            .find(|p| p.base < pos && pos < p.base + p.tokens)
            .map_or(pos, |p| p.base)
    }

    /// Cut the cache back to position `pos`: keep the injected pages that end at
    /// or below it, drop the rest and the live tail, and resume the tail at
    /// `pos`. Returns the pages dropped.
    ///
    /// For a rebuild that keeps its slot's prefix up to a piece boundary and
    /// re-injects only what follows. A page is atomic — its last row is ragged —
    /// so a cut that falls inside one is refused rather than splitting it. The
    /// caller chooses its cut through [`Self::cut_floor`], so a cut anywhere
    /// else means the two have diverged.
    pub fn truncate_to(&mut self, pos: usize) -> Result<Vec<Arc<ResidentPage>>> {
        let keep = self
            .pages
            .iter()
            .take_while(|p| p.base + p.tokens <= pos)
            .count();
        if let Some(straddles) = self.pages.get(keep).filter(|p| p.base < pos) {
            candle::bail!(
                "qsa index: a cut at {pos} falls inside the page covering {}..{} — pages \
                 are atomic, so a cut must land on a page boundary",
                straddles.base,
                straddles.base + straddles.tokens,
            );
        }
        self.keys.clear();
        self.n_blocks = 0;
        self.n_open = 0;
        self.page_rows.truncate(keep + 1);
        self.tail_base = pos;
        // A prompt is a prefix: a cut below its end keeps only what it kept.
        self.prompt_end = self.prompt_end.min(pos);
        // So is each pin: a cut keeps what of it lies below.
        self.pins = self
            .pins
            .iter()
            .map(|r| r.start.min(pos)..r.end.min(pos))
            .filter(|r| !r.is_empty())
            .collect();
        self.layout_gen += 1;
        Ok(self.pages.drain(keep..).map(|p| p.page).collect())
    }

    /// Replace the injected pages covering `[from, to)` with `pieces`, placed
    /// from `from` on, and move every page after `to` by the change in width.
    /// Returns the pages taken out.
    ///
    /// The mid-prefix counterpart of [`Self::truncate_to`] and
    /// [`Self::push_page`], and as cheap: recording only. A page's rows are
    /// rotated to its `base` as the scorer reads them, so moving a page is
    /// changing a number — which is what lets a rebuild change one piece of a
    /// slot without re-placing the pieces after it.
    ///
    /// Only an injected prefix splices: refused once the sequence holds live
    /// rows, as a page is. A page is atomic, so `from` and `to` must each fall
    /// on a page boundary ([`Self::cut_floor`] returns them unchanged); pages
    /// starting in `[from, to)` are the ones taken out.
    pub fn splice_pages(
        &mut self,
        from: usize,
        to: usize,
        pieces: Vec<SplicePiece>,
        ratio: usize,
    ) -> Result<Vec<Arc<ResidentPage>>> {
        self.check_splice(from, to)?;
        let lo = self.pages.partition_point(|p| p.base < from);
        let hi = self.pages.partition_point(|p| p.base < to);
        let mut base = from;
        let mut placed = Vec::with_capacity(pieces.len());
        for piece in pieces {
            match piece {
                SplicePiece::Page(page) => {
                    let tokens = page.tokens(ratio);
                    placed.push(PlacedPage { page, base, tokens });
                    base += tokens;
                }
                SplicePiece::Gap(tokens) => base += tokens,
            }
        }
        // Every position from `to` on moves to `base + (pos - to)`.
        let moved = |pos: usize| base + (pos - to);
        let after: Vec<PlacedPage> = self.pages.drain(hi..).collect();
        let taken: Vec<Arc<ResidentPage>> = self.pages.drain(lo..).map(|p| p.page).collect();
        self.pages.extend(placed);
        self.pages.extend(after.into_iter().map(|p| PlacedPage {
            base: moved(p.base),
            ..p
        }));
        self.page_rows.truncate(1);
        let mut rows = 0usize;
        for p in &self.pages {
            rows += p.page.rows();
            self.page_rows.push(rows);
        }
        self.tail_base = moved(self.tail_base);
        // The prompt is a prefix: it moves with what follows the splice, and a
        // splice inside it keeps only what the splice left before it.
        if self.prompt_end >= to {
            self.prompt_end = moved(self.prompt_end);
        } else if self.prompt_end > from {
            self.prompt_end = from;
        }
        // A pin keeps what of it lies outside the splice, never the new
        // content: its part before the splice stays, its part after moves with
        // what follows, and a pin across the splice becomes the two.
        self.pins = self
            .pins
            .iter()
            .flat_map(|r| {
                let before = r.start..r.end.min(from);
                let after = moved(r.start.max(to))..moved(r.end.max(to));
                [before, after]
            })
            .filter(|r| !r.is_empty())
            .collect();
        self.layout_gen += 1;
        Ok(taken)
    }

    /// Whether [`Self::splice_pages`] would accept `[from, to)`: an injected
    /// prefix with no live rows, a range inside what is placed, and both ends
    /// on page boundaries. A caller splicing several caches asks every one
    /// before changing any, so a refusal leaves them all as they stood.
    pub fn check_splice(&self, from: usize, to: usize) -> Result<()> {
        if self.n_blocks != 0 || self.n_open != 0 {
            candle::bail!(
                "qsa index: a splice after {} live block(s) and {} carried row(s) — only an \
                 injected prefix can be spliced",
                self.n_blocks,
                self.n_open,
            );
        }
        if from > to || to > self.tail_base {
            candle::bail!(
                "qsa index: a splice of {from}..{to} reaches past the {} token(s) placed",
                self.tail_base,
            );
        }
        if let Some(p) = self.pages.iter().find(|p| {
            let end = p.base + p.tokens;
            (p.base < from && from < end) || (p.base < to && to < end)
        }) {
            candle::bail!(
                "qsa index: a splice of {from}..{to} cuts the page covering {}..{} — pages are \
                 atomic, so a splice must land on page boundaries",
                p.base,
                p.base + p.tokens,
            );
        }
        Ok(())
    }

    /// Blocks this span would complete, and the rows it would leave open.
    pub(super) fn plan(&self, rows: usize, ratio: usize) -> (usize, usize) {
        let have = self.n_open + rows;
        let n_new = have / ratio;
        (n_new, have - n_new * ratio)
    }
}
