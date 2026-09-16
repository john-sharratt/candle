//! Framing for the QSA index pages a turn seals.
//!
//! A turn's index is not one page. Its rows span whichever pieces the sequence's
//! cache was divided into when the seal ran — a mid-decode reprojection closes
//! one, and a thinking turn cuts one at each end of its reasoning — so the seal
//! (`ManagedBatchedModel::seal_positional_tail_span`) hands back **one blob per
//! page, in ascending token order**, and the projection pushes them back in that
//! same order.
//!
//! # Each page carries its token width, and that is load-bearing
//!
//! The width is what lets a reader locate a page in the turn's grid without
//! parsing the page itself, which is model-opaque. The projection needs exactly
//! that: to drop the pages covering a turn's `<think>…</think>` span, it has to
//! know which pages those are, and **the count is not stable** — a turn that
//! reprojected mid-decode has more pages than one that did not, and reprojection
//! is the common case. Identifying the reasoning by position rather than by
//! ordinal is the difference between windowing every thinking turn and silently
//! windowing only the ones that never reprojected.
//!
//! # One record, not several
//!
//! The turn record stores one payload, last-writer-wins per turn stream id, and
//! five persistence sites depend on that shape. So the list is framed into that
//! single payload here rather than becoming several records: the record, the
//! substrate field, and `Scheduler::turn_positional` all keep their types, and
//! only the producer and the consumer learn about the framing.

use std::fmt::{self, Display, Formatter};

/// A malformed page payload.
///
/// Framing errors are reported rather than silently truncated: a payload that
/// does not decode is a turn whose K/V can be borrowed and whose rows cannot,
/// which is the failure the pages exist to prevent. The caller logs and injects
/// nothing rather than injecting part of a turn's index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MalformedPages {
    pub reason: String,
}

impl Display for MalformedPages {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "index page payload: {}", self.reason)
    }
}

/// One sealed page: the tokens it covers, and its model-opaque bytes.
pub type Page<'a> = (usize, &'a [u8]);

/// Frame `pages` as one payload: a `u32` count, then per page a `u32` token
/// width, a `u32` byte length, and the bytes. All little-endian.
pub fn encode(pages: &[(usize, Vec<u8>)]) -> Vec<u8> {
    let total: usize = pages.iter().map(|(_, p)| p.len() + 8).sum();
    let mut out = Vec::with_capacity(4 + total);
    out.extend_from_slice(&(pages.len() as u32).to_le_bytes());
    for (tokens, p) in pages {
        out.extend_from_slice(&(*tokens as u32).to_le_bytes());
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        out.extend_from_slice(p);
    }
    out
}

/// Read back what [`encode`] wrote, as `(token width, bytes)` borrowed from
/// `blob`.
pub fn decode(blob: &[u8]) -> Result<Vec<Page<'_>>, MalformedPages> {
    let bad = |reason: String| MalformedPages { reason };
    let u32_at = |o: usize| -> Result<u32, MalformedPages> {
        let b = blob
            .get(o..o + 4)
            .ok_or_else(|| bad(format!("truncated reading a field at offset {o}")))?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let count = u32_at(0)? as usize;
    // **Capacity from the payload's own bounds, not from its header.** `count` is
    // an unvalidated `u32` read off bytes that may be corrupt, and every page
    // costs at least its two length fields, so a blob claiming `u32::MAX` pages
    // would reserve tens of gigabytes and abort the process — defeating the
    // graceful `MalformedPages` refusal this whole function exists to give. The
    // header stays authoritative for the LOOP; it just no longer gets to size an
    // allocation on its own word.
    let mut out = Vec::with_capacity(count.min(blob.len() / 8));
    let mut off = 4usize;
    for i in 0..count {
        let tokens = u32_at(off)? as usize;
        let len = u32_at(off + 4)? as usize;
        off += 8;
        let page = blob.get(off..off + len).ok_or_else(|| {
            bad(format!(
                "page {i} of {count} declares {len} bytes but only {} remain",
                blob.len().saturating_sub(off)
            ))
        })?;
        out.push((tokens, page));
        off += len;
    }
    if off != blob.len() {
        return Err(bad(format!(
            "{count} page(s) consumed {off} of {} bytes — the payload carries \
             trailing data that no page claims",
            blob.len()
        )));
    }
    Ok(out)
}

/// What [`push_in_order`] did with a turn's payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pushed {
    /// Pages the model accepted.
    pub pages: usize,
    /// Tokens no accepted page covers — the width the slot was advanced over.
    pub gap: usize,
    /// Why the walk stopped short, when it did.
    pub refused: Option<String>,
}

/// Hand a turn's framed pages to the model one at a time, in the order the seal
/// produced them, then advance the slot over whatever no accepted page covers.
///
/// **The model reads pages, never the framing.** Every page is one model-opaque
/// blob; the payload around them is this module's. Passing the whole payload as
/// a page makes the model read the page COUNT as its own header — a three-page
/// turn arrived as "aux blob: version 3 unknown" on every memory catch-up after
/// a scope splice, and the replay ran against a prefix it never indexed.
///
/// `push` hands one page to the model; `gap` advances the slot past tokens that
/// carry no rows. A refused page stops the walk — pushing the pages after it
/// would place them where their K/V does not sit — and the rest of the turn's
/// declared width goes to `gap`, so every later piece still lands where its K/V
/// does. A payload that does not decode pushes nothing and is returned as the
/// error: the width to advance over is then the turn's whole K/V, which only the
/// caller knows.
pub fn push_in_order<E: Display>(
    blob: &[u8],
    mut push: impl FnMut(&[u8]) -> Result<(), E>,
    mut gap: impl FnMut(usize),
) -> Result<Pushed, MalformedPages> {
    let pages = decode(blob)?;
    let declared: usize = pages.iter().map(|(tokens, _)| *tokens).sum();
    let mut covered = 0usize;
    let mut accepted = 0usize;
    let mut refused = None;
    for (tokens, page) in pages {
        match push(page) {
            Ok(()) => {
                accepted += 1;
                covered += tokens;
            }
            Err(e) => {
                refused = Some(e.to_string());
                break;
            }
        }
    }
    // Only the tokens no page covered, so a partial push is not counted twice.
    let rest = declared - covered;
    if rest > 0 {
        gap(rest);
    }
    Ok(Pushed {
        pages: accepted,
        gap: rest,
        refused,
    })
}

/// Split a turn's pages at the reasoning span: the pages to keep, framed as a
/// payload, or `None` when the span does not fall on page boundaries.
///
/// **Both halves or neither.** The retained pages have to describe exactly the
/// K/V the projection injects, so the dropped ones have to be exactly the span.
/// If the pages do not line up with it — a turn sealed before the cuts existed,
/// or one whose boundary landed inside a page — there is no filtering that
/// matches the window, and the caller injects the turn WHOLE rather than pairing
/// windowed chunks with rows describing something else.
///
/// `span` is the reasoning's `[start, end)` in the turn's own grid, and the
/// pages are assumed to start at grid 0 and run in order — which is what
/// `seal_positional_tail_span` produces for a turn's own width.
pub fn without_span(blob: &[u8], span: std::ops::Range<usize>) -> Option<Vec<u8>> {
    let pages = decode(blob).ok()?;
    // **An empty span removes nothing, so everything is kept.** The final
    // equality below compares `dropped_from` against `span.start`, and a
    // zero-width span drops no page at all, so it stays `None` and the check
    // fails — turning "there was no reasoning to take out" into a hard refusal.
    // A `Thinking` span can legitimately be zero-width (a block whose reasoning
    // was clamped away), and such a turn is windowed by keeping it intact.
    if span.is_empty() {
        return Some(blob.to_vec());
    }
    let mut kept: Vec<(usize, Vec<u8>)> = Vec::with_capacity(pages.len());
    let mut dropped_from: Option<usize> = None;
    let mut dropped_tokens = 0usize;
    let mut cursor = 0usize;
    for (tokens, bytes) in pages {
        let page = cursor..cursor + tokens;
        cursor = page.end;
        // **A zero-width page describes no token, so it cannot straddle
        // anything.** It carries no K/V either way, and left to the tests below
        // one sitting inside the span satisfies the overlap check (its start is
        // before the span's end and its end after the span's start) while being
        // excluded from the drop branch by `!page.is_empty()` — a refusal on a
        // page that holds nothing, even when the real reasoning pages tile the
        // span exactly. Two `close_positional_page` calls with no token between
        // them produce one.
        if page.is_empty() {
            continue;
        }
        // Wholly inside the span — this is reasoning.
        if page.start >= span.start && page.end <= span.end {
            dropped_from.get_or_insert(page.start);
            dropped_tokens += tokens;
            continue;
        }
        // Overlapping it at all means the boundary is inside a page, and no
        // choice about this page is right: keeping it carries reasoning the
        // window removed, dropping it takes tokens the window kept.
        if page.start < span.end && page.end > span.start {
            return None;
        }
        kept.push((tokens, bytes.to_vec()));
    }
    // The dropped pages must be exactly the span — same start, same width.
    (dropped_from == Some(span.start) && dropped_tokens == span.len()).then(|| encode(&kept))
}

/// The page boundaries a blob describes, as cumulative grid positions starting
/// at 0 — what a span has to land on.
///
/// For the refusal message. "The span does not fall on page boundaries" is not
/// actionable without the boundaries: the question it always raises is *which
/// cut was missing*, and that is answered by the two positions the span sits
/// between. Returns `None` for a blob that will not decode.
pub fn boundaries(blob: &[u8]) -> Option<Vec<usize>> {
    let pages = decode(blob).ok()?;
    let mut at = 0usize;
    let mut out = Vec::with_capacity(pages.len() + 1);
    out.push(0);
    for (tokens, _) in pages {
        at += tokens;
        out.push(at);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{decode, encode, push_in_order, without_span, Pushed};

    fn pages(spec: &[(usize, &[u8])]) -> Vec<(usize, Vec<u8>)> {
        spec.iter().map(|(t, b)| (*t, b.to_vec())).collect()
    }

    /// The framing, asserted against the exact bytes — not against a round
    /// trip, which would pass just as happily on a self-consistent but wrong
    /// layout.
    #[test]
    fn encodes_to_the_declared_bytes() {
        assert_eq!(
            encode(&pages(&[(7, &[0xAA, 0xBB]), (3, &[0xCC])])),
            vec![
                2, 0, 0, 0, // count
                7, 0, 0, 0, 2, 0, 0, 0, 0xAA, 0xBB, // page 0: 7 tokens, 2 bytes
                3, 0, 0, 0, 1, 0, 0, 0, 0xCC, // page 1: 3 tokens, 1 byte
            ]
        );
    }

    /// Order is the contract: pages are pushed back in the order the seal
    /// produced them, because each carries its own ragged width and a
    /// reordering would place a turn's rows at the wrong positions.
    #[test]
    fn round_trip_preserves_order_widths_and_contents() {
        let blob = encode(&pages(&[(4, &[1, 2, 3]), (0, &[]), (9, &[9])]));
        assert_eq!(
            decode(&blob).expect("decodes"),
            vec![(4, &[1u8, 2, 3][..]), (0, &[][..]), (9, &[9u8][..])]
        );
    }

    #[test]
    fn no_pages_encodes_and_decodes_as_empty() {
        let blob = encode(&[]);
        assert_eq!(blob, vec![0, 0, 0, 0]);
        assert_eq!(decode(&blob), Ok(Vec::new()));
    }

    /// Truncation is refused rather than silently yielding the pages that did
    /// fit. Half a turn's index is worse than none: the slot would hold the
    /// turn's K/V with only part of it accounted for, which is silent below the
    /// QSA identity threshold and a refused select above it.
    #[test]
    fn truncation_is_refused_not_truncated() {
        let blob = encode(&pages(&[(2, &[1, 2, 3, 4]), (1, &[5, 6])]));
        for cut in [0, 3, 7, 11, blob.len() - 1] {
            assert!(
                decode(&blob[..cut]).is_err(),
                "a payload cut at {cut} must not decode"
            );
        }
        assert!(decode(&blob).is_ok());
    }

    #[test]
    fn trailing_data_is_refused() {
        let mut blob = encode(&pages(&[(1, &[7])]));
        blob.push(0);
        assert!(decode(&blob).is_err());
    }

    /// **The reasoning pages are found by POSITION, not by ordinal.** The count
    /// varies — a turn that reprojected mid-decode seals more pages than one
    /// that did not — so the same span must be located whatever the split.
    #[test]
    fn the_reasoning_pages_are_found_at_any_page_count() {
        // Three pages: prefill [0,10), reasoning [10,25), answer [25,40).
        let three = encode(&pages(&[(10, b"pre"), (15, b"think"), (15, b"ans")]));
        let kept = without_span(&three, 10..25).expect("aligned");
        assert_eq!(
            decode(&kept).unwrap(),
            vec![(10, &b"pre"[..]), (15, &b"ans"[..])]
        );

        // The same turn after two reprojections: the answer arrived in three
        // pieces, and the prefill in two. Same span, same survivors.
        let five = encode(&pages(&[
            (4, b"pre1"),
            (6, b"pre2"),
            (15, b"think"),
            (9, b"ans1"),
            (6, b"ans2"),
        ]));
        let kept = without_span(&five, 10..25).expect("aligned");
        assert_eq!(
            decode(&kept).unwrap(),
            vec![
                (4, &b"pre1"[..]),
                (6, &b"pre2"[..]),
                (9, &b"ans1"[..]),
                (6, &b"ans2"[..])
            ]
        );

        // Reasoning split across two pages (a reprojection inside the block).
        let split = encode(&pages(&[
            (10, b"pre"),
            (5, b"t1"),
            (10, b"t2"),
            (15, b"ans"),
        ]));
        let kept = without_span(&split, 10..25).expect("aligned");
        assert_eq!(
            decode(&kept).unwrap(),
            vec![(10, &b"pre"[..]), (15, &b"ans"[..])]
        );
    }

    /// A span that does not line up with page boundaries yields `None`, so the
    /// caller injects the turn whole. Keeping the straddling page would carry
    /// reasoning the window removed; dropping it would take answer tokens the
    /// window kept. Neither matches, so neither is chosen.
    #[test]
    fn a_span_inside_a_page_refuses_rather_than_guessing() {
        let blob = encode(&pages(&[(10, b"pre"), (20, b"body")]));
        // The reasoning ends mid-way through the second page.
        assert_eq!(without_span(&blob, 10..22), None);
        // …and starts mid-way through the first.
        assert_eq!(without_span(&blob, 6..30), None);
        // A span covering no whole page at all.
        assert_eq!(without_span(&blob, 11..14), None);
    }

    /// An unparseable payload is a refusal, not a panic — an older record, or a
    /// truncated one, injects whole.
    #[test]
    fn an_unreadable_payload_refuses() {
        assert_eq!(without_span(b"not a page list", 0..4), None);
    }

    /// **An empty span removes nothing, so it keeps everything.**
    ///
    /// The equality at the end of `without_span` compares the first dropped
    /// page's start against the span's, and a zero-width span drops none — so it
    /// stayed `None` and a turn whose reasoning had been clamped away was
    /// refused instead of passed through. A refusal here is permanent: it fails
    /// every later projection that selects the turn.
    #[test]
    fn an_empty_span_keeps_every_page() {
        let blob = encode(&pages(&[(4, b"aaaa"), (3, b"bbb")]));
        let kept = without_span(&blob, 4..4).expect("an empty span cannot misalign");
        assert_eq!(decode(&kept).unwrap().len(), 2, "both pages survive");
        assert_eq!(kept, blob, "and the payload is unchanged");
    }

    /// A zero-width page describes no token, so it cannot straddle a span.
    ///
    /// It is excluded from the "wholly inside" branch by its own emptiness, yet
    /// satisfies the overlap test — so one sitting inside the span forced a
    /// refusal even when the real pages tiled the span exactly. Two
    /// `close_positional_page` calls with no token between them produce one.
    #[test]
    fn a_zero_width_page_inside_the_span_does_not_refuse() {
        let blob = encode(&pages(&[(2, b"aa"), (0, b""), (3, b"bbb"), (4, b"cccc")]));
        // The span is the two real pages at [2..5); the empty page sits at 2.
        let kept = without_span(&blob, 2..5).expect("a zero-width page cannot straddle anything");
        let out = decode(&kept).unwrap();
        assert_eq!(
            out.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec![2, 4],
            "the span's page is dropped and the rest kept"
        );
    }

    /// A corrupt header may not size an allocation on its own word.
    ///
    /// `count` is an unvalidated `u32`; reserving from it directly let a
    /// malformed blob claim billions of pages and abort the process, which is
    /// exactly the outcome the `MalformedPages` refusal exists to avoid.
    #[test]
    fn a_corrupt_page_count_refuses_rather_than_allocating() {
        let mut blob = u32::MAX.to_le_bytes().to_vec();
        blob.extend_from_slice(&4u32.to_le_bytes());
        blob.extend_from_slice(&2u32.to_le_bytes());
        blob.extend_from_slice(b"xy");
        assert!(
            decode(&blob).is_err(),
            "a count the payload cannot back must refuse"
        );
        assert_eq!(without_span(&blob, 0..4), None);
    }

    /// Each page reaches the model as its own bytes, in seal order, and a fully
    /// accepted turn advances the slot over nothing.
    ///
    /// The first page handed over must be the page — not the payload, whose
    /// leading `u32` is the page count. A three-page payload read as one page is
    /// "aux blob: version 3", which is how every memory catch-up after a scope
    /// splice refused its turns.
    #[test]
    fn pages_are_pushed_one_at_a_time_in_seal_order() {
        let blob = encode(&pages(&[(3, &[0xA1]), (5, &[0xB1, 0xB2]), (4, &[0xC1])]));
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let mut gaps: Vec<usize> = Vec::new();
        let out = push_in_order(
            &blob,
            |p| {
                seen.push(p.to_vec());
                Ok::<(), String>(())
            },
            |g| gaps.push(g),
        )
        .expect("a well-formed payload");
        assert_eq!(seen, vec![vec![0xA1], vec![0xB1, 0xB2], vec![0xC1]]);
        assert!(
            gaps.is_empty(),
            "every token was covered, so nothing is skipped"
        );
        assert_eq!(
            out,
            Pushed {
                pages: 3,
                gap: 0,
                refused: None
            }
        );
    }

    /// A refused page stops the walk, and the slot is advanced over exactly
    /// the declared width no accepted page covered — so every later piece is
    /// still placed where its K/V sits.
    #[test]
    fn a_refused_page_stops_the_walk_and_skips_the_rest_of_the_turn() {
        let blob = encode(&pages(&[(3, &[0xA1]), (5, &[0xB1]), (4, &[0xC1])]));
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let mut gaps: Vec<usize> = Vec::new();
        let out = push_in_order(
            &blob,
            |p| {
                if p == [0xB1] {
                    return Err("refused".to_string());
                }
                seen.push(p.to_vec());
                Ok(())
            },
            |g| gaps.push(g),
        )
        .expect("a well-formed payload");
        assert_eq!(
            seen,
            vec![vec![0xA1]],
            "nothing after the refusal is pushed"
        );
        assert_eq!(gaps, vec![9], "5 + 4 tokens carry no rows");
        assert_eq!(
            out,
            Pushed {
                pages: 1,
                gap: 9,
                refused: Some("refused".to_string())
            }
        );
    }

    /// A payload that does not decode pushes nothing and skips nothing: the
    /// width to advance over is the turn's whole K/V, which only the caller
    /// knows.
    #[test]
    fn a_malformed_payload_pushes_nothing() {
        let mut pushes = 0usize;
        let mut gaps = 0usize;
        let out = push_in_order(
            b"not a page list",
            |_| {
                pushes += 1;
                Ok::<(), String>(())
            },
            |_| gaps += 1,
        );
        assert!(out.is_err());
        assert_eq!((pushes, gaps), (0, 0));
    }
}
