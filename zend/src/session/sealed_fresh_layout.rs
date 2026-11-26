//! The sealed side's slot against the fresh side's, piece by piece of the
//! prompt both decoded over: where each piece's chunks lie on each side, their
//! windows, the format of every `(head, palette)` band on every layer, and the
//! index pages each slot holds (`session/sealed_fresh.rs`).
//!
//! The question it answers is whether the two hold the same layout and the
//! same formats — not the same values, which differ wherever the K/V was
//! computed behind a different context.

use std::collections::BTreeMap;

use candle_conversation::{format_name, ChunkLayout, PromptDump, SlotLayout};
use serde::Serialize;

/// One piece of the prompt on both sides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PieceRow {
    pub piece: usize,
    pub kind: String,
    pub name: String,
    pub tokens: usize,
    /// The piece's chunk windows, `offset:tokens` each.
    pub sealed_windows: String,
    pub fresh_windows: String,
    pub windows_equal: bool,
    /// Bands, over every chunk and layer of the piece, whose format is the
    /// same on both sides, of `bands`.
    pub k_bands_equal: usize,
    pub v_bands_equal: usize,
    pub bands: usize,
    /// Each side's band formats, counted over the piece.
    pub sealed_k: String,
    pub fresh_k: String,
    pub sealed_v: String,
    pub fresh_v: String,
}

/// The two slots compared.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LayoutComparison {
    pub sealed_depth: usize,
    pub fresh_depth: usize,
    pub sealed_chunks: usize,
    pub fresh_chunks: usize,
    pub sealed_index_coverage: Option<usize>,
    pub fresh_index_coverage: Option<usize>,
    /// Each slot's index pages, `(tokens, bytes)`.
    pub sealed_index: Vec<(usize, usize)>,
    pub fresh_index: Vec<(usize, usize)>,
    pub index_lengths_equal: bool,
    pub pieces: Vec<PieceRow>,
    /// Why a side's chunks could not be cut into the prompt's pieces.
    pub layout_error: Option<String>,
}

/// Where each piece's chunks lie: `(first, end)` per piece, from the chunks'
/// token counts. Chunks after the last piece — the turn — are left out; an
/// empty chunk after a piece belongs to it; a chunk holding two pieces'
/// tokens is refused.
pub fn piece_chunks(
    chunks: &[ChunkLayout],
    piece_tokens: &[usize],
) -> Result<Vec<(usize, usize)>, String> {
    let mut out = Vec::with_capacity(piece_tokens.len());
    let mut at = 0;
    for (i, &want) in piece_tokens.iter().enumerate() {
        let first = at;
        let mut held = 0;
        while held < want {
            let c = chunks
                .get(at)
                .ok_or_else(|| format!("piece {i}: the slot ran out at {held} of {want} tokens"))?;
            held += c.tokens as usize;
            at += 1;
        }
        if held != want {
            return Err(format!(
                "piece {i} ends inside chunk {} ({held} tokens against {want})",
                at - 1
            ));
        }
        while chunks.get(at).is_some_and(|c| c.tokens == 0) {
            at += 1;
        }
        out.push((first, at));
    }
    Ok(out)
}

fn windows(chunks: &[ChunkLayout]) -> String {
    chunks
        .iter()
        .map(|c| format!("{}:{}", c.offset, c.tokens))
        .collect::<Vec<_>>()
        .join(",")
}

fn histogram<'a>(bands: impl Iterator<Item = &'a Vec<u8>>) -> String {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for layer in bands {
        if layer.is_empty() {
            *counts.entry("unrecorded".into()).or_default() += 1;
        }
        for &t in layer {
            *counts.entry(format_name(t)).or_default() += 1;
        }
    }
    counts
        .iter()
        .map(|(f, n)| format!("{f}:{n}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Bands equal on both sides, over chunk pairs and layers.
fn equal_bands(
    a: &[ChunkLayout],
    b: &[ChunkLayout],
    side: fn(&ChunkLayout) -> &Vec<Vec<u8>>,
) -> usize {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            side(x)
                .iter()
                .zip(side(y))
                .map(|(lx, ly)| lx.iter().zip(ly).filter(|(p, q)| p == q).count())
                .sum::<usize>()
        })
        .sum()
}

/// Compare `sealed` with `fresh` over `dump`'s pieces.
pub fn compare(dump: &PromptDump, sealed: &SlotLayout, fresh: &SlotLayout) -> LayoutComparison {
    let pieces: Vec<_> = dump
        .pieces
        .iter()
        .filter(|p| !p.token_ids.is_empty())
        .collect();
    let tokens: Vec<usize> = pieces.iter().map(|p| p.token_ids.len()).collect();
    let index = |s: &SlotLayout| -> Vec<(usize, usize)> {
        s.index_pages.iter().map(|p| (p.tokens, p.bytes)).collect()
    };
    let sealed_index = index(sealed);
    let fresh_index = index(fresh);
    let mut out = LayoutComparison {
        sealed_depth: sealed.depth,
        fresh_depth: fresh.depth,
        sealed_chunks: sealed.chunks.len(),
        fresh_chunks: fresh.chunks.len(),
        sealed_index_coverage: sealed.index_coverage,
        fresh_index_coverage: fresh.index_coverage,
        index_lengths_equal: sealed_index
            .iter()
            .map(|p| p.0)
            .eq(fresh_index.iter().map(|p| p.0)),
        sealed_index,
        fresh_index,
        pieces: Vec::new(),
        layout_error: None,
    };
    let (s_cut, f_cut) = match (
        piece_chunks(&sealed.chunks, &tokens),
        piece_chunks(&fresh.chunks, &tokens),
    ) {
        (Ok(s), Ok(f)) => (s, f),
        (s, f) => {
            out.layout_error = Some(format!(
                "sealed: {}; fresh: {}",
                s.err().unwrap_or_else(|| "ok".into()),
                f.err().unwrap_or_else(|| "ok".into())
            ));
            return out;
        }
    };
    for (i, ((p, &(sa, sb)), &(fa, fb))) in pieces.iter().zip(&s_cut).zip(&f_cut).enumerate() {
        let s = &sealed.chunks[sa..sb];
        let f = &fresh.chunks[fa..fb];
        let bands = s.iter().flat_map(|c| c.k.iter()).map(Vec::len).sum();
        out.pieces.push(PieceRow {
            piece: i,
            kind: format!("{:?}", p.kind),
            name: p.name.clone(),
            tokens: p.token_ids.len(),
            sealed_windows: windows(s),
            fresh_windows: windows(f),
            windows_equal: windows(s) == windows(f),
            k_bands_equal: equal_bands(s, f, |c| &c.k),
            v_bands_equal: equal_bands(s, f, |c| &c.v),
            bands,
            sealed_k: histogram(s.iter().flat_map(|c| c.k.iter())),
            fresh_k: histogram(f.iter().flat_map(|c| c.k.iter())),
            sealed_v: histogram(s.iter().flat_map(|c| c.v.iter())),
            fresh_v: histogram(f.iter().flat_map(|c| c.v.iter())),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(tokens: u16) -> ChunkLayout {
        ChunkLayout {
            offset: 0,
            tokens,
            k: vec![vec![2, 2]],
            v: vec![vec![2, 2]],
            layers_agree: true,
        }
    }

    /// **Each piece takes the chunks its tokens filled**; the turn's chunks
    /// after the last piece are left out, and a writer left empty after a
    /// piece goes with it.
    #[test]
    fn each_piece_takes_the_chunks_its_tokens_filled() {
        let chunks = [chunk(32), chunk(8), chunk(5), chunk(0), chunk(20)];
        assert_eq!(
            piece_chunks(&chunks, &[40, 5]).unwrap(),
            vec![(0, 2), (2, 4)]
        );
    }

    /// **A chunk holding two pieces' tokens is refused**, naming the chunk.
    #[test]
    fn a_chunk_shared_by_two_pieces_is_refused() {
        let err = piece_chunks(&[chunk(32), chunk(13)], &[40, 5]).unwrap_err();
        assert!(err.contains("ends inside chunk 1"), "{err}");
    }

    /// **Equal bands are counted band by band**, over chunk pairs and layers.
    #[test]
    fn equal_bands_count_band_by_band() {
        let a = [ChunkLayout {
            k: vec![vec![2, 3], vec![4, 4]],
            ..chunk(32)
        }];
        let b = [ChunkLayout {
            k: vec![vec![2, 2], vec![4, 4]],
            ..chunk(32)
        }];
        assert_eq!(equal_bands(&a, &b, |c| &c.k), 3);
    }
}
