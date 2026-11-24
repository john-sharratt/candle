//! Where a prompt's images are, and the one place their pads become virtual
//! ids: the engine's input.
//!
//! A prompt's ids hold `<|image_pad|>` for every image row, everywhere they are
//! recorded. [`engine_ids`] — called wherever a scheduler slice of ids becomes
//! the engine's input — names each pad's row instead, `VIRTUAL_BASE + …`, so the
//! engine reads the image's rows in their place and nothing recorded ever holds
//! a virtual id.

use std::sync::Arc;

use candle_transformers::models::qwen_vision::{ImageHandle, ImageRows};
use tokenizers::Tokenizer;

use super::marker::{image_blocks, marker_ids};

/// One image's rows in a run of ids: its first `<|image_pad|>` and its handle.
#[derive(Clone, Debug)]
pub struct ImageSpan {
    pub start: usize,
    pub handle: Arc<ImageHandle>,
}

impl ImageSpan {
    /// The span moved `by` positions later.
    pub fn shifted(&self, by: usize) -> Self {
        Self {
            start: self.start + by,
            handle: self.handle.clone(),
        }
    }
}

/// The vision control tokens of a tokenizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionTokens {
    pub start: u32,
    pub pad: u32,
    pub end: u32,
}

impl VisionTokens {
    /// The ids `tokenizer` gives `<|vision_start|>`, `<|image_pad|>` and
    /// `<|vision_end|>`, when it has all three.
    pub fn of(tokenizer: &Tokenizer) -> Option<Self> {
        Some(Self {
            start: tokenizer.token_to_id("<|vision_start|>")?,
            pad: tokenizer.token_to_id("<|image_pad|>")?,
            end: tokenizer.token_to_id("<|vision_end|>")?,
        })
    }

    /// One image's ids: the start, `rows` pads, the end.
    pub fn block(&self, rows: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(rows + 2);
        out.push(self.start);
        out.extend(std::iter::repeat_n(self.pad, rows));
        out.push(self.end);
        out
    }
}

/// `ids` — positions `at .. at + ids.len()` of a prompt — as the engine's
/// input: every position inside one of `spans` named by its image row's
/// virtual id.
pub fn engine_ids(ids: &[u32], at: usize, spans: &[ImageSpan]) -> Vec<u32> {
    let mut out = ids.to_vec();
    for s in spans {
        let (lo, hi) = (s.start, s.start + s.handle.len());
        let from = lo.max(at);
        let to = hi.min(at + ids.len());
        for p in from..to {
            out[p - at] = s.handle.id(p - lo);
        }
    }
    out
}

/// A stored turn's image spans, from its ids and its text: the `k`-th image
/// block in `ids` is the `k`-th marker in `text`, its rows found by `rows`.
/// A block whose marker names an image the substrate no longer holds, or a
/// turn whose blocks and markers do not pair, is refused rather than prefilled
/// as pads.
pub fn turn_spans(
    ids: &[u32],
    text: &str,
    tokens: VisionTokens,
    rows: impl Fn(&str) -> Result<Option<ImageRows>, String>,
) -> Result<Vec<ImageSpan>, String> {
    let blocks = image_blocks(ids, tokens.start, tokens.pad, tokens.end);
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    let named = marker_ids(text);
    if named.len() != blocks.len() {
        return Err(format!(
            "a turn holds {} image blocks and its text names {} images",
            blocks.len(),
            named.len()
        ));
    }
    blocks
        .into_iter()
        .zip(named)
        .map(|((start, len), id)| {
            let r =
                rows(&id)?.ok_or_else(|| format!("image {id}'s rows are not in the substrate"))?;
            if r.len() != len {
                return Err(format!("image {id} has {} rows, its block {len}", r.len()));
            }
            Ok(ImageSpan {
                start,
                handle: ImageHandle::register(r)?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::qwen_vision::geometry::FACTOR;
    use candle_transformers::models::qwen_vision::Grid;

    fn rows(cols: usize) -> ImageRows {
        let grid = Grid {
            height: FACTOR,
            width: cols * FACTOR,
        };
        ImageRows {
            grid,
            width: 1,
            values: (0..cols).map(|i| i as f32).collect(),
        }
    }

    /// A slice of a prompt that cuts through an image takes exactly the
    /// virtual ids of the rows it holds.
    #[test]
    fn engine_ids_name_the_rows_of_the_slice() {
        let h = ImageHandle::register(rows(4)).unwrap();
        let span = ImageSpan {
            start: 3,
            handle: h.clone(),
        };
        // Positions 2..6 of a prompt whose image rows are 3..7.
        let got = engine_ids(&[10, 20, 20, 20], 2, &[span]);
        assert_eq!(got, [10, h.id(0), h.id(1), h.id(2)]);
        // A slice before the image is untouched.
        let span = ImageSpan {
            start: 3,
            handle: h,
        };
        assert_eq!(engine_ids(&[1, 2], 0, &[span]), [1, 2]);
    }

    #[test]
    fn a_block_is_its_start_its_pads_and_its_end() {
        let t = VisionTokens {
            start: 7,
            pad: 8,
            end: 9,
        };
        assert_eq!(t.block(3), [7, 8, 8, 8, 9]);
    }

    /// A stored turn finds its image by its marker; one whose rows are gone, or
    /// whose blocks and markers do not pair, is refused.
    #[test]
    fn a_turns_spans_come_from_its_ids_and_its_text() {
        let t = VisionTokens {
            start: 7,
            pad: 8,
            end: 9,
        };
        // The user half as the submitter lays it: the block, then the message.
        let mut ids = t.block(2);
        ids.push(2);
        let text = "[image 0123456789abcdef 64×32]\nwhat?";
        let found = turn_spans(&ids, text, t, |id| {
            assert_eq!(id, "0123456789abcdef");
            Ok(Some(rows(2)))
        })
        .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].start, 1);
        assert_eq!(found[0].handle.len(), 2);
        assert!(turn_spans(&ids, text, t, |_| Ok(None)).is_err());
        assert!(turn_spans(&ids, "no marker", t, |_| Ok(Some(rows(2)))).is_err());
        // A message that only quotes a marker holds no image.
        assert!(turn_spans(&[2], "see [image 0123456789abcdef 64×32]\n", t, |_| unreachable!())
            .unwrap()
            .is_empty());
        assert!(turn_spans(&[1, 2], "no image", t, |_| unreachable!())
            .unwrap()
            .is_empty());
    }
}
