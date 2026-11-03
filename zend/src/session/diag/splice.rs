//! Text spliced into a diagnostic decode's target turn before it is submitted.
//!
//! A splice puts text into the target's user half at a byte offset — a page
//! where the model's call is answered, a block in place of another's place —
//! without moving the turn out of its recording. The recorded grid is spliced
//! the same way, so the recorded reply still replays onto the spliced prefill
//! and a run that forces `token` ids of it reaches the same point of the reply
//! it did live.
//!
//! A splice must fall where the tokenizer splits regardless — at a chat tag, as
//! a `<tool_response>` block's edges do — so the half's ids before it are a
//! prefix of the whole half's. One that falls inside a token is refused. The
//! conversation then checks the spliced grid against the prefill it encodes,
//! and refuses a turn whose two disagree.

use anyhow::{bail, Result};
use candle_conversation::TurnText;
use serde::Deserialize;

/// Text put into the target's user half.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Splice {
    /// A byte offset into the half the run submits — the recorded one, or the
    /// run's own `user`.
    pub at: usize,
    pub text: String,
    /// Content the model reads, whose chat tags stay text — a file. Otherwise
    /// markup, whose tags are the tokens themselves — a block's wrapper.
    #[serde(default)]
    pub literal: bool,
}

impl Splice {
    fn piece(&self) -> TurnText {
        if self.literal {
            TurnText::literal(self.text.as_str())
        } else {
            TurnText::markup(self.text.as_str())
        }
    }
}

/// `splices` in the order they fall, splices at one offset in the order given.
fn ordered<'a>(half: &str, splices: &'a [Splice]) -> Result<Vec<&'a Splice>> {
    let mut out: Vec<&Splice> = splices.iter().collect();
    out.sort_by_key(|s| s.at);
    for s in &out {
        if !half.is_char_boundary(s.at) {
            bail!(
                "a splice at byte {} falls outside the {}-byte user half or inside a character",
                s.at,
                half.len()
            );
        }
    }
    Ok(out)
}

/// `half` with each splice's text at its offset.
pub fn splice_text(half: &TurnText, splices: &[Splice]) -> Result<TurnText> {
    let text = half.text();
    let mut out = TurnText::default();
    let mut from = 0;
    for s in ordered(&text, splices)? {
        out = out.then(half.slice(from..s.at)).then(s.piece());
        from = s.at;
    }
    Ok(out.then(half.slice(from..text.len())))
}

/// The recorded `grid` of a turn submitted with `half`, with each splice's ids
/// where its bytes fall. `encode` is the conversation's own encoder.
pub fn splice_grid(
    grid: &[u32],
    half: &TurnText,
    splices: &[Splice],
    encode: impl Fn(&TurnText) -> Result<Vec<u32>>,
) -> Result<Vec<u32>> {
    let whole = encode(half)?;
    // The grid opens with the turn's head, then the half.
    let Some(head) =
        (0..=grid.len().saturating_sub(whole.len())).find(|&h| grid[h..].starts_with(&whole))
    else {
        bail!("the recorded turn does not hold the ids of the user half it is spliced into");
    };
    let text = half.text();
    let mut out = grid[..head].to_vec();
    let mut from = 0;
    for s in ordered(&text, splices)? {
        let before = encode(&half.prefix(s.at))?;
        if !whole.starts_with(&before) {
            bail!(
                "a splice at byte {} of the user half falls inside a token",
                s.at
            );
        }
        out.extend_from_slice(&whole[from..before.len()]);
        out.extend(encode(&s.piece())?);
        from = before.len();
    }
    out.extend_from_slice(&whole[from..]);
    out.extend_from_slice(&grid[head + whole.len()..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One id per character, `<` and `>` joining a tag into one id (the tag's
    /// length times 1000) the way a registered tag is one token — so a byte
    /// inside a tag is inside a token.
    fn encode(text: &TurnText) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for piece in text.pieces() {
            let mut rest = piece.text.as_str();
            while let Some(c) = rest.chars().next() {
                match (c, piece.literal, rest.find('>')) {
                    ('<', false, Some(end)) => {
                        ids.push((end as u32 + 1) * 1000);
                        rest = &rest[end + 1..];
                    }
                    _ => {
                        ids.push(c as u32);
                        rest = &rest[c.len_utf8()..];
                    }
                }
            }
        }
        Ok(ids)
    }

    fn splice(at: usize, text: &str, literal: bool) -> Splice {
        Splice {
            at,
            text: text.to_string(),
            literal,
        }
    }

    fn half() -> TurnText {
        TurnText::markup("<r>")
            .then_literal("ab")
            .then_markup("</r>")
    }

    #[test]
    fn a_splice_goes_in_at_its_offset_and_keeps_its_kind() {
        let out = splice_text(&half(), &[splice(3, "xy", true), splice(0, "<q>", false)]).unwrap();
        assert_eq!(
            out,
            TurnText::markup("<q><r>")
                .then_literal("xyab")
                .then_markup("</r>")
        );
        assert_eq!(splice_text(&half(), &[]).unwrap(), half());
    }

    /// The grid keeps its head and its reply, and takes each splice's ids
    /// where its bytes fall in the half.
    #[test]
    fn the_grid_takes_each_splice_s_ids_between_its_head_and_its_reply() {
        let head = [7u32, 7];
        let reply = [9u32, 9, 9];
        let mut grid = head.to_vec();
        grid.extend(encode(&half()).unwrap());
        grid.extend(reply);
        assert_eq!(grid, [7, 7, 3000, 97, 98, 4000, 9, 9, 9]);
        let out = splice_grid(
            &grid,
            &half(),
            &[splice(5, "<s>", false), splice(3, "x", true)],
            encode,
        )
        .unwrap();
        assert_eq!(out, [7, 7, 3000, 120, 97, 98, 3000, 4000, 9, 9, 9]);
    }

    #[test]
    fn a_splice_inside_a_token_is_refused() {
        let grid = encode(&half()).unwrap();
        let err = splice_grid(&grid, &half(), &[splice(1, "x", true)], encode).unwrap_err();
        assert_eq!(
            err.to_string(),
            "a splice at byte 1 of the user half falls inside a token"
        );
    }

    #[test]
    fn a_grid_of_another_half_is_refused() {
        let err = splice_grid(&[1, 2, 3], &half(), &[splice(0, "x", true)], encode).unwrap_err();
        assert_eq!(
            err.to_string(),
            "the recorded turn does not hold the ids of the user half it is spliced into"
        );
    }

    #[test]
    fn a_splice_past_the_half_is_refused() {
        let err = splice_text(&half(), &[splice(99, "x", true)]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "a splice at byte 99 falls outside the 9-byte user half or inside a character"
        );
    }
}
