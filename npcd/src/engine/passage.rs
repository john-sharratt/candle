//! An edit puts one passage right; it does not write the piece again.
//!
//! **A composed piece was undone by hand.** A Maker that sat down and composed
//! its story whole then `file_edit`ed it with the entire document as `old_str`
//! and its own new text as `new_str` — the piece rewritten in the middle of its
//! running conversation, which is exactly the writing `compose` exists to
//! replace, and the vault's light rings and air handlers went straight back in.
//! So on a mission's prose document an edit that takes most of the piece is
//! refused and pointed at `compose`.

/// The share of a piece's words an edit may take before it is the piece and
/// not a passage of it.
const MOST: f32 = 0.5;

/// Pieces shorter than this are edited whole without remark: a heading and a
/// line is not a piece anyone composes.
const SMALLEST_PIECE: usize = 60;

/// The refusal for an edit to `doc` whose `old` takes most of `current` —
/// `None` when it is a passage.
pub fn rewrites_the_piece(doc: &str, current: &str, old: &str) -> Option<String> {
    let whole = current.split_whitespace().count();
    let taken = old.split_whitespace().count();
    if whole < SMALLEST_PIECE || (taken as f32) <= MOST * whole as f32 {
        return None;
    }
    Some(format!(
        "That is {taken} of the {whole} words in {doc} — the piece, not a passage of it. An edit \
         puts one passage right. To write it whole again, sit down to it with `compose`."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(n: usize) -> String {
        (0..n)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn a_passage_is_edited_and_the_piece_is_not() {
        let piece = words(200);
        let passage = words(40);
        assert_eq!(rewrites_the_piece("x.md", &piece, &passage), None);
        assert_eq!(
            rewrites_the_piece("x.md", &piece, &words(150)).as_deref(),
            Some(
                "That is 150 of the 200 words in x.md — the piece, not a passage of it. An edit \
                 puts one passage right. To write it whole again, sit down to it with `compose`."
            )
        );
    }

    #[test]
    fn a_short_document_is_edited_whole_without_remark() {
        assert_eq!(rewrites_the_piece("x.md", &words(30), &words(30)), None);
    }
}
