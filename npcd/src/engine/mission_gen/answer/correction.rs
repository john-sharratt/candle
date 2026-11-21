//! Defines the call a correction is answered with: the two clashing sentences and which one is wrong.
//!
//! A correction is answered as a single call so the model names both sides of a contradiction and the engine can check each side against its document. The call's definition is kept here apart from the other kinds of work, so each kind's fields stay in one place.

use super::*;

/// The call for a correction.
pub const CORRECTION: &str = "correction";

pub(super) fn correction(
    field: &Fields,
    a: &str,
    b: &str,
    corpus: &Corpus,
) -> Result<Answer, String> {
    let (ta, tb) = (
        corpus.text(a).unwrap_or_default(),
        corpus.text(b).unwrap_or_default(),
    );
    let qa = quoted(&field.get("quote_a"), &ta, "Document A")?;
    let qb = quoted(&field.get("quote_b"), &tb, "Document B")?;
    let (wrong, right) = match field.get("wrong").as_str() {
        "A" => (a, b),
        "B" => (b, a),
        w => return Err(format!("`wrong` is `A` or `B`, not `{w}`.")),
    };
    let why = field.words("why", 8, WHY_WORDS, "why that side is the wrong one")?;
    let change = field.words("change", 6, CHANGE_WORDS, "the change to make")?;
    let brief = format!(
        "Two documents of the record contradict each other, and one must change.\n\n\
         `{a}` says: \"{qa}\"\n\n`{b}` says: \"{qb}\"\n\n\
         `{wrong}` is the one that is wrong: {why}\n\n\
         The change: {change}\n\n\
         Make it with `file_edit` — the clashing words replaced, nothing else in `{wrong}` \
         touched — so that it agrees with `{right}`."
    );
    let fields = fields([
        ("a", a.to_string()),
        ("b", b.to_string()),
        ("qa", qa),
        ("qb", qb),
        ("wrong", wrong.to_string()),
        ("right", right.to_string()),
        ("why", why),
        ("change", change),
    ]);
    Ok(Answer::Mission(Proposal {
        objective: format!("Correct `{wrong}` so it agrees with `{right}`"),
        brief,
        writes: wrong.to_string(),
        reads: vec![a.to_string(), b.to_string()],
        min_words: 0,
        fields,
    }))
}
