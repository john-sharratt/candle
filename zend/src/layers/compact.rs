//! A finished unit, written again as the one exchange it amounts to.
//!
//! A unit is read in many turns — its question, each round of tool calls and
//! what they answered, the reasoning between them — and what it stands for is
//! its last answer. Every turn of the reading stays in the substrate, and a
//! unit forked on it (the layer above, built on this one) carries the whole
//! reading as its opening.
//!
//! So once a unit has answered, it is written again as a conversation of one
//! exchange: a short question naming the layer and the unit's subject
//! ([`question`]), and the answer, with no reasoning block, prefilled rather
//! than decoded. It is minted as the unit was — the same lineage, retrieval
//! scopes and early tags — so it stands where the reading stood, and the
//! reading goes. What stays is the answer's own K/V, a small part of the
//! reading's.

use std::sync::Mutex;

use candle_conversation::Sequence;

use super::run::{mint_unit, UnitJob};
use super::spec::LayerSpec;
use crate::refresh_ctx::RefreshContext;

/// The question a unit's exchange is written under: the layer's name as words
/// with its first letter capitalised, and the subject — `code_review` on
/// `candle/zend/src/rate.rs` asks "Code review: candle/zend/src/rate.rs".
pub fn question(layer: &str, subject: &str) -> String {
    let words = layer.replace('_', " ");
    let mut chars = words.chars();
    let named = match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    };
    format!("{named}: {subject}")
}

/// The unit `job` of `spec`'s layer as one exchange: minted on `base` with the
/// unit's lineage and early tags, its [`question`] and `answer` prefilled. The
/// caller commits it and lets the reading go; on an error it tombstones what
/// was minted.
pub fn rewrite(
    ctx: &RefreshContext<'_>,
    spec: &LayerSpec,
    job: &UnitJob,
    base: &Mutex<Sequence>,
    answer: &str,
) -> anyhow::Result<Sequence> {
    let mut conv = mint_unit(ctx, spec, job, base)?;
    let prefilled = conv
        .insert_turn_staged(
            question(&spec.layer, &job.subject),
            answer,
            vec![spec.layer.clone(), job.subject.clone()],
        )
        .map_err(|e| anyhow::anyhow!("writing the unit's exchange: {e}"));
    if let Err(e) = prefilled {
        let _ = ctx
            .engine
            .lock()
            .unwrap()
            .tombstone_timeline(conv.timeline_id());
        return Err(e);
    }
    Ok(conv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_question_names_the_layer_and_the_subject() {
        assert_eq!(
            question("code_review", "candle/zend/src/rate.rs"),
            "Code review: candle/zend/src/rate.rs"
        );
        assert_eq!(
            question("dependency_analysis", "candle/zend/"),
            "Dependency analysis: candle/zend/"
        );
        assert_eq!(question("days", "2026-10-28"), "Days: 2026-10-28");
        assert_eq!(question("", "x"), ": x");
    }
}
