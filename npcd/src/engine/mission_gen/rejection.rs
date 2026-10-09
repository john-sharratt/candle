//! What a rejection must show before it is taken: the draft's own words that
//! cannot stand, and — against the storyline — the era they contradict.
//!
//! **A rejection is held to the evidence a reading is.** A Maker that had just
//! rejected one story as "structurally broken — it loops endlessly" rejected the
//! next document it was given, a life event that passed review, with the same
//! words. The reason was its own previous report, still in its history, and
//! nothing in the draft. A reason that quotes the draft cannot be carried over
//! from another one.

use std::path::Path;

use super::corpus::{era_year, heading};
use super::reading::quotes_draft;
use crate::engine::mission::{Stage, Work};

/// Why a rejection of `work` with `why` is refused, for the Maker to read and
/// answer; `None` when it shows what it must. `root` is the mind the documents
/// are read from. A draft no longer on the record needs no quote.
pub fn unsupported(root: &Path, work: &Work, stage: Stage, why: &str) -> Option<String> {
    let draft = std::fs::read_to_string(root.join(&work.writes)).ok()?;
    if !quotes_draft(why, &draft) {
        return Some(format!(
            "Your reason does not quote {doc}. Copy the words of it that cannot stand, exactly, \
             between double quotes — \"<the draft's sentence>\" — and say what is wrong with \
             them. If you cannot point at them, it can stand: mend what you can and `report_done`.",
            doc = work.writes
        ));
    }
    if stage != Stage::Canon {
        return None;
    }
    let eras: Vec<(String, Option<u32>)> = work
        .reads
        .iter()
        .filter(|r| r.starts_with("layers/eras/"))
        .filter_map(|r| std::fs::read_to_string(root.join(r)).ok())
        .filter_map(|t| Some((heading(&t)?, era_year(&t))))
        .collect();
    let lower = why.to_lowercase();
    let named = eras.iter().any(|(title, year)| {
        lower.contains(&title.to_lowercase()) || year.is_some_and(|y| why.contains(&y.to_string()))
    });
    match named || eras.is_empty() {
        true => None,
        false => Some(format!(
            "A draft is rejected against the storyline only for contradicting an era. Name the \
             era it contradicts — {} — and say what the era says against the words you quoted. \
             If no era says otherwise, it agrees with the storyline: `report_done`.",
            eras.iter()
                .map(|(t, _)| t.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRAFT: &str = "The Hart interviewer wanted to know what the room was and you could not \
                         tell her. The floor just shakes.";

    fn mind() -> (tempfile::TempDir, Work) {
        let dir = tempfile::tempdir().unwrap();
        let write = |rel: &str, text: &str| {
            let p = dir.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        };
        write("layers/life/cadence/2900 The Shelf.md", DRAFT);
        write(
            "layers/eras/the-final-battle.md",
            "# The Final Battle\n\n2792 CE. The last engagement on the surface.",
        );
        write(
            "layers/eras/the-tower-age.md",
            "# The Tower Age\n\n2937 CE. The towers rise.",
        );
        let work = Work {
            writes: "layers/life/cadence/2900 The Shelf.md".into(),
            reads: vec![
                "layers/life/cadence/2900 The Shelf.md".into(),
                "layers/eras/the-final-battle.md".into(),
                "layers/eras/the-tower-age.md".into(),
            ],
            min_words: 0,
            edit_optional: true,
            anew: false,
        };
        (dir, work)
    }

    /// **A reason carried over from another draft quotes nothing of this one**,
    /// and is refused; one that quotes it stands on review.
    #[test]
    fn a_rejection_quotes_the_draft_it_rejects() {
        let (dir, work) = mind();
        let carried = "The draft is structurally broken — it loops endlessly on the same \
                       description of the room.";
        let refused = unsupported(dir.path(), &work, Stage::Review, carried).unwrap();
        assert!(refused
            .starts_with("Your reason does not quote layers/life/cadence/2900 The Shelf.md."));
        let quoted = "\"The Hart interviewer wanted to know what the room was\" — no such \
                      interviewer exists before the towers.";
        assert_eq!(unsupported(dir.path(), &work, Stage::Review, quoted), None);
    }

    /// **Against the storyline, a rejection names the era it contradicts**, by
    /// its title or its year.
    #[test]
    fn a_canon_rejection_names_its_era() {
        let (dir, work) = mind();
        let unnamed = "\"The floor just shakes.\" is wrong.";
        assert!(unsupported(dir.path(), &work, Stage::Canon, unnamed)
            .unwrap()
            .contains("Name the era it contradicts — The Final Battle, The Tower Age —"));
        let titled = "\"The floor just shakes.\" — the final battle era has nobody left there.";
        assert_eq!(unsupported(dir.path(), &work, Stage::Canon, titled), None);
        let dated = "\"The floor just shakes.\" — by 2937 the shelf is a tower footing.";
        assert_eq!(unsupported(dir.path(), &work, Stage::Canon, dated), None);
    }

    /// A draft already gone from the record has nothing left to quote.
    #[test]
    fn a_draft_off_the_record_needs_no_quote() {
        let (dir, work) = mind();
        std::fs::remove_file(dir.path().join(&work.writes)).unwrap();
        assert_eq!(
            unsupported(
                dir.path(),
                &work,
                Stage::Review,
                "It is not on the record any more."
            ),
            None
        );
    }
}
