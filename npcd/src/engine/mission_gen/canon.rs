//! The main storyline a document answers to — the eras — and the year its work
//! is done in.
//!
//! **Read the storyline, then decide.** A workflow's check against the
//! storyline (`context: storyline`, see [`super::step`]) reads the document
//! and the eras around it — the one it is set in and those either side — and
//! names them in its prompt; the Maker accepts it, mends a contradiction an edit
//! can put right, or rejects it. The reading steps are the engine's to sign
//! off, and no verdict is taken until they are done, so a check is never a
//! judgement made without the storyline in front of it.

use super::corpus::Corpus;
use crate::sim::operations::Operation;

/// The eras a document is checked against, in the order they happened: the
/// one it is set in and the ones either side. A story's era is the one its
/// operation was opened for; a life event's is the one its year falls in.
pub fn storyline(op: &Operation, corpus: &Corpus) -> Vec<String> {
    let set_in = match op.target.strip_prefix("era:") {
        Some(era) => corpus.eras.iter().position(|e| e.path == era),
        None => op
            .document
            .rsplit('/')
            .next()
            .and_then(|f| f.get(..4))
            .and_then(|y| y.parse::<u32>().ok())
            .and_then(|y| corpus.era_of(y))
            .and_then(|era| corpus.eras.iter().position(|e| e.path == era.path)),
    };
    let Some(at) = set_in else {
        // Set in no era the record knows: the whole storyline is what it is
        // checked against.
        return corpus.eras.iter().map(|e| e.path.clone()).collect();
    };
    let from = at.saturating_sub(1);
    let to = (at + 1).min(corpus.eras.len().saturating_sub(1));
    corpus.eras[from..=to]
        .iter()
        .map(|e| e.path.clone())
        .collect()
}

/// The year a document of an operation is set in, which its Maker works in at
/// a time machine: a life event's own year, or the year a story's era opens.
/// `None` for a correction, which answers to two eras at once.
pub fn set_in(target: &str, document: &str, corpus: &Corpus) -> Option<u32> {
    if let Some(era) = target.strip_prefix("era:") {
        return corpus.eras.iter().find(|e| e.path == era)?.year;
    }
    if !document.starts_with("layers/life/") {
        return None;
    }
    document
        .rsplit('/')
        .next()
        .and_then(|f| f.get(..4))
        .and_then(|y| y.parse::<u32>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission_gen::corpus::tests::mind;
    use crate::sim::operations::tests::workflows;
    use crate::sim::operations::Operations;

    /// **A life event is checked against the era its year falls in and those
    /// either side; a story against the era it was told for and its
    /// neighbours.**
    #[test]
    fn the_storyline_is_the_era_it_is_set_in_and_its_neighbours() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let mut ops = Operations::default();
        ops.set_workflows(workflows());
        let life = ops
            .open(
                "life-event",
                None,
                "life-event",
                "life:keeper",
                "Keeper's charge",
                "layers/life/keeper/2786 The Charge.md",
            )
            .unwrap();
        assert_eq!(
            storyline(ops.get(life).unwrap(), &c),
            [
                "layers/eras/the-fall.md",
                "layers/eras/the-retreat.md",
                "layers/eras/the-salvation.md",
            ]
        );
        let story = ops
            .open(
                "story",
                None,
                "untold",
                "era:layers/eras/the-fall.md",
                "a story",
                "layers/stories/x.md",
            )
            .unwrap();
        assert_eq!(
            storyline(ops.get(story).unwrap(), &c),
            ["layers/eras/the-fall.md", "layers/eras/the-retreat.md"],
            "the first era has none before it"
        );
    }

    /// **A life event is worked in its own year, a story in the year its era
    /// opens**, and a correction — two eras at once — in none.
    #[test]
    fn a_document_is_worked_in_the_year_it_is_set_in() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        assert_eq!(
            set_in("life:keeper", "layers/life/keeper/2786 The Charge.md", &c),
            Some(2786)
        );
        assert_eq!(
            set_in("era:layers/eras/the-fall.md", "layers/stories/x.md", &c),
            Some(2487)
        );
        assert_eq!(
            set_in(
                "pair:layers/eras/the-fall.md|layers/eras/the-retreat.md",
                "layers/eras/the-fall.md",
                &c
            ),
            None
        );
    }
}
