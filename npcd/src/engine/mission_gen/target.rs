//! What a generator works on: the kinds of work, and the piece of the corpus the
//! next mission of each kind is about.
//!
//! **Chosen by the engine, written by the model.** Which character's life is
//! thinnest, which two documents have not been compared, which era nobody has
//! told a story of — these are counts over the corpus, and a model asked to pick
//! freely goes back to what it already knows (the reflection's domains measured
//! exactly that). So coverage is structural: the engine walks the corpus and the
//! ledger, and the model is asked about one target at a time.

use serde::{Deserialize, Serialize};

use super::corpus::Corpus;
use super::fingerprint::Fingerprint;

/// A kind of work a generator finds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// The next significant event in a character's life, written as a dated
    /// document in `layers/life/<who>/`.
    LifeEvent,
    /// Two documents that touch the same ground, read against each other; the
    /// one that is wrong is corrected.
    Contradiction,
    /// Something an era passes over that nobody has told, written as a story in
    /// `layers/stories/`.
    Gap,
}

/// What a target is about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Subject {
    /// A character's life, by personality id.
    Life { who: String },
    /// Two documents, by mind path.
    Pair { a: String, b: String },
    /// An era, by mind path.
    Era { path: String },
}

/// One piece of work a mission can be generated for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    /// What the ledger knows it by.
    pub key: String,
    /// What it held when chosen — a target settled while its documents held
    /// exactly this is not worked again until they change.
    pub fingerprint: u64,
    pub subject: Subject,
}

/// The next target of `kind`, skipping any `blocked` says is in hand or
/// settled. `turn` rotates among equally good candidates so successive draws
/// spread out.
pub fn next(
    kind: Kind,
    corpus: &Corpus,
    blocked: &dyn Fn(&str, u64) -> bool,
    turn: u64,
) -> Option<Target> {
    let candidates = match kind {
        Kind::LifeEvent => lives(corpus),
        Kind::Contradiction => pairs(corpus),
        Kind::Gap => eras(corpus),
    };
    let open: Vec<(usize, Target)> = candidates
        .into_iter()
        .filter(|(_, t)| !blocked(&t.key, t.fingerprint))
        .collect();
    // The best rank first, rotated among the ties.
    let best = open.iter().map(|(rank, _)| *rank).min()?;
    let tied: Vec<Target> = open
        .into_iter()
        .filter(|(rank, _)| *rank == best)
        .map(|(_, t)| t)
        .collect();
    Some(tied[(turn % tied.len() as u64) as usize].clone())
}

/// Every life, ranked by how few events it has: the thinnest is worked first,
/// so the cast fills evenly rather than one life growing while the rest stay at
/// two pages.
fn lives(corpus: &Corpus) -> Vec<(usize, Target)> {
    corpus
        .lives
        .iter()
        .map(|l| {
            (
                l.events.len(),
                Target {
                    key: format!("life:{}", l.who),
                    fingerprint: l.fingerprint(),
                    subject: Subject::Life { who: l.who.clone() },
                },
            )
        })
        .collect()
}

/// Every pair worth reading together, ranked: neighbouring eras first, then
/// each written life event against the era it falls in.
///
/// A boundary between two eras is where the record most often disagrees with
/// itself — each era was written as if it were the whole story — and a life
/// event is the place a character's history meets the world's.
fn pairs(corpus: &Corpus) -> Vec<(usize, Target)> {
    let mut out = Vec::new();
    for w in corpus.eras.windows(2) {
        out.push((0, pair(corpus, &w[0].path, &w[1].path)));
    }
    for life in &corpus.lives {
        for event in &life.events {
            if let Some(era) = event.year().and_then(|y| corpus.era_of(y)) {
                out.push((1, pair(corpus, &era.path, &event.path)));
            }
        }
    }
    out
}

fn pair(corpus: &Corpus, a: &str, b: &str) -> Target {
    let mut h = Fingerprint::new();
    h.add_opt(corpus.text(a).as_deref())
        .add_opt(corpus.text(b).as_deref());
    Target {
        key: format!("pair:{a}|{b}"),
        fingerprint: h.finish(),
        subject: Subject::Pair {
            a: a.to_string(),
            b: b.to_string(),
        },
    }
}

/// Every era, ranked by how many stories already name it — the least told
/// first.
fn eras(corpus: &Corpus) -> Vec<(usize, Target)> {
    corpus
        .eras
        .iter()
        .map(|era| {
            let told = corpus
                .stories
                .iter()
                .filter(|s| s.text.contains(&era.title))
                .count();
            let mut h = Fingerprint::new();
            h.add(&era.text);
            for s in &corpus.stories {
                h.add(&s.path);
            }
            (
                told,
                Target {
                    key: format!("era:{}", era.path),
                    fingerprint: h.finish(),
                    subject: Subject::Era {
                        path: era.path.clone(),
                    },
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission_gen::corpus::tests::mind;

    fn nothing_blocked(_: &str, _: u64) -> bool {
        false
    }

    /// **The thinnest life is worked first.** Kaelor has no events and Keeper
    /// two; with Kaelor in hand, Keeper is next.
    #[test]
    fn the_thinnest_life_is_chosen_and_one_in_hand_is_skipped() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let t = next(Kind::LifeEvent, &c, &nothing_blocked, 0).unwrap();
        assert_eq!(t.key, "life:kaelor");
        let t = next(Kind::LifeEvent, &c, &|k, _| k == "life:kaelor", 0).unwrap();
        assert_eq!(
            t.subject,
            Subject::Life {
                who: "keeper".into()
            }
        );
        assert_eq!(next(Kind::LifeEvent, &c, &|_, _| true, 0), None);
    }

    /// Neighbouring eras are compared before a life against its era, and each
    /// life event is paired with the era it falls in.
    #[test]
    fn era_boundaries_come_first_then_each_event_against_its_era() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let keys: Vec<String> = pairs(&c).into_iter().map(|(_, t)| t.key).collect();
        assert_eq!(
            keys,
            [
                "pair:layers/eras/the-fall.md|layers/eras/the-retreat.md",
                "pair:layers/eras/the-retreat.md|layers/eras/the-salvation.md",
                "pair:layers/eras/the-fall.md|layers/life/keeper/2487-03-08 The Second the Sky \
                 Went Out.md",
                "pair:layers/eras/the-retreat.md|layers/life/keeper/2786 The Charge.md",
            ]
        );
        // Ties rotate with the turn.
        let first = next(Kind::Contradiction, &c, &nothing_blocked, 0).unwrap();
        let second = next(Kind::Contradiction, &c, &nothing_blocked, 1).unwrap();
        assert_ne!(first.key, second.key);
    }

    /// **A settled pair stays settled until a document changes.** The ledger
    /// keys on the fingerprint, which moves with either text.
    #[test]
    fn a_pair_fingerprint_moves_when_either_document_changes() {
        let dir = mind();
        let before = pair(
            &Corpus::read(dir.path(), "test"),
            "layers/eras/the-fall.md",
            "layers/eras/the-retreat.md",
        );
        std::fs::write(
            dir.path().join("layers/eras/the-retreat.md"),
            "# The Retreat\n\n**Era 120 · 2607 CE**\n\nChanged.\n",
        )
        .unwrap();
        let after = pair(
            &Corpus::read(dir.path(), "test"),
            "layers/eras/the-fall.md",
            "layers/eras/the-retreat.md",
        );
        assert_eq!(before.key, after.key);
        assert_ne!(before.fingerprint, after.fingerprint);
    }

    /// The least-told era is chosen; "The Charge" names no era here, so all
    /// three tie and the turn decides.
    #[test]
    fn the_least_told_era_is_chosen() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let ranks: Vec<usize> = eras(&c).into_iter().map(|(r, _)| r).collect();
        assert_eq!(ranks, [0, 0, 0]);
        std::fs::write(
            dir.path().join("layers/stories/a-fall.md"),
            "# A Fall\n\nIn The Fall, a clerk finished the schedule.\n",
        )
        .unwrap();
        let c = Corpus::read(dir.path(), "test");
        let t = next(Kind::Gap, &c, &nothing_blocked, 0).unwrap();
        assert_ne!(t.key, "era:layers/eras/the-fall.md", "the told era waits");
    }
}
