//! The last acceptance step of an operation: a reviewed life event or story
//! checked against the main storyline — the eras — for major contradictions.
//!
//! **Read the storyline, then decide.** The review read the draft as writing;
//! this reads it as history. A Maker who has carried no other stage of the
//! operation reads the draft and the eras around it — the one it is set in and
//! those either side — and then decides for itself: accept it as it stands,
//! mend a contradiction that an edit can put right and accept it, or reject it
//! when the contradiction runs through it. The reading steps are the engine's to
//! sign off, and neither verdict is taken until they are done, so a check is
//! never a judgement made without the storyline in front of it.

use super::answer::Desk;
use super::corpus::Corpus;
use super::gates::{Form, LIFE_MIN_WORDS, STORY_MIN_WORDS};
use crate::engine::mission::{Mission, Origin, Stage, Todo, Work};
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

/// The canon-check mission for operation `op`.
pub fn canon_mission(op: &Operation, corpus: &Corpus, desk: Option<&Desk>) -> Mission {
    let doc = &op.document;
    let eras = storyline(op, corpus);
    let mut reads = vec![doc.clone()];
    reads.extend(eras.iter().cloned());
    let mut todo = Vec::new();
    if let Some(d) = desk {
        todo.push(Todo::new(format!("go to {} on {}", d.room, d.level)));
    }
    for r in &reads {
        todo.push(Todo::new(format!("read {r}")));
    }
    todo.push(Todo::report("go back to the table and report your verdict"));
    let titles: Vec<String> = eras
        .iter()
        .filter_map(|p| corpus.eras.iter().find(|e| &e.path == p))
        .map(|e| match e.year {
            Some(y) => format!("{} ({y})", e.title),
            None => e.title.clone(),
        })
        .collect();
    let brief = format!(
        "{name} — {objective}.\n\n\
         `{doc}` has been written and passed on review by two other Makers. Before it stands in \
         the record, check it against the main storyline: the eras of this world, which \
         everything else must agree with. Read the draft, then read the storyline around it — \
         {titles} — before you decide anything.\n\n\
         Look for major contradictions only: a date the eras put elsewhere — check every year \
         it names, and every claim about how things stood then (\"the war is over\", \"the \
         gates are open\", who ruled, who was gone) against the era that year falls in — a war \
         or a battle with a different outcome, somebody somewhere the storyline says they could \
         not be, something that did not exist yet or no longer did, an order of events the eras \
         reverse. Its style and voice have been reviewed already; leave them.\n\n\
         Then decide.\n\
         - It agrees with the storyline: `report_done`, saying what you checked it against.\n\
         - A contradiction an edit can put right: put it right with `file_edit`, `bench_commit` \
           it, then `report_done` saying what you changed and why.\n\
         - The contradiction runs through it — the events themselves could not have happened \
           as told: `report_rejected`, saying which era it contradicts and how. A rejected draft \
           leaves the record.",
        name = op.name,
        objective = op.objective,
        titles = titles.join(", "),
    );
    Mission::new(
        brief,
        todo,
        Origin::Generated {
            generator: op.generator.clone(),
            target: op.target.clone(),
            operation: op.id,
            stage: Stage::Canon,
        },
    )
    .with_work(Work {
        writes: doc.clone(),
        reads,
        min_words: match Form::of(doc) {
            Form::LifeEvent => LIFE_MIN_WORDS,
            Form::Story => STORY_MIN_WORDS,
            Form::Other => 0,
        },
        edit_optional: true,
        anew: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission_gen::corpus::tests::mind;
    use crate::sim::operations::Operations;

    /// **A life event is checked against the era its year falls in and those
    /// either side; a story against the era it was told for and its
    /// neighbours.** Every era is read before the verdict.
    #[test]
    fn a_check_reads_the_draft_and_the_storyline_around_it() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let mut ops = Operations::default();
        let life = ops.open(
            "life-event",
            "life:keeper",
            "Keeper's charge",
            "layers/life/keeper/2786 The Charge.md",
        );
        let m = canon_mission(ops.get(life).unwrap(), &c, None);
        let steps: Vec<&str> = m.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "read layers/life/keeper/2786 The Charge.md",
                "read layers/eras/the-fall.md",
                "read layers/eras/the-retreat.md",
                "read layers/eras/the-salvation.md",
                "go back to the table and report your verdict",
            ]
        );
        assert_eq!(m.operation(), Some((life, Stage::Canon)));
        assert!(m.written_up(), "accepting it unchanged is a verdict");
        assert!(m.prompt.contains("before you decide anything"));
        assert!(m.prompt.contains("`report_rejected`"));

        let story = ops.open(
            "untold",
            "era:layers/eras/the-fall.md",
            "a story",
            "layers/stories/x.md",
        );
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
