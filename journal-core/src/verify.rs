//! The server's half of a journal entry: what the model wrote, checked.
//!
//! A [`Draft`] is only what the model supplies — claim text, the turns each claim
//! rests on, and how each open item relates to what is already open. [`verify`]
//! turns it into an [`Entry`] or refuses it, and decides everything the model is
//! not trusted to say: how a claim came to be known ([`Kind`], read off the
//! cited turns' evidence, never off wording), whether it can go stale, and — for
//! a claim the world can be asked about — whether the world agrees.
//!
//! The same draft against the same span, open items and world always draws the
//! same verdict. Nothing here samples.

use std::collections::BTreeSet;

use crate::entry::{Claim, Entry, Item, Kind, Typed};
use crate::period::Period;
use crate::span::Span;

/// Claims one entry may carry.
pub const MAX_CLAIMS: usize = 12;
/// Intentions one entry may carry.
pub const MAX_INTEND: usize = 3;

/// What the world can say about a thing, when asked.
///
/// `None` is "the world has no answer", which is not a disagreement.
pub trait World {
    fn read(&self, subject: &str, attribute: &str) -> Option<String>;
}

/// A claim as the model wrote it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftClaim {
    pub text: String,
    pub cite: Vec<u64>,
    pub typed: Option<Typed>,
}

/// How an open item relates to the items already open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relates {
    New,
    Restates(u64),
    Resolves(u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftOpen {
    pub text: String,
    pub relates: Relates,
}

/// An entry as the model wrote it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Draft {
    /// The task or step the stretch was, in one sentence.
    pub summary: String,
    pub claims: Vec<DraftClaim>,
    pub intend: Vec<String>,
    pub open: Vec<DraftOpen>,
}

/// Why a draft was not accepted. Worded for the model, which reads it and may
/// try again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    Empty,
    TooMany { what: &'static str, max: usize },
    EmptyText { what: &'static str },
    NotCitable { claim: usize, turn: u64 },
    NotOpen { id: u64 },
    ConflictingItem { id: u64 },
}

impl Refusal {
    pub fn reason(&self) -> String {
        match self {
            Refusal::Empty => "That entry says nothing: it has no claims, no intentions and \
                               nothing opened or settled."
                .into(),
            Refusal::TooMany { what, max } => {
                format!("Too many {what}: an entry holds at most {max}.")
            }
            Refusal::EmptyText { what } => format!("A {what} has no text."),
            Refusal::NotCitable { claim, turn } => format!(
                "Claim {} cites turn {turn}, which is not something a claim can rest on: cite \
                 only the numbered turns of the stretch.",
                claim + 1
            ),
            Refusal::NotOpen { id } => format!("Item #{id} is not open."),
            Refusal::ConflictingItem { id } => {
                format!("Item #{id} is both restated and resolved in one entry.")
            }
        }
    }
}

/// What the check decided about one claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Kept,
    /// The world answered differently; the claim keeps its sentence and
    /// carries what the world read.
    Corrected,
    /// The world had no answer; the claim stands as prose.
    Demoted,
    /// A later claim about the same subject and attribute took its typed fact
    /// over: it stands as prose.
    Superseded,
}

impl Verdict {
    /// The name it is shown under.
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Kept => "kept",
            Verdict::Corrected => "corrected",
            Verdict::Demoted => "demoted",
            Verdict::Superseded => "superseded",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Note {
    pub claim: usize,
    pub verdict: Verdict,
    pub kind: Kind,
}

/// An accepted draft.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verified {
    pub entry: Entry,
    pub notes: Vec<Note>,
}

/// The strongest support among the cited turns; none is a conclusion.
fn derive_kind(span: &Span, cite: &[u64]) -> Kind {
    cite.iter()
        .filter_map(|id| span.turn(*id)?.evidence)
        .max_by_key(|k| k.rank())
        .unwrap_or(Kind::Inferred)
}

/// A yes-or-no value as one word: `yes` is `true` and `no` is `false`, so a
/// claim the world confirms is not "corrected" into the word it already meant.
fn truth(v: &str) -> String {
    let v = v.trim().to_lowercase();
    match v.as_str() {
        "yes" => "true".to_string(),
        "no" => "false".to_string(),
        _ => v,
    }
}

fn same(a: &str, b: &str) -> bool {
    truth(a) == truth(b)
}

/// Check a draft against the span it is about, the items open now, and the
/// world.
pub fn verify(
    draft: Draft,
    span: &Span,
    open: &[Item],
    world: &dyn World,
) -> Result<Verified, Refusal> {
    if draft.claims.len() > MAX_CLAIMS {
        return Err(Refusal::TooMany {
            what: "claims",
            max: MAX_CLAIMS,
        });
    }
    if draft.intend.len() > MAX_INTEND {
        return Err(Refusal::TooMany {
            what: "intentions",
            max: MAX_INTEND,
        });
    }
    let citable: BTreeSet<u64> = span.citable_ids().into_iter().collect();

    let mut claims: Vec<(usize, Claim)> = Vec::new();
    let mut notes: Vec<Note> = Vec::new();
    for (n, c) in draft.claims.into_iter().enumerate() {
        if c.text.trim().is_empty() {
            return Err(Refusal::EmptyText { what: "claim" });
        }
        if let Some(turn) = c.cite.iter().find(|t| !citable.contains(t)) {
            return Err(Refusal::NotCitable {
                claim: n,
                turn: *turn,
            });
        }
        let kind = derive_kind(span, &c.cite);
        let mut claim = Claim {
            text: c.text.trim().to_string(),
            cite: c.cite,
            kind,
            perishable: kind != Kind::Did,
            typed: c.typed,
            corrected: false,
        };
        let mut verdict = Verdict::Kept;
        if let Some(t) = claim.typed.clone() {
            match world.read(&t.subject, &t.attribute) {
                Some(actual) if same(&actual, &t.value) => {}
                // The sentence stays: it is the writer's, and often says more
                // than the fact it was typed with. The line carries what the
                // world read beside it (`Claim::line`).
                Some(actual) => {
                    claim.typed = Some(Typed { value: actual, ..t });
                    claim.corrected = true;
                    claim.perishable = true;
                    verdict = Verdict::Corrected;
                }
                None => {
                    claim.typed = None;
                    verdict = Verdict::Demoted;
                }
            }
        }
        // A newer claim about the same thing takes the typed fact over: an
        // older one stays, as prose. One the world corrected keeps its
        // reading, which is part of what its line says.
        if let Some(t) = &claim.typed {
            let key = (t.subject.to_lowercase(), t.attribute.to_lowercase());
            let at = claims.iter().position(|(_, o)| {
                !o.corrected
                    && o.typed.as_ref().is_some_and(|ot| {
                        (ot.subject.to_lowercase(), ot.attribute.to_lowercase()) == key
                    })
            });
            if let Some(i) = at {
                let older = claims[i].0;
                claims[i].1.typed = None;
                if let Some(note) = notes.iter_mut().find(|note| note.claim == older) {
                    note.verdict = Verdict::Superseded;
                }
            }
        }
        notes.push(Note {
            claim: n,
            verdict,
            kind,
        });
        claims.push((n, claim));
    }
    let claims: Vec<Claim> = claims.into_iter().map(|(_, c)| c).collect();

    let intend: Vec<String> = draft
        .intend
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let live: BTreeSet<u64> = open.iter().map(|i| i.id).collect();
    let mut opened: Vec<Item> = Vec::new();
    let mut resolved: Vec<u64> = Vec::new();
    let mut restated: BTreeSet<u64> = BTreeSet::new();
    for o in draft.open {
        if o.text.trim().is_empty() {
            return Err(Refusal::EmptyText { what: "open item" });
        }
        let text = o.text.trim().to_string();
        match o.relates {
            Relates::New => opened.push(Item { id: 0, text }),
            Relates::Restates(id) => {
                if !live.contains(&id) {
                    return Err(Refusal::NotOpen { id });
                }
                restated.insert(id);
                opened.push(Item { id, text });
            }
            Relates::Resolves(id) => {
                if !live.contains(&id) {
                    return Err(Refusal::NotOpen { id });
                }
                if !resolved.contains(&id) {
                    resolved.push(id);
                }
            }
        }
    }
    if let Some(id) = resolved.iter().find(|id| restated.contains(id)) {
        return Err(Refusal::ConflictingItem { id: *id });
    }

    if claims.is_empty() && intend.is_empty() && opened.is_empty() && resolved.is_empty() {
        return Err(Refusal::Empty);
    }
    let summary = draft.summary.trim().to_string();
    if summary.is_empty() {
        return Err(Refusal::EmptyText { what: "summary" });
    }
    Ok(Verified {
        entry: Entry {
            id: 0,
            from_turn: span.from_turn,
            to_turn: span.to_turn,
            from_ms: span.from_ms,
            to_ms: span.to_ms,
            summary,
            claims,
            intend,
            opened,
            resolved,
            period: Period::Step,
            digest: String::new(),
        },
        notes,
    })
}

#[cfg(test)]
pub(crate) mod testing {
    use std::collections::BTreeMap;

    use super::World;

    /// A world that answers from a table, case-blind.
    #[derive(Default)]
    pub struct Fake(BTreeMap<(String, String), String>);

    impl Fake {
        pub fn with(mut self, s: &str, a: &str, v: &str) -> Self {
            self.0
                .insert((s.to_lowercase(), a.to_lowercase()), v.into());
            self
        }
    }

    impl World for Fake {
        fn read(&self, subject: &str, attribute: &str) -> Option<String> {
            self.0
                .get(&(subject.to_lowercase(), attribute.to_lowercase()))
                .cloned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::Fake;
    use super::*;
    use crate::span::testing::turn;

    /// Turns: 1 heard, 2 observed, 3 own act, 4 a tool answer (observed),
    /// 5 scenery (uncitable), 6 own reflection (uncitable).
    fn span() -> Span {
        Span::of(vec![
            turn(1, 10, Some(Kind::Heard), "Pax: Fabricator 5 is lost."),
            turn(2, 20, Some(Kind::Observed), "You are in the foundry."),
            turn(3, 30, Some(Kind::Did), "You: tell Pax: on my way"),
            turn(4, 40, Some(Kind::Observed), "You: read — shields are down"),
            turn(5, 50, None, "The conduit hums."),
            turn(6, 60, None, "You: reflect — thinking"),
        ])
        .unwrap()
    }

    fn claim(text: &str, cite: &[u64]) -> DraftClaim {
        DraftClaim {
            text: text.into(),
            cite: cite.to_vec(),
            typed: None,
        }
    }

    fn typed(subject: &str, attribute: &str, value: &str, cite: &[u64]) -> DraftClaim {
        DraftClaim {
            text: format!("{subject} {attribute} is {value}"),
            cite: cite.to_vec(),
            typed: Some(Typed {
                subject: subject.into(),
                attribute: attribute.into(),
                value: value.into(),
            }),
        }
    }

    /// [`super::verify`] with a summary filled in for a draft that left it
    /// out, so each test states only what it is about.
    fn verify(
        mut d: Draft,
        span: &Span,
        open: &[Item],
        world: &dyn World,
    ) -> Result<Verified, Refusal> {
        if d.summary.is_empty() {
            d.summary = "A step.".into();
        }
        super::verify(d, span, open, world)
    }

    fn run(d: Draft) -> Result<Verified, Refusal> {
        verify(d, &span(), &[], &Fake::default())
    }

    #[test]
    fn an_entry_without_a_summary_is_refused_and_one_with_it_keeps_it() {
        let d = Draft {
            summary: "  ".into(),
            claims: vec![claim("I saw the room.", &[2])],
            ..Default::default()
        };
        assert_eq!(
            super::verify(d.clone(), &span(), &[], &Fake::default()),
            Err(Refusal::EmptyText { what: "summary" })
        );
        let kept = super::verify(
            Draft {
                summary: " Looked round the foundry. ".into(),
                ..d
            },
            &span(),
            &[],
            &Fake::default(),
        )
        .unwrap();
        assert_eq!(kept.entry.summary, "Looked round the foundry.");
    }

    fn one(c: DraftClaim) -> Claim {
        let v = run(Draft {
            claims: vec![c],
            ..Default::default()
        })
        .unwrap();
        v.entry.claims.into_iter().next().unwrap()
    }

    #[test]
    fn the_kind_comes_from_what_is_cited_and_not_from_the_wording() {
        assert_eq!(one(claim("I saw the room.", &[2])).kind, Kind::Observed);
        assert_eq!(one(claim("I saw Pax say it.", &[1])).kind, Kind::Heard);
        assert_eq!(
            one(claim("The board says shields are down.", &[4])).kind,
            Kind::Observed
        );
        assert_eq!(one(claim("I told Pax.", &[3])).kind, Kind::Did);
        assert_eq!(
            one(claim("I observed it directly.", &[])).kind,
            Kind::Inferred
        );
    }

    #[test]
    fn the_strongest_support_wins_when_several_turns_are_cited() {
        assert_eq!(one(claim("c", &[1, 2])).kind, Kind::Observed);
        assert_eq!(one(claim("c", &[3, 1])).kind, Kind::Heard);
        assert_eq!(one(claim("c", &[3])).kind, Kind::Did);
    }

    #[test]
    fn what_was_done_does_not_go_stale_and_everything_else_can() {
        assert!(!one(claim("I told Pax.", &[3])).perishable);
        assert!(one(claim("The room is quiet.", &[2])).perishable);
        assert!(one(claim("Pax is worried.", &[1])).perishable);
        assert!(one(claim("A guess.", &[])).perishable);
    }

    #[test]
    fn a_claim_may_not_rest_on_a_turn_with_no_evidence_or_outside_the_span() {
        for turn in [5, 6, 99] {
            assert_eq!(
                run(Draft {
                    claims: vec![claim("x", &[turn])],
                    ..Default::default()
                }),
                Err(Refusal::NotCitable { claim: 0, turn })
            );
        }
    }

    #[test]
    fn a_refusal_names_the_claim_by_its_position_as_a_reader_counts() {
        let r = Refusal::NotCitable { claim: 1, turn: 5 }.reason();
        assert!(r.starts_with("Claim 2 cites turn 5"), "{r}");
    }

    #[test]
    fn an_entry_that_says_nothing_is_refused() {
        assert_eq!(run(Draft::default()), Err(Refusal::Empty));
        assert_eq!(
            run(Draft {
                intend: vec!["  ".into()],
                ..Default::default()
            }),
            Err(Refusal::Empty)
        );
    }

    #[test]
    fn blank_text_is_refused_where_it_appears() {
        assert_eq!(
            run(Draft {
                claims: vec![claim("  ", &[])],
                ..Default::default()
            }),
            Err(Refusal::EmptyText { what: "claim" })
        );
        assert_eq!(
            run(Draft {
                open: vec![DraftOpen {
                    text: "".into(),
                    relates: Relates::New
                }],
                ..Default::default()
            }),
            Err(Refusal::EmptyText { what: "open item" })
        );
    }

    #[test]
    fn an_entry_is_bounded() {
        let many = Draft {
            claims: (0..=MAX_CLAIMS)
                .map(|i| claim(&format!("c{i}"), &[]))
                .collect(),
            ..Default::default()
        };
        assert_eq!(
            run(many),
            Err(Refusal::TooMany {
                what: "claims",
                max: MAX_CLAIMS
            })
        );
        let many = Draft {
            intend: vec!["a".into(); MAX_INTEND + 1],
            ..Default::default()
        };
        assert_eq!(
            run(many),
            Err(Refusal::TooMany {
                what: "intentions",
                max: MAX_INTEND
            })
        );
    }

    #[test]
    fn a_typed_claim_the_world_confirms_is_kept_and_stays_typed() {
        let world = Fake::default().with("fabricator 5", "mode", "running");
        let v = verify(
            Draft {
                claims: vec![typed("fabricator 5", "mode", "Running", &[2])],
                ..Default::default()
            },
            &span(),
            &[],
            &world,
        )
        .unwrap();
        let c = &v.entry.claims[0];
        assert!(c.typed.is_some() && !c.corrected && c.perishable);
        assert_eq!(v.notes[0].verdict, Verdict::Kept);
    }

    #[test]
    fn a_typed_claim_the_world_contradicts_keeps_its_sentence_and_carries_the_reading() {
        let world = Fake::default().with("fabricator 5", "mode", "running");
        let v = verify(
            Draft {
                claims: vec![typed("fabricator 5", "mode", "lost", &[1])],
                ..Default::default()
            },
            &span(),
            &[],
            &world,
        )
        .unwrap();
        let c = &v.entry.claims[0];
        assert_eq!(
            c.text, "fabricator 5 mode is lost",
            "the writer's sentence stays"
        );
        assert_eq!(
            c.typed.as_ref().unwrap().value,
            "running",
            "beside the reading"
        );
        assert!(c.corrected && c.perishable);
        assert_eq!(c.kind, Kind::Heard, "its support is still what it cited");
        assert_eq!(v.notes[0].verdict, Verdict::Corrected);
    }

    /// The live failure: every claim of an entry was typed `exists` with a
    /// value that was no answer to it — a constant's value, a line number —
    /// and each correction replaced the claim's sentence with the bare
    /// reading, so the entry kept one line, "file:… exists: true", of three.
    #[test]
    fn misused_typed_facts_never_cost_the_claims_their_sentences() {
        let world = Fake::default().with("file:c/journal.rs", "exists", "true");
        let said = |text: &str, value: &str, cite: u64| DraftClaim {
            text: text.into(),
            cite: vec![cite],
            typed: Some(Typed {
                subject: "file:c/journal.rs".into(),
                attribute: "exists".into(),
                value: value.into(),
            }),
        };
        let v = verify(
            Draft {
                claims: vec![
                    said("The user asked where IN_PROMPT is defined.", "IN_PROMPT", 1),
                    said("file_grep found IN_PROMPT at journal.rs:27.", "27", 2),
                    said("IN_PROMPT is 5.", "5", 4),
                ],
                ..Default::default()
            },
            &span(),
            &[],
            &world,
        )
        .unwrap();
        let texts: Vec<_> = v.entry.claims.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "The user asked where IN_PROMPT is defined.",
                "file_grep found IN_PROMPT at journal.rs:27.",
                "IN_PROMPT is 5."
            ]
        );
        assert!(v.entry.claims.iter().all(|c| c.corrected));
    }

    #[test]
    fn a_typed_claim_the_world_cannot_answer_stands_as_prose() {
        let v = verify(
            Draft {
                claims: vec![typed("the moon", "mood", "sour", &[1])],
                ..Default::default()
            },
            &span(),
            &[],
            &Fake::default(),
        )
        .unwrap();
        let c = &v.entry.claims[0];
        assert!(c.typed.is_none() && !c.corrected && c.perishable);
        assert_eq!(c.kind, Kind::Heard, "its support is what it cited");
        assert_eq!(v.notes[0].verdict, Verdict::Demoted);
    }

    /// A corrected claim keeps its reading even when a later claim states the
    /// same fact: the reading is part of what its line says.
    #[test]
    fn a_corrected_claim_keeps_its_reading_beside_a_later_claim_of_the_same_fact() {
        let world = Fake::default().with("door", "state", "open");
        let v = verify(
            Draft {
                claims: vec![
                    typed("door", "state", "jammed", &[1]),
                    claim("Something else.", &[2]),
                    typed("Door", "State", "open", &[2]),
                ],
                ..Default::default()
            },
            &span(),
            &[],
            &world,
        )
        .unwrap();
        let texts: Vec<_> = v.entry.claims.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "door state is jammed",
                "Something else.",
                "Door State is open"
            ]
        );
        assert_eq!(v.entry.claims[0].typed.as_ref().unwrap().value, "open");
        let verdicts: Vec<_> = v.notes.iter().map(|n| n.verdict).collect();
        assert_eq!(verdicts, [Verdict::Corrected, Verdict::Kept, Verdict::Kept]);
    }

    /// The live failure: every claim of an entry carried the same typed fact,
    /// and each newer one wiped out the older claim's sentence.
    #[test]
    fn claims_sharing_a_typed_fact_keep_their_sentences_and_the_newest_keeps_the_fact() {
        let world = Fake::default().with("file:repo/mod.rs", "exists", "true");
        let said = |text: &str, cite: u64| DraftClaim {
            text: text.into(),
            cite: vec![cite],
            typed: Some(Typed {
                subject: "file:repo/mod.rs".into(),
                attribute: "exists".into(),
                value: "true".into(),
            }),
        };
        let v = verify(
            Draft {
                claims: vec![
                    said("The grep found SKIP_QUESTION in style.rs.", 1),
                    said("It asks for journal_skip with a reason.", 2),
                ],
                ..Default::default()
            },
            &span(),
            &[],
            &world,
        )
        .unwrap();
        let texts: Vec<_> = v.entry.claims.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "The grep found SKIP_QUESTION in style.rs.",
                "It asks for journal_skip with a reason."
            ]
        );
        assert!(v.entry.claims[0].typed.is_none(), "the fact moved on");
        assert!(v.entry.claims[1].typed.is_some(), "the newest holds it");
        let verdicts: Vec<_> = v.notes.iter().map(|n| n.verdict).collect();
        assert_eq!(verdicts, [Verdict::Superseded, Verdict::Kept]);
    }

    #[test]
    fn yes_and_no_mean_true_and_false_to_the_check() {
        let world = Fake::default()
            .with("file:a", "exists", "true")
            .with("file:b", "exists", "false");
        let v = verify(
            Draft {
                claims: vec![
                    typed("file:a", "exists", "Yes", &[1]),
                    typed("file:b", "exists", "no", &[2]),
                ],
                ..Default::default()
            },
            &span(),
            &[],
            &world,
        )
        .unwrap();
        assert!(v.entry.claims.iter().all(|c| !c.corrected));
        assert!(v.notes.iter().all(|n| n.verdict == Verdict::Kept));
    }

    fn open_items() -> Vec<Item> {
        vec![
            Item {
                id: 3,
                text: "who holds the shield decision?".into(),
            },
            Item {
                id: 4,
                text: "is the door jammed?".into(),
            },
        ]
    }

    fn with_open(open: Vec<DraftOpen>) -> Result<Verified, Refusal> {
        verify(
            Draft {
                open,
                ..Default::default()
            },
            &span(),
            &open_items(),
            &Fake::default(),
        )
    }

    #[test]
    fn an_open_item_is_new_restated_or_resolved() {
        let v = with_open(vec![
            DraftOpen {
                text: "what is Pax hiding?".into(),
                relates: Relates::New,
            },
            DraftOpen {
                text: "is the east door still jammed?".into(),
                relates: Relates::Restates(4),
            },
            DraftOpen {
                text: "Pax holds it.".into(),
                relates: Relates::Resolves(3),
            },
        ])
        .unwrap();
        assert_eq!(
            v.entry.opened,
            vec![
                Item {
                    id: 0,
                    text: "what is Pax hiding?".into()
                },
                Item {
                    id: 4,
                    text: "is the east door still jammed?".into()
                },
            ]
        );
        assert_eq!(v.entry.resolved, vec![3]);
    }

    #[test]
    fn restating_or_resolving_an_item_that_is_not_open_is_refused() {
        assert_eq!(
            with_open(vec![DraftOpen {
                text: "x".into(),
                relates: Relates::Resolves(9)
            }]),
            Err(Refusal::NotOpen { id: 9 })
        );
        assert_eq!(
            with_open(vec![DraftOpen {
                text: "x".into(),
                relates: Relates::Restates(9)
            }]),
            Err(Refusal::NotOpen { id: 9 })
        );
    }

    #[test]
    fn one_item_cannot_be_restated_and_resolved_together() {
        assert_eq!(
            with_open(vec![
                DraftOpen {
                    text: "still".into(),
                    relates: Relates::Restates(3)
                },
                DraftOpen {
                    text: "done".into(),
                    relates: Relates::Resolves(3)
                },
            ]),
            Err(Refusal::ConflictingItem { id: 3 })
        );
    }

    #[test]
    fn the_entry_carries_the_span_it_was_drafted_from() {
        let v = run(Draft {
            intend: vec!["Go to the foundry.".into()],
            ..Default::default()
        })
        .unwrap();
        assert_eq!((v.entry.from_turn, v.entry.to_turn), (1, 6));
        assert_eq!((v.entry.from_ms, v.entry.to_ms), (10, 60));
        assert_eq!(v.entry.id, 0, "the state assigns the id at keep");
    }

    #[test]
    fn the_same_draft_draws_the_same_verdict() {
        let d = Draft {
            claims: vec![typed("door", "state", "jammed", &[1]), claim("a", &[2])],
            intend: vec!["x".into()],
            ..Default::default()
        };
        let world = Fake::default().with("door", "state", "open");
        let a = verify(d.clone(), &span(), &[], &world);
        let b = verify(d, &span(), &[], &world);
        assert_eq!(a, b);
    }

    #[test]
    fn each_verdict_has_a_name() {
        let names: Vec<&str> = [
            Verdict::Kept,
            Verdict::Corrected,
            Verdict::Demoted,
            Verdict::Superseded,
        ]
        .map(Verdict::name)
        .to_vec();
        assert_eq!(names, ["kept", "corrected", "demoted", "superseded"]);
    }
}
