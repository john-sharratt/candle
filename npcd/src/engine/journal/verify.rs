//! The server's half of a journal entry: what the model wrote, checked.
//!
//! A [`Draft`] is only what the model supplies — claim text, the turns each claim
//! rests on, and how each open item relates to what is already open. [`verify`]
//! turns it into an [`Entry`] or refuses it, and decides everything the model is
//! not trusted to say: how a claim came to be known ([`Kind`], read off the cited
//! turns' [`Origin`], never off wording), whether it can go stale, and — for a
//! claim the world can be asked about — whether the world agrees.
//!
//! The same draft against the same span, open items and world always draws the
//! same verdict. Nothing here samples.

use std::collections::BTreeSet;

use crate::engine::body;
use crate::engine::journal::entry::{Claim, Entry, Item, Kind, Typed};
use crate::engine::journal::state::Span;
use crate::engine::window::Origin;

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
                "Claim {} cites turn {turn}, which is not something a claim can rest on \
                 (scenery, standing instructions and your own reflections are not evidence).",
                claim + 1
            ),
            Refusal::NotOpen { id } => format!("Item #{id} is not open."),
            Refusal::ConflictingItem { id } => {
                format!("Item #{id} is both restated and resolved in one entry.")
            }
        }
    }
}

/// What the check decided about one claim, for the pulse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Kept,
    /// The world answered differently; the claim now says what the world said.
    Corrected,
    /// The world had no answer; the claim stands as prose.
    Demoted,
    /// A later claim about the same subject and attribute replaced it.
    Superseded,
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

/// How one turn supports a claim that cites it.
fn evidence(origin: &Origin) -> Kind {
    match origin {
        Origin::Event { tag, .. } => match *tag {
            "speech" | "message" | "announcement" | "operator" => Kind::Heard,
            _ => Kind::Observed,
        },
        // A tool whose line is something the world told the character, rather
        // than the character's own act being recorded.
        Origin::Act { verb } if body::answers(verb) => Kind::Observed,
        Origin::Act { .. } => Kind::Did,
    }
}

/// The strongest support among the cited turns; none is a conclusion.
fn derive_kind(span: &Span, cite: &[u64]) -> Kind {
    let rank = |k: Kind| match k {
        Kind::Observed => 3,
        Kind::Heard => 2,
        Kind::Did => 1,
        Kind::Inferred => 0,
    };
    cite.iter()
        .filter_map(|id| span.turn(*id))
        .map(|t| evidence(&t.origin))
        .max_by_key(|k| rank(*k))
        .unwrap_or(Kind::Inferred)
}

fn same(a: &str, b: &str) -> bool {
    a.trim().eq_ignore_ascii_case(b.trim())
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
        let mut kind = derive_kind(span, &c.cite);
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
                Some(actual) => {
                    claim.text = format!("{} {}: {actual}", t.subject, t.attribute);
                    claim.typed = Some(Typed { value: actual, ..t });
                    claim.corrected = true;
                    claim.perishable = true;
                    kind = Kind::Observed;
                    claim.kind = kind;
                    verdict = Verdict::Corrected;
                }
                None => {
                    claim.typed = None;
                    verdict = Verdict::Demoted;
                }
            }
        }
        // A newer claim about the same thing replaces the older one.
        if let Some(t) = &claim.typed {
            let key = (t.subject.to_lowercase(), t.attribute.to_lowercase());
            let at = claims.iter().position(|(_, o)| {
                o.typed.as_ref().is_some_and(|ot| {
                    (ot.subject.to_lowercase(), ot.attribute.to_lowercase()) == key
                })
            });
            if let Some(i) = at {
                let (older, _) = claims.remove(i);
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
    Ok(Verified {
        entry: Entry {
            id: 0,
            from_turn: span.from_turn,
            to_turn: span.to_turn,
            from_ms: span.from_ms,
            to_ms: span.to_ms,
            claims,
            intend,
            opened,
            resolved,
        },
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::event::{Event, EventKind, Salience};
    use crate::engine::journal::state::JournalState;
    use crate::engine::window::Window;
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Fake(BTreeMap<(String, String), String>);
    impl Fake {
        fn with(mut self, s: &str, a: &str, v: &str) -> Self {
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

    fn at(w: &mut Window, kind: EventKind, ms: u64) -> u64 {
        w.push_event(&Event::new(0, ms, Salience::NORMAL, kind))
    }

    /// Turns: 1 speech (Heard), 2 situation (Observed), 3 own `tell`
    /// (Did), 4 `read` answer (Observed), 5 ambient scenery (uncitable),
    /// 6 own `reflect` (uncitable).
    fn span() -> Span {
        let mut w = Window::with_default_cap();
        at(
            &mut w,
            EventKind::Speech {
                speaker: "Pax".into(),
                text: "Fabricator 5 is lost.".into(),
                to: Default::default(),
            },
            10,
        );
        at(
            &mut w,
            EventKind::Situation {
                text: "You are in the foundry.".into(),
            },
            20,
        );
        w.push_npc("tell — to Pax: on my way", 30);
        w.push_npc("read — the board says: shields are down", 40);
        w.push_event(&Event::new(
            0,
            50,
            Salience::IDLE,
            EventKind::Description {
                text: "The conduit hums.".into(),
            },
        ));
        w.push_npc("reflect — thinking", 60);
        JournalState::new(0).begin(&w).unwrap()
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

    fn run(d: Draft) -> Result<Verified, Refusal> {
        verify(d, &span(), &[], &Fake::default())
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
    fn what_a_character_did_does_not_go_stale_and_everything_else_can() {
        assert!(!one(claim("I told Pax.", &[3])).perishable);
        assert!(one(claim("The room is quiet.", &[2])).perishable);
        assert!(one(claim("Pax is worried.", &[1])).perishable);
        assert!(one(claim("A guess.", &[])).perishable);
    }

    #[test]
    fn a_claim_may_not_rest_on_scenery_or_its_own_reflection_or_a_turn_outside_the_span() {
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
    fn a_typed_claim_the_world_contradicts_is_replaced_by_what_the_world_says() {
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
        assert_eq!(c.text, "fabricator 5 mode: running");
        assert_eq!(c.typed.as_ref().unwrap().value, "running");
        assert!(c.corrected);
        assert_eq!(
            c.kind,
            Kind::Observed,
            "the world's reading is an observation"
        );
        assert_eq!(v.notes[0].verdict, Verdict::Corrected);
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

    #[test]
    fn two_typed_claims_about_one_thing_keep_only_the_newer() {
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
        assert_eq!(texts, vec!["Something else.", "Door State is open"]);
        assert!(v
            .notes
            .iter()
            .any(|n| n.claim == 0 && n.verdict == Verdict::Superseded));
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
        let s = span();
        let v = run(Draft {
            intend: vec!["Go to the foundry.".into()],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            (v.entry.from_turn, v.entry.to_turn),
            (s.from_turn, s.to_turn)
        );
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
}
