//! What one draft did, kept so the journal can be inspected from outside.

use std::time::Duration;

use serde::Serialize;

use crate::engine::journal::entry::Entry;
use crate::engine::journal::state::{JournalState, Span};
use crate::engine::journal::verify::{Note, Verdict};
use crate::engine::journal::workflow::{Abandon, Outcome, Timing};

/// How a draft ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Ending {
    Wrote,
    Nothing,
    Abandoned,
    /// The draft never began: no model to ask or no character to ask for.
    NotStarted,
}

/// One claim's check, as the pulse shows it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Check {
    pub claim: usize,
    pub verdict: &'static str,
    pub kind: &'static str,
}

/// One draft over one stretch of turns.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DraftRecord {
    pub from_turn: u64,
    pub to_turn: u64,
    pub turns: usize,
    pub took_ms: u64,
    /// Of `took_ms`, asking whether the stretch was worth an entry.
    pub gate_ms: u64,
    /// Of `took_ms`, asking for the entry.
    pub write_ms: u64,
    pub result: Ending,
    /// Why it ended without an entry, when it did: the character's own reason for
    /// a no, or what went wrong.
    pub detail: Option<String>,
    /// The id of the entry it kept.
    pub entry: Option<u64>,
    pub rounds: usize,
    pub checks: Vec<Check>,
}

fn verdict_name(v: Verdict) -> &'static str {
    match v {
        Verdict::Kept => "kept",
        Verdict::Corrected => "corrected",
        Verdict::Demoted => "demoted",
        Verdict::Superseded => "superseded",
    }
}

fn checks(notes: &[Note]) -> Vec<Check> {
    notes
        .iter()
        .map(|n| Check {
            claim: n.claim,
            verdict: verdict_name(n.verdict),
            kind: n.kind.name(),
        })
        .collect()
}

fn detail(why: &Abandon) -> String {
    match why {
        Abandon::Asking(e) => format!("could not ask the model: {e}"),
        Abandon::Refused(e) => format!("refused until out of rounds: {e}"),
        Abandon::Keeping(e) => format!("keeping the entry failed: {e}"),
    }
}

impl DraftRecord {
    pub fn of(span: &Span, took: Duration, outcome: &Outcome) -> Self {
        let base = Self {
            from_turn: span.from_turn,
            to_turn: span.to_turn,
            turns: span.turns.len(),
            took_ms: took.as_millis() as u64,
            gate_ms: 0,
            write_ms: 0,
            result: Ending::Nothing,
            detail: None,
            entry: None,
            rounds: 0,
            checks: Vec::new(),
        };
        match outcome {
            Outcome::Wrote {
                entry,
                notes,
                rounds,
            } => Self {
                result: Ending::Wrote,
                entry: Some(entry.id),
                rounds: *rounds,
                checks: checks(notes),
                ..base
            },
            Outcome::Nothing { why } => Self {
                detail: Some(why.clone()),
                ..base
            },
            Outcome::Abandoned(why) => Self {
                result: Ending::Abandoned,
                detail: Some(detail(why)),
                ..base
            },
        }
    }

    /// This record with where its time went.
    pub fn timed(self, timing: Timing) -> Self {
        Self {
            gate_ms: timing.gate_ms,
            write_ms: timing.write_ms,
            ..self
        }
    }

    pub fn not_started(span: &Span, took: Duration, why: impl Into<String>) -> Self {
        Self {
            result: Ending::NotStarted,
            detail: Some(why.into()),
            ..Self::of(span, took, &Outcome::Nothing { why: String::new() })
        }
    }
}

/// The entry as the API shows it: its stored form, its span and how it reads in
/// the prompt.
pub fn shown(entry: &Entry) -> serde_json::Value {
    let mut v = serde_json::to_value(entry).unwrap_or_default();
    if let Some(o) = v.as_object_mut() {
        o.insert("span".into(), entry.span().into());
        o.insert("text".into(), entry.render().into());
    }
    v
}

/// Everything the journal API says about one character's journal.
pub fn view(state: &JournalState) -> serde_json::Value {
    let in_prompt: Vec<u64> = state.entries().map(|e| e.id).collect();
    serde_json::json!({
        "written": state.written(),
        "in_prompt": in_prompt,
        "covered_to": state.covered_to(),
        "looked_to": state.looked_to(),
        "drafting": state.pending(),
        "open": state.open(),
        "entries": state.entries().map(shown).collect::<Vec<_>>(),
        "drafts": state.drafts().collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::journal::entry::Kind;
    use crate::engine::journal::state::IN_PROMPT;

    #[test]
    fn the_view_names_what_the_prompt_shows_and_what_drafts_did() {
        let mut s = JournalState::new(0);
        let e = s.kept(entry(0));
        s.record(DraftRecord::of(&span(), Duration::ZERO, &nothing("quiet")));
        let v = view(&s);
        assert_eq!(v["written"], 1);
        assert_eq!(v["in_prompt"], serde_json::json!([e.id]));
        assert_eq!(v["entries"][0]["id"], e.id);
        assert_eq!(v["drafts"][0]["result"], "nothing");
        assert_eq!(v["drafting"], false);
    }

    #[test]
    fn an_empty_journal_shows_no_entries_and_nothing_in_the_prompt() {
        let v = view(&JournalState::new(3));
        assert_eq!(v["written"], 0);
        assert_eq!(v["in_prompt"], serde_json::json!([]));
        assert_eq!(v["entries"], serde_json::json!([]));
    }

    #[test]
    fn the_prompt_holds_only_the_newest_entries() {
        let mut s = JournalState::new(0);
        for _ in 0..(IN_PROMPT + 2) {
            s.kept(entry(0));
        }
        let ids = view(&s)["in_prompt"].as_array().unwrap().len();
        assert_eq!(ids, IN_PROMPT);
    }

    fn nothing(why: &str) -> Outcome {
        Outcome::Nothing { why: why.into() }
    }

    fn span() -> Span {
        Span {
            from_turn: 3,
            to_turn: 9,
            from_ms: 0,
            to_ms: 1,
            turns: Vec::new(),
        }
    }

    fn entry(id: u64) -> Entry {
        Entry {
            id,
            from_turn: 3,
            to_turn: 9,
            from_ms: 0,
            to_ms: 1,
            claims: vec![],
            intend: vec!["find Pax".into()],
            opened: vec![],
            resolved: vec![],
        }
    }

    #[test]
    fn a_kept_entry_records_its_id_rounds_and_checks() {
        let notes = vec![
            Note {
                claim: 1,
                verdict: Verdict::Corrected,
                kind: Kind::Observed,
            },
            Note {
                claim: 2,
                verdict: Verdict::Demoted,
                kind: Kind::Heard,
            },
        ];
        let out = Outcome::Wrote {
            entry: entry(7),
            notes,
            rounds: 2,
        };
        let r = DraftRecord::of(&span(), Duration::from_millis(1500), &out);
        assert_eq!(r.result, Ending::Wrote);
        assert_eq!((r.entry, r.rounds, r.took_ms), (Some(7), 2, 1500));
        assert_eq!((r.from_turn, r.to_turn), (3, 9));
        assert_eq!(
            r.checks,
            vec![
                Check {
                    claim: 1,
                    verdict: "corrected",
                    kind: "observed"
                },
                Check {
                    claim: 2,
                    verdict: "demoted",
                    kind: "heard"
                },
            ]
        );
    }

    #[test]
    fn a_gate_no_records_nothing_kept_and_the_reason() {
        let r = DraftRecord::of(&span(), Duration::ZERO, &nothing("Already in my journal."));
        assert_eq!(r.result, Ending::Nothing);
        assert!(r.entry.is_none());
        assert_eq!(r.detail.as_deref(), Some("Already in my journal."));
    }

    #[test]
    fn a_draft_says_where_its_time_went() {
        let r = DraftRecord::of(&span(), Duration::from_millis(900), &nothing("x")).timed(Timing {
            gate_ms: 300,
            write_ms: 600,
        });
        assert_eq!((r.took_ms, r.gate_ms, r.write_ms), (900, 300, 600));
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(
            (v["gate_ms"].clone(), v["write_ms"].clone()),
            (300.into(), 600.into())
        );
    }

    #[test]
    fn each_abandon_says_why() {
        let said = |a| {
            DraftRecord::of(&span(), Duration::ZERO, &Outcome::Abandoned(a))
                .detail
                .unwrap()
        };
        assert!(said(Abandon::Asking("down".into())).contains("down"));
        assert!(said(Abandon::Refused("empty".into())).contains("empty"));
        assert!(said(Abandon::Keeping("disk".into())).contains("disk"));
    }

    #[test]
    fn a_draft_that_never_began_is_marked_so() {
        let r = DraftRecord::not_started(&span(), Duration::ZERO, "no persona");
        assert_eq!(r.result, Ending::NotStarted);
        assert_eq!(r.detail.as_deref(), Some("no persona"));
    }

    #[test]
    fn the_wire_form_uses_snake_case_results() {
        let r = DraftRecord::not_started(&span(), Duration::ZERO, "x");
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["result"], "not_started");
    }

    #[test]
    fn a_shown_entry_carries_its_span_and_prompt_text() {
        let v = shown(&entry(4));
        assert_eq!(v["id"], 4);
        assert_eq!(v["text"], entry(4).render());
        assert_eq!(v["span"], entry(4).span());
    }
}
