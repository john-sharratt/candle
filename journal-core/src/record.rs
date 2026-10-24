//! What one draft did, kept so the journal can be inspected from outside.

use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};

use crate::entry::Entry;
use crate::journal::Journal;
use crate::span::Span;
use crate::style::Style;
use crate::verify::Note;
use crate::workflow::{Abandon, Outcome, Timing};

/// How a draft ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Ending {
    Wrote,
    Nothing,
    Abandoned,
    /// The draft never began: nothing to ask, or nobody to ask it of.
    NotStarted,
}

/// One claim's check, as an inspector shows it.
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
    /// Of `took_ms`, asking for the entry.
    pub write_ms: u64,
    pub result: Ending,
    /// Why it ended without an entry, when it did: the writer's own reason for a
    /// skip, or what went wrong.
    pub detail: Option<String>,
    /// The id of the entry it kept.
    pub entry: Option<u64>,
    pub rounds: usize,
    pub checks: Vec<Check>,
}

fn checks(notes: &[Note]) -> Vec<Check> {
    notes
        .iter()
        .map(|n| Check {
            claim: n.claim,
            verdict: n.verdict.name(),
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
            took_ms: u64::try_from(took.as_millis()).unwrap_or(u64::MAX),
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

/// The entry as an inspector shows it: its stored form, its span, its line as
/// it reads in the prompt below `before`, and everything behind that line.
pub fn shown(entry: &Entry, before: Option<&Entry>, style: &Style) -> Value {
    let mut v = serde_json::to_value(entry).unwrap_or_default();
    if let Some(o) = v.as_object_mut() {
        o.insert("span".into(), entry.span(style).into());
        o.insert("text".into(), entry.render(before, style).into());
        o.insert("detail".into(), entry.detail(style).into());
    }
    v
}

/// Everything an inspector is told about one journal.
pub fn view(journal: &Journal, style: &Style) -> Value {
    let in_prompt: Vec<u64> = journal.entries().map(|e| e.id).collect();
    let held: Vec<&Entry> = journal.entries().collect();
    let entries: Vec<Value> = held
        .iter()
        .enumerate()
        .map(|(i, e)| shown(e, i.checked_sub(1).map(|b| held[b]), style))
        .collect();
    json!({
        "written": journal.written(),
        "in_prompt": in_prompt,
        "covered_to": journal.covered_to(),
        "looked_to": journal.looked_to(),
        "drafting": journal.pending(),
        "open": journal.open(),
        "entries": entries,
        "drafts": journal.drafts().collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::testing::entry as entry_at;
    use crate::entry::Kind;
    use crate::style::testing::PLAIN;
    use crate::verify::Verdict;

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
            intend: vec!["find Pax".into()],
            ..entry_at(id)
        }
    }

    #[test]
    fn the_view_names_what_the_prompt_shows_and_what_drafts_did() {
        let mut j = Journal::new();
        let e = j.kept(entry(0));
        j.record(DraftRecord::of(&span(), Duration::ZERO, &nothing("quiet")));
        let v = view(&j, &PLAIN);
        assert_eq!(v["written"], 1);
        assert_eq!(v["in_prompt"], json!([e.id]));
        assert_eq!(v["entries"][0]["id"], e.id);
        assert_eq!(v["entries"][0]["text"], e.render(None, &PLAIN));
        assert_eq!(v["drafts"][0]["result"], "nothing");
        assert_eq!(v["drafting"], false);
        assert_eq!(v["covered_to"], 9);
    }

    #[test]
    fn an_empty_journal_shows_no_entries_and_nothing_in_the_prompt() {
        let v = view(&Journal::new(), &PLAIN);
        assert_eq!(v["written"], 0);
        assert_eq!(v["in_prompt"], json!([]));
        assert_eq!(v["entries"], json!([]));
        assert_eq!(v["open"], json!([]));
    }

    #[test]
    fn the_prompt_holds_every_entry_the_journal_holds() {
        let mut j = Journal::new();
        for _ in 0..7 {
            j.kept(entry(0));
        }
        let ids = view(&j, &PLAIN)["in_prompt"].as_array().unwrap().len();
        assert_eq!(ids, 7);
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
    fn a_no_records_nothing_kept_and_the_reason() {
        let r = DraftRecord::of(&span(), Duration::ZERO, &nothing("Already in my journal."));
        assert_eq!(r.result, Ending::Nothing);
        assert!(r.entry.is_none());
        assert_eq!(r.detail.as_deref(), Some("Already in my journal."));
    }

    #[test]
    fn a_draft_says_where_its_time_went() {
        let r = DraftRecord::of(&span(), Duration::from_millis(900), &nothing("x"))
            .timed(Timing { write_ms: 600 });
        assert_eq!((r.took_ms, r.write_ms), (900, 600));
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["write_ms"], json!(600));
        assert!(v.get("gate_ms").is_none(), "{v}");
    }

    #[test]
    fn each_abandon_says_why() {
        let said = |a| {
            DraftRecord::of(&span(), Duration::ZERO, &Outcome::Abandoned(a))
                .detail
                .unwrap()
        };
        assert_eq!(
            said(Abandon::Asking("down".into())),
            "could not ask the model: down"
        );
        assert_eq!(
            said(Abandon::Refused("empty".into())),
            "refused until out of rounds: empty"
        );
        assert_eq!(
            said(Abandon::Keeping("disk".into())),
            "keeping the entry failed: disk"
        );
    }

    #[test]
    fn a_draft_that_never_began_is_marked_so() {
        let r = DraftRecord::not_started(&span(), Duration::ZERO, "no persona");
        assert_eq!(r.result, Ending::NotStarted);
        assert_eq!(r.detail.as_deref(), Some("no persona"));
        assert_eq!(serde_json::to_value(&r).unwrap()["result"], "not_started");
    }

    #[test]
    fn a_shown_entry_carries_its_span_and_prompt_text() {
        let v = shown(&entry(4), None, &PLAIN);
        assert_eq!(v["id"], 4);
        assert_eq!(v["text"], entry(4).render(None, &PLAIN));
        assert_eq!(v["detail"], entry(4).detail(&PLAIN));
        assert_eq!(v["span"], entry(4).span(&PLAIN));
    }
}
