//! Drafts a task's journal over its turn's stretch, and explains why a turn that
//! wrote no entry kept none.
//!
//! The draft reads the committed history, so the task waits for the turn to leave
//! the live set before drafting. A draft that is already running may not cover the
//! turn's last stretch, so the task waits for it and asks again. A turn with no new
//! entry gets a reason worded for the caller, taken from its closing draft's record,
//! and a draft that did not run this turn is not blamed on an earlier turn's record.

use super::*;

/// Draft the journal over the turn's stretch now, and wait for the draft to
/// end.
pub(super) async fn finish_journal(session: &ZendSession, conv_id: &str) -> Drafted {
    let deadline = Instant::now() + DRAFT_DEADLINE;
    // The reply's stream ends before its turn is committed; the draft reads
    // the committed history, so it waits for the turn to leave the live set.
    while session.live_turn(conv_id).is_some() {
        if Instant::now() >= deadline {
            return Drafted::TimedOut;
        }
        tokio::time::sleep(DRAFT_POLL).await;
    }
    loop {
        match session.journal_draft(conv_id) {
            Some(Ok(_)) => break,
            // A draft already running — the cadence's — may not cover the
            // turn's last stretch: wait it out and ask again.
            Some(Err(NoDraft::Running)) => {
                if Instant::now() >= deadline {
                    return Drafted::TimedOut;
                }
                tokio::time::sleep(DRAFT_POLL).await;
            }
            Some(Err(why)) => return Drafted::NotRun(why),
            None => return Drafted::NoJournal,
        }
    }
    while drafting(session.journal(conv_id).as_ref()) {
        if Instant::now() >= deadline {
            return Drafted::TimedOut;
        }
        tokio::time::sleep(DRAFT_POLL).await;
    }
    Drafted::Ended
}

/// Why a turn whose journal `view` shows no new entry kept none — worded for
/// the caller, who reads it beside the answer: how its closing draft went,
/// and when one ran, its record's reason.
pub fn why_no_entry(view: Option<&Value>, drafted: Drafted) -> String {
    match drafted {
        Drafted::Ended => {}
        Drafted::TimedOut => {
            return format!(
                "its journal draft did not finish within {} minutes",
                DRAFT_DEADLINE.as_secs() / 60
            )
        }
        Drafted::NotRun(NoDraft::NothingNew) => {
            return "nothing in its turn was new to its journal".into()
        }
        Drafted::NotRun(NoDraft::NotOpen) => {
            return "its conversation was not open where it ran, so no draft could run".into()
        }
        Drafted::NotRun(NoDraft::NotDue | NoDraft::Running) => {
            return "no journal draft ran".into()
        }
        Drafted::NoJournal => return "its conversation keeps no journal".into(),
    }
    let Some(last) = view
        .and_then(|v| v.get("drafts"))
        .and_then(Value::as_array)
        .and_then(|d| d.last())
    else {
        return "no journal draft ran".into();
    };
    let detail = last
        .get("detail")
        .and_then(Value::as_str)
        .unwrap_or("no reason was recorded");
    match last.get("result").and_then(Value::as_str) {
        Some("nothing") => format!("the journal judged it added nothing: {detail}"),
        Some("abandoned") => format!("its journal draft failed: {detail}"),
        Some("not_started") => format!("no journal draft ran: {detail}"),
        Some("wrote") => "an earlier entry already covers its turn".into(),
        _ => "no journal draft ran".into(),
    }
}
