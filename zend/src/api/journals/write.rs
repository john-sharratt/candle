//! `POST /v1/journals` (`docs/memory_backbone.md` §6.7): `{channel?, ops:
//! [...]}`, at most [`MAX_OPS`] operations of [`super::ops`], applied in
//! order as the caller, each its own commit.
//!
//! The whole batch is read and applied, in order, to a copy of the tree before
//! anything is committed — each operation read as the ones before it leave
//! the tree — so an operation malformed or refused by the rules — a line
//! too long, an eleventh sub-journal, a line not the caller's to edit — is
//! answered with nothing kept. Only the journals changing between that check
//! and the commits — another writer, a fetch — can stop a batch partway, and
//! then the answer names the operations kept.
//!
//! `channel` names who the person speaks through, as an MCP session's channel
//! names itself: stated by the caller, not checked; `other` when absent. The
//! person is the gateway's.

use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::Json;
use journals::actor::Actor;
use journals::ops::{applied_to, Applied, Op};
use journals::path::JournalPath;
use journals::scope::Channel;
use journals::time::Stamp;
use serde_json::{json, Value};

use super::ops::{plan, Planned};
use super::{blocking, malformed, person, status_of, status_of_journal, Answer};
use crate::mcp::memory::WriteError;
use crate::mcp::tools::mem::Mem;
use crate::session::ZendSession;

/// The most operations one request applies.
pub const MAX_OPS: usize = 500;

/// The most bytes a request's body holds: [`MAX_OPS`] lines at their cap,
/// with room for their JSON.
pub const BODY_LIMIT: usize = 8 << 20;

/// A write request: who it speaks through, and its operations.
#[derive(Debug)]
pub struct Request {
    pub channel: Channel,
    pub ops: Vec<Value>,
}

pub fn request(body: &Value) -> Result<Request, String> {
    let obj = body.as_object().ok_or("the body is a JSON object")?;
    if let Some(k) = obj
        .keys()
        .find(|k| !matches!(k.as_str(), "channel" | "ops"))
    {
        return Err(format!(
            "`{k}` is not a field of a request: send `channel` and `ops`"
        ));
    }
    let channel = match obj.get("channel") {
        None => Channel::Other,
        Some(v) => {
            let name = v.as_str().ok_or("`channel` is a channel's name")?;
            Channel::parse(name).ok_or_else(|| format!("`{name}` is not a channel"))?
        }
    };
    let ops = obj
        .get("ops")
        .and_then(Value::as_array)
        .ok_or("`ops` is a list of operations")?;
    if ops.is_empty() || ops.len() > MAX_OPS {
        return Err(format!(
            "`ops` holds {} operations: between 1 and {MAX_OPS}",
            ops.len()
        ));
    }
    Ok(Request {
        channel,
        ops: ops.clone(),
    })
}

/// The journal an operation's line is in, or the journal it made or changed.
fn journal_of<'a>(op: &'a Op, applied: &'a Applied) -> Option<&'a JournalPath> {
    applied.created.as_ref().or(match op {
        Op::Note { path, .. }
        | Op::Episode { path, .. }
        | Op::Forget { path, .. }
        | Op::Fold { path }
        | Op::Unfold { path } => Some(path),
        Op::Edit { address, .. } | Op::Accept { address } => Some(&address.path),
        Op::Merge { into, .. } => Some(into),
        _ => applied.touched.first(),
    })
}

/// What one kept operation answers.
fn kept(m: &Mem, index: usize, op: &Op, applied: &Applied) -> Value {
    let journal = journal_of(op, applied);
    json!({
        "index": index,
        "subject": applied.subject,
        "journal": journal.map(JournalPath::show),
        "line": match (journal, &applied.line) {
            (Some(j), Some(id)) => Some(format!("{j}#{}", id.as_str())),
            _ => None,
        },
        "readers": journal.map(|j| m.readers(j)),
        "folded": applied.folded.iter().map(JournalPath::show).collect::<Vec<_>>(),
    })
}

fn stopped(
    status: StatusCode,
    error: &str,
    index: usize,
    message: String,
    applied: Vec<Value>,
) -> Answer {
    (
        status,
        json!({ "error": error, "index": index, "message": message, "applied": applied }),
    )
}

/// `ops` for `m`'s caller at `now`: read, checked whole against the tree,
/// then each handed to `commit`.
pub fn batch(
    m: &Mem,
    ops: &[Value],
    now: Stamp,
    mut commit: impl FnMut(&Actor, &Op, Stamp) -> Result<Applied, WriteError>,
) -> Answer {
    let actor = m.actor();
    let mut tree = Arc::clone(&m.tree);
    let mut planned: Vec<Planned> = Vec::with_capacity(ops.len());
    for (i, v) in ops.iter().enumerate() {
        // Read against the tree as the operations before it leave it: a
        // group an earlier one made the default is where a bare path goes.
        let p = match plan(&m.seeing(Arc::clone(&tree)), v, now) {
            Ok(p) => p,
            Err(e) => return stopped(StatusCode::BAD_REQUEST, "invalid", i, e, Vec::new()),
        };
        match applied_to(&tree, &actor, &p.op, p.at, None) {
            Ok((next, _)) => tree = Arc::new(next),
            Err(e) => {
                let (status, code) = status_of_journal(&e);
                return stopped(status, code, i, e.to_string(), Vec::new());
            }
        }
        planned.push(p);
    }
    // Who reads a journal is told as the batch leaves the groups.
    let after = m.seeing(tree);
    let mut applied = Vec::with_capacity(planned.len());
    for (i, p) in planned.iter().enumerate() {
        match commit(&actor, &p.op, p.at) {
            Ok(a) => applied.push(kept(&after, i, &p.op, &a)),
            Err(e) => {
                let (status, code) = status_of(&e);
                return stopped(status, code, i, e.to_string(), applied);
            }
        }
    }
    (StatusCode::OK, json!({ "applied": applied }))
}

pub async fn write(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let Json(body) = match body {
        Ok(b) => b,
        Err(e) => return malformed(e.body_text()),
    };
    let req = match request(&body) {
        Ok(r) => r,
        Err(e) => return malformed(e),
    };
    blocking(move || match person(&session, &headers, req.channel) {
        Ok(m) => {
            let memory = Arc::clone(&m.memory);
            batch(&m, &req.ops, Stamp::now(), |actor, op, at| {
                memory.write_at(actor, op, at)
            })
        }
        Err(a) => a,
    })
    .await
}

#[cfg(test)]
mod tests {
    use journals::error::Error;
    use journals::line::{LineId, MAX_TEXT};
    use journals::ops::{apply, OnTaken, Which};
    use journals::slug::Slug;
    use journals::store::StoreError;
    use journals::tree::Tree;

    use super::*;
    use crate::mcp::tools::mem::tests::mem;

    fn now() -> Stamp {
        Stamp::parse("2026-10-30T12:00:00Z").unwrap()
    }

    /// `ops` as John, committed by applying each to a tree of the test's own,
    /// as the store would; and that tree after.
    fn run(ops: Value) -> (Answer, Tree) {
        run_as("john", ops)
    }

    fn run_as(person: &str, ops: Value) -> (Answer, Tree) {
        let m = mem(person);
        let mut tree = (*m.tree).clone();
        let answer = batch(&m, ops.as_array().unwrap(), now(), |actor, op, at| {
            apply(&mut tree, actor, op, at).map_err(|e| WriteError::Store(StoreError::Journal(e)))
        });
        (answer, tree)
    }

    #[test]
    fn a_request_names_its_channel_and_holds_its_operations() {
        let r = request(&json!({"channel": "claude-code", "ops": [{"op": "note"}]})).unwrap();
        assert_eq!(r.channel, Channel::ClaudeCode);
        assert_eq!(r.ops.len(), 1);
        assert_eq!(
            request(&json!({"ops": [{}]})).unwrap().channel,
            Channel::Other
        );
        let err = |v: Value| request(&v).unwrap_err();
        assert_eq!(err(json!([])), "the body is a JSON object");
        assert_eq!(
            err(json!({"channel": "tesla", "ops": [{}]})),
            "`tesla` is not a channel"
        );
        assert_eq!(
            err(json!({"channel": 1, "ops": [{}]})),
            "`channel` is a channel's name"
        );
        assert_eq!(err(json!({"ops": {}})), "`ops` is a list of operations");
        assert_eq!(
            err(json!({"ops": []})),
            "`ops` holds 0 operations: between 1 and 500"
        );
        let many: Vec<Value> = (0..=MAX_OPS).map(|_| json!({})).collect();
        assert_eq!(
            err(json!({"ops": many})),
            "`ops` holds 501 operations: between 1 and 500"
        );
        assert_eq!(
            err(json!({"op": "note", "ops": [{}]})),
            "`op` is not a field of a request: send `channel` and `ops`"
        );
    }

    #[test]
    fn a_batch_is_kept_in_order_each_naming_its_line_and_readers() {
        let (answer, tree) = run(json!([
            {"op": "create", "under": "topics", "name": "trip", "cue": "open for the trip"},
            {"op": "note", "journal": "topics/trip", "text": "Flights booked.", "at": "2026-10-01T09:00:00Z"},
            {"op": "note", "journal": "~", "text": "Prefers tea."},
        ]));
        assert_eq!(answer.0, StatusCode::OK);
        let applied = answer.1["applied"].as_array().unwrap();
        assert_eq!(applied.len(), 3);
        let trip = mem("john").path("topics/trip").unwrap();
        let line = &tree.get(&trip).unwrap().window[0];
        assert_eq!(line.at, Stamp::parse("2026-10-01T09:00:00Z").unwrap());
        assert_eq!(
            applied[0],
            json!({
                "index": 0,
                "subject": "Create @sharratt/topics/trip",
                "journal": "@sharratt/topics/trip",
                "line": null,
                "readers": "shared with Yen and Thanh Long",
                "folded": [],
            })
        );
        assert_eq!(
            applied[1]["line"],
            format!("@sharratt/topics/trip#{}", line.id.as_str())
        );
        assert_eq!(applied[2]["journal"], "~");
        assert_eq!(applied[2]["readers"], "only you read it");
    }

    /// **A batch the rules refuse anywhere keeps nothing**: the seventh of
    /// ten notes too long commits none of them.
    #[test]
    fn a_refused_operation_anywhere_keeps_nothing() {
        let mut ops: Vec<Value> = (0..10)
            .map(|i| json!({"op": "note", "journal": "~", "text": format!("n{i}")}))
            .collect();
        ops[6] = json!({"op": "note", "journal": "~", "text": "x".repeat(MAX_TEXT + 1)});
        let (answer, tree) = run(Value::Array(ops));
        assert_eq!(answer.0, StatusCode::BAD_REQUEST);
        assert_eq!(answer.1["error"], "invalid");
        assert_eq!(answer.1["index"], 6);
        assert_eq!(answer.1["applied"], json!([]));
        assert!(tree
            .get(&mem("john").path("~").unwrap())
            .unwrap()
            .window
            .is_empty());
        // A note on a journal an earlier operation of the batch creates is
        // checked against the tree as the batch leaves it.
        let (answer, _) = run(json!([
            {"op": "note", "journal": "topics/trip", "text": "too early"},
            {"op": "create", "under": "topics", "name": "trip", "cue": "c"},
        ]));
        assert_eq!(
            answer,
            (
                StatusCode::NOT_FOUND,
                json!({
                    "error": "not_found",
                    "index": 0,
                    "message": "there is no journal @sharratt/topics/trip",
                    "applied": [],
                })
            )
        );
    }

    /// **Only an estate admin lists an email, over HTTP too**: Yen owns
    /// @sharratt, and her batch that gives a newcomer an account keeps none
    /// of its operations; John's, the same batch, keeps them all.
    #[test]
    fn a_non_admin_owner_lists_no_email_in_a_batch() {
        let ops = json!([
            {"op": "note", "journal": "~", "text": "Adding Yen Nhi."},
            {"op": "add_member", "person": "yen-nhi", "name": "Yen Nhi", "emails": ["yen.nhi@example.com"]},
        ]);
        let (answer, tree) = run_as("yen", ops.clone());
        assert_eq!(
            answer,
            (
                StatusCode::FORBIDDEN,
                json!({
                    "error": "forbidden",
                    "index": 1,
                    "message": "a person's AI may not list yen-nhi's accounts or rename them: only an estate admin does",
                    "applied": [],
                })
            )
        );
        assert!(!tree.persons.contains_key(&Slug::parse("yen-nhi").unwrap()));
        let (answer, tree) = run(ops);
        assert_eq!(answer.0, StatusCode::OK);
        assert!(tree.persons.contains_key(&Slug::parse("yen-nhi").unwrap()));
    }

    /// **A group named the default earlier in a batch is where its later
    /// bare paths go.**
    #[test]
    fn a_default_chosen_in_a_batch_holds_for_the_rest_of_it() {
        let (answer, tree) = run(json!([
            {"op": "create_group", "group": "tokera", "name": "Tokera", "kind": "team"},
            {"op": "choose_groups", "default": "tokera", "framed": []},
            {"op": "create", "under": "projects", "name": "candle", "cue": "open for candle"},
            {"op": "add_bot", "bot": "b1"},
        ]));
        assert_eq!(answer.0, StatusCode::OK, "{}", answer.1);
        let applied = answer.1["applied"].as_array().unwrap();
        assert_eq!(applied[2]["journal"], "@tokera/projects/candle");
        // Told as the batch leaves the group: its bot, added after.
        assert_eq!(applied[2]["readers"], "read by its bot only");
        let tokera = tree.group_file(&Slug::parse("tokera").unwrap()).unwrap();
        assert!(tokera.bot(&Slug::parse("b1").unwrap()).is_some());
    }

    #[test]
    fn a_malformed_operation_is_named_and_nothing_kept() {
        let (answer, _) = run(json!([
            {"op": "note", "journal": "~", "text": "fine"},
            {"op": "note", "journal": "~"},
        ]));
        assert_eq!(
            answer,
            (
                StatusCode::BAD_REQUEST,
                json!({
                    "error": "invalid",
                    "index": 1,
                    "message": "`text` is missing: it is a non-empty string",
                    "applied": [],
                })
            )
        );
    }

    /// **A commit refused after the check** — the journals moved — stops the
    /// rest, and the answer names the ones kept.
    #[test]
    fn a_commit_refused_after_the_check_names_what_was_kept() {
        let m = mem("john");
        let ops = [
            json!({"op": "note", "journal": "~", "text": "a"}),
            json!({"op": "note", "journal": "~", "text": "b"}),
        ];
        let mut n = 0;
        let answer = batch(&m, &ops, now(), |_, _, _| {
            n += 1;
            if n == 1 {
                Ok(Applied {
                    subject: "Note on ~".into(),
                    touched: vec![m.path("~").unwrap()],
                    line: Some(LineId::parse("2026-10-30-1200-0001").unwrap()),
                    ..Applied::default()
                })
            } else {
                Err(WriteError::Store(StoreError::Journal(Error::conflict(
                    "moved",
                ))))
            }
        });
        assert_eq!(answer.0, StatusCode::CONFLICT);
        assert_eq!(answer.1["index"], 1);
        assert_eq!(answer.1["message"], "moved");
        assert_eq!(answer.1["applied"][0]["line"], "~#2026-10-30-1200-0001");
    }

    #[test]
    fn a_kept_operation_names_the_journal_it_made_or_changed() {
        let m = mem("john");
        let topics = m.path("topics").unwrap();
        let trip = topics.child(Slug::parse("trip").unwrap()).unwrap();
        let create = Op::Create {
            under: topics.clone(),
            name: Slug::parse("trip").unwrap(),
            cue: "c".into(),
            on_taken: OnTaken::Refuse,
        };
        let made = Applied {
            created: Some(trip.clone()),
            ..Applied::default()
        };
        assert_eq!(journal_of(&create, &made), Some(&trip));
        let merge = Op::Merge {
            from: trip.clone(),
            into: topics.clone(),
        };
        assert_eq!(journal_of(&merge, &Applied::default()), Some(&topics));
        let share = Op::Share {
            from: m.path("~/people").unwrap(),
            into: topics.clone(),
            name: None,
        };
        assert_eq!(journal_of(&share, &made), Some(&trip));
        let forget = Op::Forget {
            path: topics.clone(),
            which: Which::Says("x".into()),
        };
        assert_eq!(journal_of(&forget, &Applied::default()), Some(&topics));
    }
}
