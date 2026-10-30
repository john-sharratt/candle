//! A `POST /v1/journals` body applied by hand onto a journal checkout, as a
//! person, exactly as the route applies it: read and checked whole against
//! the tree, then each operation its own commit (`docs/memory_backbone.md`
//! §6.7). The `zend-journals-import` binary pushes what it applied.
//!
//! Whoever runs it holds push access to the repository, and so may write
//! anything there already (§7.2): the person is written for with an estate
//! admin's standing.

use std::sync::Arc;

use axum::http::StatusCode;
use journals::actor::home_group;
use journals::slug::Slug;
use journals::store::Store;
use journals::time::Stamp;
use serde_json::Value;

use super::write::{batch, request};
use crate::mcp::memory::who::Who;
use crate::mcp::memory::WriteError;
use crate::mcp::tools::mem::Mem;

/// `body` applied onto `store` as `person` at `now`: what each kept
/// operation answers, or the refusal that kept none — or, when the journals
/// moved between the check and the commits, the ones it kept.
pub fn apply(store: &mut Store, person: &Slug, body: &Value, now: Stamp) -> Result<Value, String> {
    let req = request(body)?;
    // The caller as the route sees them, on `store`'s tree. No memory is
    // served here — every write goes to `store` through the closure — so
    // the memory is one that serves none, and refuses any write made to it.
    let tree = store.shared();
    let m = Mem {
        memory: Arc::default(),
        group: home_group(&tree, person),
        tree,
        who: Who::Person(person.clone()),
        channel: req.channel,
        admin: true,
        here: None,
    };
    let (status, answer) = batch(&m, &req.ops, now, req.speaker.as_ref(), |actor, op, at| {
        store.write(actor, op, at).map_err(WriteError::Store)
    });
    if status == StatusCode::OK {
        Ok(answer)
    } else {
        Err(format!("{status}: {answer}"))
    }
}

#[cfg(test)]
mod tests {
    use journals::path::JournalPath;
    use serde_json::json;

    use super::*;
    use crate::mcp::memory::tests::repo;

    fn now() -> Stamp {
        Stamp::parse("2026-10-30T12:00:00Z").unwrap()
    }

    /// **A file is applied as the route applies it, and pushed**: a topic, a
    /// reference to it and a backdated note on it from its speaker reach
    /// origin, read back through a fresh store.
    #[test]
    fn a_file_is_applied_as_its_person_and_reaches_origin() {
        let (_root, dir) = repo();
        let mut store = Store::open(&dir).unwrap();
        let john = Slug::parse("john").unwrap();
        let answer = apply(
            &mut store,
            &john,
            &json!({"channel": "claude-code", "speaker": "john", "ops": [
                {"op": "create", "under": "topics", "name": "candle", "cue": "open for candle", "at": "2026-05-13T12:41:14Z"},
                {"op": "link", "under": "people/john", "name": "engine", "to": "topics/candle", "cue": "open for what John builds"},
                {"op": "note", "journal": "people/john/engine", "text": "The candle fork serves Zen Code.", "at": "2026-05-13T12:41:14Z"},
            ]}),
            now(),
        )
        .unwrap();
        assert_eq!(answer["applied"].as_array().unwrap().len(), 3);
        assert_eq!(store.sync().unwrap().pushed, 3);
        let tree = Store::open(&dir).unwrap().shared();
        let candle = JournalPath::new(
            Slug::parse("zen").unwrap(),
            vec![
                Slug::parse("topics").unwrap(),
                Slug::parse("candle").unwrap(),
            ],
        )
        .unwrap();
        let line = &tree.get(&candle).unwrap().window[0];
        assert_eq!(line.text, "The candle fork serves Zen Code.");
        assert_eq!(line.at, Stamp::parse("2026-05-13T12:41:14Z").unwrap());
        assert_eq!(line.speaker, Some(john));
    }

    /// **A refusal anywhere keeps nothing**, and says which operation.
    #[test]
    fn a_refused_file_keeps_nothing() {
        let (_root, dir) = repo();
        let mut store = Store::open(&dir).unwrap();
        let err = apply(
            &mut store,
            &Slug::parse("john").unwrap(),
            &json!({"ops": [
                {"op": "note", "journal": "people/john", "text": "kept only with the rest"},
                {"op": "note", "journal": "@nowhere/topics", "text": "x"},
            ]}),
            now(),
        )
        .unwrap_err();
        assert!(err.starts_with("404 Not Found: "), "{err}");
        assert!(err.contains("\"index\":1"), "{err}");
        assert!(!store.unpushed());
    }
}
