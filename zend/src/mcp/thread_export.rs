//! A thread's results kept in the memory (`docs/zen_memory_and_guides.md` D9,
//! `docs/memory_backbone.md` §7.3): a thread attached with `about` topics is
//! bound to its owner's group, and its notes and its tasks' ends are written
//! to the memory journal `threads/<thread>` there, which each topic it is
//! about references. The thread's own journal stays the work record; this is
//! what the group's memory — every conversation reading those topics — sees
//! of it.
//!
//! Every write runs off the request path: the call that causes it answers
//! first.

use std::sync::Arc;
use std::thread;

use journals::actor::Actor;
use journals::error::Error;
use journals::ops::wire::{path_from_wire, path_to_wire};
use journals::ops::{OnTaken, Op};
use journals::path::JournalPath;
use journals::scope::Section;
use journals::slug::Slug;
use journals::store::StoreError;
use journals::terms::Terms;
use serde::{Deserialize, Serialize};

use super::memory::{Memory, WriteError};
use super::records::Records;

/// What a thread is bound to in the memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bound {
    /// The owner's group its journal is written in.
    pub group: String,
    /// The topics it is about, as folders (`groups/zen/topics/candle`).
    pub about: Vec<String>,
}

fn key(owner: &str, thread: &str) -> String {
    format!("mcp/thread-memory/{owner}/{thread}")
}

/// What `owner`'s thread is bound to, when it is.
pub fn bound(records: &dyn Records, owner: &str, thread: &str) -> Option<Bound> {
    serde_json::from_value(records.get(&key(owner, thread))?).ok()
}

/// Bind `owner`'s thread to `group`, about `about`.
pub fn bind(
    records: &dyn Records,
    owner: &str,
    thread: &str,
    group: &Slug,
    about: &[JournalPath],
) -> anyhow::Result<()> {
    let b = Bound {
        group: group.to_string(),
        about: about.iter().map(path_to_wire).collect(),
    };
    records.put(&key(owner, thread), serde_json::to_value(b)?)
}

/// The memory journal a thread's results are kept on, in `group`.
pub fn thread_path(group: &Slug, thread: &str) -> Result<JournalPath, String> {
    let name = Slug::parse(thread)?;
    JournalPath::of_section(group, Section::Threads)
        .child(name)
        .map_err(|e| e.to_string())
}

/// Whether a write failed only because what it makes is already there.
fn already(e: &WriteError) -> bool {
    matches!(e, WriteError::Store(StoreError::Journal(Error::Conflict(_))))
}

/// Open the thread's memory journal, when it is not yet, as zend.
fn ensure(memory: &Memory, path: &JournalPath, thread: &str) -> Result<(), String> {
    if memory.tree()?.get(path).is_some() {
        return Ok(());
    }
    let op = Op::Create {
        under: JournalPath::of_section(path.group(), Section::Threads),
        name: Slug::parse(thread)?,
        cue: format!("open for the work on {thread}: its decisions and how its tasks ended"),
        on_taken: OnTaken::Refuse,
    };
    match memory.write(&Actor::Zend, &op) {
        Ok(_) => Ok(()),
        Err(e) if already(&e) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

/// Open the thread's memory journal and reference it from each topic it is
/// about — the references a person's call makes, as `person`.
pub fn attach(memory: &Memory, person: &Actor, b: &Bound, thread: &str) -> Result<(), String> {
    let group = Slug::parse(&b.group)?;
    let path = thread_path(&group, thread)?;
    ensure(memory, &path, thread)?;
    for folder in &b.about {
        let under = path_from_wire(folder).map_err(|e| e.to_string())?;
        let op = Op::Link {
            under,
            name: Slug::parse(thread)?,
            to: path.clone(),
            cue: format!("open for the work on {thread}"),
        };
        match memory.write(person, &op) {
            Ok(_) => {}
            Err(e) if already(&e) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}

/// When `owner`'s thread is bound, write `label text` on its memory journal
/// off the request path.
pub fn spawn_export(
    records: &dyn Records,
    memory: &Arc<Memory>,
    owner: &str,
    thread: &str,
    label: &str,
    text: &str,
) {
    let Some(b) = bound(records, owner, thread) else {
        return;
    };
    let memory = Arc::clone(memory);
    let (thread, line) = (thread.to_string(), (label.to_string(), text.to_string()));
    thread::spawn(move || {
        if let Err(e) = export(&memory, &b, &thread, &line.0, &line.1) {
            tracing::warn!(%thread, "mcp: a thread's note could not be kept in the memory: {e}");
        }
    });
}

/// Write `label text` on a bound thread's memory journal, as zend.
pub fn export(memory: &Memory, b: &Bound, thread: &str, label: &str, text: &str) -> Result<(), String> {
    let group = Slug::parse(&b.group)?;
    let path = thread_path(&group, thread)?;
    ensure(memory, &path, thread)?;
    let op = Op::Note {
        path,
        text: format!("{label} {text}"),
        speaker: None,
        terms: Terms::default(),
    };
    memory.write(&Actor::Zend, &op).map(|_| ()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;

    fn slug(s: &str) -> Slug {
        Slug::parse(s).unwrap()
    }

    /// **A thread is bound to its group and topics in a record of its own**,
    /// kept as folders, and its memory journal is `threads/<thread>` there.
    #[test]
    fn a_thread_is_bound_to_its_group_and_topics() {
        let r = MemRecords::default();
        assert_eq!(bound(&r, "o", "dj-set"), None);
        let topic = JournalPath::of_section(&slug("zen"), Section::Topics)
            .child(slug("candle"))
            .unwrap();
        bind(&r, "o", "dj-set", &slug("zen"), &[topic]).unwrap();
        assert_eq!(
            bound(&r, "o", "dj-set"),
            Some(Bound {
                group: "zen".into(),
                about: vec!["groups/zen/topics/candle".into()],
            })
        );
        assert_eq!(bound(&r, "other", "dj-set"), None);
        assert_eq!(
            thread_path(&slug("zen"), "dj-set").unwrap().show(),
            "@zen/threads/dj-set"
        );
    }
}
