//! Which conversations each mind document became, so a document that leaves the
//! record — rejected on review, changed, deleted — leaves memory with it.
//!
//! **A document is ingested as conversations, and nothing used to say which.**
//! A layer document becomes one conversation; a life event becomes one, and
//! each belief, relationship and intention it forms another. None of them
//! carried the path they came from, so a document changed or removed on disk
//! stayed selectable by every character's gather: rejected drafts were still
//! remembered after they had left the record.
//!
//! Every conversation a document becomes is now marked with its mind path
//! ([`META_PATH`]) and retired by it. Conversations written before the mark
//! existed are found the way they can be: a life event's episode and
//! consequences by the date and title they carry, a layer document by its
//! address — the first turn's user half.

use std::collections::HashSet;
use std::path::Path;

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

use crate::engine::life;

/// The metadata key a conversation names its mind document by.
pub const META_PATH: &str = "mind.path";

/// `path` as a mind path — relative to `mind`, forward slashes.
pub fn rel(mind: &Path, path: &Path) -> Option<String> {
    // A path already relative is a mind path: what every document read from
    // the mind's files is (`docs/substrate_git.md` §8.1).
    // An absolute path against an empty root is no mind path at all: an empty
    // prefix strips from anything, and the whole drive would become the key.
    let r = match path.is_relative() {
        true => path,
        false if mind.as_os_str().is_empty() => return None,
        false => path.strip_prefix(mind).ok()?,
    };
    Some(r.to_string_lossy().replace('\\', "/"))
}

/// Mark `timeline` as written from the mind document at `rel`.
///
/// A document's conversation is read, never continued, so it is also marked
/// as never resumed: it keeps its turns and no recurrent snapshot.
pub fn mark(engine: &ConversationEngine, timeline: TimelineId, rel: &str) {
    engine.mark_timeline_unresumed(timeline);
    if let Err(e) = engine.set_conversation_metadata(timeline, META_PATH, rel) {
        tracing::warn!("{rel}: not marked as its document's — {e:?}");
    }
}

/// The metadata key a document's conversation records its content hash under —
/// the ledger's hash ([`crate::engine::watcher::hash`]) of the text it was
/// written from.
pub const META_HASH: &str = "mind.sha";

/// [`mark`], recording also the content hash of `body` — what lets another
/// member, or this one with its ledger lost, tell the document is already in
/// the substrate ([`seed_ledger`]).
pub fn mark_version(engine: &ConversationEngine, timeline: TimelineId, rel: &str, body: &str) {
    mark(engine, timeline, rel);
    let digest = crate::engine::watcher::hash(body);
    if let Err(e) = engine.set_conversation_metadata(timeline, META_HASH, &digest) {
        tracing::warn!("{rel}: content hash not recorded — {e:?}");
    }
}

/// Give the ingest ledger every document the substrate already holds at a
/// recorded version, so a document written by another member — or by this one
/// before its ledger was lost — is not written again. Answers how many.
pub fn seed_ledger(engine: &ConversationEngine, ledger: &crate::engine::watcher::Ledger) -> usize {
    let hashes: std::collections::HashMap<TimelineId, String> = engine
        .conversations_with_metadata_key(META_HASH)
        .into_iter()
        .collect();
    let held = engine
        .conversations_with_metadata_key(META_PATH)
        .into_iter()
        .filter_map(|(t, rel)| hashes.get(&t).map(|h| (rel, h.clone())));
    ledger.adopt_held(held)
}

/// Retire every copy but one of each layer document that is in the substrate
/// more than once. Answers how many copies were retired.
///
/// A layer document becomes exactly one conversation in its group, so two live
/// ones with the same mind path in the same group are the same document written
/// twice — by two members that each found it missing from their own ledger.
/// The copy kept is the first created, the lower timeline id breaking a tie: a
/// rule every member applies to the same records the same way, so members
/// retiring at once retire the same copies. A life event's path names several
/// conversations by design and is left alone.
pub fn retire_duplicate_documents(engine: &ConversationEngine) -> usize {
    use std::collections::HashMap;
    let mut copies: HashMap<(String, u32), Vec<(u64, u64, TimelineId)>> = HashMap::new();
    for (timeline, rel) in engine.conversations_with_metadata_key(META_PATH) {
        if life_event(&rel).is_some() {
            continue;
        }
        let Some((group, order)) = engine.timeline_group_and_order(timeline) else {
            continue;
        };
        copies
            .entry((rel, group.raw()))
            .or_default()
            .push((order, timeline.raw(), timeline));
    }
    let mut retired = 0;
    for ((rel, _), mut found) in copies {
        if found.len() < 2 {
            continue;
        }
        found.sort();
        for (_, _, t) in found.into_iter().skip(1) {
            match engine.tombstone_timeline(t) {
                Ok(()) => retired += 1,
                Err(e) => tracing::warn!("{rel}: a duplicate copy was not retired — {e:?}"),
            }
        }
    }
    retired
}

/// Smallest snapshot taken to be a kept state rather than a retired tail —
/// which encodes in a few dozen bytes.
const KEPT_SNAPSHOT_BYTES: u64 = 4096;

/// Retire the recurrent snapshots of every conversation that is read and never
/// continued — a mind document's, a kept dream's — and mark them so none is
/// written again. Answers how many tails were retired.
///
/// For a store written before the mark existed, where every such conversation
/// kept a whole recurrent state as its tail, and for marks lost to a restart.
/// A tail already retired costs a lookup.
pub fn retire_unresumed_snapshots(engine: &ConversationEngine) -> usize {
    let mut seen: HashSet<TimelineId> = HashSet::new();
    let mut retired = 0;
    for key in [META_PATH, crate::engine::dreams::META_OF] {
        for (timeline, _) in engine.conversations_with_metadata_key(key) {
            if !seen.insert(timeline) {
                continue;
            }
            let kept = engine
                .recurrent_snapshot_bytes(timeline)
                .is_some_and(|b| b >= KEPT_SNAPSHOT_BYTES);
            if kept {
                engine.mark_timeline_unresumed(timeline);
                retired += 1;
            } else {
                engine.note_timeline_unresumed(timeline);
            }
        }
    }
    retired
}

/// Retire every conversation the document at `rel` became. Answers how many.
pub fn retire(engine: &ConversationEngine, rel: &str) -> usize {
    let mut found: HashSet<TimelineId> = engine
        .find_conversations_by_metadata(META_PATH, rel)
        .into_iter()
        .collect();
    if found.is_empty() {
        found.extend(unmarked(engine, rel));
    }
    let mut retired = 0;
    for t in found {
        match engine.tombstone_timeline(t) {
            Ok(()) => retired += 1,
            Err(e) => tracing::warn!("{rel}: a conversation was not retired — {e:?}"),
        }
    }
    retired
}

/// The conversations a document became before they were marked.
fn unmarked(engine: &ConversationEngine, rel: &str) -> Vec<TimelineId> {
    match life_event(rel) {
        Some((who, date, title)) => {
            let of_life = |t: &TimelineId| {
                engine
                    .turn_tag_lists(*t)
                    .iter()
                    .flatten()
                    .any(|tag| tag.ends_with(&format!(":{who}")))
            };
            let dated = |t: &TimelineId, key: &str| {
                engine
                    .conversation_metadata(*t)
                    .is_some_and(|m| m.get(key).map(String::as_str) == Some(date.as_str()))
            };
            // The episode itself, and every consequence that names it.
            engine
                .find_conversations_by_metadata("life.title", &title)
                .into_iter()
                .filter(|t| dated(t, "life.date") && of_life(t))
                .chain(
                    engine
                        .find_conversations_by_metadata("from.title", &title)
                        .into_iter()
                        .filter(|t| dated(t, "from.date") && of_life(t)),
                )
                .collect()
        }
        None => {
            let Some(address) = address_of(rel) else {
                return Vec::new();
            };
            engine
                .live_conversations()
                .into_iter()
                .filter(|t| {
                    engine
                        .turn_texts(*t)
                        .first()
                        .is_some_and(|(user, _)| user == &address)
                })
                .collect()
        }
    }
}

/// A life event's character, date and title, from its mind path.
fn life_event(rel: &str) -> Option<(String, String, String)> {
    let rest = rel.strip_prefix("layers/life/")?;
    let (who, file) = rest.split_once('/')?;
    let named = life::parse_name(Path::new(file)).ok()?;
    Some((who.to_string(), named.date, named.title))
}

/// The address a layer document is ingested under: its path inside `layers/`,
/// without the extension — `layers/stories/the-ledger.md` → `stories/the-ledger`.
fn address_of(rel: &str) -> Option<String> {
    let inner = rel.strip_prefix("layers/")?;
    [".md", ".yaml", ".yml"]
        .iter()
        .find_map(|ext| inner.strip_suffix(ext))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn a_documents_path_address_and_life_are_read_from_its_mind_path() {
        let mind = PathBuf::from("D:/prog/mind");
        assert_eq!(
            rel(
                &mind,
                &mind.join("layers").join("stories").join("the-ledger.md")
            )
            .as_deref(),
            Some("layers/stories/the-ledger.md")
        );
        assert_eq!(rel(&mind, Path::new("C:/elsewhere/x.md")), None);
        assert_eq!(rel(Path::new(""), &std::env::temp_dir().join("x.md")), None);
        assert_eq!(
            rel(Path::new(""), Path::new("layers/x.md")).as_deref(),
            Some("layers/x.md")
        );
        assert_eq!(
            address_of("layers/stories/the-ledger.md").as_deref(),
            Some("stories/the-ledger")
        );
        assert_eq!(
            address_of("layers/world/factions/hess.yaml").as_deref(),
            Some("world/factions/hess")
        );
        assert_eq!(address_of("missions.yaml"), None);
        assert_eq!(
            life_event("layers/life/creed/2950-03-12 The Silence Between Orders.md"),
            Some((
                "creed".into(),
                "2950-03-12".into(),
                "The Silence Between Orders".into()
            ))
        );
        assert_eq!(life_event("layers/stories/x.md"), None);
    }
}
