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
    let r = path.strip_prefix(mind).ok()?;
    Some(r.to_string_lossy().replace('\\', "/"))
}

/// Mark `timeline` as written from the mind document at `rel`.
pub fn mark(engine: &ConversationEngine, timeline: TimelineId, rel: &str) {
    if let Err(e) = engine.set_conversation_metadata(timeline, META_PATH, rel) {
        tracing::warn!("{rel}: not marked as its document's — {e:?}");
    }
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
