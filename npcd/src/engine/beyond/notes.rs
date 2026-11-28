//! The world's notebook (`docs/zen_moon_mind.md` §16): notes kept for good,
//! one notebook per world, read and written by everyone in it who reaches
//! `memory`. Each note is the substrate's custom object
//! `notes/<world>/<key>`, so it replicates with the world and outlives a
//! restart; writing a note empty tombstones it.

use std::cmp::Reverse;

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::SharedSubstrate;
use serde::{Deserialize, Serialize};

use crate::discord::trace::render::utc;

/// Where every world's notebook lives among the substrate's custom objects.
const PREFIX: &str = "notes/";

/// The longest key a note is kept under, in bytes.
pub const MAX_KEY: usize = 256;

/// The most a note holds, in bytes: what an answer shows of one.
pub const MAX_NOTE: usize = 6_000;

/// How much of a note a search shows.
const SNIPPET: usize = 200;

/// The most notes a search names unless it says: `notes_search`'s default.
pub const SEARCH_MOST: usize = 10;

/// The most notes a listing names unless it says: `notes_list`'s default.
pub const LIST_MOST: usize = 50;

/// What an act asks of the notebook, with `notes_*`'s own arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotesCall {
    /// Keep `content` under `key`; empty, remove the note.
    Write {
        key: String,
        content: String,
        tags: Vec<String>,
    },
    /// The note under `key`.
    Read { key: String },
    /// The notes whose key or text holds `query` and that carry one of
    /// `tags`, newest first, the first `most`.
    Search {
        query: String,
        tags: Vec<String>,
        most: usize,
    },
    /// The notes under `prefix` that carry one of `tags`, in key order, the
    /// first `most`.
    List {
        prefix: String,
        tags: Vec<String>,
        most: usize,
    },
}

/// One note, as it is kept.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Note {
    pub content: String,
    pub tags: Vec<String>,
    pub created_ms: u64,
    pub updated_ms: u64,
}

/// The record key of `world`'s note `key`.
pub fn key(world: &str, key: &str) -> String {
    format!("{PREFIX}{world}/{key}")
}

fn shelf(world: &str) -> String {
    format!("{PREFIX}{world}/")
}

/// `world`'s note `name`, if it has one.
fn get(shared: &SharedSubstrate, world: &str, name: &str) -> Option<Note> {
    shared
        .custom_object(&key(world, name))
        .filter(|o| !o.ceased)
        .and_then(|o| serde_json::from_value(o.blob).ok())
}

/// Every note in `world`'s notebook, by key, in key order.
fn all(shared: &SharedSubstrate, world: &str) -> Vec<(String, Note)> {
    let shelf = shelf(world);
    shared
        .custom_objects_under(&shelf)
        .into_iter()
        .filter(|(o, _)| !o.ceased)
        .filter_map(|(o, _)| {
            let name = o.key.strip_prefix(&shelf)?.to_string();
            Some((name, serde_json::from_value(o.blob).ok()?))
        })
        .collect()
}

/// Whether `note` carries one of `tags` — any note, with none asked for.
fn tagged(note: &Note, tags: &[String]) -> bool {
    tags.is_empty() || tags.iter().any(|t| note.tags.contains(t))
}

/// What `call` does to `world`'s notebook at `now_ms`: what the act answers,
/// or why it was refused.
pub fn run(
    shared: &SharedSubstrate,
    world: &str,
    call: &NotesCall,
    now_ms: u64,
) -> Result<String, String> {
    match call {
        NotesCall::Write {
            key: name,
            content,
            tags,
        } => {
            if name.len() > MAX_KEY || name.chars().any(char::is_control) {
                return Err(format!(
                    "A note's key is one line of at most {MAX_KEY} bytes."
                ));
            }
            if content.len() > MAX_NOTE {
                return Err(format!(
                    "A note holds at most {MAX_NOTE} bytes; this is {}. Keep it in two.",
                    content.len()
                ));
            }
            let before = get(shared, world, name);
            if content.is_empty() {
                if before.is_none() {
                    return Err(format!("There is no note `{name}` to remove."));
                }
                shared
                    .put_custom_object(CustomObjectPayload::ceased(key(world, name)))
                    .map_err(|e| format!("The note was not removed: {e:?}"))?;
                return Ok(format!("Removed the note `{name}`."));
            }
            let note = Note {
                content: content.clone(),
                tags: tags.clone(),
                created_ms: before.as_ref().map_or(now_ms, |n| n.created_ms),
                updated_ms: now_ms,
            };
            let body = serde_json::to_value(&note).expect("a note serialises");
            shared
                .put_custom_object(CustomObjectPayload::new(key(world, name), body))
                .map_err(|e| format!("The note was not kept: {e:?}"))?;
            Ok(match before {
                Some(_) => format!("Kept `{name}` again, in place of what it held."),
                None => format!("Kept `{name}`."),
            })
        }
        NotesCall::Read { key: name } => get(shared, world, name)
            .map(|n| {
                format!(
                    "`{name}`{} — written {}\n{}",
                    tags_of(&n),
                    utc(n.updated_ms),
                    n.content
                )
            })
            .ok_or_else(|| {
                format!("There is no note `{name}`: `notes_list` names those there are.")
            }),
        NotesCall::Search { query, tags, most } => {
            let q = query.to_lowercase();
            let mut hits: Vec<(String, Note)> = all(shared, world)
                .into_iter()
                .filter(|(_, n)| tagged(n, tags))
                .filter(|(name, n)| {
                    q.is_empty()
                        || name.to_lowercase().contains(&q)
                        || n.content.to_lowercase().contains(&q)
                })
                .collect();
            if hits.is_empty() {
                return Ok("No note holds that.".into());
            }
            hits.sort_by_key(|(_, n)| Reverse(n.updated_ms));
            let total = hits.len();
            let mut lines: Vec<String> = hits
                .into_iter()
                .take(*most)
                .map(|(name, n)| {
                    let snippet: String = n.content.chars().take(SNIPPET).collect();
                    format!("- `{name}`{}: {snippet}", tags_of(&n))
                })
                .collect();
            if total > *most {
                lines.push(format!("…and {} more.", total - most));
            }
            Ok(lines.join("\n"))
        }
        NotesCall::List { prefix, tags, most } => {
            let notes: Vec<(String, Note)> = all(shared, world)
                .into_iter()
                .filter(|(name, n)| name.starts_with(prefix.as_str()) && tagged(n, tags))
                .collect();
            if notes.is_empty() {
                return Ok(match prefix.is_empty() {
                    true => "The notebook holds nothing yet.".into(),
                    false => format!("The notebook holds nothing under `{prefix}`."),
                });
            }
            let total = notes.len();
            let mut lines: Vec<String> = notes
                .into_iter()
                .take(*most)
                .map(|(name, n)| {
                    format!(
                        "- `{name}`{} — {} bytes, written {}",
                        tags_of(&n),
                        n.content.len(),
                        utc(n.updated_ms)
                    )
                })
                .collect();
            if total > *most {
                lines.push(format!("…and {} more.", total - most));
            }
            Ok(lines.join("\n"))
        }
    }
}

/// A note's tags as a listing shows them: ` [a, b]`, or nothing.
fn tags_of(n: &Note) -> String {
    match n.tags.is_empty() {
        true => String::new(),
        false => format!(" [{}]", n.tags.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::TempDir;

    use super::*;
    use crate::npcs::Npcs;

    /// 2026-11-28 08:00:00 UTC.
    const AT: u64 = 1_795_852_800_000;

    fn write(content: &str, tags: &[&str]) -> NotesCall {
        NotesCall::Write {
            key: "house/wifi".into(),
            content: content.into(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
        }
    }

    /// **A note is the record `notes/<world>/<key>`**, in exactly this shape,
    /// kept again in place with its first writing's time, and removed by
    /// writing it empty.
    #[test]
    fn a_note_is_a_custom_object_of_this_shape() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        assert_eq!(
            run(&shared, "moon", &write("Moonbase-Guest", &["house"]), AT),
            Ok("Kept `house/wifi`.".into())
        );
        assert_eq!(
            shared.custom_object("notes/moon/house/wifi").unwrap().blob,
            json!({"content": "Moonbase-Guest", "tags": ["house"],
                   "created_ms": AT, "updated_ms": AT})
        );
        assert_eq!(
            run(
                &shared,
                "moon",
                &write("Moonbase-Guest-5G", &[]),
                AT + 60_000
            ),
            Ok("Kept `house/wifi` again, in place of what it held.".into())
        );
        assert_eq!(
            shared.custom_object("notes/moon/house/wifi").unwrap().blob,
            json!({"content": "Moonbase-Guest-5G", "tags": [],
                   "created_ms": AT, "updated_ms": AT + 60_000})
        );
        assert_eq!(
            run(&shared, "moon", &write("", &[]), AT),
            Ok("Removed the note `house/wifi`.".into())
        );
        assert_eq!(shared.custom_object("notes/moon/house/wifi"), None);
        assert_eq!(
            run(&shared, "moon", &write("", &[]), AT),
            Err("There is no note `house/wifi` to remove.".into())
        );
    }

    /// **Each world keeps its own notebook**: a note of the moon's is not
    /// another world's to read.
    #[test]
    fn each_world_keeps_its_own_notebook() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        run(&shared, "moon", &write("Moonbase-Guest", &["house"]), AT).unwrap();
        let read = NotesCall::Read {
            key: "house/wifi".into(),
        };
        assert_eq!(
            run(&shared, "moon", &read, AT),
            Ok("`house/wifi` [house] — written 2026-11-28 08:00:00 UTC\nMoonbase-Guest".into())
        );
        assert_eq!(
            run(&shared, "creators-vault", &read, AT),
            Err("There is no note `house/wifi`: `notes_list` names those there are.".into())
        );
    }

    /// **A search finds words in a key or a note's text, newest first, and a
    /// listing walks keys in order**, each held to a prefix or a tag, and to
    /// how many it was asked for.
    #[test]
    fn a_search_and_a_listing_find_what_is_kept() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        let put = |k: &str, c: &str, tags: &[&str], at: u64| {
            let call = NotesCall::Write {
                key: k.into(),
                content: c.into(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
            };
            run(&shared, "moon", &call, at).unwrap();
        };
        put(
            "house/router",
            "A Netgear Orbi, in the hall.",
            &["house"],
            AT,
        );
        put(
            "people/john",
            "Reads Celsius. Router password is his.",
            &[],
            AT + 1_000,
        );
        put("house/tv", "An LG C2.", &["house", "tv"], AT + 2_000);

        let search = |q: &str, tags: &[&str]| {
            let call = NotesCall::Search {
                query: q.into(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
                most: SEARCH_MOST,
            };
            run(&shared, "moon", &call, AT).unwrap()
        };
        assert_eq!(
            search("router", &[]),
            "- `people/john`: Reads Celsius. Router password is his.\n\
             - `house/router` [house]: A Netgear Orbi, in the hall."
        );
        assert_eq!(search("", &["tv"]), "- `house/tv` [house, tv]: An LG C2.");
        assert_eq!(search("piano", &[]), "No note holds that.");

        let list = |prefix: &str| {
            let call = NotesCall::List {
                prefix: prefix.into(),
                tags: Vec::new(),
                most: 1,
            };
            run(&shared, "moon", &call, AT).unwrap()
        };
        assert_eq!(
            list("house/"),
            "- `house/router` [house] — 28 bytes, written 2026-11-28 08:00:00 UTC\n…and 1 more."
        );
        assert_eq!(
            list("garden/"),
            "The notebook holds nothing under `garden/`."
        );
    }

    /// **A note too long, or a key that is no line, is refused** with what
    /// fits.
    #[test]
    fn a_note_past_its_bounds_is_refused() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        let long = NotesCall::Write {
            key: "k".into(),
            content: "x".repeat(MAX_NOTE + 1),
            tags: Vec::new(),
        };
        assert_eq!(
            run(&shared, "moon", &long, AT),
            Err("A note holds at most 6000 bytes; this is 6001. Keep it in two.".into())
        );
        let two_lines = NotesCall::Write {
            key: "a\nb".into(),
            content: "x".into(),
            tags: Vec::new(),
        };
        assert_eq!(
            run(&shared, "moon", &two_lines, AT),
            Err("A note's key is one line of at most 256 bytes.".into())
        );
    }
}
