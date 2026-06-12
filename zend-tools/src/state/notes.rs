//! Per-user persistent notes (cross-conversation memory).

use std::collections::HashMap;
use std::sync::RwLock;

use chrono::Utc;

pub struct Note {
    pub key: String,
    pub content: String,
    pub tags: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
    pub bytes: usize,
}

pub struct NoteResult {
    pub key: String,
    pub snippet: String,
    pub tags: Vec<String>,
    pub updated_at: String,
    pub rank: f64,
}

pub struct NoteListEntry {
    pub key: String,
    pub bytes: usize,
    pub tags: Vec<String>,
    pub updated_at: String,
}

#[derive(Default)]
pub struct NotesStore {
    inner: RwLock<HashMap<String, Note>>,
}

impl NotesStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns (created, bytes).
    pub fn write(&self, key: &str, content: String, tags: Vec<String>) -> (bool, usize) {
        let now = Utc::now().to_rfc3339();
        let bytes = content.len();
        let mut guard = self.inner.write().unwrap();
        let created = !guard.contains_key(key);
        let created_at = if created {
            now.clone()
        } else {
            guard[key].created_at.clone()
        };
        guard.insert(
            key.to_string(),
            Note {
                key: key.to_string(),
                content,
                tags,
                created_at,
                updated_at: now,
                bytes,
            },
        );
        (created, bytes)
    }

    pub fn read(&self, key: &str) -> Option<Note> {
        let guard = self.inner.read().unwrap();
        guard.get(key).map(|n| Note {
            key: n.key.clone(),
            content: n.content.clone(),
            tags: n.tags.clone(),
            created_at: n.created_at.clone(),
            updated_at: n.updated_at.clone(),
            bytes: n.bytes,
        })
    }

    pub fn search(&self, query: &str, tags: &[String], max: usize) -> (Vec<NoteResult>, usize) {
        let guard = self.inner.read().unwrap();
        let query_lower = query.to_lowercase();
        let mut results: Vec<NoteResult> = guard
            .values()
            .filter(|n| {
                let tag_match = tags.is_empty() || tags.iter().any(|t| n.tags.contains(t));
                let query_match = query.is_empty()
                    || n.content.to_lowercase().contains(&query_lower)
                    || n.key.to_lowercase().contains(&query_lower);
                tag_match && query_match
            })
            .map(|n| {
                let snippet = n.content.chars().take(200).collect::<String>();
                NoteResult {
                    key: n.key.clone(),
                    snippet,
                    tags: n.tags.clone(),
                    updated_at: n.updated_at.clone(),
                    rank: 1.0,
                }
            })
            .collect();
        let total = results.len();
        results.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        results.truncate(max);
        (results, total)
    }

    pub fn list(&self, prefix: &str, tags: &[String], max: usize) -> (Vec<NoteListEntry>, usize) {
        let guard = self.inner.read().unwrap();
        let mut entries: Vec<NoteListEntry> = guard
            .values()
            .filter(|n| {
                let prefix_match = prefix.is_empty() || n.key.starts_with(prefix);
                let tag_match = tags.is_empty() || tags.iter().any(|t| n.tags.contains(t));
                prefix_match && tag_match
            })
            .map(|n| NoteListEntry {
                key: n.key.clone(),
                bytes: n.bytes,
                tags: n.tags.clone(),
                updated_at: n.updated_at.clone(),
            })
            .collect();
        let total = entries.len();
        entries.sort_by(|a, b| a.key.cmp(&b.key));
        entries.truncate(max);
        (entries, total)
    }
}
