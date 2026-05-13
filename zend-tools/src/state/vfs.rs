//! Per-session in-memory virtual filesystem (VFS).
//!
//! The VFS is a `HashMap<String, String>` (normalised path → UTF-8 content)
//! shared across all `file_*` tool calls within a session.  It has no connection
//! to any real filesystem; nothing written here ever touches disk.
//!
//! # Path normalisation
//!
//! Paths are normalised before storage: leading `/` is stripped, `.` and empty
//! segments are collapsed, `..` pops the stack.  This means `./src/../main.rs`,
//! `/main.rs`, and `main.rs` all resolve to the same key `"main.rs"`.  There is
//! no traversal concern (no real FS backing), but normalisation prevents duplicate
//! entries under different spellings.
//!
//! # Size cap
//!
//! Total VFS content is capped at 10 MiB per session (enforced on each `write`
//! call).  Individual files are uncapped within that budget.  [`VfsError::Full`]
//! is returned when the cap would be exceeded; the write is not applied.

use std::collections::HashMap;
use std::sync::RwLock;

const MAX_BYTES: usize = 10 * 1024 * 1024; // 10 MiB

#[derive(Debug)]
pub enum VfsError {
    Full,
}

impl std::fmt::Display for VfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VfsError::Full => write!(f, "VFS storage limit exceeded (10 MiB)"),
        }
    }
}

/// In-memory `path -> content` map shared across `file_*` tools.
/// Capped at 10 MiB total per session.
#[derive(Default)]
pub struct VfsStore {
    inner: RwLock<HashMap<String, String>>,
}

impl VfsStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write(&self, path: &str, content: String) -> Result<bool, VfsError> {
        let norm = Self::normalize(path);
        let mut guard = self.inner.write().unwrap();
        let existing_bytes: usize = guard.values().map(|v| v.len()).sum();
        let old_len = guard.get(&norm).map(|v| v.len()).unwrap_or(0);
        let new_total = existing_bytes - old_len + content.len();
        if new_total > MAX_BYTES {
            return Err(VfsError::Full);
        }
        let created = !guard.contains_key(&norm);
        guard.insert(norm, content);
        Ok(created)
    }

    pub fn read(&self, path: &str) -> Option<String> {
        let norm = Self::normalize(path);
        self.inner.read().unwrap().get(&norm).cloned()
    }

    pub fn list(&self, prefix: &str) -> Vec<(String, usize, usize)> {
        let norm_prefix = Self::normalize(prefix);
        let guard = self.inner.read().unwrap();
        let mut entries: Vec<(String, usize, usize)> = guard
            .iter()
            .filter(|(k, _)| k.starts_with(&norm_prefix) || norm_prefix.is_empty())
            .map(|(k, v)| {
                let lines = v.lines().count();
                (k.clone(), v.len(), lines)
            })
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }

    pub fn delete(&self, path: &str) -> bool {
        let norm = Self::normalize(path);
        self.inner.write().unwrap().remove(&norm).is_some()
    }

    pub fn total_bytes(&self) -> usize {
        self.inner.read().unwrap().values().map(|v| v.len()).sum()
    }

    fn normalize(path: &str) -> String {
        let path = path.trim_start_matches('/');
        let mut parts: Vec<&str> = Vec::new();
        for segment in path.split('/') {
            match segment {
                "" | "." => {}
                ".." => { parts.pop(); }
                s => parts.push(s),
            }
        }
        parts.join("/")
    }
}
