//! Directory listings, path search and grep over both layers of a store.
//!
//! A listing is one level deep: the upper layer shadows the workspace, whiteouts
//! are omitted, and a subdirectory appears as itself. Listing never opens a file,
//! so it cannot refuse where a read would succeed. A plain file and a directory
//! sharing a name are kept as two entries rather than one hiding the other. Grep
//! skips a file it cannot scan instead of failing the pass, and reports whether
//! it stopped at its hit ceiling, so a caller knows the result is a prefix.

use super::*;

/// One matching line from [`VfsStore::grep`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepHit {
    pub path: String,
    /// 1-based line number within the file.
    pub line_no: u32,
    /// The matching line, with trailing `\r` and whitespace trimmed.
    pub line: String,
    /// `true` when the hit came from this session's own copy of the file.
    pub modified: bool,
}

/// What a [`VfsStore::grep`] pass found.
#[derive(Debug, Default)]
pub struct GrepOutcome {
    pub hits: Vec<GrepHit>,
    /// Files whose contents were actually scanned — the denominator that tells a
    /// caller whether "no matches" means "searched a lot and found nothing" or
    /// "the prefix matched nothing to search".
    pub files_searched: usize,
    /// `true` when the scan stopped at its hit ceiling, so the result is a
    /// prefix of what is there rather than all of it.
    pub truncated: bool,
}

/// One entry in a directory listing: normalised path, byte size. No line
/// count — that would cost opening every file in the walk to compute, and a
/// listing that never opens a file is the whole point of `file_list`. A file's length reaches the model
/// through `file_read`'s own opening fence instead (`page=0/13 lines=2499`),
/// which is exact, costs nothing extra, and arrives at the moment the
/// number is actually needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListEntry {
    pub path: String,
    /// `None` for a directory entry — a directory has no size of its own.
    pub bytes: Option<usize>,
    /// `true` when this entry is a subdirectory rather than a file. A listing
    /// is one level deep, so a subdirectory appears as itself — never
    /// expanded into the files it holds.
    pub dir: bool,
    /// `true` when the entry is the session's own copy (upper layer) rather than
    /// a file read straight off the workspace. Always `false` for a directory
    /// entry: a directory is not itself written, only the files inside it.
    pub modified: bool,
}

/// How the session has changed a path, against the lower layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FileState {
    /// The lower layer has no such file.
    Added,
    Modified,
    Deleted,
}

impl VfsStore {
    /// One level of `dir`'s contents, upper layer shadowing the workspace:
    /// the files and subdirectories directly inside it, never a deeper file
    /// flattened up into its listing. Whiteouted paths are omitted. Sorted by
    /// path. `Ok(None)` means `dir` does not resolve to a directory in either
    /// layer — a missing path, or one that names a file.
    pub fn list_dir(&self, dir: &str) -> Result<Option<Vec<ListEntry>>, VfsError> {
        let norm = Self::normalize(dir);
        // A protected folder named directly lists as empty — never an error
        // that would confirm or deny it, and never its contents.
        if Self::is_protected(&norm) {
            return Ok(Some(Vec::new()));
        }
        let view = self.view()?;
        // `norm` resolving as a file rules out a directory listing outright —
        // checked first, and independently of the walk below, because a
        // conflicting write elsewhere in the upper layer (e.g. a file at
        // `"a/b.rs/c.rs"` alongside a real file at `"a/b.rs"`) would otherwise
        // make `upper_has_children` true for `"a/b.rs"` and this would
        // silently resolve a file path as a directory.
        if self.resolves_as_file(&view, &norm) {
            return Ok(None);
        }
        let mut out: Vec<ListEntry> = Vec::new();
        // Keyed on `(path, is_dir)`, not just `path`: a plain file at `"a"`
        // and another upper file at `"a/b.rs"` (which collapses to a
        // directory entry also named `"a"`) are two genuinely different
        // things sharing one name — deduping on the name alone would drop
        // whichever the map happened to iterate second, silently hiding it
        // from the listing rather than just looking odd.
        let mut seen: HashSet<(String, bool)> = HashSet::new();
        let mut upper_has_children = false;

        {
            let guard = self.upper.read().unwrap();
            for (k, chain) in guard.chains.iter() {
                // A deleted path shadows the workspace's entry and lists as
                // nothing.
                let Some(size) = chain.size else {
                    seen.insert((k.clone(), false));
                    continue;
                };
                let Some((child, is_dir)) = Self::immediate_child(&norm, k) else {
                    continue;
                };
                upper_has_children = true;
                if !seen.insert((child.clone(), is_dir)) {
                    continue;
                }
                out.push(if is_dir {
                    ListEntry {
                        path: child,
                        bytes: None,
                        dir: true,
                        modified: false,
                    }
                } else {
                    ListEntry {
                        path: child,
                        bytes: Some(size),
                        dir: false,
                        modified: true,
                    }
                });
            }
        }

        let lower_is_dir = view.is_dir(&norm);
        if !lower_is_dir && !upper_has_children && !norm.is_empty() {
            return Ok(None);
        }

        if lower_is_dir || norm.is_empty() {
            for (path, bytes, is_dir) in view.children(&norm) {
                if seen.contains(&(path.clone(), is_dir)) {
                    continue;
                }
                seen.insert((path.clone(), is_dir));
                out.push(ListEntry {
                    path,
                    bytes,
                    dir: is_dir,
                    modified: false,
                });
            }
        }

        out.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(Some(out))
    }

    /// Whether `norm` resolves to a file in either layer — key/metadata
    /// existence only, never a content read (matching `list_dir`'s "never
    /// opens a file" contract). Upper-first, respecting whiteouts, the same
    /// precedence `read` uses.
    pub(super) fn resolves_as_file(&self, view: &View<'_>, norm: &str) -> bool {
        if norm.is_empty() {
            return false;
        }
        if let Some(chain) = self.upper.read().unwrap().chains.get(norm) {
            return chain.size.is_some();
        }
        view.is_file(norm)
    }

    /// Remove a path from the overlay. A file the lower layer holds is
    /// recorded as one [`FileDelta::Delete`], superseding the path's earlier
    /// deltas, so it stops resolving; a file only the session made just loses
    /// its deltas. Returns whether the path resolved before the call. The
    /// lower layer is never touched.
    pub fn delete(&self, path: &str) -> bool {
        let norm = Self::normalize(path);
        if Self::is_protected(&norm) {
            return false;
        }
        let Ok(view) = self.view() else {
            return false;
        };
        let in_lower = view.is_file(&norm);
        let mut guard = self.upper.write().unwrap();
        if guard.deleted(&norm) {
            return false;
        }
        let had_upper = guard.chains.remove(&norm).is_some();
        if in_lower {
            guard.chains.insert(
                norm,
                Chain {
                    deltas: vec![TimedDelta::now(FileDelta::Delete)],
                    size: None,
                    conflict: false,
                },
            );
        }
        had_upper || in_lower
    }

    /// Every path the session has changed, sorted, and how — against the
    /// lower layer as it stands now: what a status of the conversation's
    /// uncommitted work reports.
    pub fn status(&self) -> Vec<(String, FileState)> {
        let Ok(view) = self.view() else {
            return Vec::new();
        };
        let upper = self.upper.read().unwrap();
        let mut out: Vec<(String, FileState)> = upper
            .chains
            .iter()
            .map(|(path, chain)| {
                let state = match (chain.size, view.is_file(path)) {
                    (None, _) => FileState::Deleted,
                    (Some(_), true) => FileState::Modified,
                    (Some(_), false) => FileState::Added,
                };
                (path.clone(), state)
            })
            .collect();
        out.sort();
        out
    }

    /// Drop every change the session holds: every path reads as the lower
    /// layer holds it again. What a hard reset does to the conversation's
    /// uncommitted work.
    pub fn discard(&self) {
        let mut upper = self.upper.write().unwrap();
        upper.chains.clear();
    }

    // ── Search ───────────────────────────────────────────────────────────────

    /// Every path visible under `prefix`, upper layer shadowing the lower.
    ///
    /// Unlike [`VfsStore::list_dir`] this reads no file contents, so it stays
    /// cheap over a whole repository — line counts are what make a full
    /// listing expensive, and a path search does not need them.
    pub fn paths(&self, prefix: &str) -> Vec<String> {
        match self.view() {
            Ok(view) => self.paths_in(&view, prefix),
            Err(_) => Vec::new(),
        }
    }

    /// [`Self::paths`], against `view`.
    pub(super) fn paths_in(&self, view: &View<'_>, prefix: &str) -> Vec<String> {
        let norm_prefix = Self::normalize(prefix);
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<String> = Vec::new();
        {
            let guard = self.upper.read().unwrap();
            for (k, chain) in guard.chains.iter() {
                seen.insert(k.clone());
                if chain.size.is_some()
                    && Self::matches_prefix(k, &norm_prefix)
                    && !Self::is_protected(k)
                {
                    out.push(k.clone());
                }
            }
        }
        for path in view.files_under(&norm_prefix) {
            if !seen.contains(&path) {
                out.push(path);
            }
        }
        out.sort();
        out
    }

    /// Scan file contents under `prefix` for `re`.
    ///
    /// Files that cannot be scanned — oversize, not UTF-8, vanished between the
    /// walk and the read — are skipped rather than failing the pass: a single
    /// binary blob in a tree must not turn a whole search into an error.
    pub fn grep(
        &self,
        re: &regex::Regex,
        prefix: &str,
        max_per_file: usize,
        max_total: usize,
    ) -> GrepOutcome {
        let mut out = GrepOutcome::default();
        let Ok(view) = self.view() else {
            return out;
        };
        for path in self.paths_in(&view, prefix) {
            let modified = self.holds(&path);
            let Ok(Some(content)) = self.current(&view, &path) else {
                continue;
            };
            out.files_searched += 1;

            let mut in_file = 0usize;
            for (idx, line) in content.lines().enumerate() {
                if !re.is_match(line) {
                    continue;
                }
                if out.hits.len() >= max_total {
                    out.truncated = true;
                    return out;
                }
                out.hits.push(GrepHit {
                    path: path.clone(),
                    line_no: idx as u32 + 1,
                    line: line.trim_end().to_string(),
                    modified,
                });
                in_file += 1;
                if in_file >= max_per_file {
                    // One file monopolising the budget would hide every other
                    // file that matches, which is the answer the caller wants.
                    out.truncated = true;
                    break;
                }
            }
        }
        out
    }

    // ── Path handling ────────────────────────────────────────────────────────

    /// What a one-level listing of `dir` shows for `key`, given `key` lives
    /// somewhere under `dir` (or `dir` is the empty root prefix): `key` itself
    /// when it is a direct child, or the immediate subdirectory's own path when
    /// `key` is deeper — collapsed rather than flattened all the way down, the
    /// same way a real directory listing never expands a subdirectory's
    /// contents into itself. `None` when `key` is not under `dir` at all.
    pub(super) fn immediate_child(dir: &str, key: &str) -> Option<(String, bool)> {
        let remainder = if dir.is_empty() {
            key
        } else {
            key.strip_prefix(dir)?.strip_prefix('/')?
        };
        if remainder.is_empty() {
            return None;
        }
        Some(match remainder.find('/') {
            None => (key.to_string(), false),
            Some(slash) => {
                let first = &remainder[..slash];
                let path = if dir.is_empty() {
                    first.to_string()
                } else {
                    format!("{dir}/{first}")
                };
                (path, true)
            }
        })
    }
}
