//! A session's own changes to a file, and reads of them through the overlay.
//!
//! A write records one whole-file replacement and supersedes every earlier change
//! to the path. An edit records only the lines that differ, or the whole file
//! when most of its lines differ, and refuses to apply when the workspace copy it
//! was made against has changed underneath it. A read replays the session's chain
//! onto the workspace copy, or returns the workspace copy alone. Nothing reaches
//! disk, but a write to a protected path is refused like a read, since a later
//! read would otherwise find what was planted there.

use super::*;

impl VfsStore {
    /// Replace the whole file with `content` — recorded as one
    /// [`FileDelta::Replace`], superseding every earlier delta for the path,
    /// a deletion included. Returns whether this created a path that did not
    /// previously resolve: replacing a workspace file for the first time is an
    /// overwrite, and writing over a deletion is a creation.
    pub fn write(&self, path: &str, content: String) -> Result<bool, VfsError> {
        let norm = Self::normalize(path);
        // Nothing reaches disk, but a session planting a decoy at a protected
        // path would have later reads find it: refused like a read.
        Self::guard(&norm)?;
        let view = self.view()?;
        let created = !self.resolves_as_file(&view, &norm);
        let size = content.len();
        let mut guard = self.upper.write().unwrap();
        let conflict = Self::still_conflicted(&guard, &norm, Some(&content));
        let chain = Chain {
            deltas: vec![TimedDelta::now(FileDelta::Replace { content })],
            size: Some(size),
            conflict,
        };
        Self::set_capped(&mut guard, norm, chain)?;
        Ok(created)
    }

    /// Change the file to `content` by an edit — recorded as one
    /// [`FileDelta::Edit`] carrying only the lines that differ from what the
    /// file holds now, or, when those are more than
    /// [`REPLACE_ABOVE_PERCENT`](file_delta::REPLACE_ABOVE_PERCENT) of its
    /// lines, as the whole file, exactly as a [`Self::write`] records it. The
    /// file must exist. Returns whether anything changed: an edit that leaves
    /// the file as it was records nothing.
    pub fn edit(&self, path: &str, content: String) -> Result<bool, VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        loop {
            let seen = Seen::of(&self.upper.read().unwrap(), &norm);
            let view = self.view()?;
            let current = self
                .current(&view, &norm)?
                .ok_or_else(|| VfsError::Unreadable(format!("{norm} does not exist to edit")))?;
            if current == content {
                // Keeping a file in conflict exactly as it stands — the side
                // of an edit that met a delete, say — settles it, markers
                // aside.
                let mut guard = self.upper.write().unwrap();
                if !seen.stands(&guard, &norm) {
                    continue;
                }
                if let Some(chain) = guard.chains.get_mut(&norm) {
                    if chain.conflict && !has_markers(&content) {
                        chain.conflict = false;
                    }
                }
                return Ok(false);
            }
            let size = Some(content.len());
            let delta = TimedDelta::now(file_delta::delta(&current, &content));
            let mut guard = self.upper.write().unwrap();
            if !seen.stands(&guard, &norm) {
                continue;
            }
            let conflict = Self::still_conflicted(&guard, &norm, Some(&content));
            let chain = match delta.delta {
                // A replacement supersedes the chain, as a write does.
                FileDelta::Replace { .. } => Chain {
                    deltas: vec![delta],
                    size,
                    conflict,
                },
                _ => {
                    let mut chain = seen.chain.unwrap_or(Chain {
                        deltas: Vec::new(),
                        size: None,
                        conflict: false,
                    });
                    chain.deltas.push(delta);
                    chain.size = size;
                    chain.conflict = conflict;
                    chain
                }
            };
            Self::set_capped(&mut guard, norm, chain)?;
            return Ok(true);
        }
    }

    /// Resolve a path through the overlay: the session's deltas replayed onto
    /// the workspace's copy, or the workspace's copy alone when the session has
    /// not changed it. `Ok(None)` means the path does not exist in either
    /// layer, or the session deleted it.
    pub fn read(&self, path: &str) -> Result<Option<String>, VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        let view = self.view()?;
        self.current(&view, &norm)
    }

    /// Whether the session has changed `path` — written, edited or deleted it —
    /// so that what a read returns is no longer the lower layer's copy.
    pub fn is_modified(&self, path: &str) -> bool {
        self.holds(&Self::normalize(path))
    }

    /// Every path the session has changed, sorted.
    pub fn changed_paths(&self) -> Vec<String> {
        let mut paths: Vec<String> = self.upper.read().unwrap().chains.keys().cloned().collect();
        paths.sort();
        paths
    }

    /// Whether the session layer holds a chain for `norm`.
    pub(super) fn holds(&self, norm: &str) -> bool {
        self.upper.read().unwrap().chains.contains_key(norm)
    }

    /// `norm`'s content as this store resolves it: the session's chain
    /// replayed, or the lower layer's copy.
    ///
    /// A chain opening with an edit is replayed onto the lower layer's copy as
    /// it is now. When that copy has changed underneath the edit, the edit no
    /// longer fits it and the read is refused as [`VfsError::Diverged`] — the
    /// alternative is splicing the session's change into the wrong place.
    pub(super) fn current(&self, view: &View<'_>, norm: &str) -> Result<Option<String>, VfsError> {
        let chain = self.upper.read().unwrap().chains.get(norm).cloned();
        let Some(chain) = chain else {
            return view.read_text(norm);
        };
        let base = match chain.deltas.first().map(|t| &t.delta) {
            Some(FileDelta::Edit { .. }) => view.read_text(norm)?,
            _ => None,
        };
        file_delta::replay(base, chain.deltas.iter().map(|t| &t.delta)).map_err(|e| match e {
            ReplayError::Diverged(_) => Self::diverged(norm),
            ReplayError::NotText => Self::not_text(norm),
        })
    }

    /// [`Self::current`] as bytes: what a binary file — an upload, a build's
    /// output — holds as well as a text one.
    pub(super) fn current_bytes(
        &self,
        view: &View<'_>,
        norm: &str,
    ) -> Result<Option<Vec<u8>>, VfsError> {
        let chain = self.upper.read().unwrap().chains.get(norm).cloned();
        let Some(chain) = chain else {
            return view.read_bytes(norm);
        };
        let base = match chain.deltas.first().map(|t| &t.delta) {
            Some(FileDelta::Edit { .. }) => view.read_bytes(norm)?,
            _ => None,
        };
        file_delta::replay_bytes(base, chain.deltas.iter().map(|t| &t.delta))
            .map_err(|_| Self::diverged(norm))
    }

    /// The deltas this store holds for `path` — the session's chain, oldest
    /// first, each with the moment its operation executed — or `None` when the
    /// session has not changed it.
    pub fn deltas(&self, path: &str) -> Option<Vec<TimedDelta>> {
        let norm = Self::normalize(path);
        self.upper
            .read()
            .unwrap()
            .chains
            .get(&norm)
            .map(|c| c.deltas.clone())
    }

    /// When the session's chain for `path` began and when its latest change
    /// was made, or `None` when the session has not changed it.
    pub fn times(&self, path: &str) -> Option<FileTimes> {
        let norm = Self::normalize(path);
        FileTimes::of(&self.upper.read().unwrap().chains.get(&norm)?.deltas)
    }

    /// Apply `deltas` — made against the file as this store holds it now — to
    /// `path`, in order, recorded as extending the path's chain exactly as the
    /// edits and writes that made them would have.
    ///
    /// Each delta keeps the moment it was made. Checked before anything
    /// changes: deltas that do not fit the file as it stands are refused as
    /// [`VfsError::Diverged`] and the store is left as it was.
    pub fn apply(&self, path: &str, deltas: &[TimedDelta]) -> Result<(), VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        if deltas.is_empty() {
            return Ok(());
        }
        loop {
            let seen = Seen::of(&self.upper.read().unwrap(), &norm);
            let view = self.view()?;
            let replayed = file_delta::replay_bytes(
                self.current_bytes(&view, &norm)?,
                deltas.iter().map(|t| &t.delta),
            );
            let result = replayed.map_err(|_| {
                VfsError::Diverged(format!(
                    "the changes do not fit {norm} as it stands — read it and build them again"
                ))
            })?;
            let in_lower = view.is_file(&norm);
            let mut guard = self.upper.write().unwrap();
            if !seen.stands(&guard, &norm) {
                continue;
            }
            let text = result.as_deref().and_then(|b| std::str::from_utf8(b).ok());
            let conflict = Self::still_conflicted(&guard, &norm, text);
            let mut chain = seen.chain.unwrap_or(Chain {
                deltas: Vec::new(),
                size: None,
                conflict: false,
            });
            chain.deltas.extend(deltas.iter().cloned());
            // A replace or a delete supersedes everything before it.
            if let Some(last) = chain.deltas.iter().rposition(|t| t.delta.supersedes()) {
                chain.deltas.drain(..last);
            }
            chain.size = result.as_ref().map(Vec::len);
            chain.conflict = conflict;
            // Deleting a file only the session made leaves nothing to record.
            if chain.size.is_none() && !in_lower {
                guard.chains.remove(&norm);
                return Ok(());
            }
            return Self::set_capped(&mut guard, norm, chain);
        }
    }

    /// A file's bytes, resolved through the overlay the same way
    /// [`Self::read`] is, for a caller that records content rather than
    /// showing it — `git_commit`'s `take`. Neither the UTF-8 requirement nor
    /// [`MAX_LOWER_FILE_BYTES`] applies, since nothing here reaches a tool
    /// response; every rule on which file a path may open does, because it
    /// reads through the same lower layer every other read does.
    pub fn read_bytes(&self, path: &str) -> Result<Option<Vec<u8>>, VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        let view = self.view()?;
        self.current_bytes(&view, &norm)
    }

    /// Replace the whole file with `bytes` — one [`FileDelta::Replace`] when
    /// they are text, one [`FileDelta::ReplaceBinary`] otherwise — as
    /// [`Self::write`] does. An upload is written this way (§10.2).
    pub fn write_bytes(&self, path: &str, bytes: Vec<u8>) -> Result<bool, VfsError> {
        let content = match String::from_utf8(bytes) {
            Ok(text) => return self.write(path, text),
            Err(e) => e.into_bytes(),
        };
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        let view = self.view()?;
        let created = !self.resolves_as_file(&view, &norm);
        let size = content.len();
        let mut guard = self.upper.write().unwrap();
        let chain = Chain {
            deltas: vec![TimedDelta::now(FileDelta::ReplaceBinary { content })],
            size: Some(size),
            conflict: false,
        };
        Self::set_capped(&mut guard, norm, chain)?;
        Ok(created)
    }
}
