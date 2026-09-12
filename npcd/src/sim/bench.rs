//! Documents a body has changed and not yet committed.
//!
//! # In memory until it commits
//!
//! A working set is a body's uncommitted edits, held here and nowhere else. A
//! read falls through to the disk; a write does not touch the disk at all until
//! [`Benches::commit`], which writes every changed document in one go. That is
//! the whole mechanism, and it is what makes "your work is yours alone until you
//! offer it" a fact about the program rather than a line in a description —
//! there is no path by which another body can observe a working set.
//!
//! # A collision is a comparison, not a lock
//!
//! Every entry remembers what the file held when this working set first touched
//! it. Committing re-reads the disk and requires it to still match. So two
//! Makers editing the same document do not deadlock and do not silently
//! overwrite each other: the second one to commit is refused, and told whose
//! work it ran into.
//!
//! Holding a lock instead would be worse in the way that matters. A lock makes
//! the second Maker wait, which is nothing to play; a refusal hands it a person
//! and a disagreement, which is the trigger condition the bench exists for.
//!
//! # Paths are [`MindPath`]s, and never anything else
//!
//! A path here arrives from a language model, which is an untrusted author in
//! exactly the way a URL is. `..`, a drive letter, a NUL, a symlink out of the
//! tree — all of it is refused by [`MindPath`], which was written for the same
//! attack surface on the console's editor. Nothing in this module takes a
//! `&Path` from outside, and the extension check keeps a working set to the
//! `.md` and `.yaml` documents the mind is actually made of.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::mind::path::MindPath;
use crate::registry::yaml_edit;

/// One document's pending change.
///
/// `base` and `now` are both `Option`, and absent means *the file is not there*
/// — which makes creating, changing and removing one shape instead of three.
/// A create is `base: None`, a removal is `now: None`, and the collision check
/// is one comparison for all of them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Change {
    /// What the document held when this working set first touched it.
    pub base: Option<String>,
    /// What it will hold once this is committed.
    pub now: Option<String>,
}

impl Change {
    /// Whether committing this would actually alter the disk. A write that
    /// restores a file to exactly what it already said is not a change, and
    /// counting it as one would manufacture collisions out of nothing.
    pub fn changed(&self) -> bool {
        self.base != self.now
    }

    /// How this reads in a diff.
    fn sign(&self) -> &'static str {
        match (&self.base, &self.now) {
            (None, Some(_)) => "new",
            (Some(_), None) => "gone",
            _ => "changed",
        }
    }
}

/// What one body has open at a bench.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Working {
    /// What it was opened on, in the character's own words.
    pub about: String,
    /// Pending changes by path. Ordered, so a diff reads the same twice.
    pub files: BTreeMap<String, Change>,
    /// Whether it has been offered for somebody to look at.
    pub offered: bool,
}

impl Working {
    /// The paths this would actually write, in order.
    pub fn changed(&self) -> Vec<&String> {
        self.files
            .iter()
            .filter(|(_, c)| c.changed())
            .map(|(p, _)| p)
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.changed().is_empty()
    }
}

/// Every bench in a world, and the documents under them.
///
/// A world with no root has no documents — the vault has one because the mind
/// is what its Makers write, and Battle Cities does not because there is
/// nothing out there to edit. That is the same rule the rest of [`crate::sim`]
/// runs on: a world instantiates only what it is, and an absent facet refuses
/// rather than pretending.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Benches {
    /// Where documents live. `None` in a world that has none.
    root: Option<PathBuf>,
    /// body → what it has open.
    open: BTreeMap<String, Working>,
    /// body → what it set aside.
    stashed: BTreeMap<String, Working>,
    /// path → who last committed it, so a collision names somebody.
    last: BTreeMap<String, String>,
}

impl Benches {
    pub fn new() -> Benches {
        Benches::default()
    }

    /// Point the benches at the documents they work on.
    pub fn set_root(&mut self, root: impl Into<PathBuf>) {
        self.root = Some(root.into());
    }

    /// Whether this world has documents at all.
    pub fn has_root(&self) -> bool {
        self.root.is_some()
    }

    fn root(&self) -> Result<&Path, String> {
        self.root
            .as_deref()
            .ok_or_else(|| "There are no documents in this world to work on.".to_string())
    }

    /// Parse and resolve a path naming an editable document.
    fn doc(&self, raw: &str) -> Result<(String, PathBuf), String> {
        let root = self.root()?;
        let p = MindPath::parse(raw).map_err(|e| format!("{raw} is not a document here: {e}."))?;
        p.check_editable()
            .map_err(|e| format!("{raw} is not a document here: {e}."))?;
        if !editable_area(&p) {
            return Err(format!(
                "{raw} is not somewhere anybody works. What can be changed is {}.",
                EDITABLE_AREAS.join(", ")
            ));
        }
        let full = p
            .resolve(root)
            .map_err(|e| format!("{raw} is not a document here: {e}."))?;
        refuse_link(&full, raw)?;
        Ok((p.as_str(), full))
    }

    /// Parse and resolve a path naming a place to look, which has no extension.
    fn dir(&self, raw: &str) -> Result<(String, PathBuf), String> {
        let root = self.root()?;
        let p = MindPath::parse(raw).map_err(|e| format!("{raw} is nowhere here: {e}."))?;
        if !editable_area(&p) {
            return Err(format!(
                "{raw} is not somewhere anybody works. What can be reached is {}.",
                EDITABLE_AREAS.join(", ")
            ));
        }
        let full = p
            .resolve(root)
            .map_err(|e| format!("{raw} is nowhere here: {e}."))?;
        refuse_link(&full, raw)?;
        Ok((p.as_str(), full))
    }

    /// The working set for a body, opened on `about` if it has none.
    ///
    /// **Opening is implicit here on purpose.** The fiction wants `bench_branch`
    /// to be a deliberate act and it stays one, but refusing every write until
    /// somebody remembered to branch would spend a character's turn on a
    /// procedural scolding it cannot learn from — and the collision check does
    /// not depend on branching, because it compares against the disk rather
    /// than against a claim.
    fn working(&mut self, body: &str, about: &str) -> &mut Working {
        self.open
            .entry(body.to_string())
            .or_insert_with(|| Working {
                about: about.to_string(),
                ..Working::default()
            })
    }

    /// What a body has open, if anything.
    pub fn opened(&self, body: &str) -> Option<&Working> {
        self.open.get(body)
    }

    /// Open a working set deliberately. `Ok(false)` when one was already open on
    /// the same thing, which is not a failure and not a second set.
    pub fn open_on(&mut self, body: &str, about: &str) -> Result<bool, String> {
        if let Some(w) = self.open.get(body) {
            if w.about.eq_ignore_ascii_case(about) {
                return Ok(false);
            }
            return Err(format!(
                "You still have {} open. Commit it, set it aside, or throw it away before you \
                 start something else.",
                w.about
            ));
        }
        self.open.insert(
            body.to_string(),
            Working {
                about: about.to_string(),
                ..Working::default()
            },
        );
        Ok(true)
    }

    // ── the documents ───────────────────────────────────────────────────────

    /// Read a document as this body currently has it — its own uncommitted
    /// version when it has one, and what stands otherwise.
    pub fn read(&self, body: &str, raw: &str) -> Result<String, String> {
        let (key, full) = self.doc(raw)?;
        if let Some(c) = self.open.get(body).and_then(|w| w.files.get(&key)) {
            return match &c.now {
                Some(t) => Ok(t.clone()),
                None => Err(format!(
                    "You have taken {key} out. It goes when you commit."
                )),
            };
        }
        std::fs::read_to_string(&full).map_err(|_| format!("There is no document at {key}."))
    }

    /// Read a document as a numbered excerpt, capped at [`MAX_READ_LINES`].
    ///
    /// What the `file_read` act returns. [`Benches::read`] stays raw because an
    /// edit matches against the document's own text, and matching against a
    /// rendering with line numbers stuck to the front of every line would never
    /// find anything.
    pub fn excerpt(&self, body: &str, raw: &str, start_line: usize) -> Result<String, String> {
        let (key, _) = self.doc(raw)?;
        let content = self.read(body, raw)?;
        Ok(numbered_excerpt(&key, &content, start_line))
    }

    /// Write a document whole.
    pub fn write(
        &mut self,
        body: &str,
        about: &str,
        raw: &str,
        text: &str,
    ) -> Result<String, String> {
        let (key, full) = self.doc(raw)?;
        // Only touch the disk when this path is new to the working set —
        // re-reading it on every write would let a later read become the base
        // and quietly swallow somebody else's commit.
        let fresh = !self
            .open
            .get(body)
            .is_some_and(|w| w.files.contains_key(&key));
        let base = fresh.then(|| on_disk(&full)).flatten();
        let w = self.working(body, about);
        match w.files.get_mut(&key) {
            Some(c) => c.now = Some(text.to_string()),
            None => {
                w.files.insert(
                    key.clone(),
                    Change {
                        base,
                        now: Some(text.to_string()),
                    },
                );
            }
        }
        Ok(key)
    }

    /// Add to the end of a document, making it if it is not there.
    ///
    /// **A record is added to, not replaced.** This is what the station acts
    /// do — an entry goes into an era, a draft goes into a silence — and it is
    /// the same argument [`crate::sim::record::Record::write`] makes for
    /// appending: a tool that silently overwrote a colleague's paragraph would
    /// be indistinguishable from one that worked.
    pub fn append(
        &mut self,
        body: &str,
        about: &str,
        raw: &str,
        text: &str,
    ) -> Result<String, String> {
        let (key, full) = self.doc(raw)?;
        let existing = match self.read(body, raw) {
            Ok(t) => t,
            // Not there yet is not an error here: writing the first entry into
            // an era nobody has opened is how an era begins.
            Err(_) if !full.exists() => String::new(),
            Err(why) => return Err(why),
        };
        let next = match existing.trim_end().is_empty() {
            true => format!("{}\n", text.trim_end()),
            false => format!("{}\n\n{}\n", existing.trim_end(), text.trim_end()),
        };
        self.write(body, about, &key, &next)
    }

    /// Replace one span of a document, where it appears exactly once.
    ///
    /// The uniqueness rule is what makes an edit safe without reading the whole
    /// document back: an ambiguous target is refused rather than applied to
    /// whichever of three the scan happened to reach first.
    pub fn edit(
        &mut self,
        body: &str,
        about: &str,
        raw: &str,
        old: &str,
        new: &str,
    ) -> Result<String, String> {
        if old.is_empty() {
            return Err("Replacing nothing would put your text everywhere at once.".into());
        }
        let (key, _) = self.doc(raw)?;
        let current = self.read(body, raw)?;
        match current.matches(old).count() {
            0 => Err(format!(
                "That does not appear in {key}. Read it back — what you are replacing has to be \
                 what is actually there."
            )),
            1 => {
                let next = current.replacen(old, new, 1);
                self.write(body, about, raw, &next)
            }
            n => Err(format!(
                "That appears {n} times in {key}, so changing it would change the wrong one. Give \
                 more of the text around it."
            )),
        }
    }

    /// Read one field out of a YAML document.
    ///
    /// `at` is the path to it — `["portrait", "prompt"]` — because the fields
    /// worth reading are not all at the top level and a dotted string would
    /// have to be parsed back into exactly this.
    pub fn read_field(&self, body: &str, raw: &str, at: &[&str]) -> Result<String, String> {
        let text = self.read(body, raw)?;
        let doc: Value = serde_yaml::from_str(&text)
            .map_err(|e| format!("{raw} is not a document this can read: {e}"))?;
        let mut here = &doc;
        for key in at {
            here = here
                .get(key)
                .ok_or_else(|| format!("{raw} has no {}.", at.join(" ")))?;
        }
        match here {
            Value::String(s) => Ok(s.clone()),
            other => Ok(other.to_string()),
        }
    }

    /// Change one field of a YAML document, leaving every other byte alone.
    ///
    /// **Through [`yaml_edit::splice`], never a round trip.** An authored
    /// document carries its reasoning in its comments — why a mood reads the
    /// way it does, why a personality's biography is not in the file — and
    /// `serde_yaml` cannot see a comment, so parsing and re-serialising turns a
    /// commented block-scalar document into a flat list of quoted strings. It
    /// still loads perfectly, and half of what a person wrote is gone. The
    /// splice edits a concrete syntax tree instead: the one value that changed
    /// is replaced and every other byte is carried through.
    ///
    /// A document the splice declines is *refused* rather than written whole,
    /// which is the whole point of asking it.
    pub fn write_field(
        &mut self,
        body: &str,
        about: &str,
        raw: &str,
        at: &[&str],
        value: &str,
    ) -> Result<String, String> {
        let (key, _) = self.doc(raw)?;
        let text = self.read(body, raw)?;
        let mut doc: Map<String, Value> = serde_yaml::from_str(&text)
            .map_err(|e| format!("{key} is not a document this can change: {e}"))?;

        let (last, parents) = at.split_last().ok_or("Nothing was named to change.")?;
        let mut here = &mut doc;
        for step in parents {
            let slot = here
                .entry(step.to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            here = slot
                .as_object_mut()
                .ok_or_else(|| format!("{key} has a {step} that is not a block."))?;
        }
        here.insert(last.to_string(), Value::String(value.to_string()));

        let next = yaml_edit::splice(&text, &doc).ok_or_else(|| {
            format!(
                "{key} could not be changed without rewriting it, and rewriting it would lose \
                 what somebody wrote around the parts that matter."
            )
        })?;
        self.write(body, about, &key, &next)
    }

    /// Take a document out. It goes from the disk when this is committed.
    pub fn remove(&mut self, body: &str, about: &str, raw: &str) -> Result<String, String> {
        let (key, full) = self.doc(raw)?;
        let held = self
            .open
            .get(body)
            .and_then(|w| w.files.get(&key))
            .map(|c| c.now.is_some());
        let there = match held {
            Some(present) => present,
            None => full.exists(),
        };
        if !there {
            return Err(format!("There is no document at {key}."));
        }
        let fresh = held.is_none();
        let base = fresh.then(|| on_disk(&full)).flatten();
        let w = self.working(body, about);
        match w.files.get_mut(&key) {
            Some(c) => c.now = None,
            None => {
                w.files.insert(key.clone(), Change { base, now: None });
            }
        }
        Ok(key)
    }

    /// What documents are in a place, as this body has them — including ones it
    /// has made and not committed, and without ones it has taken out.
    pub fn list(&self, body: &str, raw: &str) -> Result<Vec<String>, String> {
        let (key, full) = self.dir(raw)?;
        let prefix = match key.is_empty() {
            true => String::new(),
            false => format!("{key}/"),
        };
        // At the root, only the areas that can actually be worked in. Listing
        // the rest would advertise `projection.yaml` and then refuse every act
        // that reached for it, which teaches a character nothing except that
        // the world is arbitrary.
        if key.is_empty() {
            return Ok(EDITABLE_AREAS.iter().map(|a| format!("{a}/")).collect());
        }
        let mut names: BTreeSet<String> = BTreeSet::new();
        if let Ok(entries) = std::fs::read_dir(&full) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                match e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    true => names.insert(format!("{name}/")),
                    false => names.insert(name),
                };
            }
        }
        if let Some(w) = self.open.get(body) {
            for (path, c) in &w.files {
                let Some(rest) = path.strip_prefix(&prefix) else {
                    continue;
                };
                // Only this directory's own entries; anything deeper belongs to
                // a listing of the directory it is actually in.
                if rest.is_empty() || rest.contains('/') {
                    continue;
                }
                match &c.now {
                    Some(_) => {
                        names.insert(rest.to_string());
                    }
                    None => {
                        names.remove(rest);
                    }
                }
            }
        }
        Ok(names.into_iter().collect())
    }

    // ── the working loop ────────────────────────────────────────────────────

    /// What this body has changed and not committed, in the order it reads.
    pub fn diff(&self, body: &str) -> Vec<String> {
        let Some(w) = self.open.get(body) else {
            return Vec::new();
        };
        w.files
            .iter()
            .filter(|(_, c)| c.changed())
            .map(|(p, c)| format!("{p} ({})", c.sign()))
            .collect()
    }

    /// Set the working set aside, whole. `false` when there was nothing to set
    /// aside — which is a refusal upstream, not an error here.
    pub fn stash(&mut self, body: &str) -> bool {
        match self.open.remove(body) {
            Some(mut w) => {
                w.offered = false;
                self.stashed.insert(body.to_string(), w);
                true
            }
            None => false,
        }
    }

    /// Pick a set-aside working set back up. `Err` when picking it up would
    /// bury work that is already open.
    pub fn pop(&mut self, body: &str) -> Result<String, String> {
        let Some(w) = self.stashed.get(body) else {
            return Err("You have nothing set aside.".into());
        };
        if let Some(open) = self.open.get(body) {
            if !open.is_empty() {
                return Err(format!(
                    "You have changes to {} open. Picking your set-aside work up now would bury \
                     them.",
                    open.about
                ));
            }
        }
        let about = w.about.clone();
        let w = self.stashed.remove(body).expect("just read");
        self.open.insert(body.to_string(), w);
        Ok(about)
    }

    /// Throw the working set away. `false` when there was nothing to throw.
    pub fn discard(&mut self, body: &str) -> bool {
        self.open.remove(body).is_some()
    }

    /// Offer the work, or take the offer back.
    pub fn set_offered(&mut self, body: &str, offered: bool) -> bool {
        match self.open.get_mut(body) {
            Some(w) => {
                w.offered = offered;
                true
            }
            None => false,
        }
    }

    /// Write everything this body has changed, all at once.
    ///
    /// The check runs over every document before anything is written, so a
    /// collision on the last of six leaves the first five untouched and the
    /// working set intact — the character still has all of its work, and a
    /// person to talk to about it.
    pub fn commit(&mut self, body: &str) -> Result<Vec<String>, String> {
        let Some(w) = self.open.get(body) else {
            return Err("You have nothing open.".into());
        };
        let changed: Vec<(String, Change)> = w
            .files
            .iter()
            .filter(|(_, c)| c.changed())
            .map(|(p, c)| (p.clone(), c.clone()))
            .collect();
        if changed.is_empty() {
            // Committing nothing still closes the bench. A body that opened
            // work, changed no document and committed is finished with it, and
            // leaving the set open would refuse its next `bench_branch`.
            self.open.remove(body);
            return Ok(Vec::new());
        }

        for (key, c) in &changed {
            let (_, full) = self.doc(key)?;
            if on_disk(&full) != c.base {
                return Err(match self.last.get(key) {
                    Some(who) => format!(
                        "{key} has moved under you — {who} committed over the same ground while \
                         you were working."
                    ),
                    None => format!("{key} has changed since you opened it."),
                });
            }
        }

        let mut written = Vec::with_capacity(changed.len());
        for (key, c) in &changed {
            let (_, full) = self.doc(key)?;
            match &c.now {
                Some(text) => put(&full, text)?,
                None => {
                    std::fs::remove_file(&full).map_err(|e| format!("{key} would not go: {e}"))?
                }
            }
            self.last.insert(key.clone(), body.to_string());
            written.push(key.clone());
        }
        self.open.remove(body);
        Ok(written)
    }

    /// Who last committed a document, when anybody has.
    pub fn last_hand(&self, path: &str) -> Option<&str> {
        self.last.get(path).map(String::as_str)
    }
}

/// Refuse a path that is a link rather than a document.
///
/// **Containment is not enough on its own.** [`MindPath::resolve`] canonicalises,
/// so a link pointing *out* of the root is already caught — but one pointing at
/// another document *inside* it resolves cleanly and would silently redirect a
/// read to a file the character did not name. The console's editor
/// ([`crate::mind::doc`]) refuses links outright for the same reason, and a
/// bench that did not would be the weaker of two doors into one tree.
///
/// A path that is not there yet is not a link, which is the ordinary case for
/// creating a document.
///
/// On Windows the reparse-point attribute is checked as well as the portable
/// symlink type. A **junction** is the redirection an unprivileged process can
/// actually create there — `mklink /J` needs no Developer Mode where
/// `mklink` does — so testing only `is_symlink` would leave the reachable half
/// of the hazard unguarded on the platform this daemon runs on.
fn refuse_link(full: &Path, raw: &str) -> Result<(), String> {
    let Ok(meta) = full.symlink_metadata() else {
        return Ok(());
    };
    let redirected = meta.file_type().is_symlink() || is_reparse_point(&meta);
    match redirected {
        true => Err(format!("{raw} is a link, not a document.")),
        false => Ok(()),
    }
}

/// Whether the filesystem will send this path somewhere else.
#[cfg(windows)]
fn is_reparse_point(meta: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
    meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn is_reparse_point(_meta: &std::fs::Metadata) -> bool {
    false
}

/// The parts of the mind a bench may reach.
///
/// **An allow-list, because the mind is also this daemon's own configuration.**
/// `projection.yaml` is the schema every character is assembled from,
/// `mind.yaml` names the mind itself, and `schema/` holds the validators. A bad
/// edit to any of those is not a bad document somebody can see and revert — it
/// is a daemon that will not come back up, which cannot be fixed from inside
/// the world that broke it.
///
/// Everything named here is content: the canon, the cast, the craft libraries,
/// and the buildings. Adding an area is one line; the guard is
/// [`MindPath::area`], which is the first path segment and cannot be spoofed
/// because the path is parsed into segments before anything sees it.
pub const EDITABLE_AREAS: &[&str] = &[
    "layers",
    "map",
    "moods",
    "personalities",
    "responses",
    "worlds",
];

/// Whether an area may be edited at all.
fn editable_area(p: &MindPath) -> bool {
    match p.area() {
        // The root itself, which is a listing rather than a document.
        None => true,
        Some(a) => EDITABLE_AREAS.contains(&a),
    }
}

/// The most lines one read returns.
///
/// **A tool result lands in the conversation verbatim**, so an uncapped read is
/// a context hazard: one long `.yaml` would fill a character's whole window and
/// push out the turns that gave it a reason to be reading. The cap applies
/// whether or not a start was named, so it cannot be walked around by asking
/// for a wide range, and the header carries the continuation signal.
pub const MAX_READ_LINES: usize = 200;

/// Header, fence and `cat -n` body — the shape an editing tool returns.
///
/// Numbered because the number is what makes the *next* act precise: an edit
/// names text, and a character that can see which line it is on picks a target
/// that appears once instead of one that appears three times.
fn numbered_excerpt(path: &str, content: &str, start_line: usize) -> String {
    // Split on '\n' rather than `lines()`: a trailing newline must not shift the
    // numbering, and the phantom last element it leaves is dropped by count.
    let all: Vec<&str> = content.split('\n').collect();
    let total = match all.last() {
        Some(&"") => all.len().saturating_sub(1),
        _ => all.len(),
    };
    if total == 0 {
        return format!("\n{path} (empty):\n\n```\n```\n");
    }

    // A start past the end reads the last line rather than returning nothing,
    // which a model reads as "the file is empty" and acts on.
    let start = start_line.clamp(1, total);
    let end = (start + MAX_READ_LINES - 1).min(total);
    let width = end.to_string().len();
    let mut numbered = String::with_capacity(content.len() + 8);
    for (idx, line) in all[start - 1..end].iter().enumerate() {
        let no = start + idx;
        numbered.push_str(&format!("{no:width$}  {line}\n"));
    }

    let range = match end >= total {
        true => format!("lines {start}-{end}"),
        false => format!("lines {start}-{end} of {total}"),
    };
    let tag = match path.rsplit_once('.') {
        Some((_, "yaml")) => "yaml",
        Some((_, "md")) => "markdown",
        _ => "",
    };
    format!("\n{path} ({range}):\n\n```{tag}\n{numbered}```\n")
}

/// What a file holds, or nothing when it is not there.
///
/// A file that exists and cannot be read is `None` on purpose: it is not the
/// base this working set was built on either way, so the commit check refuses
/// rather than writing over something it could not see.
fn on_disk(full: &Path) -> Option<String> {
    std::fs::read_to_string(full).ok()
}

/// Write a document, making its directory if it is new.
///
/// Through a temporary and a rename, because the obvious spelling truncates
/// before it writes: a machine that dies in that window leaves the document
/// gone rather than merely unchanged, and on NTFS it can leave one of the right
/// length full of NULs. A rename either happened or did not.
fn put(full: &Path, text: &str) -> Result<(), String> {
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("{} would not open: {e}", parent.display()))?;
    }
    let tmp = full.with_extension("npcd-writing");
    std::fs::write(&tmp, text).map_err(|e| format!("{} would not write: {e}", full.display()))?;
    std::fs::rename(&tmp, full).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{} would not land: {e}", full.display())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A root with two documents in it, at a path unique to this test.
    fn rooted(name: &str) -> (Benches, PathBuf) {
        let root = std::env::temp_dir().join(format!("npcd-bench-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("layers/eras")).unwrap();
        std::fs::write(
            root.join("layers/eras/third.md"),
            "the third era\nburned in the spring\n",
        )
        .unwrap();
        std::fs::write(root.join("layers/eras/fourth.md"), "the fourth era\n").unwrap();
        let mut b = Benches::new();
        b.set_root(&root);
        (b, root)
    }

    #[test]
    fn a_world_with_no_root_has_no_documents_and_says_so() {
        let mut b = Benches::new();
        assert!(!b.has_root());
        let err = b.read("m1", "layers/eras/third.md").unwrap_err();
        assert!(err.contains("no documents"), "{err}");
        assert!(b.write("m1", "x", "layers/eras/third.md", "…").is_err());
        assert!(b.list("m1", "layers/eras").is_err());
    }

    #[test]
    fn a_read_falls_through_to_the_disk() {
        let (b, _) = rooted("read");
        let got = b.read("m1", "layers/eras/third.md").unwrap();
        assert!(got.starts_with("the third era"), "{got}");
    }

    /// The whole point of the module: a write is invisible on the disk and
    /// visible to its own author.
    #[test]
    fn a_write_stays_in_memory_until_it_is_committed() {
        let (mut b, root) = rooted("memory");
        b.write("m1", "the third era", "layers/eras/third.md", "rewritten\n")
            .unwrap();

        assert_eq!(b.read("m1", "layers/eras/third.md").unwrap(), "rewritten\n");
        assert_eq!(
            std::fs::read_to_string(root.join("layers/eras/third.md")).unwrap(),
            "the third era\nburned in the spring\n",
            "the disk moved before the commit"
        );
        // And invisible to anybody else, which is what makes it *yours*.
        assert!(b
            .read("m2", "layers/eras/third.md")
            .unwrap()
            .starts_with("the third era"));

        b.commit("m1").unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join("layers/eras/third.md")).unwrap(),
            "rewritten\n"
        );
    }

    #[test]
    fn committing_clears_the_working_set_and_names_what_it_wrote() {
        let (mut b, _) = rooted("clears");
        b.write("m1", "layers/eras", "layers/eras/third.md", "a\n")
            .unwrap();
        b.write("m1", "layers/eras", "layers/eras/fourth.md", "b\n")
            .unwrap();
        let written = b.commit("m1").unwrap();
        assert_eq!(
            written,
            vec!["layers/eras/fourth.md", "layers/eras/third.md"]
        );
        assert!(b.opened("m1").is_none());
        assert!(b.diff("m1").is_empty());
    }

    #[test]
    fn a_new_document_is_created_with_its_directory() {
        let (mut b, root) = rooted("create");
        b.write(
            "m1",
            "a silence",
            "layers/stories/deep/new.md",
            "a night at the gate\n",
        )
        .unwrap();
        b.commit("m1").unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join("layers/stories/deep/new.md")).unwrap(),
            "a night at the gate\n"
        );
    }

    #[test]
    fn removing_a_document_takes_it_off_the_disk_at_the_commit() {
        let (mut b, root) = rooted("remove");
        b.remove("m1", "layers/eras", "layers/eras/fourth.md")
            .unwrap();
        assert!(
            root.join("layers/eras/fourth.md").exists(),
            "gone before the commit"
        );
        assert!(b
            .read("m1", "layers/eras/fourth.md")
            .unwrap_err()
            .contains("taken"));
        b.commit("m1").unwrap();
        assert!(!root.join("layers/eras/fourth.md").exists());
    }

    #[test]
    fn removing_something_that_is_not_there_is_refused() {
        let (mut b, _) = rooted("remove-absent");
        assert!(b.remove("m1", "x", "layers/eras/never.md").is_err());
    }

    // ── the uniqueness rule ─────────────────────────────────────────────────

    #[test]
    fn an_edit_replaces_the_one_place_its_target_appears() {
        let (mut b, _) = rooted("edit-one");
        b.edit(
            "m1",
            "layers/eras",
            "layers/eras/third.md",
            "spring",
            "autumn",
        )
        .unwrap();
        assert_eq!(
            b.read("m1", "layers/eras/third.md").unwrap(),
            "the third era\nburned in the autumn\n"
        );
    }

    /// An ambiguous edit is the one that silently does the wrong thing, so it
    /// is refused and says how many it found.
    #[test]
    fn an_edit_whose_target_appears_twice_is_refused_and_counts_them() {
        let (mut b, root) = rooted("edit-two");
        std::fs::write(root.join("layers/eras/third.md"), "a fire\nand a fire\n").unwrap();
        let err = b
            .edit(
                "m1",
                "layers/eras",
                "layers/eras/third.md",
                "a fire",
                "a flood",
            )
            .unwrap_err();
        assert!(err.contains('2'), "{err}");
        assert!(
            b.opened("m1").is_none_or(|w| w.is_empty()),
            "a refused edit still landed"
        );
    }

    #[test]
    fn an_edit_whose_target_is_not_there_is_refused() {
        let (mut b, _) = rooted("edit-none");
        let err = b
            .edit(
                "m1",
                "layers/eras",
                "layers/eras/third.md",
                "a flood",
                "a fire",
            )
            .unwrap_err();
        assert!(err.contains("does not appear"), "{err}");
    }

    #[test]
    fn replacing_the_empty_string_is_refused() {
        let (mut b, _) = rooted("edit-empty");
        assert!(b
            .edit("m1", "layers/eras", "layers/eras/third.md", "", "x")
            .is_err());
    }

    /// Edits compound in memory: the second reads the first's result, not the
    /// disk's.
    #[test]
    fn a_second_edit_sees_the_first() {
        let (mut b, _) = rooted("edit-compound");
        b.edit(
            "m1",
            "layers/eras",
            "layers/eras/third.md",
            "third",
            "thirteenth",
        )
        .unwrap();
        b.edit(
            "m1",
            "layers/eras",
            "layers/eras/third.md",
            "spring",
            "autumn",
        )
        .unwrap();
        assert_eq!(
            b.read("m1", "layers/eras/third.md").unwrap(),
            "the thirteenth era\nburned in the autumn\n"
        );
    }

    // ── collisions ──────────────────────────────────────────────────────────

    /// The act somebody else's work can refuse, and the refusal names them.
    #[test]
    fn a_commit_over_somebody_elses_commit_is_refused_and_names_them() {
        let (mut b, _) = rooted("collide");
        b.write("m1", "layers/eras", "layers/eras/third.md", "mine\n")
            .unwrap();
        b.write("m2", "layers/eras", "layers/eras/third.md", "mine too\n")
            .unwrap();

        assert_eq!(b.commit("m1").unwrap(), vec!["layers/eras/third.md"]);
        let err = b.commit("m2").unwrap_err();
        assert!(err.contains("m1"), "the other party was not named: {err}");
        assert!(
            b.opened("m2").is_some(),
            "a refused commit threw the work away"
        );
    }

    /// A refusal on one document leaves the others alone — the character still
    /// has every change it made.
    #[test]
    fn a_collision_writes_none_of_the_batch() {
        let (mut b, root) = rooted("collide-batch");
        b.write("m1", "layers/eras", "layers/eras/fourth.md", "safe\n")
            .unwrap();
        b.write("m1", "layers/eras", "layers/eras/third.md", "contested\n")
            .unwrap();
        // Somebody else moves one of the two under it.
        std::fs::write(root.join("layers/eras/third.md"), "theirs\n").unwrap();

        assert!(b.commit("m1").is_err());
        assert_eq!(
            std::fs::read_to_string(root.join("layers/eras/fourth.md")).unwrap(),
            "the fourth era\n",
            "the uncontested half was written anyway"
        );
        assert_eq!(b.diff("m1").len(), 2, "the working set did not survive");
    }

    /// Two bodies on different documents are not each other's business.
    #[test]
    fn two_bodies_on_different_documents_both_commit() {
        let (mut b, _) = rooted("parallel");
        b.write("m1", "layers/eras", "layers/eras/third.md", "a\n")
            .unwrap();
        b.write("m2", "layers/eras", "layers/eras/fourth.md", "b\n")
            .unwrap();
        assert!(b.commit("m1").is_ok());
        assert!(b.commit("m2").is_ok(), "an unrelated document collided");
    }

    /// A write that says exactly what the file already said is not a change,
    /// and must not manufacture a collision for somebody else.
    #[test]
    fn writing_what_is_already_there_changes_nothing() {
        let (mut b, _) = rooted("noop");
        b.write(
            "m1",
            "layers/eras",
            "layers/eras/fourth.md",
            "the fourth era\n",
        )
        .unwrap();
        assert!(b.diff("m1").is_empty());
        assert!(b.commit("m1").unwrap().is_empty());
    }

    // ── the working loop ────────────────────────────────────────────────────

    #[test]
    fn setting_aside_keeps_the_work_and_picking_it_up_restores_it() {
        let (mut b, _) = rooted("stash");
        b.write("m1", "the third era", "layers/eras/third.md", "half done\n")
            .unwrap();
        assert!(b.stash("m1"));
        assert!(b.diff("m1").is_empty(), "set-aside work is still open");
        assert!(b
            .read("m1", "layers/eras/third.md")
            .unwrap()
            .starts_with("the third era"));

        assert_eq!(b.pop("m1").unwrap(), "the third era");
        assert_eq!(b.read("m1", "layers/eras/third.md").unwrap(), "half done\n");
    }

    #[test]
    fn picking_up_set_aside_work_over_live_changes_is_refused() {
        let (mut b, _) = rooted("stash-bury");
        b.write("m1", "the third era", "layers/eras/third.md", "first\n")
            .unwrap();
        b.stash("m1");
        b.write("m1", "the fourth era", "layers/eras/fourth.md", "second\n")
            .unwrap();
        let err = b.pop("m1").unwrap_err();
        assert!(err.contains("bury"), "{err}");
        assert_eq!(b.read("m1", "layers/eras/fourth.md").unwrap(), "second\n");
    }

    #[test]
    fn throwing_the_work_away_puts_the_document_back() {
        let (mut b, _) = rooted("discard");
        b.write(
            "m1",
            "layers/eras",
            "layers/eras/third.md",
            "wrong from the start\n",
        )
        .unwrap();
        assert!(b.discard("m1"));
        assert!(!b.discard("m1"), "threw away twice");
        assert!(b
            .read("m1", "layers/eras/third.md")
            .unwrap()
            .starts_with("the third era"));
    }

    #[test]
    fn opening_a_second_thing_while_one_is_open_is_refused() {
        let (mut b, _) = rooted("open-twice");
        assert!(b.open_on("m1", "the third era").unwrap());
        assert!(
            !b.open_on("m1", "the third era").unwrap(),
            "opened it twice"
        );
        let err = b.open_on("m1", "the fourth era").unwrap_err();
        assert!(err.contains("the third era"), "{err}");
    }

    #[test]
    fn offering_and_withdrawing_only_work_on_something_open() {
        let (mut b, _) = rooted("offer");
        assert!(!b.set_offered("m1", true));
        b.open_on("m1", "the third era").unwrap();
        assert!(b.set_offered("m1", true));
        assert!(b.opened("m1").unwrap().offered);
        assert!(b.set_offered("m1", false));
        assert!(!b.opened("m1").unwrap().offered);
    }

    #[test]
    fn a_diff_names_what_kind_of_change_each_one_is() {
        let (mut b, _) = rooted("diff");
        b.write("m1", "layers/eras", "layers/eras/third.md", "changed\n")
            .unwrap();
        b.write("m1", "layers/eras", "layers/eras/new.md", "made\n")
            .unwrap();
        b.remove("m1", "layers/eras", "layers/eras/fourth.md")
            .unwrap();
        assert_eq!(
            b.diff("m1"),
            vec![
                "layers/eras/fourth.md (gone)",
                "layers/eras/new.md (new)",
                "layers/eras/third.md (changed)",
            ]
        );
    }

    // ── listing ─────────────────────────────────────────────────────────────

    #[test]
    fn a_listing_shows_the_disk_plus_your_own_uncommitted_work() {
        let (mut b, _) = rooted("list");
        assert_eq!(
            b.list("m1", "layers/eras").unwrap(),
            vec!["fourth.md", "third.md"]
        );
        // The root lists the areas that can be worked in, not the mind's own
        // configuration sitting beside them.
        assert_eq!(
            b.list("m1", "").unwrap(),
            EDITABLE_AREAS
                .iter()
                .map(|a| format!("{a}/"))
                .collect::<Vec<_>>()
        );

        b.write("m1", "layers/eras", "layers/eras/fifth.md", "new\n")
            .unwrap();
        b.remove("m1", "layers/eras", "layers/eras/fourth.md")
            .unwrap();
        assert_eq!(
            b.list("m1", "layers/eras").unwrap(),
            vec!["fifth.md", "third.md"]
        );
        // …and nobody else's view moved.
        assert_eq!(
            b.list("m2", "layers/eras").unwrap(),
            vec!["fourth.md", "third.md"]
        );
    }

    /// A listing of one directory does not leak the ones below it.
    #[test]
    fn a_listing_stops_at_its_own_directory() {
        let (mut b, _) = rooted("list-depth");
        b.write("m1", "x", "layers/eras/deep/buried.md", "…")
            .unwrap();
        assert!(!b
            .list("m1", "layers/eras")
            .unwrap()
            .contains(&"buried.md".to_string()));
        assert_eq!(b.list("m1", "layers/eras/deep").unwrap(), vec!["buried.md"]);
    }

    // ── the path guard ──────────────────────────────────────────────────────

    /// The paths here are written by a language model, so every spelling of
    /// "leave the directory" has to be refused on the way in.
    #[test]
    fn a_path_that_leaves_the_root_is_refused_by_every_act() {
        let (mut b, root) = rooted("escape");
        for bad in [
            "../secrets.md",
            "layers/eras/../../secrets.md",
            "c:/windows/system.md",
            r"..\..\secrets.md",
            "layers/eras/a\0b.md",
        ] {
            assert!(b.read("m1", bad).is_err(), "read {bad}");
            assert!(b.write("m1", "x", bad, "owned").is_err(), "write {bad}");
            assert!(b.remove("m1", "x", bad).is_err(), "remove {bad}");
        }
        assert!(!root.parent().unwrap().join("secrets.md").exists());
    }

    /// The mind holds `.md` and `.yaml`. Anything else is not a document a
    /// bench writes, and letting one through would put an executable into a
    /// directory the daemon reads.
    #[test]
    fn only_text_documents_can_be_written() {
        let (mut b, _) = rooted("ext");
        for bad in ["layers/eras/x.exe", "layers/eras/x.png", "layers/eras/x"] {
            assert!(b.write("m1", "x", bad, "…").is_err(), "{bad}");
        }
        assert!(b.write("m1", "x", "layers/eras/x.yaml", "a: 1\n").is_ok());
    }

    // ── links ───────────────────────────────────────────────────────────────

    /// Make a symlink, or `false` where the OS will not allow one.
    ///
    /// Windows needs Developer Mode or an elevated process, so the tests below
    /// say out loud when the link half did not run. A silent skip that reads as
    /// a pass is worse than no test at all.
    fn try_symlink(target: &Path, link: &Path) -> bool {
        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_file(target, link).is_ok()
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, link).is_ok()
        }
    }

    /// Make a directory link, by the one route Windows allows unprivileged.
    ///
    /// `mklink /J` makes a junction without Developer Mode, where every symlink
    /// API needs it. Shelling out in a test is worth it here because this is
    /// the *only* redirection reachable on the platform the daemon runs on, and
    /// a guard whose reachable case is untested is a guard nobody has checked.
    fn try_dir_link(target: &Path, link: &Path) -> bool {
        #[cfg(windows)]
        {
            if std::os::windows::fs::symlink_dir(target, link).is_ok() {
                return true;
            }
            std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(link)
                .arg(target)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, link).is_ok()
        }
    }

    /// **The case that actually runs on Windows.** A junction redirects a whole
    /// directory, so a listing walked through one would report documents from
    /// somewhere the character never named.
    #[test]
    fn a_directory_link_is_refused_rather_than_walked() {
        let (b, root) = rooted("junction");
        if !try_dir_link(&root.join("layers/eras"), &root.join("layers/alias")) {
            eprintln!("SKIPPED: this OS would not create a directory link");
            return;
        }
        let err = b.list("m1", "layers/alias").unwrap_err();
        assert!(
            err.contains("link"),
            "a listing walked through a junction: {err}"
        );
        // The real directory is still readable — the guard is about the link,
        // not about the documents behind it.
        assert_eq!(
            b.list("m1", "layers/eras").unwrap(),
            vec!["fourth.md", "third.md"]
        );
    }

    /// A link pointing **out** of the root is caught by containment — resolving
    /// it lands outside and the path is refused.
    #[test]
    fn a_link_out_of_the_root_cannot_be_read_through() {
        let (b, root) = rooted("link-out");
        let outside = root
            .parent()
            .unwrap()
            .join(format!("npcd-bench-outside-{}.md", std::process::id()));
        std::fs::write(&outside, "not yours\n").unwrap();
        if !try_symlink(&outside, &root.join("layers/eras/out.md")) {
            eprintln!("SKIPPED the link half: this OS would not create a symlink");
            return;
        }
        let got = b.read("m1", "layers/eras/out.md");
        assert!(got.is_err(), "read straight through a link out of the root");
        assert!(
            !got.unwrap_or_default().contains("not yours"),
            "content from outside the root came back"
        );
    }

    /// A link pointing **inside** the root resolves cleanly, so containment
    /// says nothing about it — and it would still redirect a read to a document
    /// the character did not name. This is the case the leaf check is for.
    #[test]
    fn a_link_inside_the_root_is_refused_rather_than_followed() {
        let (mut b, root) = rooted("link-in");
        if !try_symlink(
            &root.join("layers/eras/third.md"),
            &root.join("layers/eras/alias.md"),
        ) {
            eprintln!("SKIPPED the link half: this OS would not create a symlink");
            return;
        }
        let err = b.read("m1", "layers/eras/alias.md").unwrap_err();
        assert!(err.contains("link"), "{err}");
        assert!(b.write("m1", "x", "layers/eras/alias.md", "…").is_err());
        assert!(b.remove("m1", "x", "layers/eras/alias.md").is_err());
        // …and the document it pointed at is untouched.
        assert!(b
            .read("m1", "layers/eras/third.md")
            .unwrap()
            .starts_with("the third era"));
    }

    // ── the read cap ────────────────────────────────────────────────────────

    /// A document of `n` numbered lines, for exercising the cap.
    fn long_document(root: &Path, lines: usize) {
        let body: String = (1..=lines).map(|i| format!("line {i}\n")).collect();
        std::fs::write(root.join("layers/eras/long.md"), body).unwrap();
    }

    /// **An uncapped read is a context hazard.** A tool result lands in the
    /// conversation verbatim, so one long document would fill a character's
    /// whole window and push out the turns that gave it a reason to read.
    #[test]
    fn a_read_returns_at_most_the_capped_number_of_lines() {
        let (b, root) = rooted("cap");
        long_document(&root, 900);
        let out = b.excerpt("m1", "layers/eras/long.md", 1).unwrap();
        assert!(out.contains("(lines 1-200 of 900)"), "{out}");
        assert!(out.contains("  line 1\n"), "{out}");
        assert!(out.contains("line 200"), "{out}");
        assert!(!out.contains("line 201"), "the cap did not hold");
    }

    /// The header names the absolute range, so the next read asks for the next
    /// line directly rather than having to work it out.
    #[test]
    fn a_read_continues_from_the_line_it_is_given() {
        let (b, root) = rooted("cap-continue");
        long_document(&root, 900);
        let out = b.excerpt("m1", "layers/eras/long.md", 201).unwrap();
        assert!(out.contains("(lines 201-400 of 900)"), "{out}");
        assert!(out.contains("line 201"), "{out}");
        assert!(!out.contains("  line 200\n"), "{out}");
    }

    /// The last page drops the `of n`, which is how a reader knows it has the
    /// end and not another 200 waiting.
    #[test]
    fn the_last_page_says_it_is_the_last() {
        let (b, root) = rooted("cap-last");
        long_document(&root, 250);
        let out = b.excerpt("m1", "layers/eras/long.md", 201).unwrap();
        assert!(out.contains("(lines 201-250)"), "{out}");
        assert!(!out.contains(" of 250"), "{out}");
    }

    /// A start past the end reads the last line rather than coming back empty,
    /// which a model reads as "the document is empty" and then acts on.
    #[test]
    fn a_start_past_the_end_reads_the_end() {
        let (b, root) = rooted("cap-past");
        long_document(&root, 10);
        let out = b.excerpt("m1", "layers/eras/long.md", 9_999).unwrap();
        assert!(out.contains("line 10"), "{out}");
    }

    #[test]
    fn a_short_document_comes_back_whole_and_numbered() {
        let (b, _) = rooted("excerpt");
        let out = b.excerpt("m1", "layers/eras/third.md", 1).unwrap();
        assert_eq!(
            out,
            "\nlayers/eras/third.md (lines 1-2):\n\n```markdown\n1  the third era\n2  burned in the spring\n```\n"
        );
    }

    #[test]
    fn an_empty_document_says_so_rather_than_naming_a_range() {
        let (b, root) = rooted("excerpt-empty");
        std::fs::write(root.join("layers/eras/blank.md"), "").unwrap();
        let out = b.excerpt("m1", "layers/eras/blank.md", 1).unwrap();
        assert!(out.contains("(empty)"), "{out}");
        assert!(
            !out.contains("lines 1-0"),
            "a range that reads as a bug: {out}"
        );
    }

    /// The excerpt shows the reader's own uncommitted work, like a plain read.
    #[test]
    fn an_excerpt_shows_your_own_changes() {
        let (mut b, _) = rooted("excerpt-mine");
        b.write("m1", "x", "layers/eras/third.md", "one\ntwo\nthree\n")
            .unwrap();
        let mine = b.excerpt("m1", "layers/eras/third.md", 1).unwrap();
        assert!(mine.contains("3  three"), "{mine}");
        let theirs = b.excerpt("m2", "layers/eras/third.md", 1).unwrap();
        assert!(theirs.contains("burned in the spring"), "{theirs}");
    }

    /// The numbering is right-aligned to the widest number in the range, which
    /// is what keeps the text column straight across a page boundary.
    #[test]
    fn line_numbers_are_aligned_to_the_widest_in_the_range() {
        let (b, root) = rooted("align");
        long_document(&root, 120);
        let out = b.excerpt("m1", "layers/eras/long.md", 1).unwrap();
        assert!(out.contains("\n  1  line 1\n"), "{out}");
        assert!(out.contains("\n120  line 120\n"), "{out}");
    }

    /// An edit matches the document, not its rendering — numbering the text and
    /// then searching it would never find anything.
    #[test]
    fn an_edit_matches_the_document_and_not_its_numbering() {
        let (mut b, _) = rooted("raw");
        b.edit(
            "m1",
            "x",
            "layers/eras/third.md",
            "the third era",
            "the fourth era",
        )
        .unwrap();
        assert!(b
            .read("m1", "layers/eras/third.md")
            .unwrap()
            .starts_with("the fourth era"));
    }

    // ── isolation ───────────────────────────────────────────────────────────

    /// **No uncommitted change is visible to anybody but the body that made
    /// it.** Every mutating act, run by two bodies over the same documents at
    /// once, with each one's view asserted after every step.
    ///
    /// The workspace is keyed by body and every entry point takes one, so this
    /// holds by construction — but "by construction" is exactly the claim that
    /// stops being true the first time somebody adds a method and forgets the
    /// argument, and the failure would be one character reading another's
    /// half-finished work as though it were canon.
    #[test]
    fn nothing_uncommitted_leaks_between_bodies() {
        let (mut b, root) = rooted("isolation");
        std::fs::write(root.join("layers/eras/shared.md"), "as it stands\n").unwrap();

        // Both open work on the same thing, and both change the same document.
        assert!(b.open_on("m1", "the third era").unwrap());
        assert!(b.open_on("m2", "the third era").unwrap());
        b.write(
            "m1",
            "the third era",
            "layers/eras/shared.md",
            "m1's version\n",
        )
        .unwrap();
        b.write(
            "m2",
            "the third era",
            "layers/eras/shared.md",
            "m2's version\n",
        )
        .unwrap();

        assert_eq!(
            b.read("m1", "layers/eras/shared.md").unwrap(),
            "m1's version\n"
        );
        assert_eq!(
            b.read("m2", "layers/eras/shared.md").unwrap(),
            "m2's version\n"
        );
        assert_eq!(
            b.read("m3", "layers/eras/shared.md").unwrap(),
            "as it stands\n",
            "an uninvolved body saw somebody's draft"
        );

        // An edit compounds only on its own author's version.
        b.edit(
            "m1",
            "x",
            "layers/eras/shared.md",
            "m1's",
            "m1 has edited its",
        )
        .unwrap();
        assert_eq!(
            b.read("m2", "layers/eras/shared.md").unwrap(),
            "m2's version\n"
        );

        // A creation is invisible, and so is a removal.
        b.write("m1", "x", "layers/eras/mine.md", "only m1 has this\n")
            .unwrap();
        b.remove("m2", "x", "layers/eras/fourth.md").unwrap();
        assert!(
            b.read("m2", "layers/eras/mine.md").is_err(),
            "a new document leaked"
        );
        assert!(
            b.read("m1", "layers/eras/fourth.md").is_ok(),
            "a removal leaked"
        );
        assert!(b
            .list("m2", "layers/eras")
            .unwrap()
            .iter()
            .all(|n| n != "mine.md"));
        assert!(b
            .list("m1", "layers/eras")
            .unwrap()
            .iter()
            .any(|n| n == "fourth.md"));

        // Diffs, excerpts and offers are each their own body's.
        assert!(b.diff("m1").iter().any(|d| d.contains("mine.md")));
        assert!(b.diff("m2").iter().all(|d| !d.contains("mine.md")));
        assert!(b
            .excerpt("m1", "layers/eras/shared.md", 1)
            .unwrap()
            .contains("m1 has edited"));
        assert!(b
            .excerpt("m2", "layers/eras/shared.md", 1)
            .unwrap()
            .contains("m2's version"));
        b.set_offered("m1", true);
        assert!(b.opened("m1").unwrap().offered);
        assert!(!b.opened("m2").unwrap().offered, "an offer leaked");

        // Setting aside empties only one bench, and picking it up fills only it.
        b.stash("m1");
        assert!(b.diff("m1").is_empty());
        assert!(!b.diff("m2").is_empty(), "the other bench was cleared too");
        b.pop("m1").unwrap();
        assert!(b.diff("m1").iter().any(|d| d.contains("mine.md")));

        // Throwing one away leaves the other standing.
        b.discard("m2");
        assert!(b.diff("m2").is_empty());
        assert!(
            !b.diff("m1").is_empty(),
            "one body's discard took another's work"
        );

        // And a commit publishes exactly one body's changes.
        b.commit("m1").unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join("layers/eras/shared.md")).unwrap(),
            "m1 has edited its version\n"
        );
        assert!(root.join("layers/eras/mine.md").exists());
        assert!(
            root.join("layers/eras/fourth.md").exists(),
            "a discarded removal reached the disk"
        );
    }

    /// The same, for the YAML field acts — they go through the working set too,
    /// so a mood somebody is part-way through changing is not what anybody else
    /// reads.
    #[test]
    fn a_half_changed_field_is_not_what_anybody_else_reads() {
        let (mut b, root) = rooted("isolation-field");
        std::fs::create_dir_all(root.join("moods")).unwrap();
        std::fs::write(
            root.join("moods/undone.yaml"),
            "id: undone\n# why it reads this way\ndescription: As it stands.\n",
        )
        .unwrap();

        b.write_field(
            "m1",
            "x",
            "moods/undone.yaml",
            &["description"],
            "m1's wording.",
        )
        .unwrap();
        assert_eq!(
            b.read_field("m1", "moods/undone.yaml", &["description"])
                .unwrap(),
            "m1's wording."
        );
        assert_eq!(
            b.read_field("m2", "moods/undone.yaml", &["description"])
                .unwrap(),
            "As it stands.",
            "an uncommitted field change leaked"
        );
        assert!(std::fs::read_to_string(root.join("moods/undone.yaml"))
            .unwrap()
            .contains("As it stands."));
    }

    // ── determinism ─────────────────────────────────────────────────────────

    /// The property the tool tests rest on: the same acts leave the same state.
    #[test]
    fn the_same_acts_leave_byte_identical_state() {
        let build = |name: &str| {
            let (mut b, _) = rooted(name);
            b.open_on("m1", "the third era").unwrap();
            b.write("m1", "the third era", "layers/eras/third.md", "a\n")
                .unwrap();
            b.write("m1", "the third era", "layers/eras/new.md", "b\n")
                .unwrap();
            b.remove("m1", "the third era", "layers/eras/fourth.md")
                .unwrap();
            b.set_offered("m1", true);
            b
        };
        let (one, two) = (build("det-a"), build("det-b"));
        // The roots differ by design; everything the acts produced does not.
        assert_eq!(one.open, two.open);
        assert_eq!(one.diff("m1"), two.diff("m1"));
        assert_eq!(
            serde_json::to_string(&one.open).unwrap(),
            serde_json::to_string(&two.open).unwrap()
        );
    }
}
