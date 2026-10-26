//! Documents a body has changed and not yet committed.
//!
//! # In memory until it commits
//!
//! A working set is a body's uncommitted edits, held here and nowhere else. A
//! read falls through to the body's [`Shelf`] — the store of the operation its
//! mission is a step of, or the published mind — and a write touches nothing
//! else until [`Benches::commit`], which records every changed document into
//! that store in one go (`docs/replicated_substrate.md` §10.3). That is the
//! whole mechanism, and it is what makes "your work is yours alone until you
//! offer it" a fact about the program rather than a line in a description —
//! there is no path by which another body can observe a working set. Nothing
//! reaches the published mind from here: an operation's publish step merges
//! its store into the record.
//!
//! # A collision is a comparison, not a lock
//!
//! Every entry remembers what the document held when this working set first
//! touched it. Committing re-reads the shelf and requires it to still match:
//! there the operation's earlier steps are what moved the document. So two
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

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use doc_units::render::{around, pages, units, PAGE_LINES};
use doc_units::{apply, Doc, Edit, Language};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::mind::files::MindFiles;
use crate::mind::path::MindPath;
use crate::mind::Mind;
use crate::registry::yaml_edit;
use crate::sim::missing::no_document;
use crate::sim::shelf::Shelf;

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
    /// Whether committing this would actually alter what stands. A write that
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

    /// What this would write to `path`, named in any case: `None` when it
    /// leaves `path` as it stands or takes it out.
    pub fn changed_text(&self, path: &str) -> Option<&str> {
        self.files
            .iter()
            .find(|(key, c)| c.changed() && key.eq_ignore_ascii_case(path))?
            .1
            .now
            .as_deref()
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
    /// The mind whose documents these are. `None` in a world that has none.
    /// Never saved: a restored world keeps the mind this daemon was given.
    #[serde(skip)]
    mind: Option<Mind>,
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
    pub fn set_mind(&mut self, mind: Mind) {
        self.mind = Some(mind);
    }

    /// Take back the working sets `saved` held — what every body had open, set
    /// aside and last committed — keeping the root this daemon was given.
    ///
    /// **Uncommitted work outlives a reboot.** Only the missions were saved,
    /// so every restart threw away each Maker's working copy, and a Maker that
    /// had composed its piece and was about to commit it came back to an empty
    /// set and composed it all over again.
    pub fn restore(&mut self, saved: Benches) {
        self.open = saved.open;
        self.stashed = saved.stashed;
        self.last = saved.last;
    }

    /// Whether this world has documents at all.
    pub fn has_root(&self) -> bool {
        self.mind.is_some()
    }

    /// The mind this world's documents are, if it has any.
    pub fn mind(&self) -> Option<&Mind> {
        self.mind.as_ref()
    }

    /// This world's documents as they stand now, if it has any.
    pub fn files(&self) -> Option<Arc<MindFiles>> {
        self.mind.as_ref()?.files()
    }

    fn root(&self) -> Result<(), String> {
        self.mind
            .as_ref()
            .map(|_| ())
            .ok_or_else(|| "There are no documents in this world to work on.".to_string())
    }

    /// Parse a path naming an editable document: its key.
    fn doc(&self, raw: &str) -> Result<String, String> {
        self.root()?;
        let p = MindPath::parse(raw).map_err(|e| format!("{raw} is not a document here: {e}."))?;
        p.check_editable()
            .map_err(|e| format!("{raw} is not a document here: {e}."))?;
        if !editable_area(&p) {
            return Err(format!(
                "{raw} is not somewhere anybody works. What can be changed is {}.",
                EDITABLE_AREAS.join(", ")
            ));
        }
        Ok(p.as_str())
    }

    /// Parse a path naming a place to look, which has no extension: its key.
    fn dir(&self, raw: &str) -> Result<String, String> {
        self.root()?;
        let p = MindPath::parse(raw).map_err(|e| format!("{raw} is nowhere here: {e}."))?;
        if !editable_area(&p) {
            return Err(format!(
                "{raw} is not somewhere anybody works. What can be reached is {}.",
                EDITABLE_AREAS.join(", ")
            ));
        }
        Ok(p.as_str())
    }

    /// The refusal for `key`, not on `shelf`, naming what its folder holds.
    fn missing(&self, key: &str, shelf: &dyn Shelf) -> String {
        let folder = key.rsplit_once('/').map_or("", |(d, _)| d);
        no_document(key, &shelf.list(folder))
    }

    /// The working set for a body, opened on `about` if it has none.
    ///
    /// **Opening is implicit here on purpose.** The fiction wants `bench_branch`
    /// to be a deliberate act and it stays one, but refusing every write until
    /// somebody remembered to branch would spend a character's turn on a
    /// procedural scolding it cannot learn from — and the collision check does
    /// not depend on branching, because it compares against the shelf rather
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
    /// version when it has one, and what stands on its `shelf` otherwise.
    pub fn read(&self, body: &str, raw: &str, shelf: &dyn Shelf) -> Result<String, String> {
        let key = self.doc(raw)?;
        if let Some(c) = self.open.get(body).and_then(|w| w.files.get(&key)) {
            return match &c.now {
                Some(t) => Ok(t.clone()),
                None => Err(format!(
                    "You have taken {key} out. It goes when you commit."
                )),
            };
        }
        shelf.read(&key).ok_or_else(|| self.missing(&key, shelf))
    }

    /// Read one page of a document, its parts under their anchors
    /// (`docs/npcd_anchored_edits.md` §5).
    ///
    /// What the `file_read` act returns. [`Benches::read`] stays raw because an
    /// edit is applied to the document's own text, never to its rendering.
    pub fn excerpt(
        &self,
        body: &str,
        raw: &str,
        page: usize,
        shelf: &dyn Shelf,
    ) -> Result<String, String> {
        let key = self.doc(raw)?;
        let content = self.read(body, raw, shelf)?;
        Ok(anchored_excerpt(&key, &content, page))
    }

    /// Write a document whole.
    pub fn write(
        &mut self,
        body: &str,
        about: &str,
        raw: &str,
        text: &str,
        shelf: &dyn Shelf,
    ) -> Result<String, String> {
        let key = self.doc(raw)?;
        // Only read the shelf when this path is new to the working set —
        // re-reading it on every write would let a later read become the base
        // and quietly swallow somebody else's commit.
        let fresh = !self
            .open
            .get(body)
            .is_some_and(|w| w.files.contains_key(&key));
        let base = fresh.then(|| shelf.read(&key)).flatten();
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
        shelf: &dyn Shelf,
    ) -> Result<String, String> {
        let key = self.doc(raw)?;
        let existing = match self.read(body, raw, shelf) {
            Ok(t) => t,
            // Not there yet is not an error here: writing the first entry into
            // an era nobody has opened is how an era begins.
            Err(_) if shelf.read(&key).is_none() => String::new(),
            Err(why) => return Err(why),
        };
        let next = match existing.trim_end().is_empty() {
            true => format!("{}\n", text.trim_end()),
            false => format!("{}\n\n{}\n", existing.trim_end(), text.trim_end()),
        };
        self.write(body, about, &key, &next, shelf)
    }

    /// Make one anchored edit to a document as this body has it
    /// (`docs/npcd_anchored_edits.md` §6), and answer with what was done and
    /// the parts it wrote under their new anchors, between the parts either
    /// side of them.
    ///
    /// **An anchor the document no longer has is refused** — its part changed
    /// since it was read — so an edit is never applied to text the character
    /// did not see; the refusal says to read it again.
    pub fn edit(
        &mut self,
        body: &str,
        about: &str,
        raw: &str,
        edit: &Edit,
        shelf: &dyn Shelf,
    ) -> Result<String, String> {
        let key = self.doc(raw)?;
        let current = self.read(body, raw, shelf)?;
        let done = apply(&Doc::new(&key, &current), edit).map_err(|e| format!("In {key}, {e}."))?;
        self.write(body, about, raw, &done.doc.text(), shelf)?;
        Ok(format!(
            "{key}: {}\n{}",
            done.summary,
            around(&done.doc, &done.touched, done.at_line)
        ))
    }

    /// Read one field out of a YAML document.
    ///
    /// `at` is the path to it — `["portrait", "prompt"]` — because the fields
    /// worth reading are not all at the top level and a dotted string would
    /// have to be parsed back into exactly this.
    pub fn read_field(
        &self,
        body: &str,
        raw: &str,
        at: &[&str],
        shelf: &dyn Shelf,
    ) -> Result<String, String> {
        let text = self.read(body, raw, shelf)?;
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
        shelf: &dyn Shelf,
    ) -> Result<String, String> {
        let key = self.doc(raw)?;
        let text = self.read(body, raw, shelf)?;
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
        self.write(body, about, &key, &next, shelf)
    }

    /// Take a document out. It goes from the shelf when this is committed.
    pub fn remove(
        &mut self,
        body: &str,
        about: &str,
        raw: &str,
        shelf: &dyn Shelf,
    ) -> Result<String, String> {
        let key = self.doc(raw)?;
        let held = self
            .open
            .get(body)
            .and_then(|w| w.files.get(&key))
            .map(|c| c.now.is_some());
        let standing = shelf.read(&key);
        let there = match held {
            Some(present) => present,
            None => standing.is_some(),
        };
        if !there {
            return Err(self.missing(&key, shelf));
        }
        let fresh = held.is_none();
        let base = fresh.then_some(standing).flatten();
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
    pub fn list(&self, body: &str, raw: &str, shelf: &dyn Shelf) -> Result<Vec<String>, String> {
        let key = self.dir(raw)?;
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
        let mut names = shelf.list(&key);
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

    /// Record everything this body has changed onto its `shelf`, all at once.
    ///
    /// The check runs over every document before anything is recorded, so a
    /// collision on the last of six leaves the first five untouched and the
    /// working set intact — the character still has all of its work, and a
    /// person to talk to about it. A shelf that takes no commit — the
    /// published mind — refuses the lot, and the working set stays.
    pub fn commit(&mut self, body: &str, shelf: &dyn Shelf) -> Result<Vec<String>, String> {
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
            self.doc(key)?;
            // **Nobody collides with themselves.** A reviewer back from a
            // restart with its working copy restored was refused its commit
            // because the document had "moved under" it — and the hand named
            // was its own.
            let own = self.last.get(key).is_some_and(|who| who == body);
            if shelf.read(key) != c.base && !own {
                return Err(match self.last.get(key) {
                    Some(who) => format!(
                        "{key} has moved under you — {who} committed over the same ground while \
                         you were working."
                    ),
                    None => format!("{key} has changed since you opened it."),
                });
            }
        }

        let changes: Vec<(String, Option<String>)> = changed
            .iter()
            .map(|(key, c)| (key.clone(), c.now.clone()))
            .collect();
        shelf.commit(&changes)?;
        let written: Vec<String> = changed.into_iter().map(|(key, _)| key).collect();
        for key in &written {
            self.last.insert(key.clone(), body.to_string());
        }
        self.open.remove(body);
        Ok(written)
    }

    /// Take what stands on `shelf` now as the base of every open or set-aside
    /// change to `raw` — for the engine's own clerical rewrite of a document
    /// (`gates::tidy`), which is nobody's commit and must not read as one: a
    /// working copy opened before it would otherwise be refused at its commit
    /// for a change nobody made.
    pub fn rebase(&mut self, raw: &str, shelf: &dyn Shelf) {
        let Ok(key) = self.doc(raw) else {
            return;
        };
        let now = shelf.read(&key);
        for working in self.open.values_mut().chain(self.stashed.values_mut()) {
            if let Some(c) = working.files.get_mut(&key) {
                c.base = now.clone();
            }
        }
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
/// shelf that did not would be the weaker of two doors into one tree.
///
/// A path that is not there yet is not a link, which is the ordinary case for
/// creating a document.
///
/// On Windows the reparse-point attribute is checked as well as the portable
/// symlink type. A **junction** is the redirection an unprivileged process can
/// actually create there — `mklink /J` needs no Developer Mode where
/// `mklink` does — so testing only `is_symlink` would leave the reachable half
/// of the hazard unguarded on the platform this daemon runs on.
pub(crate) fn refuse_link(full: &Path, raw: &str) -> Result<(), String> {
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

/// Header, fence and anchored parts — the shape a read returns.
///
/// **A tool result lands in the conversation verbatim**, so a read is one
/// page: whole parts up to [`PAGE_LINES`] lines. One long `.yaml` would
/// otherwise fill a character's whole window and push out the turns that gave
/// it a reason to be reading. A page past the end reads the last one rather
/// than returning nothing, which a model reads as "the document is empty" and
/// acts on; the header and a closing line carry the continuation signal.
///
/// Each part shows under its anchor, which is what makes the *next* act
/// precise: an edit names the part by it, so the character never has to copy
/// the words it is changing to say where they are.
fn anchored_excerpt(path: &str, content: &str, page: usize) -> String {
    let doc = Doc::new(path, content);
    if doc.units().is_empty() {
        return format!("\n{path} (empty):\n\n```\n```\n");
    }
    let all = pages(&doc, PAGE_LINES);
    let page = page.clamp(1, all.len());
    let range = all[page - 1].clone();
    let n = doc.units().len();
    let parts = match doc.language() {
        Language::Markdown | Language::PlainText => "paragraphs",
        Language::Yaml => "lines",
        _ => "parts",
    };
    let header = match all.len() {
        1 if n == 1 => format!("{path} (1 {})", parts.trim_end_matches('s')),
        1 => format!("{path} ({n} {parts})"),
        total => format!(
            "{path} (page {page} of {total}: {parts} {}–{} of {n})",
            range.start + 1,
            range.end
        ),
    };
    let more = match page < all.len() {
        true => format!("There is more: the next read asks for page {}.\n", page + 1),
        false => String::new(),
    };
    let tag = match doc.language() {
        Language::Yaml => "yaml",
        Language::Markdown => "markdown",
        _ => "",
    };
    format!("\n{header}:\n\n```{tag}\n{}```\n{more}", units(&doc, range))
}

#[cfg(test)]
mod tests;
