//! What the world is made of, as the Makers hold it.
//!
//! # One store, not six
//!
//! An era's pages, a story's gap, a portrait's plate, a place's entry and a
//! character's sheet are different things in the fiction and the *same* thing to
//! every tool that touches them: something with a name, held by somebody or not,
//! in a state, with a body of text and some judgements standing against it.
//!
//! Six near-identical stores would have meant six near-identical bugs. What
//! actually differs between a chronicle entry and a portrait is which station
//! can reach it — and that is the map's business, declared on the part, not a
//! shape in here.
//!
//! # A state machine, because the tools are its transitions
//!
//! Nearly every station act is a move between states:
//!
//! ```text
//!   Unwritten ──claim──▶ Held ──draft──▶ Draft ──offer──▶ Offered ──file──▶ Filed
//!                          ▲               │                 │                │
//!                          └───────────────┴─────────────────┘         retire/let go
//!                                     (withdraw)                              ▼
//!                                                                          Retired
//! ```
//!
//! Holding that shape here rather than in each tool is what stops a story being
//! filed twice, a portrait being drawn over somebody else's claim, or an entry
//! being retired while it is still being argued about.
//!
//! # Deterministic
//!
//! Ordered by id throughout, no clock, no randomness — so what a station offers
//! is the same on two runs and a test can assert the exact set.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What a document calls itself: its first `# ` heading.
///
/// Read a line at a time and stopped at the first one that answers, so
/// indexing a directory does not pull every document into memory to learn its
/// title — the heading is on the first line of every one of them.
fn heading_of(file: &Path) -> Option<String> {
    let f = std::fs::File::open(file).ok()?;
    for line in BufReader::new(f).lines().map_while(Result::ok).take(20) {
        if let Some(h) = line.strip_prefix("# ") {
            let h = h.trim();
            if !h.is_empty() {
                return Some(h.to_string());
            }
        }
    }
    None
}

/// What sort of thing it is. Decides nothing mechanical — the state does that —
/// and everything about which station reaches it and how it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A named span of the record. Bound by a chronicle terminal.
    Era,
    /// One dated thing inside an era.
    Entry,
    /// A silence in the record — somewhere a story should be and is not.
    Gap,
    /// A piece of writing filling a gap.
    Story,
    /// Somebody the world contains.
    Character,
    /// A likeness of one.
    Portrait,
    /// Somewhere the world contains.
    Place,
    /// A question from outside the vault.
    Enquiry,
    /// A piece laid out as its scenes.
    Structure,
    /// Something that arrived and has to be taken charge of.
    Accession,
}

impl Kind {
    /// What this sort of thing *is*, in the words an index entry uses.
    ///
    /// The way in for somebody who does not yet know what they are looking for
    /// — which is what an index is for, and what `record_tidy_index` writes
    /// when it brings one back to what it actually indexes.
    pub fn describes(self) -> &'static str {
        match self {
            Kind::Era => "a named span of the record",
            Kind::Entry => "one dated thing inside an era",
            Kind::Gap => "a silence where a story should be",
            Kind::Story => "a piece of writing filling a gap",
            Kind::Character => "somebody the world contains",
            Kind::Portrait => "a likeness of somebody",
            Kind::Place => "somewhere the world contains",
            Kind::Enquiry => "a question from outside the vault",
            Kind::Structure => "a piece laid out as its scenes",
            Kind::Accession => "something taken in, to be taken charge of",
        }
    }
}

impl Kind {
    pub fn noun(&self) -> &'static str {
        match self {
            Kind::Era => "era",
            Kind::Entry => "entry",
            Kind::Gap => "gap",
            Kind::Story => "story",
            Kind::Character => "character",
            Kind::Portrait => "portrait",
            Kind::Place => "place",
            Kind::Enquiry => "enquiry",
            Kind::Structure => "piece",
            Kind::Accession => "intake",
        }
    }
}

/// Where a thing has got to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// Named and nothing more — a gap on the ledger, a face nobody has drawn.
    #[default]
    Unwritten,
    /// Somebody has taken it on. Nobody else can.
    Held,
    /// Made, not yet offered to anybody.
    Draft,
    /// Put up to be judged.
    Offered,
    /// Part of the record.
    Filed,
    /// Was part of the record and is not any more, deliberately.
    Retired,
}

/// How well a thing has survived, which is a judgement rather than a fact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Condition {
    #[default]
    Sound,
    /// True when it was written and quietly wrong now.
    Stale,
    /// Mended, and the mend is visible — a repair passed off as an original is
    /// worse than the damage.
    Mended,
}

/// One thing in the record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Item {
    pub id: String,
    /// What a character calls it.
    pub name: String,
    pub kind: Kind,
    pub state: State,
    pub condition: Condition,
    /// Who holds it, when somebody does.
    pub holder: Option<String>,
    /// What it says. Empty until somebody writes it.
    pub body: String,
    /// Where it came from, written before anybody touches it — a thing whose
    /// origin was never written down cannot be trusted afterwards.
    pub provenance: Option<String>,
    /// The way in, for somebody who does not yet know what they are looking for.
    pub description: Option<String>,
    /// What it points at, and what points at it.
    pub refers_to: Vec<String>,
    /// The reason it was let go, kept so somebody can disagree later.
    pub let_go_because: Option<String>,
    /// Which era, story or place it belongs under.
    pub within: Option<String>,
    /// Where it lives in the mind, for the things that are documents.
    ///
    /// **This is what makes an act real.** An item with a path is a file: what
    /// is written into it goes through the bench's working set and reaches the
    /// disk at the commit, with the collision check and the history that come
    /// with that. An item without one — an order, an appraisal, a thing that is
    /// a judgement rather than a document — stays in this store, which is where
    /// it belongs.
    pub path: Option<String>,
}

impl Item {
    pub fn new(id: impl Into<String>, name: impl Into<String>, kind: Kind) -> Item {
        Item {
            id: id.into(),
            name: name.into(),
            kind,
            state: State::Unwritten,
            condition: Condition::Sound,
            holder: None,
            body: String::new(),
            provenance: None,
            description: None,
            refers_to: Vec::new(),
            let_go_because: None,
            within: None,
            path: None,
        }
    }

    /// Name the document this is.
    pub fn at_path(mut self, path: impl Into<String>) -> Item {
        self.path = Some(path.into());
        self
    }

    pub fn within(mut self, of: impl Into<String>) -> Item {
        self.within = Some(of.into());
        self
    }

    pub fn in_state(mut self, s: State) -> Item {
        self.state = s;
        self
    }

    /// Whether somebody other than `who` is holding it.
    pub fn held_by_other(&self, who: &str) -> bool {
        self.holder.as_deref().is_some_and(|h| h != who)
    }
}

/// Everything the world is made of.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    items: BTreeMap<String, Item>,
}

impl Record {
    pub fn new() -> Record {
        Record::default()
    }

    pub fn put(&mut self, item: Item) {
        self.items.insert(item.id.clone(), item);
    }

    pub fn get(&self, id: &str) -> Option<&Item> {
        self.items.get(id)
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Item> {
        self.items.values()
    }

    /// Find by the name a character would use, or by id.
    pub fn by_name(&self, name: &str) -> Option<&Item> {
        let want = name.trim().to_lowercase();
        self.items
            .values()
            .find(|i| i.name.to_lowercase() == want || i.id == want)
    }

    fn by_name_mut(&mut self, name: &str) -> Option<&mut Item> {
        let want = name.trim().to_lowercase();
        self.items
            .values_mut()
            .find(|i| i.name.to_lowercase() == want || i.id == want)
    }

    /// The names of everything of a kind, in a stable order.
    pub fn names_of(&self, kind: Kind) -> Vec<String> {
        self.items
            .values()
            .filter(|i| i.kind == kind)
            .map(|i| i.name.clone())
            .collect()
    }

    /// Of a kind, and in a state.
    pub fn names_in(&self, kind: Kind, state: State) -> Vec<String> {
        self.items
            .values()
            .filter(|i| i.kind == kind && i.state == state)
            .map(|i| i.name.clone())
            .collect()
    }

    /// What `who` is holding, of a kind.
    pub fn held_of(&self, who: &str, kind: Kind) -> Vec<String> {
        self.items
            .values()
            .filter(|i| i.kind == kind && i.holder.as_deref() == Some(who))
            .map(|i| i.name.clone())
            .collect()
    }

    /// Everything `who` is holding, whatever it is.
    pub fn held_by(&self, who: &str) -> Vec<String> {
        self.items
            .values()
            .filter(|i| i.holder.as_deref() == Some(who))
            .map(|i| i.name.clone())
            .collect()
    }

    /// What may be taken on: nothing already held, of a kind.
    pub fn claimable_of(&self, kind: Kind) -> Vec<String> {
        self.items
            .values()
            .filter(|i| i.kind == kind && i.holder.is_none() && i.state != State::Retired)
            .map(|i| i.name.clone())
            .collect()
    }

    /// Where a thing lives, when it is a document.
    pub fn path_of(&self, name: &str) -> Option<String> {
        self.by_name(name)?.path.clone()
    }

    /// Give a document-shaped thing somewhere to live, and say where.
    ///
    /// **What makes the first entry into an era possible.** An era nobody has
    /// written yet is a name with no file; writing into it is what creates the
    /// file, so the path is settled at the moment of the first write rather
    /// than by somebody remembering to make a document first. Only the kinds
    /// that *are* documents get one — an order or an appraisal is a judgement
    /// and has nowhere to be.
    pub fn settle_path(&mut self, name: &str) -> Option<String> {
        let i = self.by_name_mut(name)?;
        if let Some(p) = &i.path {
            return Some(p.clone());
        }
        let dir = match i.kind {
            Kind::Era => "layers/eras",
            Kind::Story | Kind::Gap => "layers/stories",
            _ => return None,
        };
        let slug: String = i
            .name
            .to_lowercase()
            .chars()
            .map(|c| match c.is_ascii_alphanumeric() {
                true => c,
                false => '-',
            })
            .collect();
        let slug = slug.trim_matches('-').replace("--", "-");
        let path = format!("{dir}/{slug}.md");
        i.path = Some(path.clone());
        Some(path)
    }

    /// Read the canon off the disk and put it in the record.
    ///
    /// **The join between the stations and the mind.** Before this, the record
    /// held fixtures — `the third era` was an object in memory with nothing
    /// behind it, so every act at a chronicle terminal reported success and
    /// changed nothing that survived a restart. Now an era is the document it
    /// is, and writing into it writes into that file.
    ///
    /// A document names itself: the first `# ` heading is what a character
    /// calls it, falling back to the file's own stem. Reading the heading
    /// rather than deriving a name from the filename means the two cannot
    /// drift — the thing a Maker asks for is the title printed on it.
    ///
    /// Called whenever the world learns where the mind is, and idempotent: a
    /// document already indexed keeps the state and custody it has, because
    /// re-indexing must not release work somebody is holding.
    pub fn index_canon(&mut self, root: &Path) {
        for (dir, kind) in [("layers/eras", Kind::Era), ("layers/stories", Kind::Story)] {
            let Ok(entries) = std::fs::read_dir(root.join(dir)) else {
                continue;
            };
            // Sorted, because a directory's order is the filesystem's and the
            // record has to be the same on two machines for a test to assert it.
            let mut files: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "md"))
                .collect();
            files.sort();
            for file in files {
                let Some(stem) = file.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
                    continue;
                };
                let path = format!("{dir}/{stem}.md");
                if self.items.values().any(|i| i.path.as_deref() == Some(&path)) {
                    continue;
                }
                let name = heading_of(&file).unwrap_or_else(|| stem.replace(['-', '_'], " "));
                // A thing the world already names *becomes* this document
                // rather than gaining a twin beside it. Two items called the
                // same thing — one real and one not — is exactly the trap this
                // join exists to close: a Maker would reach for whichever the
                // lookup happened to find first.
                if let Some(existing) = self.by_name_mut(&name) {
                    existing.path = Some(path);
                    continue;
                }
                // **Namespaced by the document's own location.** An id built
                // from the kind and the stem alone collided with the ids the
                // world is seeded with — `era_third` was both the fixture and
                // `layers/eras/third.md` — and `put` inserts by id, so indexing
                // silently *replaced* the thing everybody names with one named
                // after a filename. The failure looked like the era had never
                // existed.
                let id = format!("doc_{}_{stem}", dir.replace('/', "_"));
                let item = Item::new(id, name, kind).in_state(State::Filed).at_path(path);
                self.put(item);
            }
        }
    }

    // ── transitions ─────────────────────────────────────────────────────────
    //
    // Each returns the second-person reason it did not happen, because a
    // character is going to read it.

    /// Take something on, and hold it against everybody else.
    pub fn take(&mut self, name: &str, who: &str) -> Result<String, String> {
        let Some(i) = self.by_name_mut(name) else {
            return Err(format!("There is nothing called {name} here."));
        };
        if i.held_by_other(who) {
            return Err(format!(
                "{} is holding {}.",
                i.holder.clone().unwrap_or_default(),
                i.name
            ));
        }
        if i.state == State::Retired {
            return Err(format!("{} was retired. It is not yours to take.", i.name));
        }
        i.holder = Some(who.to_string());
        if i.state == State::Unwritten {
            i.state = State::Held;
        }
        Ok(i.name.clone())
    }

    /// Check the custody rules for a write that lands in a document, and take
    /// it.
    ///
    /// The same gates [`Record::write`] applies, minus the writing — the text
    /// is going to the bench's working set instead of into `body`. The one
    /// difference is what *filed* means for a document: a filed thing is not
    /// written over casually, but it is not immutable either, so it refuses
    /// until the writer has opened it. That is the bench loop, said by the
    /// store: branch it, change it, commit it.
    pub fn claim_for_write(&mut self, name: &str, who: &str) -> Result<String, String> {
        let Some(i) = self.by_name_mut(name) else {
            return Err(format!("There is nothing called {name} here."));
        };
        if i.held_by_other(who) {
            return Err(format!(
                "{} is holding {} — you cannot write into it.",
                i.holder.clone().unwrap_or_default(),
                i.name
            ));
        }
        if i.state == State::Retired {
            return Err(format!("{} was retired. It is not yours to write.", i.name));
        }
        if i.state == State::Filed && i.holder.as_deref() != Some(who) {
            return Err(format!(
                "{} is part of the record. Open it first, so everybody else can see you are in \
                 it.",
                i.name
            ));
        }
        i.holder = Some(who.to_string());
        if i.state == State::Unwritten {
            i.state = State::Held;
        }
        Ok(i.name.clone())
    }

    /// Put it back for somebody else.
    pub fn give_back(&mut self, name: &str, who: &str) -> Result<String, String> {
        let Some(i) = self.by_name_mut(name) else {
            return Err(format!("There is nothing called {name} here."));
        };
        if i.holder.as_deref() != Some(who) {
            return Err(format!("You are not holding {}.", i.name));
        }
        i.holder = None;
        if i.state == State::Held {
            i.state = State::Unwritten;
        }
        Ok(i.name.clone())
    }

    /// Write into something you hold. Appends rather than replaces: a record is
    /// added to, and a tool that silently overwrote a colleague's paragraph
    /// would be indistinguishable from one that worked.
    pub fn write(&mut self, name: &str, who: &str, text: &str) -> Result<String, String> {
        let Some(i) = self.by_name_mut(name) else {
            return Err(format!("There is nothing called {name} here."));
        };
        if i.held_by_other(who) {
            return Err(format!(
                "{} is holding {} — you cannot write into it.",
                i.holder.clone().unwrap_or_default(),
                i.name
            ));
        }
        if i.state == State::Filed {
            return Err(format!(
                "{} is filed. Change it by making the change and warning whoever it breaks, not \
                 by writing over it.",
                i.name
            ));
        }
        if i.holder.is_none() {
            i.holder = Some(who.to_string());
        }
        if !i.body.is_empty() {
            i.body.push_str("\n\n");
        }
        i.body.push_str(text);
        i.state = State::Draft;
        Ok(i.name.clone())
    }

    /// Move something to a state, when the move is one the shape allows.
    pub fn set_state(&mut self, name: &str, who: &str, to: State) -> Result<String, String> {
        let Some(i) = self.by_name_mut(name) else {
            return Err(format!("There is nothing called {name} here."));
        };
        if i.held_by_other(who) {
            return Err(format!(
                "{} is holding {}.",
                i.holder.clone().unwrap_or_default(),
                i.name
            ));
        }
        let ok = matches!(
            (i.state, to),
            (State::Draft, State::Offered)
                | (State::Offered, State::Draft)
                | (State::Offered, State::Filed)
                | (State::Draft, State::Filed)
                | (State::Filed, State::Retired)
                | (State::Held, State::Draft)
        );
        if !ok {
            return Err(format!(
                "{} is {:?}. That is not a thing you can do to it from there.",
                i.name, i.state
            ));
        }
        i.state = to;
        if to == State::Filed || to == State::Retired {
            i.holder = None;
        }
        Ok(i.name.clone())
    }

    /// Attach the way in.
    pub fn describe(&mut self, name: &str, how: &str) -> Result<String, String> {
        let Some(i) = self.by_name_mut(name) else {
            return Err(format!("There is nothing called {name} here."));
        };
        i.description = Some(how.to_string());
        Ok(i.name.clone())
    }

    /// Write down where it came from, before anybody touches it.
    pub fn set_provenance(&mut self, name: &str, from: &str) -> Result<String, String> {
        let Some(i) = self.by_name_mut(name) else {
            return Err(format!("There is nothing called {name} here."));
        };
        i.provenance = Some(from.to_string());
        Ok(i.name.clone())
    }

    /// Point one part of the record at another that bears on it.
    pub fn cross_reference(&mut self, name: &str, to: &str) -> Result<String, String> {
        if self.by_name(to).is_none() {
            return Err(format!("There is nothing called {to} to point at."));
        }
        let target = self.by_name(to).map(|i| i.name.clone()).unwrap_or_default();
        let Some(i) = self.by_name_mut(name) else {
            return Err(format!("There is nothing called {name} here."));
        };
        if !i.refers_to.contains(&target) {
            i.refers_to.push(target.clone());
        }
        Ok(target)
    }

    /// Judge how well it has survived.
    pub fn set_condition(&mut self, name: &str, c: Condition) -> Result<String, String> {
        let Some(i) = self.by_name_mut(name) else {
            return Err(format!("There is nothing called {name} here."));
        };
        i.condition = c;
        Ok(i.name.clone())
    }

    /// Refuse to keep something, and say why clearly enough to be argued with.
    pub fn let_go(&mut self, name: &str, because: &str) -> Result<String, String> {
        if because.trim().is_empty() {
            return Err("A thing let go without a reason cannot be argued with later.".into());
        }
        let Some(i) = self.by_name_mut(name) else {
            return Err(format!("There is nothing called {name} here."));
        };
        i.state = State::Retired;
        i.let_go_because = Some(because.to_string());
        i.holder = None;
        Ok(i.name.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> Record {
        let mut r = Record::new();
        r.put(Item::new("gap_third", "the third silence", Kind::Gap));
        r.put(Item::new("era_third", "the third era", Kind::Era).in_state(State::Filed));
        r.put(Item::new("face_wren", "a face nobody has drawn", Kind::Portrait));
        r
    }

    // ── indexing the canon off the disk ─────────────────────────────────────

    /// A mind with eras and stories in it, at a path unique to this test.
    fn canon(name: &str) -> PathBuf {
        let root = std::env::temp_dir()
            .join(format!("npcd-canon-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("layers/eras")).unwrap();
        std::fs::create_dir_all(root.join("layers/stories")).unwrap();
        std::fs::write(
            root.join("layers/eras/the-third-era.md"),
            "# the third era\n\nIt burned.\n",
        )
        .unwrap();
        std::fs::write(root.join("layers/eras/fourth.md"), "no heading here\n").unwrap();
        std::fs::write(
            root.join("layers/stories/gate.md"),
            "# A Night At The Gate\n\nNobody came through.\n",
        )
        .unwrap();
        // Not a document, and must not become one.
        std::fs::write(root.join("layers/eras/.gitkeep"), "why this exists\n").unwrap();
        root
    }

    #[test]
    fn both_layers_are_indexed_and_carry_their_paths() {
        let root = canon("both");
        let mut r = Record::new();
        r.index_canon(&root);

        assert_eq!(
            r.path_of("A Night At The Gate").as_deref(),
            Some("layers/stories/gate.md")
        );
        assert_eq!(r.by_name("A Night At The Gate").unwrap().kind, Kind::Story);
        assert_eq!(r.by_name("the third era").unwrap().kind, Kind::Era);
        // What is indexed is part of the record, not somebody's draft.
        assert_eq!(r.by_name("the third era").unwrap().state, State::Filed);
    }

    /// A document names itself by its heading, so the thing a Maker asks for is
    /// the title printed on it. Without a heading the filename has to do.
    #[test]
    fn a_document_is_named_by_its_heading_and_falls_back_to_its_filename() {
        let root = canon("naming");
        let mut r = Record::new();
        r.index_canon(&root);
        assert!(r.by_name("the third era").is_some(), "the heading was not read");
        assert!(r.by_name("fourth").is_some(), "the filename was not the fallback");
    }

    /// **A thing the world already names becomes that document**, rather than
    /// gaining a twin beside it — two items of one name, one real and one not,
    /// is the trap the whole join exists to close.
    #[test]
    fn an_existing_thing_takes_the_document_rather_than_being_duplicated() {
        let root = canon("adopt");
        let mut r = record();
        assert!(r.path_of("the third era").is_none());
        r.index_canon(&root);

        let named: Vec<&Item> = r.iter().filter(|i| i.name == "the third era").collect();
        assert_eq!(named.len(), 1, "indexing made a twin");
        assert_eq!(named[0].id, "era_third", "the world's own item was replaced");
        assert_eq!(named[0].path.as_deref(), Some("layers/eras/the-third-era.md"));
    }

    /// Re-indexing must not release work somebody is holding — the world learns
    /// where the mind is more than once.
    #[test]
    fn indexing_twice_changes_nothing_and_keeps_custody() {
        let root = canon("idempotent");
        let mut r = Record::new();
        r.index_canon(&root);
        r.take("the third era", "m1").unwrap();
        let before = r.len();

        r.index_canon(&root);
        assert_eq!(r.len(), before, "a second pass added items");
        assert_eq!(
            r.by_name("the third era").unwrap().holder.as_deref(),
            Some("m1"),
            "re-indexing released held work"
        );
    }

    #[test]
    fn only_documents_are_indexed() {
        let root = canon("filter");
        let mut r = Record::new();
        r.index_canon(&root);
        assert!(r.iter().all(|i| i.name != ".gitkeep"), "a keep-file became an era");
        assert_eq!(r.len(), 3, "{:?}", r.iter().map(|i| &i.name).collect::<Vec<_>>());
    }

    #[test]
    fn a_mind_with_no_such_layers_indexes_nothing_and_does_not_fail() {
        let mut r = record();
        let before = r.len();
        r.index_canon(Path::new("/no/such/mind/anywhere"));
        assert_eq!(r.len(), before);
    }

    /// Only the kinds that *are* documents get somewhere to live.
    #[test]
    fn a_path_is_settled_for_documents_and_refused_for_judgements() {
        let mut r = record();
        assert_eq!(
            r.settle_path("the third era").as_deref(),
            Some("layers/eras/the-third-era.md")
        );
        assert_eq!(
            r.settle_path("the third silence").as_deref(),
            Some("layers/stories/the-third-silence.md")
        );
        // A portrait is not a text document and gets none.
        assert_eq!(r.settle_path("a face nobody has drawn"), None);
        assert_eq!(r.settle_path("nothing called this"), None);
    }

    /// Settling twice keeps the first answer — a document does not move because
    /// somebody wrote into it again.
    #[test]
    fn settling_a_path_twice_keeps_the_first_one() {
        let root = canon("settle-twice");
        let mut r = record();
        r.index_canon(&root);
        let first = r.settle_path("the third era");
        assert_eq!(first.as_deref(), Some("layers/eras/the-third-era.md"));
        assert_eq!(r.settle_path("the third era"), first);
    }

    #[test]
    fn taking_something_holds_it_against_everybody_else() {
        let mut r = record();
        assert!(r.take("the third silence", "m1").is_ok());
        let err = r.take("the third silence", "m2").unwrap_err();
        assert!(err.contains("m1"), "the holder was not named: {err}");
    }

    #[test]
    fn what_is_held_stops_being_claimable_and_comes_back_when_given_up() {
        let mut r = record();
        assert_eq!(r.claimable_of(Kind::Gap), vec!["the third silence"]);
        r.take("the third silence", "m1").unwrap();
        assert!(r.claimable_of(Kind::Gap).is_empty());
        r.give_back("the third silence", "m1").unwrap();
        assert_eq!(r.claimable_of(Kind::Gap).len(), 1);
    }

    #[test]
    fn giving_back_what_you_do_not_hold_is_refused() {
        let mut r = record();
        r.take("the third silence", "m1").unwrap();
        assert!(r.give_back("the third silence", "m2").is_err());
    }

    #[test]
    fn writing_appends_rather_than_overwriting() {
        let mut r = record();
        r.write("the third silence", "m1", "first").unwrap();
        r.write("the third silence", "m1", "second").unwrap();
        let body = &r.get("gap_third").unwrap().body;
        assert!(body.contains("first") && body.contains("second"), "{body}");
    }

    #[test]
    fn writing_into_somebody_elses_work_is_refused() {
        let mut r = record();
        r.take("the third silence", "m1").unwrap();
        let err = r.write("the third silence", "m2", "mine now").unwrap_err();
        assert!(err.contains("m1"), "{err}");
    }

    #[test]
    fn a_filed_thing_is_not_written_over() {
        let mut r = record();
        let err = r.write("the third era", "m1", "…").unwrap_err();
        assert!(err.contains("filed"), "{err}");
    }

    #[test]
    fn the_state_machine_refuses_a_move_it_does_not_have() {
        let mut r = record();
        // Nothing goes straight from unwritten to filed.
        assert!(r.set_state("the third silence", "m1", State::Filed).is_err());

        r.write("the third silence", "m1", "a draft").unwrap();
        assert!(r.set_state("the third silence", "m1", State::Offered).is_ok());
        assert!(r.set_state("the third silence", "m1", State::Filed).is_ok());
        assert_eq!(r.get("gap_third").unwrap().state, State::Filed);
        // Filing releases it, so the next thing can be taken.
        assert!(r.get("gap_third").unwrap().holder.is_none());
    }

    #[test]
    fn letting_something_go_needs_a_reason_somebody_can_disagree_with() {
        let mut r = record();
        assert!(r.let_go("the third era", "   ").is_err());
        assert!(r.let_go("the third era", "kept for years, never once asked for").is_ok());
        let i = r.get("era_third").unwrap();
        assert_eq!(i.state, State::Retired);
        assert!(i.let_go_because.is_some());
    }

    #[test]
    fn a_retired_thing_cannot_be_taken_up_again() {
        let mut r = record();
        r.let_go("the third era", "no longer carried").unwrap();
        assert!(r.take("the third era", "m1").is_err());
    }

    #[test]
    fn a_cross_reference_must_point_at_something_that_exists() {
        let mut r = record();
        assert!(r.cross_reference("the third era", "a thing nobody wrote").is_err());
        assert!(r.cross_reference("the third era", "the third silence").is_ok());
        assert_eq!(r.get("era_third").unwrap().refers_to, vec!["the third silence"]);
    }

    #[test]
    fn provenance_and_description_and_condition_attach_to_the_thing() {
        let mut r = record();
        r.set_provenance("the third era", "came in with the western intake").unwrap();
        r.describe("the third era", "what to read first, and why").unwrap();
        r.set_condition("the third era", Condition::Stale).unwrap();
        let i = r.get("era_third").unwrap();
        assert!(i.provenance.is_some() && i.description.is_some());
        assert_eq!(i.condition, Condition::Stale);
    }

    #[test]
    fn names_are_offered_in_a_stable_order() {
        let r = record();
        assert_eq!(r.names_of(Kind::Gap), vec!["the third silence"]);
        assert_eq!(r.names_in(Kind::Era, State::Filed), vec!["the third era"]);
        assert_eq!(record(), record(), "two builds of one seed differed");
    }
}
