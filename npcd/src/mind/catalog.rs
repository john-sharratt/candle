//! What is inside a place in the corpus.
//!
//! This lists an [`Address`], not a directory — and the two are different
//! shapes, so this is where the second becomes the first:
//!
//! - **A topic appears once.** `layers/world/ammo.md` and `layers/world/ammo/`
//!   are one topic stored as two things, so they are one row with text of its
//!   own, rather than a folder and a mysterious file beside it.
//! - **Extensions are gone**, because they were never the reader's business.
//! - **Only the nine sections exist.** A folder that is not one cannot be
//!   named, so `node_modules` and the daemon's `.substrate` are not filtered
//!   out of the listing — they are not in the vocabulary.
//! - **Sizes are what the thing is.** A collection has a count of what is in
//!   it; an entry has a length. Not one column meaning two things.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use super::address::{Address, Section, SECTIONS};
use super::doc;
use super::parts;
use super::path::MindPath;
use super::scope::Scope;

/// A thing in the corpus: something that holds other things, or something to
/// read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Collection,
    Entry,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Kind::Collection => "collection",
            Kind::Entry => "entry",
        }
    }
}

/// One row of a listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub id: String,
    pub title: String,
    pub kind: Kind,
    /// How many things are inside, for a collection.
    pub count: Option<u64>,
    /// How long it is, for something with text.
    pub chars: Option<u64>,
    /// A collection that also has text of its own — a topic with an overview.
    pub has_text: bool,
    /// One line saying what lives here. Only the nine sections have one; a
    /// topic is described by what is in it.
    pub blurb: Option<&'static str>,
}

impl Node {
    pub fn wire(&self) -> Value {
        json!({
            "id": self.id,
            "title": self.title,
            "kind": self.kind.as_str(),
            "count": self.count,
            "chars": self.chars,
            "has_text": self.has_text,
            "blurb": self.blurb,
        })
    }
}

/// Why a listing failed.
#[derive(Debug)]
pub enum CatalogError {
    /// Nothing is there — a topic that has been removed, or never existed.
    NotFound,
    /// The world named does not admit it.
    OutOfScope,
    Io(std::io::Error),
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::NotFound => write!(f, "there is nothing there"),
            CatalogError::OutOfScope => write!(f, "this world does not include that"),
            CatalogError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// The corpus itself — the nine sections, each with what it holds.
pub fn sections(
    root: &Path,
    scope: &Scope,
    category_of: &dyn Fn(&MindPath) -> Option<String>,
) -> Vec<Node> {
    SECTIONS
        .into_iter()
        .map(|s| {
            let addr = Address::of_section(s);
            let count = children(root, &addr, scope, category_of)
                .map(|c| c.len() as u64)
                .unwrap_or(0);
            Node {
                id: addr.as_str(),
                title: s.title().to_owned(),
                kind: Kind::Collection,
                count: Some(count),
                chars: None,
                has_text: false,
                blurb: Some(s.blurb()),
            }
        })
        .collect()
}

/// What is inside `addr`.
pub fn children(
    root: &Path,
    addr: &Address,
    scope: &Scope,
    category_of: &dyn Fn(&MindPath) -> Option<String>,
) -> Result<Vec<Node>, CatalogError> {
    if !admits(addr, scope, category_of) {
        return Err(CatalogError::OutOfScope);
    }

    // The settings are a named set, not a folder — see `address::SETTINGS`.
    if addr.section() == Section::Settings && addr.is_section() {
        return Ok(Address::settings()
            .into_iter()
            .map(|a| {
                let chars = a
                    .entry_path()
                    .and_then(|p| p.resolve(root).ok())
                    .and_then(|f| fs::metadata(f).ok())
                    .map(|m| m.len());
                // A settings document with parts is a collection *and* an
                // entry: the projection schema holds its layers, and still
                // reads whole.
                let count = a
                    .parts()
                    .map(|_| children(root, &a, scope, category_of).map_or(0, |c| c.len() as u64));
                Node {
                    title: a.title(),
                    id: a.as_str(),
                    kind: if count.is_some() {
                        Kind::Collection
                    } else {
                        Kind::Entry
                    },
                    count,
                    chars,
                    has_text: true,
                    blurb: None,
                }
            })
            .collect());
    }

    // A settings document whose parts are addressable lists them from inside
    // itself — `settings/projection` holds the nine layers. It is a collection
    // with a body, exactly as a canon topic is: it still reads whole.
    if let Some((list_key, id_key)) = addr.parts() {
        let text = read(root, addr).map_err(|_| CatalogError::NotFound)?.text;
        // A document that declares parts but has none — or cannot be read as a
        // mapping — is simply an entry with no children, not a failure. The
        // text view still opens it.
        let items = parts::list(&text, list_key, id_key).unwrap_or_default();
        return Ok(items
            .into_iter()
            .filter_map(|(name, value)| {
                let child = addr.child(&name).ok()?;
                Some(Node {
                    id: child.as_str(),
                    title: child.title(),
                    kind: Kind::Entry,
                    count: None,
                    // How big the part is, measured the way an entry is — the
                    // text it would open as.
                    chars: serde_yaml::to_string(&value).ok().map(|y| y.len() as u64),
                    has_text: true,
                    blurb: None,
                })
            })
            .collect());
    }

    let Some(dir) = addr.collection_path() else {
        return Err(CatalogError::NotFound);
    };
    let Ok(full) = dir.resolve(root) else {
        return Err(CatalogError::NotFound);
    };
    if !full.is_dir() {
        // A section whose folder is simply absent is empty, not broken: a mind
        // need not have written any beliefs yet.
        return if addr.is_section() {
            Ok(Vec::new())
        } else {
            Err(CatalogError::NotFound)
        };
    }

    // Gathered by name so a topic's folder and its overview page meet: both
    // reduce to the same name, and the pair becomes one row.
    let mut found: BTreeMap<String, (bool, bool, u64, u64)> = BTreeMap::new();
    let ext = addr.section().format();
    for entry in fs::read_dir(&full).map_err(CatalogError::Io)? {
        let entry = entry.map_err(CatalogError::Io)?;
        let Some(raw) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Ok(meta) = entry.path().symlink_metadata() else {
            continue;
        };
        // A link is not followed: it would be a second name for a document that
        // already has one, and writing through it would replace the link.
        if meta.file_type().is_symlink() {
            continue;
        }
        let (name, is_dir) = if meta.is_dir() {
            (raw.clone(), true)
        } else {
            // Only this section's own format is an entry. Anything else in the
            // folder is not part of the corpus and is not shown — there is no
            // greyed row to explain, because there is no address for it.
            match raw.strip_suffix(&format!(".{}", ext_str(ext))) {
                Some(stem) => (stem.to_owned(), false),
                None => continue,
            }
        };
        if name.is_empty() {
            continue;
        }
        let Ok(child) = addr.child(&name) else {
            continue;
        };
        if !admits(&child, scope, category_of) {
            continue;
        }
        let slot = found.entry(name).or_insert((false, false, 0, 0));
        if is_dir {
            slot.0 = true;
            slot.2 = fs::read_dir(entry.path())
                .map(|d| d.count() as u64)
                .unwrap_or(0);
        } else {
            slot.1 = true;
            slot.3 = meta.len();
        }
    }

    let mut out: Vec<Node> = found
        .into_iter()
        .filter_map(|(name, (is_dir, is_file, count, len))| {
            let child = addr.child(&name).ok()?;
            Some(Node {
                id: child.as_str(),
                title: child.title(),
                kind: if is_dir {
                    Kind::Collection
                } else {
                    Kind::Entry
                },
                count: is_dir.then_some(count),
                chars: is_file.then_some(len),
                has_text: is_file,
                blurb: None,
            })
        })
        .collect();
    // Collections first, then entries, each alphabetically — a sidebar that
    // reorders itself between visits is one nobody builds a habit in.
    out.sort_by(|a, b| {
        (a.kind == Kind::Entry)
            .cmp(&(b.kind == Kind::Entry))
            .then_with(|| a.title.cmp(&b.title))
    });
    Ok(out)
}

/// Read the text at `addr`, if it has any.
///
/// A *part* reads as a document of its own — one layer of the projection
/// schema, not the seven hundred lines it sits in.
pub fn read(root: &Path, addr: &Address) -> Result<doc::Doc, doc::DocError> {
    let path = addr.entry_path().ok_or(doc::DocError::NotFound)?;
    let whole = doc::read(root, &path)?;
    let Some(part) = addr.part() else {
        return Ok(whole);
    };
    let value = parts::read(&whole.text, part.list, part.id_key, part.name)
        .map_err(|_| doc::DocError::NotFound)?;
    let text = serde_yaml::to_string(&value).map_err(|_| doc::DocError::NotFound)?;
    Ok(doc::Doc {
        bytes: text.len() as u64,
        text,
        ..whole
    })
}

/// Write the text at `addr`.
///
/// A part is spliced back into the document that holds it, so the bytes around
/// it — including the author's banner comments between the parts — are exactly
/// what they were. Creating one is refused: an address names a part that
/// exists, and there is nowhere in the document for a new one to go without
/// rewriting the list. See [`parts`].
pub fn write(
    root: &Path,
    addr: &Address,
    text: &str,
    must_be_new: bool,
) -> Result<doc::Wrote, doc::DocError> {
    let path = addr.entry_path().ok_or(doc::DocError::NotFound)?;
    let Some(part) = addr.part() else {
        return doc::write(root, &path, text, must_be_new);
    };
    if must_be_new {
        return Err(doc::DocError::NotFound);
    }
    let value: serde_json::Value =
        serde_yaml::from_str(text).map_err(|_| doc::DocError::NotFound)?;
    let object = value.as_object().ok_or(doc::DocError::NotFound)?;
    let whole = doc::read(root, &path)?;
    let next = parts::write(&whole.text, part.list, part.id_key, part.name, object).map_err(
        |e| match e {
            parts::PartError::CannotPatch => doc::DocError::CannotPatch,
            _ => doc::DocError::NotFound,
        },
    )?;
    doc::write(root, &path, &next, false)
}

/// Remove what is at `addr`.
///
/// The text only. A topic that also holds entries keeps them — removing those
/// too would be a recursive delete behind one button, and they have addresses
/// of their own to be removed by.
///
/// A part cannot be removed, and the whole document it lives in must not be
/// removed by naming one: `settings/projection/world` is a layer, and deleting
/// `projection.yaml` because somebody pressed Delete on a layer would be the
/// worst kind of surprise.
pub fn remove(root: &Path, addr: &Address) -> Result<(), doc::DocError> {
    if addr.part().is_some() {
        return Err(doc::DocError::NotFound);
    }
    let path = addr.entry_path().ok_or(doc::DocError::NotFound)?;
    doc::remove(root, &path)
}

/// Whether the world admits this address, asked of whichever path it has.
fn admits(
    addr: &Address,
    scope: &Scope,
    category_of: &dyn Fn(&MindPath) -> Option<String>,
) -> bool {
    // A topic has both, and they answer the same — `selects` names the tag, and
    // the extension is stripped before it is compared. Either is enough.
    if let Some(p) = addr.collection_path() {
        return scope.admits(&p, category_of);
    }
    match addr.entry_path() {
        Some(p) => scope.admits(&p, category_of),
        None => true,
    }
}

fn ext_str(f: super::address::Format) -> &'static str {
    match f {
        super::address::Format::Markdown => "md",
        super::address::Format::Yaml => "yaml",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("npcd-catalog-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn seed(root: &Path) {
        for dir in [
            "layers/world/ammo",
            "layers/world/armor",
            "layers/memory/cindy-tan",
            "responses",
            "moods",
            "personalities",
            "worlds",
            // Not part of the corpus, and not addressable.
            "node_modules/x",
            ".substrate",
            "scratchpad",
        ] {
            fs::create_dir_all(root.join(dir)).unwrap();
        }
        fs::write(root.join("layers/world/ammo.md"), "all about ammo").unwrap();
        fs::write(root.join("layers/world/ammo/bolt.md"), "a bolt").unwrap();
        fs::write(root.join("layers/world/ammo/shell.md"), "a shell").unwrap();
        fs::write(root.join("layers/world/armor/plate.md"), "plate").unwrap();
        // A tag with a page and no folder.
        fs::write(root.join("layers/world/alliances.md"), "pacts").unwrap();
        fs::write(
            root.join("responses/blush_then_own.yaml"),
            "category: intimate",
        )
        .unwrap();
        fs::write(root.join("responses/fight.yaml"), "category: combat").unwrap();
        fs::write(root.join("personalities/cindy-tan.yaml"), "id: cindy-tan").unwrap();
        fs::write(root.join("worlds/earth.yaml"), "id: earth").unwrap();
        fs::write(root.join("mind.yaml"), "char_name: Keeper").unwrap();
        fs::write(root.join("projection.yaml"), "layers: []").unwrap();
        fs::write(root.join("CLAUDE.md"), "guidance").unwrap();
        // Junk that must never appear.
        fs::write(root.join("package.json"), "{}").unwrap();
        fs::write(root.join("work.json"), "{}").unwrap();
        fs::write(root.join("layers/world/notes.txt"), "not canon").unwrap();
    }

    fn no_cat(_: &MindPath) -> Option<String> {
        None
    }

    fn cat(path: &MindPath) -> Option<String> {
        match path.name() {
            "blush_then_own.yaml" => Some("intimate".into()),
            "fight.yaml" => Some("combat".into()),
            _ => None,
        }
    }

    fn addr(s: &str) -> Address {
        Address::parse(s).unwrap().unwrap()
    }

    fn titles(nodes: &[Node]) -> Vec<&str> {
        nodes.iter().map(|n| n.title.as_str()).collect()
    }

    /// **The abstraction, in one assertion.** A topic stored as a page and a
    /// folder is one row that has both — not a folder and a stray file.
    #[test]
    fn a_topic_is_one_row_with_text_of_its_own() {
        let root = tmp("topic");
        seed(&root);
        let got = children(&root, &addr("canon"), &Scope::unscoped(), &no_cat).unwrap();
        let ammo = got.iter().find(|n| n.id == "canon/ammo").expect("listed");
        assert_eq!(ammo.kind, Kind::Collection);
        assert_eq!(ammo.count, Some(2), "two entries inside");
        assert!(ammo.has_text, "and an overview of its own");
        assert_eq!(ammo.chars, Some("all about ammo".len() as u64));
        // Exactly one row for it, not two.
        assert_eq!(got.iter().filter(|n| n.title == "Ammo").count(), 1);

        // A tag with a page and no folder is simply an entry.
        let all = got.iter().find(|n| n.id == "canon/alliances").unwrap();
        assert_eq!(all.kind, Kind::Entry);
        assert_eq!(all.count, None);
        assert!(all.has_text);
    }

    /// Nothing on the wire says `.md`, `.yaml`, or names a directory.
    #[test]
    fn no_file_names_reach_the_listing() {
        let root = tmp("nofiles");
        seed(&root);
        for place in ["canon", "canon/ammo", "responses", "characters", "settings"] {
            let got = children(&root, &addr(place), &Scope::unscoped(), &no_cat).unwrap();
            for n in &got {
                assert!(!n.id.contains(".md"), "{}", n.id);
                assert!(!n.id.contains(".yaml"), "{}", n.id);
                assert!(!n.id.contains("layers/"), "{}", n.id);
                assert!(!n.title.contains('.'), "{}", n.title);
            }
        }
    }

    /// A file that is not this section's format is not part of the corpus, so
    /// there is no row for it at all — not even a greyed one.
    #[test]
    fn a_foreign_file_is_not_in_the_corpus() {
        let root = tmp("foreign");
        seed(&root);
        let got = children(&root, &addr("canon"), &Scope::unscoped(), &no_cat).unwrap();
        assert!(!titles(&got).contains(&"Notes"), "notes.txt was listed");
    }

    /// The nine sections, and nothing else the mind directory happens to hold.
    #[test]
    fn the_corpus_is_the_nine_sections() {
        let root = tmp("sections");
        seed(&root);
        let got = sections(&root, &Scope::unscoped(), &no_cat);
        assert_eq!(got.len(), 9);
        assert_eq!(
            titles(&got),
            [
                "World knowledge",
                "Agency",
                "Beliefs",
                "Memory",
                "Responses",
                "Moods",
                "Characters",
                "Worlds",
                "Settings",
            ]
        );
        for n in &got {
            assert_eq!(n.kind, Kind::Collection);
            assert!(!n.id.contains('/'), "a section id is one token: {}", n.id);
        }
        let canon = got.iter().find(|n| n.id == "canon").unwrap();
        assert_eq!(canon.count, Some(3), "ammo, armor, alliances");
    }

    #[test]
    fn a_section_with_no_folder_yet_is_empty_rather_than_missing() {
        let root = tmp("empty");
        seed(&root);
        // `layers/beliefs` was never created by the seed.
        let got = children(&root, &addr("beliefs"), &Scope::unscoped(), &no_cat).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn the_settings_are_listed_from_their_names() {
        let root = tmp("settings");
        seed(&root);
        let got = children(&root, &addr("settings"), &Scope::unscoped(), &no_cat).unwrap();
        assert_eq!(
            titles(&got),
            ["Mind", "Projection schema", "Authoring guidance"]
        );
        for n in &got {
            assert!(n.has_text, "{} has no text", n.id);
            assert!(n.chars.unwrap_or(0) > 0, "{} has no text", n.id);
        }
        // The projection schema is the one that also holds things — its layers.
        // It is a collection *and* an entry, the way a canon topic is.
        let kinds: Vec<Kind> = got.iter().map(|n| n.kind).collect();
        assert_eq!(kinds, [Kind::Entry, Kind::Collection, Kind::Entry]);
    }

    /// The projection schema's layers are addressable one by one, and each
    /// opens as a document of its own rather than as the file it sits in.
    #[test]
    fn the_projection_layers_are_addressable_parts() {
        let root = tmp("parts");
        seed(&root);
        fs::write(
            root.join("projection.yaml"),
            "layers:\n  # ── World ──\n  - name: world\n    window: 8000\n  \
             # ── Beliefs ──\n  - name: beliefs\n    window: 4000\n",
        )
        .unwrap();

        let got = children(
            &root,
            &addr("settings/projection"),
            &Scope::unscoped(),
            &no_cat,
        )
        .unwrap();
        assert_eq!(
            titles(&got),
            ["World", "Beliefs"],
            "in the document's order"
        );
        assert_eq!(
            got.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(),
            ["settings/projection/world", "settings/projection/beliefs"]
        );

        // One part reads as itself.
        let one = read(&root, &addr("settings/projection/world")).unwrap();
        assert!(one.text.contains("window: 8000"), "{}", one.text);
        assert!(
            !one.text.contains("beliefs"),
            "read the whole file:\n{}",
            one.text
        );

        // And saving it changes that layer and leaves the comments.
        write(
            &root,
            &addr("settings/projection/world"),
            "name: world\nwindow: 9000\n",
            false,
        )
        .unwrap();
        let after = read(&root, &addr("settings/projection")).unwrap().text;
        assert!(after.contains("window: 9000"), "{after}");
        assert!(
            after.contains("# ── Beliefs ──"),
            "lost a comment:\n{after}"
        );
        assert!(
            after.contains("window: 4000"),
            "touched the other layer:\n{after}"
        );
    }

    /// Delete on a part must never reach the document it lives in.
    #[test]
    fn deleting_a_part_is_refused_rather_than_deleting_its_document() {
        let root = tmp("part-delete");
        seed(&root);
        fs::write(root.join("projection.yaml"), "layers:\n  - name: world\n").unwrap();
        assert!(remove(&root, &addr("settings/projection/world")).is_err());
        assert!(root.join("projection.yaml").exists(), "the document went");
    }

    #[test]
    fn the_world_scope_still_applies() {
        let root = tmp("scope");
        seed(&root);
        let scope = Scope::of_world(&json!({
            "selects": ["ammo"],
            "excludes": ["intimate"],
            "personalities": ["cindy-tan"],
        }));
        let canon = children(&root, &addr("canon"), &scope, &no_cat).unwrap();
        assert_eq!(titles(&canon), ["Ammo"], "armor and alliances are out");

        let responses = children(&root, &addr("responses"), &scope, &cat).unwrap();
        assert_eq!(titles(&responses), ["Fight"]);

        // And naming an excluded topic directly is refused.
        assert!(matches!(
            children(&root, &addr("canon/armor"), &scope, &no_cat),
            Err(CatalogError::OutOfScope)
        ));
    }

    #[test]
    fn reading_and_writing_go_through_the_address() {
        let root = tmp("rw");
        seed(&root);
        let a = addr("canon/ammo/bolt");
        assert_eq!(read(&root, &a).unwrap().text, "a bolt");

        write(&root, &a, "a heavier bolt", false).unwrap();
        assert_eq!(
            fs::read_to_string(root.join("layers/world/ammo/bolt.md")).unwrap(),
            "a heavier bolt"
        );

        // A new entry, with no extension anywhere in the address.
        let n = addr("canon/ammo/tracer");
        write(&root, &n, "a tracer", true).unwrap();
        assert!(root.join("layers/world/ammo/tracer.md").is_file());

        remove(&root, &n).unwrap();
        assert!(!root.join("layers/world/ammo/tracer.md").exists());
    }

    /// Removing a topic's overview leaves its entries alone: one button must
    /// not become a recursive delete.
    #[test]
    fn removing_a_topics_text_keeps_its_entries() {
        let root = tmp("rmtopic");
        seed(&root);
        remove(&root, &addr("canon/ammo")).unwrap();
        assert!(!root.join("layers/world/ammo.md").exists());
        assert!(root.join("layers/world/ammo/bolt.md").is_file());
        // And it is still a topic, now without an overview.
        let got = children(&root, &addr("canon"), &Scope::unscoped(), &no_cat).unwrap();
        let ammo = got.iter().find(|n| n.id == "canon/ammo").unwrap();
        assert!(!ammo.has_text);
        assert_eq!(ammo.count, Some(2));
    }
}
