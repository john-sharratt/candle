//! What a world admits, as a filter over paths.
//!
//! A world is a **filter over one shared corpus**, not a corpus of its own
//! (`docs/npc_api_gui_design.md`). Its document says so in three fields, and
//! this turns those three into one question — *may this world see this path?* —
//! asked once per entry when a directory is listed.
//!
//! | Field | Filters |
//! |---|---|
//! | `selects` | which tags of `layers/world/` this setting admits |
//! | `excludes` | which section categories it refuses, in `responses/` and `moods/` |
//! | `personalities` | its cast, in `personalities/` and in the per-character layers |
//!
//! # Why the filter is here and not in the browser
//!
//! A list the client narrows is a list the client was first sent whole. The
//! same reasoning already governs the hidden-document filter in
//! [`crate::visibility`]: filtering after sending is a presentation choice that
//! has already lost. So a world scope is applied while the directory is read,
//! and a path the scope excludes is never named on the wire.
//!
//! # The unscoped case is not the empty case
//!
//! [`Scope::unscoped`] admits everything, and it is what a request with no
//! `?world=` gets. That is the right default for an editor: the mind is one
//! tree, a world is a lens on it, and someone editing the corpus itself should
//! not have to pick a lens first. A world whose document names none of the
//! three fields also admits everything, for the same reason it does everywhere
//! else — a document that says nothing is not a document that forbids
//! everything.

use serde_json::Value;

use super::path::MindPath;

/// The three lists a world filters by, resolved from its document.
#[derive(Debug, Clone, Default)]
pub struct Scope {
    /// Tags of `layers/world/` this world admits. `None` when the document
    /// names none — every tag.
    selects: Option<Vec<String>>,
    /// Section categories this world refuses. Empty admits all.
    excludes: Vec<String>,
    /// The cast. `None` when the document names none — every personality.
    cast: Option<Vec<String>>,
}

impl Scope {
    /// A scope that admits the whole mind.
    pub fn unscoped() -> Self {
        Scope::default()
    }

    /// Read the three lists from a world document.
    pub fn of_world(body: &Value) -> Self {
        Scope {
            selects: strings(body, "selects"),
            excludes: strings(body, "excludes").unwrap_or_default(),
            cast: strings(body, "personalities"),
        }
    }

    /// Whether this world admits `path`.
    ///
    /// Answered for the path as a whole, so a directory that is excluded takes
    /// everything beneath it: a caller cannot reach `layers/world/ammo/bolt.md`
    /// by asking for it directly when `ammo` is not selected.
    pub fn admits(
        &self,
        path: &MindPath,
        category_of: &dyn Fn(&MindPath) -> Option<String>,
    ) -> bool {
        let seg = |i: usize| path.segments().get(i).map(String::as_str);

        match path.area() {
            // `layers/world/<tag>/…` — the tag must be selected.
            //
            // A tag is both a file and a folder: `ammo.md` is its canon page
            // and `ammo/` holds its items, and `selects` names the tag once —
            // `ammo`. So the extension comes off before the comparison, or
            // every page in the world would be filtered out while its folder
            // stayed, which is how 37 of a real world's 66 entries vanished.
            Some("layers") if seg(1) == Some("world") => match (self.selects.as_ref(), seg(2)) {
                (Some(selects), Some(tag)) => selects.iter().any(|s| s == stem(tag)),
                // The world names no `selects`, or the path is `layers/world`
                // itself, which is the door to the tags rather than one of them.
                _ => true,
            },
            // `layers/beliefs/<npc>/…` and `layers/memory/<npc>/…` are a
            // character's own, so they follow the cast. `layers/agency` and any
            // other layer is shared and unfiltered.
            Some("layers") if matches!(seg(1), Some("beliefs") | Some("memory")) => {
                match (self.cast.as_ref(), seg(2)) {
                    (Some(cast), Some(npc)) => cast.iter().any(|c| c == npc),
                    _ => true,
                }
            }
            Some("personalities") => match (self.cast.as_ref(), seg(1)) {
                (Some(cast), Some(file)) => cast.iter().any(|c| c == stem(file)),
                _ => true,
            },
            // A section's category is inside the file, so the caller supplies
            // it — this module does not read from disk.
            Some("responses") | Some("moods") => match category_of(path) {
                Some(cat) => !self
                    .excludes
                    .iter()
                    .any(|e| e.trim().eq_ignore_ascii_case(cat.trim())),
                // Not a section file (a directory, or one that would not
                // parse). Admitted: a world filters what it can name, and
                // hiding a file because it could not be read would make a
                // parse error look like a permission.
                None => true,
            },
            _ => true,
        }
    }

    /// Whether the world names any filter at all. Worth reporting on the wire
    /// so the console can say "showing everything" rather than implying a lens
    /// that is not there.
    pub fn is_unscoped(&self) -> bool {
        self.selects.is_none() && self.excludes.is_empty() && self.cast.is_none()
    }
}

/// A string list from a document, or `None` when the key is absent.
///
/// Absent and empty are **different** here, and the difference is load-bearing:
/// `selects: []` is a world that admits no world-layer tags, which is exactly
/// what `earth.yaml` says and means. A missing key is a world that has not
/// been given the field.
fn strings(body: &Value, key: &str) -> Option<Vec<String>> {
    let arr = body.get(key)?.as_array()?;
    Some(
        arr.iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
    )
}

/// `commander.yaml` → `commander`.
fn stem(file: &str) -> &str {
    file.rsplit_once('.').map(|(s, _)| s).unwrap_or(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn p(s: &str) -> MindPath {
        MindPath::parse(s).expect("test path parses")
    }

    /// No category lookup — for the paths that do not need one.
    fn no_cat(_: &MindPath) -> Option<String> {
        None
    }

    #[test]
    fn an_unscoped_scope_admits_everything() {
        let s = Scope::unscoped();
        assert!(s.is_unscoped());
        for path in [
            "layers/world/ammo/bolt.md",
            "personalities/cindy-tan.yaml",
            "responses/blush_then_own.yaml",
            "layers/memory/cindy-tan/a.md",
        ] {
            assert!(s.admits(&p(path), &no_cat), "{path}");
        }
    }

    /// The real `battle-cities.yaml` shape.
    #[test]
    fn selects_gates_the_world_layer_tags() {
        let s = Scope::of_world(&json!({ "selects": ["ammo", "armor"] }));
        assert!(s.admits(&p("layers/world/ammo"), &no_cat));
        assert!(s.admits(&p("layers/world/ammo/bolt.md"), &no_cat));
        assert!(s.admits(&p("layers/world/armor"), &no_cat));
        assert!(!s.admits(&p("layers/world/alliances"), &no_cat));
        assert!(!s.admits(&p("layers/world/alliances/x.md"), &no_cat));
        // The door to the tags is not itself a tag.
        assert!(s.admits(&p("layers/world"), &no_cat));
        assert!(s.admits(&p("layers"), &no_cat));
        // Another layer entirely is unfiltered by `selects`.
        assert!(s.admits(&p("layers/agency/x.md"), &no_cat));
    }

    /// A tag exists twice in a real mind: `ammo.md` is its canon page and
    /// `ammo/` holds its items. `selects` names it once, so both must match —
    /// comparing the raw entry name dropped every page while keeping every
    /// folder, which on the real corpus hid 37 of 66 entries.
    #[test]
    fn a_selected_tag_admits_both_its_page_and_its_folder() {
        let s = Scope::of_world(&json!({ "selects": ["ammo", "alliances"] }));
        assert!(s.admits(&p("layers/world/ammo"), &no_cat), "the folder");
        assert!(s.admits(&p("layers/world/ammo.md"), &no_cat), "the page");
        assert!(s.admits(&p("layers/world/alliances.md"), &no_cat));
        assert!(
            s.admits(&p("layers/world/ammo/bolt.md"), &no_cat),
            "an item"
        );
        // And an unselected tag is still out, in both of its forms.
        assert!(!s.admits(&p("layers/world/armor"), &no_cat));
        assert!(!s.admits(&p("layers/world/armor.md"), &no_cat));
    }

    /// `earth.yaml` says `selects: []` — it admits no world-layer canon at all,
    /// and that is a statement rather than an omission.
    #[test]
    fn an_empty_selects_admits_no_tag() {
        let s = Scope::of_world(&json!({ "selects": [] }));
        assert!(!s.is_unscoped());
        assert!(!s.admits(&p("layers/world/ammo"), &no_cat));
        assert!(!s.admits(&p("layers/world/ammo/bolt.md"), &no_cat));
        // Still a door, and still not a tag.
        assert!(s.admits(&p("layers/world"), &no_cat));
    }

    /// An absent key is not an empty list: a world with no `selects` has not
    /// been given the field, and admits every tag.
    #[test]
    fn an_absent_selects_is_not_an_empty_one() {
        let s = Scope::of_world(&json!({ "name": "W" }));
        assert!(s.is_unscoped());
        assert!(s.admits(&p("layers/world/ammo"), &no_cat));
    }

    #[test]
    fn the_cast_gates_personalities_and_their_layers() {
        let s = Scope::of_world(&json!({ "personalities": ["cindy-tan"] }));
        assert!(s.admits(&p("personalities/cindy-tan.yaml"), &no_cat));
        assert!(!s.admits(&p("personalities/commander.yaml"), &no_cat));
        assert!(s.admits(&p("layers/memory/cindy-tan/a.md"), &no_cat));
        assert!(!s.admits(&p("layers/memory/commander/a.md"), &no_cat));
        assert!(s.admits(&p("layers/beliefs/cindy-tan/a.md"), &no_cat));
        assert!(!s.admits(&p("layers/beliefs/keeper/a.md"), &no_cat));
        // The directories above are doors.
        assert!(s.admits(&p("personalities"), &no_cat));
        assert!(s.admits(&p("layers/memory"), &no_cat));
    }

    #[test]
    fn excludes_gates_sections_by_their_category() {
        let s = Scope::of_world(&json!({ "excludes": ["sexual", "intimate"] }));
        let cat = |path: &MindPath| match path.name() {
            "adult.yaml" => Some("sexual".to_string()),
            "tender.yaml" => Some("Intimate".to_string()),
            "fight.yaml" => Some("combat".to_string()),
            _ => None,
        };
        assert!(!s.admits(&p("responses/adult.yaml"), &cat));
        // Case and spacing are the world's, not the file's.
        assert!(!s.admits(&p("responses/tender.yaml"), &cat));
        assert!(s.admits(&p("responses/fight.yaml"), &cat));
        assert!(s.admits(&p("moods/fight.yaml"), &cat));
        // A file whose category could not be read is admitted: a parse failure
        // must not read as a permission.
        assert!(s.admits(&p("responses/unreadable.yaml"), &cat));
    }

    /// The three filters are independent — one does not imply another.
    #[test]
    fn the_filters_do_not_leak_into_each_other() {
        let s = Scope::of_world(&json!({
            "selects": ["ammo"],
            "excludes": ["sexual"],
            "personalities": ["commander"],
        }));
        assert!(!s.is_unscoped());
        // A personality is gated by the cast, not by `selects`.
        assert!(s.admits(&p("personalities/commander.yaml"), &no_cat));
        // A layer tag is gated by `selects`, not by the cast.
        assert!(s.admits(&p("layers/world/ammo/x.md"), &no_cat));
        assert!(!s.admits(&p("layers/world/armor/x.md"), &no_cat));
        // Anything outside the three areas is unfiltered.
        assert!(s.admits(&p("worlds/battle-cities.yaml"), &no_cat));
    }

    #[test]
    fn a_document_with_junk_in_its_lists_takes_the_strings_it_can_read() {
        let s = Scope::of_world(&json!({ "selects": ["ammo", 7, null, "armor"] }));
        assert!(s.admits(&p("layers/world/ammo"), &no_cat));
        assert!(s.admits(&p("layers/world/armor"), &no_cat));
        assert!(!s.admits(&p("layers/world/other"), &no_cat));
    }
}
