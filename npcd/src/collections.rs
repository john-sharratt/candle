//! The authored section libraries, read from the mind.
//!
//! # What this replaces
//!
//! `/v1/world/{id}/collections` used to fall through to `web::mock::npcd`,
//! which answered with six invented response templates and five invented moods.
//! The mind on this machine holds **596 responses and 116 moods**, curated one
//! file at a time. The console showed the six, with nothing to say they were
//! fabricated, on exactly the page somebody would open to look at their own
//! library — and the reasonable conclusion from that screen was that the other
//! 700 files had been lost.
//!
//! Nothing had been lost. The daemon simply never read them. That is the worst
//! shape a fixture can take: not obviously fake, in the place the real thing
//! belongs.
//!
//! # What a section is here
//!
//! One YAML file in `responses/` or `moods/`, carrying an `id`, a `category`, a
//! `description`, the `template` that is installed as the section's content,
//! and the provenance `examples` that train its selection. This reads all of
//! them at boot and keeps a **summary** — the template, and how many examples
//! there are, not the examples themselves. The examples are the bulk of the
//! bytes (a single mood runs to tens of kilobytes of four-turn lead-ins) and
//! the console never displays them.
//!
//! # No invented numbers
//!
//! The fixture reported a `tokens` count per section. There is no tokenizer in
//! this daemon, so any figure it produced here would be a plausible-looking
//! guess — which is the habit that caused the problem above. It reports
//! `chars`, which is a thing it can actually count.

use std::path::Path;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::projection::Source;

/// A section's file, as far as the console cares.
///
/// Every field is optional but `id`: these are hand-written and a library of
/// six hundred will not be uniform. A file missing its `template` is still
/// worth listing — seeing it there with nothing in it is how somebody finds it.
#[derive(Debug, Deserialize)]
struct SectionFile {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    template: Option<String>,
    /// Counted, never carried. See the module note.
    #[serde(default)]
    examples: Vec<serde_yaml::Value>,
}

/// One section, summarised for the console.
#[derive(Debug, Clone)]
pub struct Section {
    pub id: String,
    pub category: String,
    pub chars: usize,
    pub examples: usize,
    pub template: String,
    pub description: String,
}

impl Section {
    fn wire(&self) -> Value {
        json!({
            "id": self.id,
            "category": self.category,
            "chars": self.chars,
            "examples": self.examples,
            "template": self.template,
            "description": self.description,
        })
    }
}

/// A folder of sections, loaded once.
#[derive(Debug, Clone)]
pub struct Library {
    pub name: &'static str,
    pub folder: String,
    pub sections: Vec<Section>,
}

impl Library {
    /// Read every `*.yaml` in `dir`.
    ///
    /// A file that does not parse is skipped with a warning rather than taking
    /// the library down — the same call the registry makes, and for the same
    /// reason: one malformed mood out of a hundred and sixteen must not cost
    /// the other hundred and fifteen.
    pub fn load(name: &'static str, dir: &Path) -> Self {
        let folder = dir
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or(name)
            .to_string()
            + "/";

        let mut sections = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut paths: Vec<_> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
                .collect();
            // Sorted, so the console's listing does not reshuffle between loads.
            paths.sort();

            for path in paths {
                let Ok(text) = std::fs::read_to_string(&path) else {
                    tracing::warn!(path = %path.display(), "{name}: unreadable, skipping");
                    continue;
                };
                match serde_yaml::from_str::<SectionFile>(&text) {
                    Ok(f) => {
                        let stem = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or_default()
                            .to_string();
                        let template = f.template.unwrap_or_default();
                        sections.push(Section {
                            // The file name is the id when the document does not
                            // give one. It is the thing an author actually types
                            // to find the file again.
                            id: f.id.unwrap_or(stem),
                            category: f.category.unwrap_or_else(|| name.to_string()),
                            chars: template.chars().count(),
                            examples: f.examples.len(),
                            template,
                            description: f.description.unwrap_or_default(),
                        });
                    }
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e, "{name}: does not parse, skipping");
                    }
                }
            }
        }

        tracing::info!("{name}: {} sections from {}", sections.len(), dir.display());
        Self {
            name,
            folder,
            sections,
        }
    }

    pub fn len(&self) -> usize {
        self.sections.len()
    }

    /// How many of these carry provenance examples.
    ///
    /// Worth surfacing: a section with none is selected worse than its
    /// neighbours, and in a library this size the ones nobody finished are not
    /// findable by scrolling.
    pub fn with_examples(&self) -> usize {
        self.sections.iter().filter(|s| s.examples > 0).count()
    }
}

/// The section libraries a world's lens is assembled from.
#[derive(Debug, Clone)]
pub struct Libraries {
    pub responses: Library,
    pub moods: Library,
}

impl Libraries {
    /// Load `responses/` and `moods/` from beside the projection schema.
    ///
    /// Both are ingested **untagged**, which is what makes them shared by every
    /// world — the text and its KV both. That is why they load once here rather
    /// than per world.
    ///
    /// The folders are resolved through [`Source::collection_dir`], which is the
    /// schema's own answer to "where does a folder-backed collection read
    /// from". Joining the mind path here instead would be a second answer to
    /// that question, free to disagree the day the layout changes.
    pub fn load(schema: &Source) -> Self {
        let open = |name: &'static str, folder: &str| match schema.collection_dir(folder) {
            Some(dir) => Library::load(name, &dir),
            // No mind, or a mind without that folder. An empty library, which
            // the console renders as a collection with no sections — the truth,
            // and visibly different from one it failed to load.
            None => Library {
                name,
                folder: format!("{folder}/"),
                sections: Vec::new(),
            },
        };
        Self {
            responses: open("response", "responses"),
            moods: open("mood", "moods"),
        }
    }

    /// The wire shape the console renders, for a world that excludes these
    /// section categories.
    ///
    /// The `rule` and `locked` fields mirror what the projection schema
    /// declares: exactly one response is chosen at turn start and frozen for
    /// the decode, and exactly one mood is held until a spike replaces it.
    pub fn world_wire(&self, excludes: &[String]) -> Value {
        json!({ "collections": [
            self.collection(
                &self.responses,
                excludes,
                "top-k 1 · frozen for the decode",
                "The structural mode of a reply. Selected once at interaction start by top-k \
                 provenance match against the incoming exchange, then frozen — structural mode \
                 cannot change mid-reply without breaking coherence.",
            ),
            self.collection(
                &self.moods,
                excludes,
                "top-k 1 · spiking",
                "Event-driven and threshold-gated rather than drifting. Holds its register until \
                 provenance scores a different one above the spike threshold at a barrier, then \
                 snaps — no blend.",
            ),
        ]})
    }

    fn collection(
        &self,
        lib: &Library,
        excludes: &[String],
        rule: &str,
        description: &str,
    ) -> Value {
        // The libraries are shared and this is a *projection* of them, so the
        // filter happens here rather than at load: `earth` and `battle-cities`
        // read the same 596 files and admit different subsets of them.
        let kept: Vec<Value> = lib
            .sections
            .iter()
            .filter(|s| !excluded(&s.category, excludes))
            .map(Section::wire)
            .collect();
        let dropped = lib.sections.len() - kept.len();

        json!({
            "name": lib.name,
            "folder": lib.folder,
            "rule": rule,
            // Authored in files and shared by every world, so the console does
            // not offer to edit them here. Nothing about them is immutable —
            // they are edited where they live.
            "locked": true,
            "source": "the mind, untagged and shared by every world",
            "description": description,
            // Reported rather than silently absent. A collection that is 546 of
            // 596 because this world excludes two categories is a fact about
            // the world, and a reader counting rows should be able to see why
            // the number is not the one in the startup log.
            "excluded": dropped,
            "excludes": excludes,
            "sections": kept,
        })
    }
}

/// What a personality contributes to a projection, read from its own document.
///
/// The counterpart of [`Libraries::world_wire`], and the same idea: the console
/// asks what a character's shared layer is made of, and the answer is in the
/// files. This one takes the document rather than a library, because a
/// personality's sections are *in* it — `anchor` is the always-resident self,
/// `personality:` is a map of facets, and `doctrine` is the part that evolves.
///
/// This replaced a fixture that invented an anchor, four identity facets and a
/// doctrine for every character alike — the same failure the module header
/// describes for worlds, on the page where an author checks what they wrote.
/// Of the mind's 74 personalities, all 74 carry an `anchor`, 69 carry facets
/// and 5 carry a doctrine; a document without one gets a collection with no
/// sections, which is the truth and is visibly different from a failure.
pub fn personality_wire(body: &Value) -> Value {
    let mut out = vec![collection_of(
        "identity_anchor",
        "personalities/<id>.yaml · anchor",
        "always-visible",
        "The always-on compressed self. Structurally resident — it never competes for the \
         gather budget, because it is the prefix the budget is read inside.",
        anchor_sections(body),
    )];

    let facets = facet_sections(body);
    out.push(collection_of(
        "identity",
        "personalities/<id>.yaml · personality",
        "top-k 3",
        "Detail facets of the same self, surfaced only when relevant to the exchange.",
        facets,
    ));

    // Only the documents that have one. An empty doctrine collection on the 69
    // characters without a doctrine would be five rows of nothing to read.
    if body.get("doctrine").and_then(Value::as_str).is_some() {
        out.push(collection_of(
            "doctrine",
            "personalities/<id>.yaml · doctrine",
            "always-visible",
            "The one part of the shared layer designed to change, aggregated from strategic \
             learning and published as a version.",
            doctrine_sections(body),
        ));
    }

    json!({ "collections": out })
}

fn collection_of(
    name: &str,
    folder: &str,
    rule: &str,
    description: &str,
    sections: Vec<Value>,
) -> Value {
    json!({
        "name": name,
        "folder": folder,
        "rule": rule,
        // Authored in a file, and edited on the personality page rather than
        // here — the same rule the world libraries follow.
        "locked": true,
        "source": "personality",
        "description": description,
        "excluded": 0,
        "excludes": Vec::<String>::new(),
        "sections": sections,
    })
}

/// One section per named facet under `personality:`, in the document's order.
fn facet_sections(body: &Value) -> Vec<Value> {
    let Some(map) = body.get("personality").and_then(Value::as_object) else {
        return Vec::new();
    };
    map.iter()
        .filter_map(|(key, value)| {
            let text = value.as_str()?;
            Some(section_wire(key, "identity", text))
        })
        .collect()
}

fn anchor_sections(body: &Value) -> Vec<Value> {
    body.get("anchor")
        .and_then(Value::as_str)
        .map(|t| vec![section_wire("anchor", "identity", t)])
        .unwrap_or_default()
}

fn doctrine_sections(body: &Value) -> Vec<Value> {
    body.get("doctrine")
        .and_then(Value::as_str)
        .map(|t| vec![section_wire("current", "doctrine", t)])
        .unwrap_or_default()
}

/// The same shape [`Section::wire`] produces, so the console renders a
/// personality's collections with the code it already has.
///
/// `chars`, never tokens — there is no tokenizer here, and the module header
/// says why a plausible-looking guess is the thing to avoid. `examples` is 0
/// rather than absent: a personality facet has no provenance lead-ins at all,
/// which is a real zero and not a missing measurement.
fn section_wire(id: &str, category: &str, text: &str) -> Value {
    json!({
        "id": id,
        "category": category,
        "chars": text.chars().count(),
        "examples": 0,
        "template": text,
        "description": "",
    })
}

/// Whether a section's category is one this world does not admit.
///
/// Compared case-insensitively and trimmed, because the list is hand-written in
/// a world's YAML and `Sexual ` should not silently admit what `sexual`
/// excludes.
fn excluded(category: &str, excludes: &[String]) -> bool {
    excludes
        .iter()
        .any(|e| e.trim().eq_ignore_ascii_case(category.trim()))
}

/// The section categories a world's document says it does not admit.
///
/// An absent or malformed `excludes` is an empty list — every category. A world
/// that says nothing admits everything, which is what a world that predates
/// this field means.
pub fn excludes_of(body: &Value) -> Vec<String> {
    body.get("excludes")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-collections-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    #[test]
    fn a_folder_of_sections_loads_with_its_examples_counted() {
        let dir = tmp("load");
        write(
            &dir,
            "berserk.yaml",
            "id: berserk\ncategory: mood\ndescription: Pure force.\n\
             template: |\n  The tactical self is gone.\nexamples:\n  - note: a\n  - note: b\n",
        );
        write(
            &dir,
            "calm.yaml",
            "id: calm\ncategory: mood\ntemplate: Steady.\n",
        );

        let lib = Library::load("mood", &dir);
        assert_eq!(lib.len(), 2);
        assert_eq!(lib.with_examples(), 1);

        let b = &lib.sections[0];
        assert_eq!(b.id, "berserk");
        assert_eq!(b.examples, 2, "examples are counted, not carried");
        assert!(b.chars > 0);
        assert!(b.template.contains("tactical self"));
        assert_eq!(b.description, "Pure force.");
    }

    /// The file name is the id when the document omits one — it is what an
    /// author types to find the file again.
    #[test]
    fn a_file_without_an_id_is_named_by_its_file() {
        let dir = tmp("stem");
        write(&dir, "bait_then_trap.yaml", "template: Lure them in.\n");
        let lib = Library::load("response", &dir);
        assert_eq!(lib.sections[0].id, "bait_then_trap");
        assert_eq!(lib.sections[0].category, "response");
    }

    /// One malformed file out of six hundred must not cost the other 599.
    #[test]
    fn a_broken_file_is_skipped_and_the_rest_still_load() {
        let dir = tmp("broken");
        write(&dir, "good.yaml", "id: good\ntemplate: Fine.\n");
        write(&dir, "broken.yaml", "id: [unclosed\n");
        let lib = Library::load("mood", &dir);
        assert_eq!(lib.len(), 1);
        assert_eq!(lib.sections[0].id, "good");
    }

    /// Listing order is stable, because a list that reshuffles between loads
    /// looks broken even when it is not.
    #[test]
    fn listing_order_is_stable() {
        let dir = tmp("order");
        for n in ["zeta", "alpha", "mid"] {
            write(
                &dir,
                &format!("{n}.yaml"),
                &format!("id: {n}\ntemplate: x\n"),
            );
        }
        let ids: Vec<_> = Library::load("response", &dir)
            .sections
            .iter()
            .map(|s| s.id.clone())
            .collect();
        assert_eq!(ids, ["alpha", "mid", "zeta"]);
    }

    /// A daemon with no mind has no libraries, and says so by being empty
    /// rather than by failing to start.
    #[test]
    fn no_mind_is_an_empty_library_not_a_failure() {
        let libs = Libraries::load(&Source::resolve(None).unwrap());
        assert_eq!(libs.responses.len(), 0);
        assert_eq!(libs.moods.len(), 0);
        let wire = libs.world_wire(&[]);
        assert_eq!(wire["collections"].as_array().unwrap().len(), 2);
    }

    /// **A world admits a subset of the shared libraries.**
    ///
    /// The libraries are one copy, read once; which categories reach a given
    /// world is a projection of them. `earth` takes the adult categories and
    /// `battle-cities` does not, from the same 596 files.
    #[test]
    fn a_world_excludes_the_categories_it_names() {
        let dir = tmp("excl");
        write(
            &dir,
            "chase.yaml",
            "id: chase\ncategory: sexual\ntemplate: a\n",
        );
        write(
            &dir,
            "caress.yaml",
            "id: caress\ncategory: intimate\ntemplate: b\n",
        );
        write(
            &dir,
            "flank.yaml",
            "id: flank\ncategory: combat\ntemplate: c\n",
        );
        let libs = Libraries {
            responses: Library::load("response", &dir),
            moods: Library::load("mood", Path::new("nowhere")),
        };

        // No exclusions: everything.
        let all = libs.world_wire(&[]);
        assert_eq!(
            all["collections"][0]["sections"].as_array().unwrap().len(),
            3
        );
        assert_eq!(all["collections"][0]["excluded"], 0);

        // The adult categories withheld.
        let ex = ["sexual".to_string(), "intimate".to_string()];
        let some = libs.world_wire(&ex);
        let kept = some["collections"][0]["sections"].as_array().unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0]["id"], "flank");
        assert_eq!(
            some["collections"][0]["excluded"], 2,
            "the count of what was withheld is reported, not silently absent"
        );
    }

    /// The list is hand-written in a world's YAML, so casing and stray spaces
    /// must not silently admit what the author meant to exclude.
    #[test]
    fn an_exclusion_is_matched_loosely_enough_to_be_typed_by_hand() {
        assert!(excluded("sexual", &["Sexual".into()]));
        assert!(excluded("sexual", &["  sexual  ".into()]));
        assert!(excluded(" intimate", &["INTIMATE".into()]));
        assert!(!excluded("combat", &["sexual".into()]));
        // But not a prefix — `sex` must not exclude `sexual` by accident, nor
        // the reverse.
        assert!(!excluded("sexual", &["sex".into()]));
    }

    #[test]
    fn a_world_that_says_nothing_admits_everything() {
        assert!(excludes_of(&json!({ "name": "Sandbox" })).is_empty());
        assert!(excludes_of(&json!({ "excludes": "sexual" })).is_empty());
        assert_eq!(
            excludes_of(&json!({ "excludes": ["sexual", "intimate"] })),
            ["sexual", "intimate"]
        );
    }

    /// The console's shape, and no invented token count — `chars` is a thing
    /// this daemon can actually count.
    #[test]
    fn the_wire_shape_reports_only_what_was_measured() {
        let dir = tmp("wire");
        write(&dir, "a.yaml", "id: a\ncategory: affect\ntemplate: abc\n");
        let libs = Libraries {
            responses: Library::load("response", &dir),
            moods: Library::load("mood", &dir),
        };
        let w = libs.world_wire(&[]);
        let s = &w["collections"][0]["sections"][0];
        assert_eq!(s["id"], "a");
        assert_eq!(s["category"], "affect");
        assert_eq!(s["chars"], 3);
        assert_eq!(s["examples"], 0);
        assert!(s.get("tokens").is_none(), "a guessed token count came back");
    }
}
