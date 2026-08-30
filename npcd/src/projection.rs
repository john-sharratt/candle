//! Where the projection schema comes from.
//!
//! One rule, borrowed from `zend`: a `projection.yaml` in the *mind directory*
//! overrides the one compiled into the binary. `zend` spells this
//! `--working-dir`, which for it also relocates the substrate; `npcd` already
//! has `--data` for that, so the two concerns stay separate and this flag moves
//! only the schema.
//!
//! # The schema and its folders travel together
//!
//! A collection in the schema declares its discipline and leaves `sections: []`
//! — the members are read from a directory at startup. Those directories are
//! resolved relative to *the directory holding the schema*, never to the data
//! directory or the cwd.
//!
//! That is the whole reason this is a directory flag rather than a file path.
//! `--mind ../mind` moves `projection.yaml` and the `responses/`, `moods/`,
//! `personalities/` beside it as one unit; pointing the schema at one folder and
//! its content at another is not expressible, which is the point. A schema
//! separated from its libraries parses perfectly and projects nothing.

use std::path::{Path, PathBuf};

/// The schema compiled into the binary — `npcd/projection.yaml`, the checked-in
/// default. Embedded rather than read from the source tree so a released binary
/// carries a working schema with no files beside it.
pub const BUNDLED: &str = include_str!("../projection.yaml");

/// The file a mind directory is recognised by. A directory without one is not a
/// mind, and naming it is an error rather than a silent fallback to the bundled
/// default — see [`Source::resolve`].
pub const SCHEMA_FILE: &str = "projection.yaml";

/// Where the active schema came from, and the directory its collections resolve
/// against.
#[derive(Debug, Clone)]
pub struct Source {
    /// The schema text, ready to parse.
    ///
    /// No production reader yet — the engine that consumes a projection schema
    /// is not written, and `resolve` only proves the text is YAML. It is kept
    /// because the tests below build a real `Builder` from it and assert the
    /// schema's *shape*: nine layers, in order, at the declared thresholds.
    /// That is a much stronger check than parsing, and this field is how it is
    /// reached.
    #[allow(dead_code)]
    pub yaml: String,
    /// The directory holding the schema — the root for every folder-backed
    /// collection. `None` for the bundled schema, which has no folders: a
    /// binary with no mind directory has nothing to read them from, and
    /// defaulting to the data directory would let an unrelated `worlds/` folder
    /// be mistaken for a collection.
    pub dir: Option<PathBuf>,
    /// For logging. The path, or the word `bundled`.
    pub label: String,
}

/// The layers the mind's schema declares, in the order it writes them.
///
/// Read through the mind's own address for `settings/projection`, so this and
/// the layer editor are the same document read the same way — there is no
/// second copy of the nine layers to drift from the first. `None` when the
/// daemon has no mind, or its schema declares no layers.
///
/// Used by `/v1/schema/layers`, and by the substrate view, which needs to know
/// what the layers *are* before it can report that every one of them is empty.
pub fn layers(mind: &crate::mind::Mind) -> Option<Vec<serde_json::Value>> {
    let root = mind.root()?;
    let addr = crate::mind::Address::parse("settings/projection").ok()??;
    let (list_key, id_key) = addr.parts()?;
    let doc = crate::mind::catalog::read(root, &addr).ok()?;
    let items = crate::mind::parts::list(&doc.text, list_key, id_key).ok()?;
    Some(items.into_iter().map(|(_, v)| v).collect())
}

#[derive(Debug)]
pub enum SchemaError {
    /// `--mind` named a directory that does not exist, or is a file.
    NotADirectory(PathBuf),
    /// The directory exists but holds no `projection.yaml`.
    ///
    /// Deliberately fatal. Falling back to the bundled schema here would start
    /// the daemon with none of the mind's content and no obvious sign of it —
    /// the failure would surface much later as characters that behave as though
    /// their libraries were empty, because they were.
    NoSchema(PathBuf),
    Unreadable(PathBuf, std::io::Error),
    /// The schema is there and is not YAML. Fatal for the same reason as
    /// [`SchemaError::NoSchema`]: starting anyway would defer the discovery to
    /// whenever something first tried to read it.
    Malformed(PathBuf, String),
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotADirectory(p) => {
                write!(f, "--mind {} is not a directory", p.display())
            }
            Self::NoSchema(p) => write!(
                f,
                "--mind {} holds no {SCHEMA_FILE} — a mind directory is recognised by that file",
                p.display()
            ),
            Self::Unreadable(p, e) => write!(f, "cannot read {}: {e}", p.display()),
            Self::Malformed(p, e) => write!(f, "{} is not valid YAML: {e}", p.display()),
        }
    }
}

impl std::error::Error for SchemaError {}

impl Source {
    /// Resolve the schema: the mind directory's if one was named, else the
    /// bundled default.
    pub fn resolve(mind: Option<&Path>) -> Result<Self, SchemaError> {
        let Some(dir) = mind else {
            return Ok(Self {
                yaml: BUNDLED.to_string(),
                dir: None,
                label: "bundled".to_string(),
            });
        };

        if !dir.is_dir() {
            return Err(SchemaError::NotADirectory(dir.to_path_buf()));
        }
        let path = dir.join(SCHEMA_FILE);
        if !path.is_file() {
            return Err(SchemaError::NoSchema(dir.to_path_buf()));
        }
        let yaml =
            std::fs::read_to_string(&path).map_err(|e| SchemaError::Unreadable(path.clone(), e))?;

        // Parsed here purely to prove it parses.
        //
        // Nothing in this daemon consumes the schema yet — the engine that
        // will is not written — so a mind whose `projection.yaml` had been left
        // mid-edit would start cleanly and only reveal it whenever that engine
        // first ran. Checking at resolve makes a broken schema a startup
        // failure beside the flag that named it, which is where somebody can
        // still remember what they changed.
        serde_yaml::from_str::<serde_yaml::Value>(&yaml)
            .map_err(|e| SchemaError::Malformed(path.clone(), e.to_string()))?;

        // Absolute, so a later relative folder lookup cannot depend on the
        // process's cwd — which nothing here changes, and which a service
        // manager is free to set anywhere.
        let dir = plain(dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()));
        Ok(Self {
            yaml,
            dir: Some(dir),
            label: path.display().to_string(),
        })
    }

    /// The directory a folder-backed collection named `name` reads from, if
    /// this schema has a home on disk and that folder exists.
    ///
    /// `None` covers both "bundled schema, no folders anywhere" and "this mind
    /// declares no such collection folder", because the caller does the same
    /// thing in either case: the collection stays empty.
    pub fn collection_dir(&self, name: &str) -> Option<PathBuf> {
        let d = self.dir.as_ref()?.join(name);
        d.is_dir().then_some(d)
    }
}

/// Drop Windows' extended-length `\\?\` prefix from a canonicalised path.
///
/// `canonicalize` returns it on Windows, and it leaks: into every log line, and
/// into any path this is later joined onto and handed to something that does
/// not understand the form. It carries no information a plain absolute path
/// lacks here — the prefix exists to lift the 260-character limit, and a mind
/// directory nested that deep has a different problem.
fn plain(p: PathBuf) -> PathBuf {
    match p.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(stripped) => PathBuf::from(stripped),
        None => p,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use candle_conversation::models::Dialect;
    use candle_conversation::projection::{Builder, GatherScope};

    use super::*;

    fn tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-projection-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Parse the way the daemon does.
    ///
    /// A dialect is REQUIRED, not optional: the schema's `kind: template` items
    /// (`system_open` / `system_close`) resolve to a model family's structural
    /// tokens at parse time, and without one the build fails with
    /// `DialectRequired`. Qwen3 is a ChatML family, so that is what the daemon
    /// supplies and what a test of the daemon's schema must supply too.
    fn build() -> Builder {
        let src = Source::resolve(None).unwrap();
        Builder::from_yaml_with_vars_and_dialect(&src.yaml, &[], Some(&Dialect::chat_ml()))
            .expect("bundled projection.yaml must parse")
    }

    #[test]
    fn the_bundled_schema_parses() {
        let src = Source::resolve(None).unwrap();
        assert_eq!(src.label, "bundled");
        assert!(src.dir.is_none());
        build();
    }

    /// The nine layers, by name and order. Order is meaningful — a higher layer
    /// attends to everything below it — so this pins the sequence, not just the
    /// set.
    #[test]
    fn the_schema_declares_the_nine_layers_in_order() {
        let b = build();
        let names: Vec<&str> = b.schema().layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "perception",
                "action",
                "interaction",
                "memory",
                "world",
                "environment",
                "agency",
                "relationships",
                "beliefs",
            ]
        );
    }

    /// `world` is the only cross-timeline layer, and the whole worlds design
    /// turns on it: shared gather scope is what makes a fact common knowledge
    /// between characters rather than one character's recollection. Every other
    /// layer must stay self-local, or one character reads another's memory.
    #[test]
    fn only_the_world_layer_is_cross_timeline() {
        let b = build();
        for l in &b.schema().layers {
            let shared = matches!(l.gather_scope, GatherScope::Shared);
            assert_eq!(
                shared,
                l.name == "world",
                "{} has the wrong gather scope — only `world` may be shared",
                l.name
            );
        }
    }

    /// Thresholds rise with the cost of being wrong. `beliefs` is the highest
    /// because a belief admitted on thin evidence justifies everything
    /// downstream of it, and nothing later re-examines it; observation layers
    /// are ungated because what happened, happened.
    #[test]
    fn thresholds_rise_with_the_cost_of_being_wrong() {
        let b = build();
        let of = |n: &str| {
            b.schema()
                .layers
                .iter()
                .find(|l| l.name == n)
                .unwrap_or_else(|| panic!("no layer {n}"))
                .score_threshold
        };
        for observed in ["perception", "action", "interaction", "environment"] {
            assert_eq!(of(observed), 0.0, "{observed} must be ungated");
        }
        assert!(of("memory") < of("world"));
        assert!(of("world") < of("agency"));
        assert!(of("agency") < of("beliefs"));
        assert_eq!(of("beliefs"), 0.40);
    }

    /// The folder-backed collections are declared empty — the daemon fills them
    /// from the mind's directories at startup. A section authored inline here
    /// would be one the GUI cannot edit and the mind cannot override.
    #[test]
    fn the_folder_backed_collections_are_empty_in_the_schema() {
        use candle_conversation::projection::SystemPromptItem;
        let b = build();
        let mut seen = Vec::new();
        for item in &b.schema().system_prompt.items {
            if let SystemPromptItem::Collection(c) = item {
                assert!(
                    c.sections.is_empty(),
                    "collection `{}` has inline sections — it is folder-backed",
                    c.name
                );
                seen.push(c.name.clone());
            }
        }
        for want in ["identity_anchor", "identity", "response", "mood"] {
            assert!(seen.iter().any(|n| n == want), "missing collection {want}");
        }
    }

    #[test]
    fn a_mind_directory_overrides_the_bundled_schema() {
        let dir = tmp();
        std::fs::write(
            dir.join(SCHEMA_FILE),
            "layers: []\nsystem_prompt:\n  sections:\n    - id: mine\n      content: |+\n        from the mind\n",
        )
        .unwrap();

        let src = Source::resolve(Some(&dir)).unwrap();
        assert!(src.yaml.contains("from the mind"));
        assert_eq!(src.dir, Some(plain(dir.canonicalize().unwrap())));
        Builder::from_yaml(&src.yaml).expect("override must parse");
    }

    /// The reported directory must be a path other tools accept, not the
    /// `\\?\` form `canonicalize` hands back on Windows — it ends up in every
    /// log line and in every folder path joined onto it.
    #[test]
    fn the_resolved_directory_has_no_extended_length_prefix() {
        let dir = tmp();
        std::fs::write(
            dir.join(SCHEMA_FILE),
            "layers: []\nsystem_prompt:\n  sections:\n    - id: a\n",
        )
        .unwrap();
        let src = Source::resolve(Some(&dir)).unwrap();
        let shown = src.dir.as_ref().unwrap().display().to_string();
        assert!(!shown.starts_with(r"\\?\"), "{shown}");
        // Still absolute, which is the property that mattered.
        assert!(src.dir.as_ref().unwrap().is_absolute());
    }

    /// Collections resolve against the schema's own directory.
    #[test]
    fn collection_folders_resolve_beside_the_schema() {
        let dir = tmp();
        std::fs::write(
            dir.join(SCHEMA_FILE),
            "layers: []\nsystem_prompt:\n  sections:\n    - id: a\n",
        )
        .unwrap();
        std::fs::create_dir_all(dir.join("responses")).unwrap();

        let src = Source::resolve(Some(&dir)).unwrap();
        assert!(src.collection_dir("responses").is_some());
        // A folder the mind does not have is absent, not an error — the
        // collection simply stays empty.
        assert!(src.collection_dir("moods").is_none());
    }

    /// The bundled schema has no folders at all, so no collection can resolve.
    /// Falling back to the data directory here would let `worlds/` be read as a
    /// collection.
    #[test]
    fn the_bundled_schema_has_no_collection_folders() {
        let src = Source::resolve(None).unwrap();
        assert!(src.collection_dir("responses").is_none());
        assert!(src.collection_dir("worlds").is_none());
    }

    /// A directory without the file is an error, never a quiet fallback: the
    /// daemon would otherwise run with an empty cast of libraries and look fine.
    #[test]
    fn a_directory_without_a_schema_is_fatal() {
        let dir = tmp();
        match Source::resolve(Some(&dir)) {
            Err(SchemaError::NoSchema(p)) => assert_eq!(p, dir),
            other => panic!("expected NoSchema, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_directory_is_named_in_the_error() {
        let missing = tmp().join("nope");
        match Source::resolve(Some(&missing)) {
            Err(SchemaError::NotADirectory(p)) => assert_eq!(p, missing),
            other => panic!("expected NotADirectory, got {other:?}"),
        }
        // And the message names the path, so the operator can see the typo.
        let msg = Source::resolve(Some(&missing)).unwrap_err().to_string();
        assert!(msg.contains("nope"), "{msg}");
    }
}
