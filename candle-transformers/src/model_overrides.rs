//! Local model overrides — `models.override.yaml` at the workspace root.
//!
//! # Why this exists
//!
//! A deployment's checkpoint is not always the repository's business. A private
//! fine-tune, a licensed conversion, a model somebody has no right to
//! redistribute: each is a legitimate thing to *run* and a bad thing to commit.
//! Without somewhere to put them, the choice is to name them in the source and
//! publish coordinates that were never meant to be public, or to keep a patched
//! working tree forever and re-apply it after every pull.
//!
//! So the repository names the models it can name, and one gitignored file
//! replaces them on the machine that runs them. Nothing about the override is
//! visible in the source tree, and nothing about the source tree has to change
//! to use one.
//!
//! # How it reaches the process
//!
//! Read once, at first use, from the workspace root. The path is this crate's
//! manifest directory's parent, fixed at compile time, so it does not depend on
//! the working directory a binary or a test runs from. An absent file is an
//! empty document, and an empty document overrides nothing; an edit takes effect
//! at the next process start, with nothing to rebuild.
//!
//! **Not a build script.** Embedding the file through `OUT_DIR` needs
//! `cargo:rerun-if-changed` on it, and cargo treats a watched path that does not
//! exist as stale on every invocation — so on every machine without an override
//! every cargo command reran the script and recompiled this crate, 70–80 s in
//! release. A binary copied to another machine reads the path it was built at:
//! where nothing is there it runs the repository's own checkpoints, the same
//! answer a clone without the file gets; where the path cannot even be checked
//! (another user's home directory), it stops rather than guess.
//!
//! # Two sections, because there are two kinds of coordinate
//!
//! ```yaml
//! # Presets in candle-conversation's `Model` registry, addressed by variant
//! # name. These carry a whole `ModelSpec`, adapters included.
//! models:
//!   Qwen35_9B_Q6:
//!     repo: SomeOrg/Some-Finetune-GGUF
//!     filename: Some-Finetune-Q6_K.gguf
//!     bytes: 8758786272
//!     loras:
//!       - name: rp
//!         repo: SomeOrg/Some-LoRA
//!         revision: main
//!
//! # Checkpoints named directly by the engine and its gates — the eval and
//! # integration-test models, which have no `ModelSpec` and are just a repo, a
//! # revision and a file.
//! checkpoints:
//!   Llama3_2_3B:
//!     repo: SomeOrg/Some-Llama-Conversion-GGUF
//!     filename: model-Q4_K_M.gguf
//! ```
//!
//! Every field is optional except the key, and an override only replaces what it
//! names. What it *cannot* reach is deliberate: architecture, tokenizer and
//! sampling defaults stay with the preset, because a file that could change
//! those could turn a preset into a different model while every test in the
//! repository still described the old one.

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::OnceLock;

use serde::Deserialize;

/// Where the override lives: the workspace root, one level above this crate.
const OVERRIDE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models.override.yaml");

/// The override document, read once per process.
///
/// Empty when no `models.override.yaml` exists, which is the ordinary case for
/// anyone who has not written one.
fn document_text() -> &'static str {
    static TEXT: OnceLock<String> = OnceLock::new();
    TEXT.get_or_init(|| read_document(Path::new(OVERRIDE_PATH)))
}

/// The document at `path`, or an empty one when there is no file there.
///
/// Any other failure panics rather than reading as empty — a file that is there
/// and unreadable, or a path that cannot be checked at all — for the reason
/// [`document`] gives: an override that is silently not applied loads the wrong
/// model.
fn read_document(path: &Path) -> String {
    match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == ErrorKind::NotFound => String::new(),
        Err(e) => panic!(
            "models.override.yaml at {} could not be read: {e}. Fix its permissions \
             or remove it; an override that cannot be read is not applied, and \
             running the wrong model silently is worse than failing here.",
            path.display()
        ),
    }
}

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Document {
    #[serde(default)]
    models: HashMap<String, ModelOverride>,
    #[serde(default)]
    checkpoints: HashMap<String, CheckpointOverride>,
}

/// One registry preset's replacement. Every field optional; an absent one keeps
/// what the preset declared.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelOverride {
    pub repo: Option<String>,
    pub filename: Option<String>,
    /// The commit to resolve [`Self::repo`] at. Absent means `main`, which moves.
    ///
    /// Naming a `repo` **clears** the preset's own pin rather than inheriting it: a SHA
    /// identifies a commit in the repository that produced it and names nothing in another.
    /// So an override that replaces the checkpoint and wants it held still names a revision
    /// here; one that names a revision and no repo pins the preset's own checkpoint.
    pub revision: Option<String>,
    /// The published file's exact length. Downloaders use it for the progress
    /// total when the server omits `Content-Length`, so an override that
    /// changes the file and not this draws a bar that finishes early or never.
    pub bytes: Option<u64>,
    /// Replaces the preset's adapters entirely rather than adding to them.
    ///
    /// Replacement, not merge, because the common case is "this is my model and
    /// these are its adapters" — a merge would leave the repository's example
    /// adapter attached to a checkpoint it was not trained against, which loads
    /// and then quietly degrades whatever asks for it.
    pub loras: Option<Vec<LoraOverride>>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoraOverride {
    pub name: String,
    pub repo: String,
    #[serde(default = "main_revision")]
    pub revision: String,
}

/// A directly-named checkpoint: what the eval binaries and the integration
/// gates fetch.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CheckpointOverride {
    repo: Option<String>,
    revision: Option<String>,
    filename: Option<String>,
}

/// A checkpoint's three coordinates, after any override.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub repo: String,
    pub revision: String,
    pub filename: String,
}

impl Checkpoint {
    pub fn new(repo: &str, revision: &str, filename: &str) -> Self {
        Self {
            repo: repo.to_owned(),
            revision: revision.to_owned(),
            filename: filename.to_owned(),
        }
    }
}

fn main_revision() -> String {
    "main".to_owned()
}

/// Parse the workspace's document, or panic with the parse error.
///
/// **Panicking is correct here.** This runs while resolving which checkpoint to
/// load, and the alternatives are worse in the same way: a silently-ignored
/// override loads the wrong model, and an error returned this deep would have to
/// be swallowed by callers with no way to act on it. A malformed override is a
/// configuration error on the machine that wrote it, discovered on the first
/// run, with the parse error in hand.
fn document() -> Document {
    let text = document_text();
    // Whitespace-only covers both the absent file (read as an empty document)
    // and a file holding only comments.
    if text.trim().is_empty() {
        return Document::default();
    }
    serde_yaml::from_str(text).unwrap_or_else(|e| {
        panic!(
            "models.override.yaml is malformed: {e}\n\
             It is read from the workspace root at startup. Fix it or delete it; \
             an override that cannot be parsed is not applied, and running the wrong \
             model silently is worse than failing here."
        )
    })
}

/// The override for a registry preset, if the document names one.
///
/// Returned raw rather than applied, because the type it applies to
/// (`candle_conversation::models::ModelSpec`) lives a crate above this one.
pub fn model(key: &str) -> Option<ModelOverride> {
    document().models.get(key).cloned()
}

/// Resolve a directly-named checkpoint, falling back to the repository's own
/// coordinates.
///
/// `default` is what the repository declares; whatever the document names
/// replaces it field by field. So an override may move a checkpoint to a
/// different repo while keeping its filename, which is the common case when a
/// conversion is re-hosted.
pub fn checkpoint(key: &str, default: Checkpoint) -> Checkpoint {
    let Some(o) = document().checkpoints.get(key).cloned() else {
        return default;
    };
    Checkpoint {
        repo: o.repo.unwrap_or(default.repo),
        revision: o.revision.unwrap_or(default.revision),
        filename: o.filename.unwrap_or(default.filename),
    }
}

/// Every key the document overrides, prefixed by section, sorted.
///
/// For logging at startup. An operator looking at a console that names a model
/// they did not expect should be able to find out in one line whether an
/// override did it, rather than going looking for the file.
pub fn active_keys() -> Vec<String> {
    let d = document();
    let mut k: Vec<String> = d
        .models
        .keys()
        .map(|s| format!("models.{s}"))
        .chain(d.checkpoints.keys().map(|s| format!("checkpoints.{s}")))
        .collect();
    k.sort();
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(yaml: &str) -> Document {
        serde_yaml::from_str(yaml).expect("parse")
    }

    /// **The workspace's document must parse.** It is read from the workspace
    /// root, so this runs against whatever is really on this machine — an
    /// override file with a typo fails here rather than at the first model load.
    #[test]
    fn the_workspace_document_parses() {
        let _ = document();
        let _ = active_keys();
    }

    /// **An absent file is an empty document; a present one is read verbatim.**
    /// The first is every clone without an override, so it must not be an
    /// error — and it must not need a rebuild to notice the file appearing.
    #[test]
    fn a_missing_file_is_an_empty_document_and_a_present_one_is_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("models.override.yaml");
        assert_eq!(read_document(&path), "");
        std::fs::write(&path, "models: {}\n").expect("write the override");
        assert_eq!(read_document(&path), "models: {}\n");
    }

    /// No file, an empty one, or one that is only comments: no entries.
    #[test]
    fn an_empty_document_overrides_nothing() {
        for y in ["", "# just a comment\n", "models: {}", "checkpoints: {}"] {
            let d = parse(y);
            assert!(d.models.is_empty() && d.checkpoints.is_empty(), "{y:?}");
        }
    }

    /// A checkpoint with no override resolves to exactly what the repository
    /// declared. This is the path every clone without an override file takes.
    #[test]
    fn an_unoverridden_checkpoint_is_the_repositorys_own() {
        let d = Checkpoint::new("Org/Repo-GGUF", "main", "model-Q4_K_M.gguf");
        // A key nothing could plausibly name, so this holds even on a machine
        // that does have an override file.
        let got = checkpoint("__no_such_checkpoint_key__", d.clone());
        assert_eq!(got, d);
    }

    /// A checkpoint override replaces field by field, so a re-hosted conversion
    /// can move repo while keeping its filename.
    #[test]
    fn a_checkpoint_override_replaces_only_what_it_names() {
        let doc = parse("checkpoints:\n  Llama3_2_3B:\n    repo: Other/Repo-GGUF\n");
        let o = doc.checkpoints.get("Llama3_2_3B").cloned().unwrap();
        let default = Checkpoint::new("Org/Repo-GGUF", "main", "model-Q4_K_M.gguf");
        let got = Checkpoint {
            repo: o.repo.unwrap_or(default.repo.clone()),
            revision: o.revision.unwrap_or(default.revision.clone()),
            filename: o.filename.unwrap_or(default.filename.clone()),
        };
        assert_eq!(got.repo, "Other/Repo-GGUF");
        assert_eq!(got.revision, "main", "unnamed fields keep the default");
        assert_eq!(got.filename, "model-Q4_K_M.gguf");
    }

    /// Both sections parse together, and neither requires the other.
    #[test]
    fn the_two_sections_are_independent() {
        let d = parse(
            r#"
models:
  Qwen35_9B_Q6:
    repo: Org/Model-GGUF
checkpoints:
  Llama3_2_3B:
    repo: Org/Llama-GGUF
    filename: model-Q4_K_M.gguf
"#,
        );
        assert_eq!(d.models.len(), 1);
        assert_eq!(d.checkpoints.len(), 1);
        assert_eq!(
            d.models["Qwen35_9B_Q6"].repo.as_deref(),
            Some("Org/Model-GGUF")
        );
        assert_eq!(
            d.checkpoints["Llama3_2_3B"].filename.as_deref(),
            Some("model-Q4_K_M.gguf")
        );
        // Only `models` — `checkpoints` defaults empty rather than failing.
        assert!(parse("models: {}").checkpoints.is_empty());
    }

    /// An adapter with no pin gets `main`, spelled out rather than left for a
    /// consumer to guess.
    #[test]
    fn an_unpinned_adapter_defaults_to_main() {
        let d = parse("models:\n  M:\n    loras:\n      - name: x\n        repo: a/b\n");
        assert_eq!(d.models["M"].loras.as_ref().unwrap()[0].revision, "main");
    }

    /// **A misspelled field is an error, not a no-op.**
    ///
    /// `deny_unknown_fields` throughout. Without it, `filenmae:` parses happily
    /// and does nothing — the loader fetches the repository's file while the
    /// operator reads their override and believes it took effect. The whole
    /// value of this file is that it is obeyed.
    #[test]
    fn an_unknown_field_is_refused() {
        for bad in [
            "modles: {}",
            "models:\n  M:\n    filenmae: typo.gguf\n",
            "checkpoints:\n  C:\n    fileame: typo.gguf\n",
            "models:\n  M:\n    loras:\n      - name: x\n        repo: a/b\n        rev: c\n",
        ] {
            assert!(
                serde_yaml::from_str::<Document>(bad).is_err(),
                "silently accepted {bad:?}"
            );
        }
    }
}
