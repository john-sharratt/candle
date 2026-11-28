//! Each tool's definition: the declarative YAML a model is shown and the
//! constrained decoder compiles — its name, category, description,
//! parameters, calibration examples and question seeds — kept beside the code
//! that runs it (`definitions/*.yaml`) and embedded at compile time.
//!
//! zend shows these to its own conversations (`zend::tool_def`, which adds what
//! belongs to the deployment: the workspace's repositories, the modes, a
//! mind's own tools folder), and npcd builds the acts a personality reaches
//! from the same ones, so a tool is described once. Only a YAML file that is
//! not a dot-file is a definition: folder metadata sits beside them and would
//! fail to parse as one.

use std::path::Path;
use std::sync::LazyLock;

use include_dir::{include_dir, Dir};
use serde::Deserialize;
use serde_json::Value;

/// The definitions, embedded at compile time.
static DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/definitions");

/// The bundled definitions, parsed once, sorted by name.
static BUNDLED: LazyLock<Vec<Definition>> = LazyLock::new(load_bundled);

/// One tool definition parsed from a `definitions/*.yaml` file.
#[derive(Debug, Clone, Deserialize)]
pub struct Definition {
    pub name: String,
    #[serde(default)]
    pub category: String,
    pub description: String,
    #[serde(default)]
    pub high_risk: bool,
    /// Projected on every turn, outside the `tools` collection's top-k:
    /// provenance still selects its usual number of tools, and a mandatory one
    /// is added on top rather than taking a slot. For the tools a coding turn
    /// needs whatever the question — finding a file, finding code inside one,
    /// reading and listing, and creating one.
    ///
    /// `write` is in the set because the edit tool cannot create a file: a
    /// "create scratch/x.py" turn that was projected `file_edit` but not `write`
    /// spent ten calls patching a file that did not exist, then printed the code
    /// and gave up.
    ///
    /// `web_search` and `web_fetch` are in it because a tool shown only by name
    /// in the catalog listing is, to the model, not really there: asked to
    /// "search the web and cite your source", a turn projected without
    /// `web_search` reasoned that the tool was "only in the descriptions, not my
    /// actual tool list" and answered that it could not search. Looking something
    /// up is as ordinary a coding-assistant step as reading a file. A mode that
    /// does not grant the network drops them with the rest of its exclusions.
    ///
    /// `journal_read` and `journal_grep` are in it because the journal shows
    /// one line per entry and its intro tells the model the turns behind each
    /// are a call away: a turn asked about earlier work that was not offered
    /// them would read the line, and answer from it, as if that were all there
    /// is.
    ///
    /// The find tools are there with reading and listing, not left out. A turn
    /// that can only enumerate and read answers "where is this" by guessing a
    /// directory at `file_list` and reading whole files to check, which is what
    /// `file_search` and `file_grep` were added to stop; leaving them to win a
    /// belief slot meant the turns that most needed them — the ones with no
    /// path in the question to score against — were exactly the turns that did
    /// not get them.
    #[serde(default)]
    pub mandatory: bool,
    /// Called at most once a turn: once a call names it, the tool-call grammar
    /// offers it to no later call of the same turn
    /// (`candle_conversation::stencil::NeverTwice`). For a tool whose second
    /// call in a turn is never what the turn needs — `journal_note`, the note a
    /// round opens with, which a model wrote twice, word for word, in place of
    /// the call its thinking had planned.
    #[serde(default)]
    pub never_twice: bool,
    /// JSON Schema for the call arguments (the tool's Request type).
    pub parameters: Value,
    /// ChatML selection-calibration trajectories (prompt + `<|im_end|>
    /// <|im_start|>assistant` + think→call with projection markers), or a bare
    /// prompt for a tool whose trajectories are not recorded yet. A trajectory
    /// is prefilled on its own; a bare prompt has no answer to prefill, so it
    /// is calibrated as the question it is.
    #[serde(default)]
    pub examples: Vec<String>,
    /// Extra ways a user might ASK for this tool — one plain question each, no
    /// reasoning block and no answer.
    ///
    /// Routing happens on the question. An `examples` trajectory is ~95% think
    /// block and call, so its question — the part a live probe resembles — is a
    /// few percent of the exemplar; the phase lens
    /// (`docs/provenance_score_normalization.md` §3.4) reads exactly that
    /// region, and these are entries that are *only* that region — so a tool can
    /// carry many more phrasings than it could afford as full examples.
    ///
    /// **All of a tool's questions are calibrated in ONE submission**, a turn
    /// group whose cases prefill together, each masked to itself, and seal one
    /// turn each (`candle_conversation::turn_group`). They are therefore resumed
    /// as a unit: the group's marker covers every question, so editing any one
    /// of them regenerates the whole set.
    #[serde(default)]
    pub questions: Vec<String>,
}

impl Definition {
    /// The Hermes `{"name","description","parameters"}` line rendered into the
    /// `<tools>…</tools>` block. Flat shape — no `{"type":"function",…}`
    /// wrapper.
    pub fn json_line(&self) -> String {
        let blob = serde_json::json!({
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        });
        serde_json::to_string(&blob).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Whether `path` names a tool definition: a `*.yaml` file that is not a
/// dot-file.
///
/// The dot-file exclusion is not cosmetic. The folder-metadata pass writes a
/// `.substrate.yaml` (a folder summary with its questions grouped by kind)
/// into every folder it describes — this one included, since it lives in the
/// source tree — and the compile-time embed picks it up with the definitions.
/// Parsed as a tool it fails on every boot (`questions` is a map there, a
/// list here), and no tool definition is ever a dot-file.
pub fn is_definition_file(path: &Path) -> bool {
    let hidden = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with('.'));
    !hidden && path.extension().and_then(|e| e.to_str()) == Some("yaml")
}

/// Parse one tool definition from `text`, read from `loc`; a malformed one is
/// logged and skipped.
pub fn parse(loc: &str, text: &str) -> Option<Definition> {
    match serde_yaml::from_str::<Definition>(text) {
        Ok(def) => Some(def),
        Err(e) => {
            tracing::error!(file = %loc, "tool definition parse failed: {e}");
            None
        }
    }
}

/// Every bundled definition's file, as `(path, text)`: what [`bundled`] parses,
/// for a check that every one of them does.
pub fn files() -> impl Iterator<Item = (&'static Path, Option<&'static str>)> {
    DIR.files()
        .filter(|f| is_definition_file(f.path()))
        .map(|f| (f.path(), f.contents_utf8()))
}

fn load_bundled() -> Vec<Definition> {
    let mut defs: Vec<Definition> = files()
        .filter_map(|(path, text)| parse(&path.display().to_string(), text?))
        .collect();
    defs.sort_by(|a, b| a.name.cmp(&b.name));
    defs
}

/// The bundled definitions, sorted by name.
pub fn bundled() -> &'static [Definition] {
    &BUNDLED
}

/// The bundled definition of tool `name`.
pub fn find(name: &str) -> Option<&'static Definition> {
    bundled().iter().find(|d| d.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry;

    /// **Every bundled file parses**, and the definitions of tools this crate
    /// runs name one it does — a definition with no tool behind it is a call
    /// nothing answers. (Some definitions are of tools another server runs:
    /// npcd's pictures and bots, zend's own.)
    #[test]
    fn every_definition_parses_and_names_a_tool() {
        let unparsed: Vec<String> = files()
            .filter(|(path, text)| {
                text.is_none_or(|t| parse(&path.display().to_string(), t).is_none())
            })
            .map(|(path, _)| path.display().to_string())
            .collect();
        assert_eq!(unparsed, Vec::<String>::new());
        assert_eq!(bundled().len(), files().count());
        for name in [
            "weather",
            "web_search",
            "http_request",
            "notes_write",
            "guide_read",
        ] {
            let def = find(name).unwrap_or_else(|| panic!("{name} has no definition"));
            assert!(registry::find(&def.name).is_some(), "{name} runs nowhere");
        }
    }

    /// The folder-metadata pass leaves a `.substrate.yaml` in the definitions
    /// folder itself, and the compile-time embed carries it along; read as a
    /// tool it failed to parse on every boot. A definition is a `*.yaml` that
    /// is not a dot-file.
    #[test]
    fn folder_metadata_is_not_a_tool_definition() {
        assert!(!is_definition_file(Path::new(".substrate.yaml")));
        assert!(!is_definition_file(Path::new("tools/.substrate.yaml")));
        assert!(!is_definition_file(Path::new("tools/.hidden.yaml")));
        assert!(is_definition_file(Path::new("tools/calculator.yaml")));
        assert!(!is_definition_file(Path::new("tools/calculator.yml")));
        assert!(!is_definition_file(Path::new("tools/README.md")));
    }

    /// **A definition's line is its name, description and parameters, flat.**
    #[test]
    fn a_definition_line_is_flat() {
        let line: Value = serde_json::from_str(&find("calculator").unwrap().json_line()).unwrap();
        assert_eq!(line["name"], "calculator");
        assert!(line.get("type").is_none());
        assert_eq!(line["parameters"]["type"], "object");
    }
}
