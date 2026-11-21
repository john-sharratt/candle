//! The bundled tool definitions: the declarative YAML source for each tool the model
//! is shown and the constrained decoder compiles.
//!
//! Every tool YAML file is embedded at compile time and defines one tool's name,
//! category, description, parameters, calibration examples and question seeds. A
//! behaviour change reaches the model only through its definition. A mind workspace with
//! its own projection may override the catalog from its own tools folder, while a coding
//! workspace always uses the bundled one. Every `repo` parameter is constrained to the
//! workspace's repositories. Only a YAML file that is not a dot-file is a definition, since
//! folder metadata sits beside the definitions and would fail to parse as one.
//!
//! - `tests_calibration`: tests of calibration markers and question seeds.
//! - `tests_definitions`: tests of definition shape, values and repository parameters.
//! - `tests_modes`: tests of what each tool mode offers.

use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;

use host::calibration::Calibrated;
#[cfg(doc)]
use host::calibration::PROJECTION_MARKER;
use include_dir::{include_dir, Dir};
use serde::Deserialize;
use serde_json::Value;
use zend_tools::registry;
use zend_vfs::Workspace;

use crate::access;
use crate::types::ToolMode;

/// The tool definitions, embedded at compile time. The `tools` collection in
/// `prompts/projection.yaml` is filled from here.
static TOOLS_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/src/prompts/tools");

/// One tool definition parsed from a `tools/*.yaml` file.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolDef {
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
    /// <|im_start|>assistant` + think→call with [`PROJECTION_MARKER`]s), or a bare
    /// prompt for a tool whose trajectories are not recorded yet. A trajectory
    /// is prefilled on its own ([`Calibrated::trajectories`]); a bare prompt
    /// has no answer to prefill, so it is calibrated as the question it is
    /// ([`Calibrated::calibration_questions`]).
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

impl ToolDef {
    /// The Hermes `{"name","description","parameters"}` line rendered into the
    /// `<tools>…</tools>` block. Flat shape (no `{"type":"function",…}` wrapper) —
    /// see the rationale on the former `tools::render_tool_json_line`.
    pub fn json_line(&self) -> String {
        let blob = serde_json::json!({
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        });
        serde_json::to_string(&blob).unwrap_or_else(|_| "{}".to_string())
    }
}

/// What the calibration phase reads of a definition: its name, the
/// [`ToolDef::json_line`] the model reads when choosing it (name, description,
/// parameters — so rewording any of them regenerates its cases), and the
/// authored examples and questions.
impl Calibrated for ToolDef {
    fn name(&self) -> &str {
        &self.name
    }

    fn line(&self) -> String {
        self.json_line()
    }

    fn examples(&self) -> &[String] {
        &self.examples
    }

    fn questions(&self) -> &[String] {
        &self.questions
    }
}

/// The effective tool catalog, resolved once per daemon.
static DEFS: OnceLock<Vec<ToolDef>> = OnceLock::new();

/// Resolve the effective tool catalog for this daemon *before* any [`all`] call:
/// the working-dir `<workspace>/tools/*.yaml` when that folder exists (a mind /
/// game's own tools, which fully override the built-ins — an empty folder yields
/// a deliberately tool-free mind), else the bundled built-in catalog (the coding
/// assistant). First call wins; later calls are ignored.
///
/// Every `repo` parameter is then constrained to `workspace`'s repositories
/// ([`constrain_repos`]).
pub fn init(workspace: &Workspace) {
    let mut defs = load_effective(workspace.root());
    constrain_repos(&mut defs, &workspace.names());
    if DEFS.set(defs).is_err() {
        // A prior `all()` already resolved (and locked in) the fallback catalog,
        // so this override is lost. That means a consumer ran before `init` — an
        // ordering bug worth shouting about rather than silently mis-tooling.
        tracing::error!(
            "tool_def::init called after the catalog was already resolved — \
             workspace tool override ignored (a consumer ran before init)"
        );
    }
}

/// The name of the parameter that selects a repository, on every tool that
/// takes one.
pub const REPO_PARAM: &str = "repo";

/// Put `repos` at the head of the `enum` of every tool's [`REPO_PARAM`]
/// parameter.
///
/// The repositories are fixed for the daemon's life, so a `repo` value is one
/// of a known list — exactly what an `enum` states. The names belong to the
/// deployment, so the tool YAML does not carry them and this writes them in.
/// What the YAML *does* declare stays after them: the tools that can cover
/// the whole workspace declare `enum: ["*"]`
/// ([`ALL_REPOS`](zend_vfs::ALL_REPOS)), so their choice
/// is a repository or `*`, and every other tool's is a repository alone.
///
/// One schema feeds both readers: [`ToolDef::json_line`] renders it into the
/// prompt, and `tools::tool_catalog` compiles it into the constrained decoder,
/// where a string `enum` is a choice between the listed values
/// (`candle_conversation::stencil::ToolSpec::from_json_schema`). So the model
/// is shown the values it may use and cannot decode any other.
fn constrain_repos(defs: &mut [ToolDef], repos: &[String]) {
    for def in defs {
        let Some(param) = def
            .parameters
            .get_mut("properties")
            .and_then(|p| p.get_mut(REPO_PARAM))
            .and_then(Value::as_object_mut)
        else {
            continue;
        };
        let declared: Vec<Value> = param
            .get("enum")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let values = repos
            .iter()
            .cloned()
            .map(Value::String)
            .chain(declared)
            .collect();
        param.insert("enum".to_string(), Value::Array(values));
    }
}

/// The effective tool definitions, sorted by name. Falls back to the bundled
/// catalog if [`init`] was never called (tests, or a host that never overrides).
pub fn all() -> &'static [ToolDef] {
    DEFS.get_or_init(load_bundled)
}

fn load_effective(workspace: &Path) -> Vec<ToolDef> {
    // Only a *mind* draws tools from a working-dir `tools/` folder — and a mind is
    // exactly a workspace that carries its own `projection.yaml` override (the same
    // signal `build_projection_builder` uses). A coding-agent workspace is an
    // arbitrary user project that may well contain an unrelated `tools/` directory
    // (e.g. build tooling), so it always uses the bundled built-in catalog.
    let is_mind = workspace.join("projection.yaml").is_file();
    let dir = workspace.join("tools");
    if is_mind && dir.is_dir() {
        load_disk(&dir)
    } else {
        load_bundled()
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
fn is_definition_file(path: &Path) -> bool {
    let hidden = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with('.'));
    !hidden && path.extension().and_then(|e| e.to_str()) == Some("yaml")
}

/// Parse the embedded (compile-time) built-in catalog.
fn load_bundled() -> Vec<ToolDef> {
    let mut defs: Vec<ToolDef> = TOOLS_DIR
        .files()
        .filter(|f| is_definition_file(f.path()))
        .filter_map(|f| parse_def(&f.path().display().to_string(), f.contents_utf8()?))
        .collect();
    defs.sort_by(|a, b| a.name.cmp(&b.name));
    defs
}

/// Parse a working-dir `tools/` folder off disk (a host override).
fn load_disk(dir: &Path) -> Vec<ToolDef> {
    let mut paths: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| is_definition_file(p))
            .collect(),
        Err(_) => return Vec::new(),
    };
    paths.sort();
    let mut defs: Vec<ToolDef> = paths
        .iter()
        .filter_map(|p| parse_def(&p.display().to_string(), &std::fs::read_to_string(p).ok()?))
        .collect();
    defs.sort_by(|a, b| a.name.cmp(&b.name));
    defs
}

/// Parse one tool definition; a malformed file is logged and skipped.
fn parse_def(loc: &str, text: &str) -> Option<ToolDef> {
    match serde_yaml::from_str::<ToolDef>(text) {
        Ok(def) => Some(def),
        Err(e) => {
            tracing::error!(file = %loc, "tool definition parse failed: {e}");
            None
        }
    }
}

/// Look up a tool definition by name.
pub fn find(name: &str) -> Option<&'static ToolDef> {
    all().iter().find(|d| d.name == name)
}

/// The names of the tools projected in `mode` — those [`access::offers`]
/// admits. A mode never offers a tool its grants would refuse: offering it
/// would spend a projection slot on a `not_permitted`.
pub fn names_for(mode: ToolMode) -> HashSet<String> {
    all()
        .iter()
        .filter(|d| {
            registry::find(&d.name).is_some_and(|t| access::offers(mode, t.requires, d.high_risk))
        })
        .map(|d| d.name.clone())
        .collect()
}

/// The catalog category a tool belongs to (the `## <category>` grouping in the
/// tool-catalog summary), or `"Other"` for an unknown name.
pub fn category_for(name: &str) -> &'static str {
    find(name).map(|d| d.category.as_str()).unwrap_or("Other")
}

#[cfg(test)]
mod tests_calibration;
#[cfg(test)]
mod tests_definitions;
#[cfg(test)]
mod tests_modes;
