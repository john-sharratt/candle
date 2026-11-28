//! The tool catalog a zend conversation is shown: each tool's definition
//! (`zend_tools::definitions`, the YAML beside the code that runs it), with
//! what belongs to this deployment written in.
//!
//! A behaviour change reaches the model only through its definition. A mind
//! workspace with its own projection may override the catalog from its own
//! tools folder, while a coding workspace always uses the bundled one. Every
//! `repo` parameter is constrained to the workspace's repositories.
//!
//! - `tests_calibration`: tests of calibration markers and question seeds.
//! - `tests_definitions`: tests of definition shape, values and repository parameters.
//! - `tests_modes`: tests of what each tool mode offers.

use std::collections::HashSet;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::OnceLock;

use host::calibration::Calibrated;
use serde_json::Value;
use zend_tools::definitions::{self, is_definition_file, Definition};
use zend_tools::registry;
use zend_vfs::Workspace;

use crate::access;
use crate::types::ToolMode;

/// One tool's definition, as this daemon shows and calibrates it.
///
/// Its own type around [`Definition`] because the calibration phase is the
/// host's ([`Calibrated`]), which a definition — zend-tools' — cannot
/// implement where it is declared; everything else is the definition's.
#[derive(Debug, Clone)]
pub struct ToolDef(pub Definition);

impl Deref for ToolDef {
    type Target = Definition;

    fn deref(&self) -> &Definition {
        &self.0
    }
}

impl DerefMut for ToolDef {
    fn deref_mut(&mut self) -> &mut Definition {
        &mut self.0
    }
}

/// What the calibration phase reads of a definition: its name, the
/// [`Definition::json_line`] the model reads when choosing it (name,
/// description, parameters — so rewording any of them regenerates its cases),
/// and the authored examples and questions.
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

/// The bundled catalog: every tool's own definition.
fn load_bundled() -> Vec<ToolDef> {
    definitions::bundled()
        .iter()
        .cloned()
        .map(ToolDef)
        .collect()
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
        .filter_map(|p| {
            definitions::parse(&p.display().to_string(), &std::fs::read_to_string(p).ok()?)
        })
        .map(ToolDef)
        .collect();
    defs.sort_by(|a, b| a.name.cmp(&b.name));
    defs
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
