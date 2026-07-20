//! Bundled tool **definitions** — the declarative, folder-driven source of truth.
//!
//! Every `*.yaml` under `src/prompts/tools/` (embedded at compile time) is one
//! tool definition following `src/prompts/tool.schema.json`: name, category,
//! description, high-risk flag, the `parameters` JSON Schema, and ChatML
//! selection-calibration `examples`. These drive everything the prompt and
//! calibration need — the tool catalog surfaced into the `tools` collection, the
//! constrained-decode stencil, the safe-subset for Restricted mode, and the
//! per-tool trajectories.
//!
//! This is the definition half of the tool system. Execution stays in
//! [`zend_tools::registry`], bound to a definition by `name`. The definitions
//! were generated from the registry (see `cargo run -p zend --example
//! export_tools`), so the two agree field-for-field — asserted by
//! `defs_match_registry` in the tests.

use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;

use include_dir::{include_dir, Dir};
use serde::Deserialize;
use serde_json::Value;

/// The tool definitions, embedded at compile time. The `tools` collection in
/// `prompts/projection.yaml` is filled from here.
static TOOLS_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/src/prompts/tools");

/// The marker embedded in a calibration example's trajectory at each projection
/// (wide-Q capture) point. Not a real token — the prefill path strips it and
/// records each occurrence's token offset so a projection fires there,
/// reproducing the decode's per-segment projection sequence.
pub const PROJECTION_MARKER: &str = "<|projection|>";

/// One tool definition parsed from a `tools/*.yaml` file.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolDef {
    pub name: String,
    #[serde(default)]
    pub category: String,
    pub description: String,
    #[serde(default)]
    pub high_risk: bool,
    /// JSON Schema for the call arguments (the tool's Request type).
    pub parameters: Value,
    /// ChatML selection-calibration trajectories (prompt + `<|im_end|>
    /// <|im_start|>assistant` + think→call with [`PROJECTION_MARKER`]s), or a bare
    /// prompt for an uncalibrated tool. Prefilled by the calibration phase.
    #[serde(default)]
    pub examples: Vec<String>,
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

/// The effective tool catalog, resolved once per daemon.
static DEFS: OnceLock<Vec<ToolDef>> = OnceLock::new();

/// Resolve the effective tool catalog for this daemon *before* any [`all`] call:
/// the working-dir `<workspace>/tools/*.yaml` when that folder exists (a mind /
/// game's own tools, which fully override the built-ins — an empty folder yields
/// a deliberately tool-free mind), else the bundled built-in catalog (the coding
/// assistant). First call wins; later calls are ignored.
pub fn init(workspace: &Path) {
    if DEFS.set(load_effective(workspace)).is_err() {
        // A prior `all()` already resolved (and locked in) the fallback catalog,
        // so this override is lost. That means a consumer ran before `init` — an
        // ordering bug worth shouting about rather than silently mis-tooling.
        tracing::error!(
            "tool_def::init called after the catalog was already resolved — \
             workspace tool override ignored (a consumer ran before init)"
        );
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

/// Parse the embedded (compile-time) built-in catalog.
fn load_bundled() -> Vec<ToolDef> {
    let mut defs: Vec<ToolDef> = TOOLS_DIR
        .files()
        .filter(|f| f.path().extension().and_then(|e| e.to_str()) == Some("yaml"))
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
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("yaml"))
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

/// The names of every non-high-risk tool — the subset projected in "Restricted"
/// tools mode.
pub fn safe_names() -> HashSet<String> {
    all()
        .iter()
        .filter(|d| !d.high_risk)
        .map(|d| d.name.clone())
        .collect()
}

/// The catalog category a tool belongs to (the `## <category>` grouping in the
/// tool-catalog summary), or `"Other"` for an unknown name.
pub fn category_for(name: &str) -> &'static str {
    find(name).map(|d| d.category.as_str()).unwrap_or("Other")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Execution (the registry) and definition (this folder) must agree on the
    /// tool set: every executable tool has a bundled definition, and every
    /// definition has an executor — so no model call binds to a missing runner
    /// and no surfaced tool lacks one. The registry is execution-only now, so this
    /// is the binding invariant (there are no definition fields left to compare).
    #[test]
    fn every_tool_has_definition_and_executor() {
        let defs = all();
        let reg = zend_tools::registry::all_tools();
        assert_eq!(
            defs.len(),
            reg.len(),
            "definition count {} != executor count {}",
            defs.len(),
            reg.len()
        );
        for tool in reg {
            assert!(
                find(tool.name).is_some(),
                "no definition for executable tool {}",
                tool.name
            );
        }
        for def in defs {
            assert!(
                zend_tools::registry::find(&def.name).is_some(),
                "no executor for defined tool {}",
                def.name
            );
        }
    }

    /// A coding-agent workspace (no `projection.yaml`) uses the bundled catalog
    /// even when it contains an unrelated `tools/` directory (the regression that
    /// shadowed the built-ins with an empty catalog); a mind workspace (with a
    /// `projection.yaml`) draws from its own `tools/` — empty means tool-free.
    #[test]
    fn tools_override_only_applies_to_a_mind_workspace() {
        let base = std::env::temp_dir().join(format!("zend_toolws_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);

        let coding = base.join("coding");
        std::fs::create_dir_all(coding.join("tools")).unwrap();
        std::fs::write(coding.join("tools").join("Cargo.toml"), "x").unwrap();
        assert_eq!(
            load_effective(&coding).len(),
            load_bundled().len(),
            "coding-agent workspace must use the bundled catalog, not an unrelated tools/ dir"
        );

        let mind = base.join("mind");
        std::fs::create_dir_all(mind.join("tools")).unwrap();
        std::fs::write(mind.join("projection.yaml"), "layers: []").unwrap();
        assert!(
            load_effective(&mind).is_empty(),
            "a mind's empty tools/ folder is deliberately tool-free"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn safe_names_excludes_high_risk() {
        let safe = safe_names();
        assert!(safe.contains("datetime"), "datetime is safe");
        assert!(!safe.contains("code_run"), "code_run is high-risk");
    }

    /// Every definition carries a real category (never the `"Other"` fallback the
    /// summary groups under) and a well-formed `"type": "object"` JSON Schema for
    /// `parameters` — the shape the tool-catalog summary and the constrained-decode
    /// stencil rely on. This is the bounded drift guard: it catches a `parameters`
    /// block that has degraded to a non-object or lost its schema shape. Exact
    /// conformance to the executor's `Tool::Request` is enforced at call time
    /// (args deserialize into `Request`, yielding `invalid_arguments` on mismatch).
    #[test]
    fn every_def_has_category_and_object_schema() {
        for def in all() {
            assert!(
                !def.category.trim().is_empty() && def.category != "Other",
                "tool {:?} has no real category",
                def.name
            );
            assert_eq!(
                def.parameters.get("type").and_then(|t| t.as_str()),
                Some("object"),
                "tool {:?} parameters is not an object JSON Schema",
                def.name
            );
        }
    }
}
