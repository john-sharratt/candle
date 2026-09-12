//! Bundled tool **definitions** — the declarative, folder-driven source of truth.
//!
//! Every `*.yaml` under `src/prompts/tools/` (embedded at compile time) is one
//! tool definition following `src/prompts/tool.schema.json`: name, category,
//! description, high-risk flag, the `parameters` JSON Schema, ChatML
//! selection-calibration `examples`, and the plain-question `questions` seeds.
//! These drive everything the prompt and calibration need — the tool catalog
//! surfaced into the `tools` collection, the constrained-decode stencil, the
//! safe-subset for Restricted mode, and the per-tool trajectories.
//!
//! This is the definition half of the tool system, and the only half the model
//! ever sees: the `description` and `parameters` here are what get rendered into
//! the prompt. Execution stays in [`zend_tools::registry`], bound to a definition
//! by `name` — the registry carries no description of its own, so a behaviour
//! change in a tool has to be reflected here to reach the model.
//! `every_tool_has_definition_and_executor` in the tests asserts every executable
//! tool has a definition and that no definition names a tool that cannot run.

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
    /// **All of a tool's questions are calibrated in ONE submission**, stuffed
    /// into a single prefill grid and carved back into one turn each
    /// (`candle_conversation::stuffed_grid`). They are therefore resumed as a
    /// unit: the group's marker covers every question, so editing any one of
    /// them regenerates the whole set.
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

    /// Content marker for one calibration case: this tool paired with one of its
    /// trajectories — an `examples` entry or a rendered `questions` one. Covers
    /// **everything that shapes the exemplar**: the projected [`Self::json_line`]
    /// (name, description, parameters — exactly the text the model reads when
    /// choosing a tool) and the trajectory itself.
    ///
    /// The calibration phase keys its resume cache on this, so editing a
    /// description, a parameter's description, an example, or a question changes
    /// the marker and that case re-runs against the current text. Hashing the
    /// trajectory alone would leave a reworded tool answering the belief scan
    /// with signatures captured against wording that no longer exists.
    pub fn calibration_marker(&self, example: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(self.json_line().as_bytes());
        h.update([0u8]);
        h.update(example.as_bytes());
        format!("{}|{:x}", self.name, h.finalize())
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

    fn def(description: &str, params: Value) -> ToolDef {
        ToolDef {
            name: "file_list".to_string(),
            category: "Files".to_string(),
            description: description.to_string(),
            high_risk: false,
            parameters: params,
            examples: vec!["list the files<|im_end|>".to_string()],
            questions: vec!["what files are here".to_string()],
        }
    }

    /// A tool's question GROUP and its examples are distinct cases, so the
    /// resume cache cannot confuse them and editing one leaves the other alone.
    ///
    /// The group's marker covers every question at once because the group is
    /// regenerated as a unit: all of a tool's questions go in as ONE stuffed
    /// prefill, so a per-question marker could never be resumed independently.
    #[test]
    fn a_question_groups_marker_differs_from_an_examples() {
        let d = def("List the files.", serde_json::json!({}));
        assert_ne!(
            d.calibration_marker(&d.questions.join("\u{0}")),
            d.calibration_marker(&d.examples[0])
        );
    }

    /// **Editing ANY question in the group must move the group's marker.**
    /// Otherwise a reworded question resumes against exemplars captured from
    /// text that no longer exists — the same staleness the whole-definition hash
    /// prevents, one level down.
    #[test]
    fn a_question_groups_marker_moves_when_any_question_changes() {
        let mut d = def("List the files.", serde_json::json!({}));
        d.questions = vec!["what files are here".into(), "show me the directory".into()];
        let baseline = d.calibration_marker(&d.questions.join("\u{0}"));

        // Reword the SECOND question — the one a marker built from only the
        // first, or from the count, would miss.
        let mut edited = d.clone();
        edited.questions[1] = "show me this directory".into();
        assert_ne!(
            edited.calibration_marker(&edited.questions.join("\u{0}")),
            baseline,
        );

        // Adding one moves it too.
        let mut added = d.clone();
        added.questions.push("list everything".into());
        assert_ne!(
            added.calibration_marker(&added.questions.join("\u{0}")),
            baseline,
        );

        // Reordering moves it as well: order decides which block a question
        // lands in, so two orderings are genuinely different corpora.
        let mut swapped = d.clone();
        swapped.questions.swap(0, 1);
        assert_ne!(
            swapped.calibration_marker(&swapped.questions.join("\u{0}")),
            baseline,
        );
    }

    /// The calibration marker must move whenever anything the model reads about
    /// the tool moves. Hashing only the example (the earlier behaviour) let a
    /// reworded description resume its old exemplars, so the belief gallery kept
    /// answering with signatures captured against text that no longer existed.
    #[test]
    fn calibration_marker_covers_the_whole_projected_definition() {
        let params = serde_json::json!({"properties": {"prefix": {"type": "string"}}});
        let base = def("List the files.", params.clone());
        let example = &base.examples[0];
        let baseline = base.calibration_marker(example);

        // Same definition, same example → same marker (the resume cache hits).
        assert_eq!(base.calibration_marker(example), baseline);

        // Description reworded → marker moves, so the case re-runs.
        let reworded = def("List the files visible to this session.", params.clone());
        assert_ne!(reworded.calibration_marker(example), baseline);

        // A *parameter's* description is projected too — it is part of the
        // rendered schema the model reads when choosing arguments.
        let reparam = def(
            "List the files.",
            serde_json::json!({
                "properties": {"prefix": {"type": "string", "description": "project root"}}
            }),
        );
        assert_ne!(reparam.calibration_marker(example), baseline);

        // A different example under an unchanged definition is its own case.
        assert_ne!(base.calibration_marker("something else"), baseline);

        // The marker stays name-prefixed so it is greppable per tool.
        assert!(baseline.starts_with("file_list|"));
    }

    /// Two tools sharing a description and example must not collide.
    #[test]
    fn calibration_markers_are_per_tool() {
        let params = serde_json::json!({});
        let mut a = def("same text", params.clone());
        let mut b = def("same text", params);
        a.name = "file_read".into();
        b.name = "file_delete".into();
        assert_ne!(
            a.calibration_marker("ex"),
            b.calibration_marker("ex"),
            "the name must participate in the marker",
        );
    }

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

    /// **Every tool carries question-only seeds, and they really are questions.**
    ///
    /// A tool with none is invisible to any probe shaped like a question until
    /// its full trajectories happen to match — which is the failure the seeds
    /// exist to remove, and it is silent: the catalog still renders, the tool
    /// still executes, it just stops being *selected*. The shape assertions
    /// catch a trajectory pasted in by mistake, which would prefill an authored
    /// answer instead of a bare question.
    #[test]
    fn every_tool_seeds_the_question_gallery() {
        for def in all() {
            assert!(
                def.questions.len() >= 16,
                "tool {:?} has {} question seeds; the schema requires at least 16",
                def.name,
                def.questions.len()
            );
            for q in &def.questions {
                assert!(
                    !q.trim().is_empty(),
                    "tool {:?} has a blank question",
                    def.name
                );
                for marker in ["<|im_end|>", "<|im_start|>", "<think>", "<tool_call>"] {
                    assert!(
                        !q.contains(marker),
                        "tool {:?} question contains {marker:?} — a question is plain \
                         text; calibration adds the ChatML framing itself",
                        def.name
                    );
                }
            }
        }
    }

    /// **No two seeds may be the same or near-identical.**
    ///
    /// A seed earns its prefill only by covering phrasing the others do not. Two
    /// questions that differ by a filler word occupy two gallery slots, cost two
    /// prefills, and match the same probes — so the corpus grows without the
    /// coverage growing, which is the exact failure a bulk expansion invites.
    /// Jaccard over content tokens catches the paraphrase a diff review misses.
    #[test]
    fn question_seeds_are_distinct_from_one_another() {
        fn content(q: &str) -> HashSet<String> {
            q.to_lowercase()
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| w.len() > 1)
                .map(str::to_string)
                .collect()
        }
        let mut seen: HashSet<String> = HashSet::new();
        for def in all() {
            let sets: Vec<HashSet<String>> = def.questions.iter().map(|q| content(q)).collect();
            for (i, a) in sets.iter().enumerate() {
                assert!(
                    seen.insert(def.questions[i].to_lowercase()),
                    "tool {:?} repeats the question {:?}",
                    def.name,
                    def.questions[i]
                );
                for (j, b) in sets.iter().enumerate().skip(i + 1) {
                    if a.is_empty() || b.is_empty() {
                        continue;
                    }
                    let overlap = a.intersection(b).count() as f32;
                    let union = a.union(b).count() as f32;
                    assert!(
                        overlap / union < 0.8,
                        "tool {:?}: {:?} and {:?} are near-identical ({:.0}% token \
                         overlap) — a paraphrase costs a prefill and adds no coverage",
                        def.name,
                        def.questions[i],
                        def.questions[j],
                        100.0 * overlap / union,
                    );
                }
            }
        }
    }

    /// **A seed must name something that distinguishes its own tool.**
    ///
    /// "close it" is a valid English request and a useless exemplar: fifteen
    /// tools close something, so the seed pulls every one of them toward every
    /// closing probe. That is precisely the promiscuous gallery entry the
    /// hit-level normalizer exists to discount, and it is cheaper not to author
    /// it. The check is deliberately weak — one token that at most a quarter of
    /// the catalog uses — because the goal is to reject the genuinely contentless
    /// seed, not to police phrasing.
    #[test]
    fn every_question_seed_names_something_distinctive() {
        let defs = all();
        let mut doc_freq: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let words = |q: &str| -> Vec<String> {
            q.to_lowercase()
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| w.len() > 2)
                .map(str::to_string)
                .collect()
        };
        for def in defs {
            let mut here: HashSet<String> = HashSet::new();
            for q in &def.questions {
                here.extend(words(q));
            }
            for w in here {
                *doc_freq.entry(w).or_default() += 1;
            }
        }
        let cap = defs.len() / 4;
        for def in defs {
            for q in &def.questions {
                let ws = words(q);
                assert!(
                    ws.iter().any(|w| doc_freq[w] <= cap),
                    "tool {:?}: {q:?} uses only words common across the catalog — it \
                     cannot route to this tool rather than its siblings. Name the \
                     protocol, object or verb that makes this tool the answer.",
                    def.name,
                );
            }
        }
    }
}
