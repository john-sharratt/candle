//! Structure-derived load-plan resolution.
//!
//! `projection.yaml` is purely declarative — it describes the runtime projection
//! (layers, collections, selection rules) and never how those get populated. This
//! module DERIVES the load plan from that declared structure; nothing about
//! loading is annotated in the schema.
//!
//! Two kinds of sink are derived:
//!
//! * **Turn-sinks** ([`ingest_layers`]) — a layer whose group takes loaded turns.
//!   The live conversation layer (identified by a `Sequence`-rule group) is
//!   excluded. The rest resolve by convention: the built-in `repo_map`
//!   (folder-scan of the workspace root) and `code_reading` (per-file carve of the
//!   workspace root) pipelines are recognised by name; every other turn-sink layer
//!   reads ChatML records from a folder named after it (`<name>/`).
//! * **Section-collection sinks** ([`section_sinks`]) — an empty collection in a
//!   layer's system prompt, other than the registry-backed `tools`. Each is filled
//!   with calibrated sections from a folder named after it (`<name>s/`).

use std::path::Path;

use candle_conversation::projection::{LayerSchema, Schema, SelectionRule, SystemPromptItem};
use candle_conversation::Sequence;

use crate::code_read::CodeReadState;
use crate::raw_read::RawState;
use crate::repo_scan::ClusterState;

/// The built-in section collection filled from a non-folder source (the tool
/// registry), so it is never treated as a folder-backed section sink.
const TOOLS_COLLECTION: &str = "tools";

/// How a turn-sink layer is populated — derived from the layer's declared
/// identity, never annotated in the schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IngestMode {
    /// Walk + cluster the content root's directories (the built-in `repo_map`).
    Folders,
    /// Per-file scope-aware carve of the content root (the built-in `code_reading`).
    Files,
    /// Hold ChatML records directly from the content folder.
    Raw,
}

/// One turn-sink layer resolved from the schema: identity, the group it populates,
/// how to populate it, the content folder (relative to the workspace), and the
/// loading-overlay label.
#[derive(Clone, Debug)]
pub struct IngestLayer {
    pub name: String,
    pub group: String,
    pub mode: IngestMode,
    pub folder: String,
    pub display: String,
}

/// A live turn-sink's mutable state, held between refreshes and keyed by layer
/// name in the session's registry. The variant mirrors the layer's [`IngestMode`].
#[allow(clippy::large_enum_variant)]
pub enum IngestConv {
    /// A folder-scan layer: the owning conversation (held so its sealed K/V stays
    /// reachable by dialogue retrieval) plus the per-cluster hash record.
    Folders {
        sequence: Sequence,
        state: ClusterState,
    },
    /// A per-file layer: only the merged per-file content-hash record — the
    /// per-file conversations are freed after their turns seal into the substrate.
    Files { state: CodeReadState },
    /// A raw-ChatML layer: the owning conversation (holding the prefilled record
    /// turns) plus the per-file content-hash record for refresh.
    Raw {
        sequence: Sequence,
        state: RawState,
    },
}

/// A section-collection sink: an empty collection to be filled with calibrated
/// sections read from a content folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectionSink {
    /// Layer the collection lives in.
    pub layer: String,
    /// Collection name (e.g. `response`).
    pub collection: String,
    /// Content folder relative to the workspace, derived by pluralising the
    /// collection name (e.g. `responses`).
    pub folder: String,
}

/// Whether a layer is the live conversation layer — its turns are the running
/// dialogue, not loaded content. Identified by a `Sequence`-rule group (the
/// recent-N + historical-top-K rule the live layer alone uses).
fn is_live_conversation(layer: &LayerSchema) -> bool {
    layer
        .groups
        .iter()
        .any(|g| matches!(g.selection, SelectionRule::Sequence { .. }))
}

/// Derive the turn-sink load plan from the declared schema + the content present
/// under `workspace`, in schema order. Skips the live conversation layer and any
/// layer with no group to populate. The built-in `repo_map` / `code_reading`
/// pipelines always resolve (they read the workspace root); every other non-live
/// layer is a raw ChatML sink **only in a mind, and only if a folder named after
/// it exists**.
///
/// The mind gate matters: a coding-agent workspace is an arbitrary user project
/// whose directories (`bug_analysis/`, `daily_history/`, …) must never be read as
/// ChatML just because a layer of that name is declared for a not-yet-online
/// pipeline. Only a *mind* — a workspace carrying its own `projection.yaml`
/// override, the same signal `build_projection_builder`/`tool_def` use — draws
/// raw turn-sinks from its (controlled) folders. This keeps the derivation
/// generic: no per-layer allow/deny list, just built-ins plus the mind gate.
pub fn ingest_layers(schema: &Schema, workspace: &Path) -> Vec<IngestLayer> {
    let is_mind = workspace.join("projection.yaml").is_file();
    let mut out = Vec::new();
    for layer in &schema.layers {
        if is_live_conversation(layer) {
            continue;
        }
        let Some(group) = layer.groups.first() else {
            continue;
        };
        let (mode, folder, display) = match layer.name.as_str() {
            "repo_map" => (
                IngestMode::Folders,
                ".".to_string(),
                "Scanning repository".to_string(),
            ),
            "code_reading" => (
                IngestMode::Files,
                ".".to_string(),
                "Reading code".to_string(),
            ),
            other => {
                if !is_mind || !workspace.join(other).is_dir() {
                    continue;
                }
                (IngestMode::Raw, other.to_string(), format!("Loading {other}"))
            }
        };
        out.push(IngestLayer {
            name: layer.name.clone(),
            group: group.name.clone(),
            mode,
            folder,
            display,
        });
    }
    out
}

/// Derive the section-collection sinks: empty collections (other than the
/// registry-backed `tools`) across every layer's system prompt, in declaration
/// order. Each is filled from `<collection-name>s/`.
pub fn section_sinks(schema: &Schema) -> Vec<SectionSink> {
    let mut out = Vec::new();
    for layer in &schema.layers {
        for item in &layer.system_prompt.items {
            let SystemPromptItem::Collection(c) = item else {
                continue;
            };
            if c.name == TOOLS_COLLECTION || !c.sections.is_empty() {
                continue;
            }
            out.push(SectionSink {
                layer: layer.name.clone(),
                collection: c.name.clone(),
                folder: pluralize(&c.name),
            });
        }
    }
    out
}

/// Content-folder name for a section collection: append `s` unless it already
/// ends in one (`response` → `responses`, `mood` → `moods`).
fn pluralize(name: &str) -> String {
    if name.ends_with('s') {
        name.to_string()
    } else {
        format!("{name}s")
    }
}

#[cfg(test)]
mod tests {
    use super::pluralize;

    #[test]
    fn pluralizes_collection_folders() {
        assert_eq!(pluralize("response"), "responses");
        assert_eq!(pluralize("mood"), "moods");
        // Already plural — left as-is.
        assert_eq!(pluralize("responses"), "responses");
    }
}
