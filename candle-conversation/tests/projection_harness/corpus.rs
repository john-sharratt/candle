//! Test data: manifest types, fixture loader, YAML path helper, tool stubs.

use std::path::PathBuf;

use candle_conversation::provenance::{ProvenanceFile, RawProvenanceFile};
use serde::Deserialize;

// ── Constants ──────────────────────────────────────────────────────────────────

pub const TOOLS: &[&str] = &[
    "weather",
    "web_search",
    "file_write",
    "file_read",
    "code_run",
    "datetime",
    "calculator",
    "random",
];

// ── Manifest types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseType {
    Positive,
    Boundary,
    Negative,
    NoTool,
}

#[derive(Debug, Deserialize)]
pub struct Scenario {
    pub id: String,
    pub tool: Option<String>,
    pub case_type: CaseType,
    /// Decode-phase Q-vector entry.
    pub byte_offset: u64,
    pub token_count: u16,
    /// Prefill-phase Q-vector entry (present only when generated with the
    /// dual-capture path of `gen_real_provenance_data`).
    #[serde(default)]
    pub prefill_byte_offset: Option<u64>,
    #[serde(default)]
    pub prefill_token_count: Option<u16>,
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub scenarios: Vec<Scenario>,
}

// ── Raw manifest types ─────────────────────────────────────────────────────────

/// One scenario entry in `RAW_MANIFEST.json`.
#[derive(Debug, Deserialize)]
pub struct RawScenario {
    pub id: String,
    pub tool: Option<String>,
    pub case_type: CaseType,
    pub raw_byte_offset: u64,
    pub raw_token_count: u16,
}

/// Top-level `RAW_MANIFEST.json` produced by `gen_real_provenance_data`.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct RawManifest {
    pub version: u32,
    pub model: String,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub n_layers_per_band: u32,
    pub band_half_width: u32,
    pub band_centers: [u32; 3],
    pub n_total_layers: u32,
    pub scenarios: Vec<RawScenario>,
}

// ── Loaders ────────────────────────────────────────────────────────────────────

pub fn load_fixtures() -> (Manifest, ProvenanceFile) {
    let dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/tool_provenance_real_data",
    ));
    let json = std::fs::read_to_string(dir.join("MANIFEST.json"))
        .expect("tool_provenance_real_data/MANIFEST.json not found");
    let manifest: Manifest = serde_json::from_str(&json).expect("MANIFEST.json parse failed");
    let pf =
        ProvenanceFile::open(dir.join("signatures.prov")).expect("signatures.prov open failed");
    (manifest, pf)
}

/// Load the prefill-phase fixtures, if present.
///
/// Returns a `Manifest` whose `byte_offset`/`token_count` are remapped to the
/// prefill entries, so every existing [`Harness`] scan path works unchanged —
/// the harness just sees a corpus of prefill Q vectors instead of decode ones.
///
/// Scenarios without prefill data are dropped.  Returns `None` when
/// `prefill_signatures.prov` is absent or no scenario carries prefill offsets
/// (e.g. data generated before the dual-capture path landed).
pub fn try_load_prefill_fixtures() -> Option<(Manifest, ProvenanceFile)> {
    let dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/tool_provenance_real_data",
    ));
    let prefill_prov = dir.join("prefill_signatures.prov");
    if !prefill_prov.exists() {
        return None;
    }
    let json = std::fs::read_to_string(dir.join("MANIFEST.json"))
        .expect("tool_provenance_real_data/MANIFEST.json not found");
    let manifest: Manifest = serde_json::from_str(&json).expect("MANIFEST.json parse failed");

    let remapped: Vec<Scenario> = manifest
        .scenarios
        .into_iter()
        .filter_map(|s| {
            let byte_offset = s.prefill_byte_offset?;
            let token_count = s.prefill_token_count?;
            Some(Scenario {
                id: s.id,
                tool: s.tool,
                case_type: s.case_type,
                byte_offset,
                token_count,
                prefill_byte_offset: None,
                prefill_token_count: None,
            })
        })
        .collect();
    if remapped.is_empty() {
        return None;
    }

    let pf = ProvenanceFile::open(prefill_prov).expect("prefill_signatures.prov open failed");
    Some((
        Manifest {
            scenarios: remapped,
        },
        pf,
    ))
}

pub fn try_load_raw_fixtures() -> Option<(RawManifest, RawProvenanceFile)> {
    let dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/tool_provenance_real_data",
    ));
    let raw_prov = dir.join("raw_kvq.prov");
    if !raw_prov.exists() {
        return None;
    }
    let json = std::fs::read_to_string(dir.join("RAW_MANIFEST.json"))
        .expect("tool_provenance_real_data/RAW_MANIFEST.json not found");
    let manifest: RawManifest =
        serde_json::from_str(&json).expect("RAW_MANIFEST.json parse failed");
    let pf = RawProvenanceFile::open(raw_prov).expect("raw_kvq.prov open failed");
    Some((manifest, pf))
}

pub fn projection_yaml_text() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../zend/src/prompts/projection.yaml",
    ))
    .expect("zend/src/prompts/projection.yaml not found")
}

// ── Layer manifest types ───────────────────────────────────────────────────────

/// Case type used in layer provenance manifests (no_match instead of no_tool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerCaseType {
    Positive,
    Boundary,
    Negative,
    NoMatch,
}

#[derive(Debug, Deserialize)]
pub struct LayerScenario {
    pub id: String,
    pub item: Option<String>,
    pub case_type: LayerCaseType,
    pub byte_offset: u64,
    pub token_count: u16,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct LayerManifest {
    pub version: u32,
    pub content_type: String,
    pub scenarios: Vec<LayerScenario>,
}

pub fn load_layer_fixtures(dir_name: &str) -> (LayerManifest, ProvenanceFile) {
    let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/")).join(dir_name);
    let json = std::fs::read_to_string(dir.join("MANIFEST.json"))
        .unwrap_or_else(|e| panic!("{}/MANIFEST.json not found: {}", dir_name, e));
    let manifest: LayerManifest = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("{}/MANIFEST.json parse failed: {}", dir_name, e));
    let pf = ProvenanceFile::open(dir.join("signatures.prov"))
        .unwrap_or_else(|e| panic!("{}/signatures.prov open failed: {}", dir_name, e));
    (manifest, pf)
}

// ── Tool section stub ──────────────────────────────────────────────────────────

/// Minimal JSON stub used as tool-section content when building the harness
/// schema.  The actual text is irrelevant for scoring — only the BDP
/// sig_entries drive section selection.
pub fn tool_stub(tool: &str) -> String {
    format!(
        r#"{{"type":"function","function":{{"name":"{tool}","description":"{tool}","parameters":{{"type":"object","properties":{{}}}}}}}}"#
    )
}
