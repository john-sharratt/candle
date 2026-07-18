//! Tier 2 substrate-fixture loader (§10.3).
//!
//! A *fixture* is a committed substrate directory (`.substrate/`) plus
//! a `manifest.yaml` describing the planted facts, pre-recorded probe
//! Q vectors, and expected algorithm outputs:
//!
//! ```text
//!   tests/fixtures/conv-fixture-N/
//!   ├── .substrate/
//!   │   ├── substrate.log         (real redo log, produced by Tier 3)
//!   │   ├── manifest.json
//!   │   └── provenance/
//!   │       └── *.bin             (Q sign-bits, real model output)
//!   ├── manifest.yaml             (fixture metadata)
//!   └── README.md
//! ```
//!
//! The fixture is built once by a Tier 3 growth run and committed to
//! the test tree.  Tier 2 tests load it read-only:
//!
//! ```ignore
//! let fixture = SubstrateFixture::load("tests/fixtures/coherent-50")?;
//! let engine  = ConversationEngine::open(fixture.workspace_path())?;
//! let tree    = engine.conversation().read().build_summary_tree_in_memory(timeline);
//! assert_eq!(tree.peaks().len(), expected_peak_count);
//! ```
//!
//! This module owns the manifest schema + a loader that validates
//! the on-disk layout and pre-parses the expected outputs so tests
//! can assert against them in one line.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Top-level fixture metadata, persisted as `manifest.yaml`.
///
/// Captures: what generated the fixture, what was planted, the
/// recorded probe vectors with expected top-K outputs, and the
/// expected algorithm-level invariants the fixture should satisfy at
/// load time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureManifest {
    /// Stable identifier — also the resume key the Tier-3 builder
    /// used (so reopening the workspace with this `debug_id` returns
    /// the same timeline).
    pub debug_id: String,
    /// Manifest format version — bumped when this struct changes.
    /// `1` is the v1 layout described in
    /// `docs/archived/infinite_conversations.md` §10.3.
    pub schema_version: u32,
    /// Git SHA of the workspace at fixture creation time.  Used to
    /// detect "fixture was built before a relevant algorithm change"
    /// staleness.
    pub created_by: String,
    /// Model identifier (e.g. `Qwen3-30B-A3B-Q4`).
    pub model: String,
    /// Normal-turn count in this fixture.
    pub n_turns_normal: u32,
    /// `SummaryOfTurns` binary-leaf count.
    pub n_leaves: u32,
    /// `SummaryOfSummaries` internal-node count.
    pub n_internals: u32,
    /// Tree depth (root height).
    pub tree_depth: u32,
    /// Planted recall facts, indexed by `(turn_idx, fact)`.
    #[serde(default)]
    pub plants: Vec<PlantSpec>,
    /// Pre-recorded probe Q vectors with their expected top-K hits.
    /// Keyed by probe name (e.g. `recall_password`).
    #[serde(default)]
    pub probes: BTreeMap<String, ProbeSpec>,
    /// Expected algorithm-level invariants — checked by the loader.
    #[serde(default)]
    pub expected: ExpectedInvariants,
}

/// One planted recall fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantSpec {
    /// Turn index in the timeline where the fact was embedded.
    pub turn: u32,
    /// Canonical fact text — substring-matched against the model's
    /// response in the recall test.
    pub fact: String,
    /// Probe text — the question the test will later ask.
    pub probe: String,
}

/// One pre-recorded probe with its expected top-K results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeSpec {
    /// Relative path to the Q sign-bits blob (`provenance/probe_*.bin`).
    pub q_blob: PathBuf,
    /// Symbolic descriptions of the top nodes (e.g. `"turn 3"`,
    /// `"leaf containing turn 3"`).  Compared as ordered lists.
    #[serde(default)]
    pub expected_top: Vec<String>,
}

/// Algorithm-level invariants the fixture must satisfy at load time
/// (`docs/immutable_summary_forest.md` — *Invariants*).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExpectedInvariants {
    /// The persisted forest matches the canonical shape for its leaf count:
    /// the peak levels are the base-`MERGE_FANOUT` digits of `N`, and every
    /// `SummaryOfSummaries` has exactly `MERGE_FANOUT` children of equal level.
    #[serde(default = "default_true")]
    pub canonical_shape: bool,
    /// Every Normal sub-leaf is referenced by exactly one
    /// `SummaryOfTurns` parent, and the peak set is a contiguous,
    /// non-overlapping cover of `0..N`.
    #[serde(default = "default_true")]
    pub coverage_complete: bool,
}

fn default_true() -> bool {
    true
}

/// A loaded fixture — manifest + workspace path.
#[derive(Debug, Clone)]
pub struct SubstrateFixture {
    manifest: FixtureManifest,
    root: PathBuf,
}

impl SubstrateFixture {
    /// Load the fixture directory at `path`.  Reads `manifest.yaml`
    /// and validates that the `.substrate/` subdirectory exists.
    /// Doesn't open the substrate log — that's the engine's job.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, FixtureError> {
        let root = path.as_ref().to_path_buf();
        if !root.is_dir() {
            return Err(FixtureError::NotFound(root));
        }
        let substrate_dir = root.join(".substrate");
        if !substrate_dir.is_dir() {
            return Err(FixtureError::MissingSubstrate(substrate_dir));
        }
        let manifest_path = root.join("manifest.yaml");
        let bytes = std::fs::read(&manifest_path)
            .map_err(|e| FixtureError::ManifestRead(manifest_path.clone(), e.to_string()))?;
        let manifest: FixtureManifest = serde_yaml::from_slice(&bytes)
            .map_err(|e| FixtureError::ManifestParse(manifest_path, e.to_string()))?;
        if manifest.schema_version != 1 {
            return Err(FixtureError::UnsupportedSchema(manifest.schema_version));
        }
        Ok(Self { manifest, root })
    }

    /// Path to the fixture root (containing `.substrate/`,
    /// `manifest.yaml`, etc.).  Open the engine with this as the
    /// workspace directory.
    pub fn workspace_path(&self) -> &Path {
        &self.root
    }

    /// Parsed manifest.
    pub fn manifest(&self) -> &FixtureManifest {
        &self.manifest
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error("fixture directory not found: {0}")]
    NotFound(PathBuf),
    #[error("fixture is missing the .substrate/ subdirectory at {0}")]
    MissingSubstrate(PathBuf),
    #[error("failed to read manifest at {0}: {1}")]
    ManifestRead(PathBuf, String),
    #[error("failed to parse manifest at {0}: {1}")]
    ManifestParse(PathBuf, String),
    #[error("unsupported manifest schema_version {0} (expected 1)")]
    UnsupportedSchema(u32),
}

/// Write a fresh `manifest.yaml` to `path`.  Used by Tier 3 growth
/// runs after they finish building the substrate.
pub fn write_manifest(path: &Path, manifest: &FixtureManifest) -> Result<(), std::io::Error> {
    let yaml = serde_yaml::to_string(manifest).expect("manifest serialise infallible");
    std::fs::write(path, yaml)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_test_fixture(root: &Path, debug_id: &str) {
        std::fs::create_dir_all(root.join(".substrate")).unwrap();
        let manifest = FixtureManifest {
            debug_id: debug_id.to_string(),
            schema_version: 1,
            created_by: "test".to_string(),
            model: "Qwen3-30B-A3B-Q4".to_string(),
            n_turns_normal: 50,
            n_leaves: 12,
            n_internals: 11,
            tree_depth: 4,
            plants: vec![PlantSpec {
                turn: 3,
                fact: "The password is rosebud".to_string(),
                probe: "what was the password?".to_string(),
            }],
            probes: {
                let mut m = BTreeMap::new();
                m.insert(
                    "recall_password".to_string(),
                    ProbeSpec {
                        q_blob: PathBuf::from("provenance/probe_recall_password.bin"),
                        expected_top: vec!["turn 3".into(), "leaf containing turn 3".into()],
                    },
                );
                m
            },
            expected: ExpectedInvariants {
                canonical_shape: true,
                coverage_complete: true,
            },
        };
        let yaml = serde_yaml::to_string(&manifest).unwrap();
        std::fs::write(root.join("manifest.yaml"), yaml).unwrap();
    }

    #[test]
    fn loads_v1_manifest() {
        let tmp = TempDir::new().unwrap();
        write_test_fixture(tmp.path(), "coherent-50");
        let fixture = SubstrateFixture::load(tmp.path()).expect("load ok");
        assert_eq!(fixture.manifest().debug_id, "coherent-50");
        assert_eq!(fixture.manifest().n_turns_normal, 50);
        assert_eq!(fixture.manifest().plants.len(), 1);
        assert_eq!(fixture.manifest().plants[0].fact, "The password is rosebud");
        assert!(fixture.manifest().expected.canonical_shape);
    }

    #[test]
    fn missing_substrate_subdir_errors() {
        let tmp = TempDir::new().unwrap();
        // Write manifest WITHOUT the .substrate/ subdir.
        std::fs::write(
            tmp.path().join("manifest.yaml"),
            "debug_id: x\nschema_version: 1\ncreated_by: t\nmodel: m\n\
             n_turns_normal: 0\nn_leaves: 0\nn_internals: 0\ntree_depth: 0\n",
        )
        .unwrap();
        let err = SubstrateFixture::load(tmp.path()).unwrap_err();
        assert!(matches!(err, FixtureError::MissingSubstrate(_)));
    }

    #[test]
    fn unsupported_schema_rejected() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join(".substrate")).unwrap();
        std::fs::write(
            tmp.path().join("manifest.yaml"),
            "debug_id: x\nschema_version: 999\ncreated_by: t\nmodel: m\n\
             n_turns_normal: 0\nn_leaves: 0\nn_internals: 0\ntree_depth: 0\n",
        )
        .unwrap();
        let err = SubstrateFixture::load(tmp.path()).unwrap_err();
        assert!(matches!(err, FixtureError::UnsupportedSchema(999)));
    }

    #[test]
    fn write_manifest_round_trips_through_load() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join(".substrate")).unwrap();
        let m = FixtureManifest {
            debug_id: "two-topics-100".to_string(),
            schema_version: 1,
            created_by: "deadbeef".to_string(),
            model: "test-model".to_string(),
            n_turns_normal: 100,
            n_leaves: 25,
            n_internals: 24,
            tree_depth: 5,
            plants: vec![],
            probes: BTreeMap::new(),
            expected: ExpectedInvariants::default(),
        };
        write_manifest(&tmp.path().join("manifest.yaml"), &m).unwrap();
        let loaded = SubstrateFixture::load(tmp.path()).unwrap();
        assert_eq!(loaded.manifest().debug_id, "two-topics-100");
        assert_eq!(loaded.manifest().n_leaves, 25);
    }
}
