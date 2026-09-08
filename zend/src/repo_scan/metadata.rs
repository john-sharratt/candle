//! `.substrate.yaml` — per-folder metadata, checked in beside the code it
//! describes.
//!
//! The `repo_map` layer's expensive artifacts are its **decoded summary** and
//! its **probe questions** ([`super::probe`]). Both are model output, both take
//! minutes of GPU time per directory, and both were previously reachable only
//! through the substrate — so wiping the substrate threw them away and the next
//! boot regenerated every one of them from scratch.
//!
//! This file is where they live instead. It sits in the folder it describes, it
//! is plain YAML, and it is meant to be **read and edited by hand**: a question
//! a person writes is worth more than one the model writes, because a person
//! knows what someone would actually ask. The daemon fills in what is missing
//! and never overwrites what is already there.
//!
//! ```yaml
//! folder:
//!   path: candle-core/src/vram/
//!   summary: >
//!     Tracks GPU memory…
//!   distinctive_terms: [governor, HostProbe]
//! questions:
//!   locational:
//!     - Where is the VRAM governor first created?
//!   mechanistic: []
//!   conceptual: []
//!   systemic: []
//! ```
//!
//! **Staleness is recorded, not enforced.** `content_hash` says what the file
//! was generated against, so a reader can tell when the folder has moved on —
//! but a stale hash does not invalidate hand-written questions, because the
//! person who wrote them knew more than the hash does.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::dir_unit::DirUnit;
use super::probe::{Probe, Register, PROBES_PER_REGISTER};

/// The file's name, in every folder that holds walked files.
pub const METADATA_FILE: &str = ".substrate.yaml";

/// Header written above the YAML so a person opening the file knows what it is
/// and that editing it is expected.
const HEADER: &str = "\
# Folder metadata for the Zen Code substrate.
#
# The daemon reads this file before it generates anything. Questions listed here
# are used as-is and are NOT regenerated — so hand-written questions win, and
# they survive `--wipe-substrate`.
#
# The `questions` are what makes this folder findable. They are the queries a
# developer would type whose ANSWER is this folder, written by someone who has
# NOT yet found it: no paths, no file names, no \"this folder\".
#
#   locational   where something lives
#   mechanistic  how something works
#   conceptual   what a name means
#   systemic     the same subject in plain English, no jargon at all
#
# Delete a list to have the daemon regenerate it. `--wipe-metadata` deletes every
# one of these files and starts over.
";

/// One folder's metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FolderMetadata {
    pub folder: FolderSection,
    /// Probe questions per register id (`locational`, `mechanistic`, …).
    ///
    /// A `BTreeMap` rather than four named fields so the file stays readable
    /// when a register is absent, and so an unknown key in a hand-edited file is
    /// preserved rather than rejected.
    #[serde(default)]
    pub questions: BTreeMap<String, Vec<String>>,
}

/// The descriptive half — what this folder is, for a human reading the file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FolderSection {
    /// Workspace-relative path, with its trailing slash.
    #[serde(default)]
    pub path: String,
    /// The folder's decoded summary — the payload the projection injects when
    /// this folder is retrieved. Cached here so a fresh substrate does not have
    /// to decode it again.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub summary: String,
    /// The terms the directory-frequency index found distinctive. Informational
    /// for a reader, and the vocabulary a hand-written question should use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub distinctive_terms: Vec<String>,
    /// The unit hash this metadata was generated against — a staleness marker,
    /// never an invalidation.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content_hash: String,
}

impl FolderMetadata {
    /// An empty skeleton for `unit`: the descriptive section filled in, every
    /// register present but empty.
    ///
    /// Written for every folder at boot so the shape is discoverable — a person
    /// browsing the tree finds the file already there, with the registers named,
    /// rather than having to know it could exist.
    pub fn skeleton(unit: &DirUnit, distinctive_terms: Vec<String>) -> Self {
        let mut questions = BTreeMap::new();
        for register in Register::ALL {
            questions.insert(register.id().to_string(), Vec::new());
        }
        Self {
            folder: FolderSection {
                path: unit.dir.clone(),
                summary: String::new(),
                distinctive_terms,
                content_hash: unit.content_hash.clone(),
            },
            questions,
        }
    }

    /// Questions for one register, trimmed and with blanks dropped.
    pub fn register(&self, register: Register) -> Vec<String> {
        self.questions
            .get(register.id())
            .map(|qs| {
                qs.iter()
                    .map(|q| q.trim().to_string())
                    .filter(|q| !q.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Whether every register holds at least [`PROBES_PER_REGISTER`] questions.
    ///
    /// This is the "is it complete?" test that decides whether the model runs.
    /// Deliberately per-register: a file with forty locational questions and no
    /// systemic ones is not complete, and the register it is missing is the one
    /// that serves a newcomer.
    pub fn is_complete(&self) -> bool {
        Register::ALL
            .into_iter()
            .all(|r| self.register(r).len() >= PROBES_PER_REGISTER)
    }

    /// Registers that still need generating.
    pub fn incomplete_registers(&self) -> Vec<Register> {
        Register::ALL
            .into_iter()
            .filter(|r| self.register(*r).len() < PROBES_PER_REGISTER)
            .collect()
    }

    /// Every question in the file, as probes, in register order.
    pub fn probes(&self) -> Vec<Probe> {
        let mut out = Vec::new();
        for register in Register::ALL {
            for text in self.register(register) {
                out.push(Probe { text, register });
            }
        }
        out
    }

    /// Replace one register's questions.
    pub fn set_register(&mut self, register: Register, questions: Vec<String>) {
        self.questions.insert(register.id().to_string(), questions);
    }

    /// Total questions across every register.
    pub fn question_count(&self) -> usize {
        Register::ALL
            .into_iter()
            .map(|r| self.register(r).len())
            .sum()
    }
}

/// Where this folder's metadata file lives.
pub fn path_for(root: &Path, unit: &DirUnit) -> PathBuf {
    let dir = unit.dir.trim_end_matches('/');
    if dir == "." || dir.is_empty() {
        root.join(METADATA_FILE)
    } else {
        root.join(dir).join(METADATA_FILE)
    }
}

/// Read a folder's metadata, or `None` when the file is absent or unreadable.
///
/// A malformed file is reported and treated as absent rather than failing the
/// ingest: it is a hand-edited file, and a stray tab in one folder must not stop
/// the workspace scan. It is **not** thereby forfeit — [`save`] refuses to write
/// over a file it cannot parse, so this folder's questions are regenerated for
/// this run and the person's file stays exactly as they left it.
pub fn load(root: &Path, unit: &DirUnit) -> Option<FolderMetadata> {
    let path = path_for(root, unit);
    let body = std::fs::read_to_string(&path).ok()?;
    match serde_yaml::from_str::<FolderMetadata>(&body) {
        Ok(meta) => Some(meta),
        Err(e) => {
            tracing::warn!(
                target: "zend::repo_scan::metadata",
                path = %path.display(),
                "ignoring malformed folder metadata (treated as absent): {e}",
            );
            None
        }
    }
}

/// Why an existing file must not be written over, if it exists and must not be.
///
/// `None` for a file that is absent, empty, or parses — all three safe to write:
/// there is nothing there a person could lose.
fn refuses_overwrite(path: &Path) -> Option<String> {
    match std::fs::read_to_string(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        // Anything else — a sharing violation on Windows, a permissions error —
        // means a file is there and this process cannot see what is in it.
        Err(e) => Some(format!("could not be read ({e})")),
        Ok(body) if body.trim().is_empty() => None,
        Ok(body) => match serde_yaml::from_str::<FolderMetadata>(&body) {
            Ok(_) => None,
            Err(e) => Some(format!("is not valid YAML ({e})")),
        },
    }
}

/// Write a folder's metadata, header included.
///
/// **Refuses to write over a file it cannot parse.** `.substrate.yaml` is meant
/// to be hand-edited — the module header promises the daemon "never overwrites
/// what is already there" — and a file that fails to parse is the one case where
/// that promise mattered most and was broken: [`load`] reported a stray tab as
/// *absent*, the folder took the skeleton path, the model regenerated the
/// questions, and this function replaced the person's file with model output.
/// The same held for a transiently unreadable file, where nothing was wrong with
/// the contents at all.
///
/// The caller treats a refusal as "these questions will be regenerated", which
/// is the right trade: a folder re-generates its questions each boot until the
/// file is fixed, and the work someone did by hand is still there to fix.
pub fn save(root: &Path, unit: &DirUnit, meta: &FolderMetadata) -> anyhow::Result<()> {
    let path = path_for(root, unit);
    if let Some(reason) = refuses_overwrite(&path) {
        anyhow::bail!(
            "{} {reason} — refusing to overwrite it. Fix or delete the file; \
             `--wipe-metadata` clears every one of them.",
            path.display(),
        );
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let body = serde_yaml::to_string(meta)?;
    std::fs::write(&path, format!("{HEADER}{body}"))
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", path.display()))?;
    Ok(())
}

/// Write skeletons for every unit that has no metadata file yet.
///
/// Returns how many were created. Existing files are left completely alone —
/// this must never touch a file a person has edited.
pub fn seed_skeletons(
    root: &Path,
    units: &[DirUnit],
    terms: impl Fn(&DirUnit) -> Vec<String>,
) -> usize {
    let mut created = 0usize;
    for unit in units {
        if path_for(root, unit).exists() {
            continue;
        }
        let skeleton = FolderMetadata::skeleton(unit, terms(unit));
        match save(root, unit, &skeleton) {
            Ok(()) => created += 1,
            Err(e) => tracing::warn!(
                target: "zend::repo_scan::metadata",
                dir = %unit.dir,
                "could not seed folder metadata: {e:#}",
            ),
        }
    }
    created
}

/// Delete every `.substrate.yaml` under `root` — the `--wipe-metadata` path.
///
/// Walks the units rather than the filesystem, so it can only ever remove files
/// in directories this workspace actually walked. A recursive delete keyed on
/// the filename would be one typo away from sweeping a path nobody intended.
pub fn wipe(root: &Path, units: &[DirUnit]) -> usize {
    let mut removed = 0usize;
    for unit in units {
        let path = path_for(root, unit);
        if path.exists() && std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::dir_unit::build_units;
    use crate::repo_scan::types::{FileEntry, Language, RepoMap};

    fn units_for(paths: &[&str]) -> (tempfile::TempDir, Vec<DirUnit>) {
        let map = RepoMap {
            files: paths
                .iter()
                .map(|p| FileEntry {
                    path: p.to_string(),
                    line_count: 1,
                    language: Language::Rust,
                    size_bytes: 1,
                    module_hint: None,
                })
                .collect(),
            ..Default::default()
        };
        let d = tempfile::tempdir().unwrap();
        let units = build_units(&map, d.path());
        (d, units)
    }

    /// **A file someone hand-edited into invalid YAML is not forfeit.** It is
    /// read as absent — a stray tab in one folder must not stop the scan — and
    /// that is exactly what made it destroyable: the folder took the skeleton
    /// path, the model regenerated, and the save replaced the person's work with
    /// model output. The refusal is what keeps the module header's promise.
    #[test]
    fn a_malformed_file_is_never_written_over() {
        let (d, units) = units_for(&["a/x.rs"]);
        let path = path_for(d.path(), &units[0]);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // A real hand-edit failure: a tab where YAML demands spaces.
        let hand_written = "questions:\n  locational:\n\t- Where is the widget made?\n";
        std::fs::write(&path, hand_written).unwrap();

        assert!(
            load(d.path(), &units[0]).is_none(),
            "a malformed file still reads as absent, so the scan continues",
        );
        let generated = FolderMetadata::skeleton(&units[0], vec!["Widget".to_string()]);
        let err = save(d.path(), &units[0], &generated).unwrap_err();
        assert!(err.to_string().contains("refusing to overwrite"), "{err:#}",);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            hand_written,
            "the file on disk must be byte-identical to what the person left",
        );
    }

    /// Absent and empty are both "nothing to lose", so both are written.
    #[test]
    fn an_absent_or_empty_file_is_written_normally() {
        let (d, units) = units_for(&["a/x.rs"]);
        let path = path_for(d.path(), &units[0]);
        let meta = FolderMetadata::skeleton(&units[0], vec!["Widget".to_string()]);

        save(d.path(), &units[0], &meta).unwrap();
        assert_eq!(load(d.path(), &units[0]).unwrap(), meta);

        std::fs::write(&path, "\n  \n").unwrap();
        save(d.path(), &units[0], &meta).unwrap();
        assert_eq!(load(d.path(), &units[0]).unwrap(), meta);
    }

    /// A well-formed file is rewritten as usual — that is how a register the
    /// model just generated gets persisted beside the ones already there.
    #[test]
    fn a_well_formed_file_is_still_updated() {
        let (d, units) = units_for(&["a/x.rs"]);
        let mut meta = FolderMetadata::skeleton(&units[0], vec!["Widget".to_string()]);
        meta.set_register(Register::Locational, vec!["Where is it made?".to_string()]);
        save(d.path(), &units[0], &meta).unwrap();

        meta.set_register(Register::Systemic, vec!["What is this for?".to_string()]);
        save(d.path(), &units[0], &meta).unwrap();
        let back = load(d.path(), &units[0]).unwrap();
        assert_eq!(back.register(Register::Locational).len(), 1);
        assert_eq!(back.register(Register::Systemic).len(), 1);
    }

    #[test]
    fn a_skeleton_names_every_register_even_when_empty() {
        let (_d, units) = units_for(&["a/x.rs"]);
        let meta = FolderMetadata::skeleton(&units[0], vec!["Widget".to_string()]);
        for register in Register::ALL {
            assert!(
                meta.questions.contains_key(register.id()),
                "{} missing — the shape must be discoverable",
                register.id(),
            );
        }
        assert_eq!(meta.folder.path, "a/");
        assert_eq!(meta.folder.distinctive_terms, vec!["Widget".to_string()]);
        assert!(!meta.is_complete());
    }

    #[test]
    fn metadata_round_trips_through_yaml() {
        let (d, units) = units_for(&["a/x.rs"]);
        let mut meta = FolderMetadata::skeleton(&units[0], vec!["Widget".to_string()]);
        meta.folder.summary = "Holds the widgets.".to_string();
        meta.set_register(
            Register::Locational,
            vec!["Where are widgets built?".to_string()],
        );
        save(d.path(), &units[0], &meta).unwrap();

        let back = load(d.path(), &units[0]).expect("reads back");
        assert_eq!(back, meta);
        assert_eq!(
            back.register(Register::Locational),
            vec!["Where are widgets built?".to_string()],
        );
    }

    /// The file is meant to be opened and edited, so it must explain itself.
    #[test]
    fn the_written_file_carries_an_explanatory_header() {
        let (d, units) = units_for(&["a/x.rs"]);
        let meta = FolderMetadata::skeleton(&units[0], Vec::new());
        save(d.path(), &units[0], &meta).unwrap();
        let body = std::fs::read_to_string(path_for(d.path(), &units[0])).unwrap();
        assert!(body.starts_with("# Folder metadata"), "{body}");
        assert!(body.contains("hand-written questions win"), "{body}");
        assert!(body.contains("locational"), "{body}");
    }

    /// Completeness is PER REGISTER. A file stuffed with locational questions
    /// and no systemic ones is missing exactly the register that serves someone
    /// who cannot name anything yet.
    #[test]
    fn completeness_is_measured_per_register_not_in_total() {
        let (_d, units) = units_for(&["a/x.rs"]);
        let mut meta = FolderMetadata::skeleton(&units[0], Vec::new());
        meta.set_register(
            Register::Locational,
            (0..40).map(|i| format!("Question {i}?")).collect(),
        );
        assert!(!meta.is_complete(), "40 of one register is not complete");
        assert_eq!(meta.incomplete_registers().len(), 3);

        for register in [
            Register::Mechanistic,
            Register::Conceptual,
            Register::Systemic,
        ] {
            meta.set_register(
                register,
                (0..PROBES_PER_REGISTER)
                    .map(|i| format!("{} {i}?", register.id()))
                    .collect(),
            );
        }
        assert!(meta.is_complete());
        assert!(meta.incomplete_registers().is_empty());
    }

    /// Blank and whitespace-only entries are what a hand-edited file collects;
    /// they must not count toward completeness or reach the corpus.
    #[test]
    fn blank_hand_edited_entries_are_dropped() {
        let (_d, units) = units_for(&["a/x.rs"]);
        let mut meta = FolderMetadata::skeleton(&units[0], Vec::new());
        meta.set_register(
            Register::Locational,
            vec![
                "  Where is it?  ".to_string(),
                "   ".to_string(),
                String::new(),
            ],
        );
        assert_eq!(
            meta.register(Register::Locational),
            vec!["Where is it?".to_string()]
        );
    }

    #[test]
    fn probes_carry_their_register() {
        let (_d, units) = units_for(&["a/x.rs"]);
        let mut meta = FolderMetadata::skeleton(&units[0], Vec::new());
        meta.set_register(Register::Systemic, vec!["How does it start?".to_string()]);
        meta.set_register(Register::Conceptual, vec!["What is a Widget?".to_string()]);
        let probes = meta.probes();
        assert_eq!(probes.len(), 2);
        // Register order, not map order.
        assert_eq!(probes[0].register, Register::Conceptual);
        assert_eq!(probes[1].register, Register::Systemic);
    }

    /// A hand-edited file with a stray key must not stop the workspace scan.
    #[test]
    fn a_malformed_file_reads_as_absent_rather_than_failing() {
        let (d, units) = units_for(&["a/x.rs"]);
        let path = path_for(d.path(), &units[0]);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "folder: [this is not a map\n").unwrap();
        assert_eq!(load(d.path(), &units[0]), None);
    }

    /// The root folder's file sits at the workspace root, not in a directory
    /// literally named `.`.
    #[test]
    fn the_root_folders_file_lands_at_the_workspace_root() {
        let (d, units) = units_for(&["top.rs"]);
        assert_eq!(units[0].dir, ".");
        assert_eq!(path_for(d.path(), &units[0]), d.path().join(METADATA_FILE));
    }

    /// Seeding must never touch a file that already exists — it is the one
    /// operation that runs over every folder on every boot, and the files it
    /// walks over are hand-edited.
    #[test]
    fn seeding_never_overwrites_an_existing_file() {
        let (d, units) = units_for(&["a/x.rs", "b/y.rs"]);
        let mut authored = FolderMetadata::skeleton(&units[0], Vec::new());
        authored.set_register(Register::Systemic, vec!["Hand written?".to_string()]);
        save(d.path(), &units[0], &authored).unwrap();

        let created = seed_skeletons(d.path(), &units, |_| Vec::new());
        assert_eq!(created, 1, "only the folder without a file");
        assert_eq!(
            load(d.path(), &units[0])
                .unwrap()
                .register(Register::Systemic),
            vec!["Hand written?".to_string()],
            "the authored file survived seeding",
        );
    }

    #[test]
    fn wipe_removes_every_file_it_seeded() {
        let (d, units) = units_for(&["a/x.rs", "b/y.rs"]);
        assert_eq!(seed_skeletons(d.path(), &units, |_| Vec::new()), 2);
        assert_eq!(wipe(d.path(), &units), 2);
        assert!(load(d.path(), &units[0]).is_none());
        assert_eq!(wipe(d.path(), &units), 0, "wiping twice is harmless");
    }
}
