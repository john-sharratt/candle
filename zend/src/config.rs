use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use candle_conversation::models::Model;

/// Runtime configuration for the zend daemon.
#[derive(Clone, Debug, Default)]
pub struct DaemonConfig {
    /// Absolute path to the root of the workspace being served.
    pub workspace: PathBuf,
    /// TCP port the HTTP server listens on.
    pub port: u16,
    /// Projection layers taken OUT OF SERVICE (`--disable-layer <name>`,
    /// repeatable). A disabled layer still exists in the schema, but it is
    /// inert: not populated at boot, not refreshed by the watcher, **excluded
    /// from the provenance gather** (`Builder::set_layer_gathered`), not
    /// normalization-warmed, and not swept for crashed partials. Its turns
    /// remain in the substrate untouched — nothing is deleted and dropping the
    /// flag restores them — but while disabled they cannot be selected into any
    /// projection.
    ///
    /// The one deliberate exception is an EXPLICIT UPLOAD: a bounded `read_file`
    /// into a disabled per-file layer still runs, seeding that layer's registry
    /// entry (see `InferenceState::ingest_uploaded_files`), because a user who
    /// uploads a file has asked for it to be read. Those turns land in the
    /// substrate like any other and become selectable once the flag is dropped.
    ///
    /// Also names section **collections** (`response`, `mood`), which have no
    /// ingest pass of their own.
    pub disabled_layers: HashSet<String>,
    /// Turn-sink layers kept IN SERVICE but not loaded (`--skip-layer <name>`,
    /// repeatable). Disjoint from [`Self::disabled_layers`] — the stronger flag
    /// wins, and `main` subtracts it — so consumers never have to encode the
    /// precedence.
    ///
    /// A skipped layer is fully live: its existing turns compete in the gather,
    /// its hit levels are warmed every boot, and its crashed-partial
    /// conversations are retired. Only the READING is skipped: no startup
    /// ingest pass and no watcher-driven refresh. The flag for "the corpus is
    /// built, stop re-reading the disk".
    pub skipped_layers: HashSet<String>,
    /// Content-root overrides for derived ingest layers (`--ingest-dir
    /// <layer>=<path>`, repeatable), keyed by layer name. Each replaces the
    /// folder that layer ingests from — relative to the workspace, or absolute.
    /// Scopes a rebuild to a subtree (e.g. `code_reading=zend/src`) so the
    /// substrate stays small instead of absorbing the whole workspace.
    pub ingest_dirs: HashMap<String, String>,
    /// `--max-depth <N>`: how deep, in path components below each layer's
    /// content root, the `repo_map` and `code_reading` walks and the watcher
    /// read (`1` = the root's own files, `2` = one folder down). Content already
    /// ingested from deeper is FROZEN — kept and still retrievable, but never
    /// re-read and never retired by the deleted-path sweeps. `None` = unbounded.
    pub max_depth: Option<usize>,
    /// Force a whole-store redo-log compaction once during load, after the
    /// substrate reload and before serving. Normally reclaim is incremental and
    /// background (the persistence-thread maintenance pass); this flag forces
    /// the eager whole-store rewrite instead of deferring it. Opt-in
    /// (`--compact-substrate`).
    pub compact_substrate: bool,
    /// Which model the daemon runs (`--model <PRESET>`). Defaults to the
    /// measured-VRAM ladder in `model_choice`.
    pub model: ModelChoice,
}

/// Which model a daemon runs.
#[derive(Clone, Debug, Default)]
pub enum ModelChoice {
    /// Pick from the card's measured VRAM — `model_choice`'s ladder. What a
    /// daemon launched without `--model` runs.
    #[default]
    MeasuredVram,
    /// Run this preset whatever the card. Boxed because `Model` carries a whole
    /// `ModelSpec` in its `Custom` variant, which would otherwise size every
    /// `DaemonConfig` to it.
    Preset(Box<Model>),
}

/// Split the two layer flags into their final, DISJOINT sets: `(disabled,
/// skipped)`.
///
/// `--disable-layer` is the stronger of the two and subsumes `--skip-layer` —
/// inert beats merely unread — so a layer named in both is simply disabled, and
/// the subtraction happens HERE, once, at the edge. Every consumer downstream
/// then treats the sets as disjoint instead of re-deriving the precedence
/// itself, which is the kind of duplicated rule that gets honoured on one branch
/// and forgotten on the next.
pub fn layer_flag_sets(disable: &[String], skip: &[String]) -> (HashSet<String>, HashSet<String>) {
    let disabled: HashSet<String> = disable.iter().cloned().collect();
    let skipped: HashSet<String> = skip
        .iter()
        .filter(|n| !disabled.contains(n.as_str()))
        .cloned()
        .collect();
    (disabled, skipped)
}

#[cfg(test)]
mod tests {
    use super::layer_flag_sets;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn a_layer_named_by_both_flags_is_disabled_not_skipped() {
        let (disabled, skipped) = layer_flag_sets(&names(&["repo_map"]), &names(&["repo_map"]));
        assert!(disabled.contains("repo_map"));
        assert!(
            skipped.is_empty(),
            "disable subsumes skip, so the skip set must not also carry the layer: {skipped:?}",
        );
    }

    #[test]
    fn each_set_keeps_its_own_members_and_they_stay_disjoint() {
        let (disabled, skipped) = layer_flag_sets(
            &names(&["code_reading"]),
            &names(&["repo_map", "code_reading"]),
        );
        assert_eq!(disabled.len(), 1);
        assert!(disabled.contains("code_reading"));
        // `repo_map` was only skipped, so it survives as skipped; `code_reading`
        // is subtracted from the skip set because it is disabled outright.
        assert_eq!(skipped.len(), 1);
        assert!(skipped.contains("repo_map"));
        assert!(disabled.is_disjoint(&skipped));
    }

    #[test]
    fn no_flags_yields_two_empty_sets() {
        let (disabled, skipped) = layer_flag_sets(&[], &[]);
        assert!(disabled.is_empty() && skipped.is_empty());
    }
}
