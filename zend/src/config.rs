use std::collections::HashSet;
use std::path::PathBuf;

/// Runtime configuration for the zend daemon.
#[derive(Clone, Debug, Default)]
pub struct DaemonConfig {
    /// Absolute path to the root of the workspace being served.
    pub workspace: PathBuf,
    /// TCP port the HTTP server listens on.
    pub port: u16,
    /// Projection layers whose startup ingest pass is suppressed (`--disable-layer
    /// <name>`, repeatable). A disabled layer still exists in the schema — it is
    /// simply not populated at boot and is skipped by the watcher-driven refresh
    /// and the upload path. Used to bring the daemon up fast without a heavy
    /// folder/file ingest sweep, or to run a projection whose ingest layers are
    /// intentionally left empty.
    pub disabled_layers: HashSet<String>,
    /// Do not spawn the background summariser thread, and do not register new
    /// conversations for summarisation. Brings the engine up without the AVL
    /// summary forest running — useful for bulk corpus prefill. Opt-in.
    pub disable_summariser: bool,
    /// Force a whole-store redo-log compaction once during load, after the
    /// substrate reload and before serving. Normally reclaim is incremental and
    /// background (the persistence-thread maintenance pass); this flag forces
    /// the eager whole-store rewrite instead of deferring it. Opt-in
    /// (`--compact-substrate`).
    pub compact_substrate: bool,
}
