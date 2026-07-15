use std::path::PathBuf;

/// Runtime configuration for the zend daemon.
#[derive(Clone, Debug, Default)]
pub struct DaemonConfig {
    /// Absolute path to the root of the workspace being served.
    pub workspace: PathBuf,
    /// TCP port the HTTP server listens on.
    pub port: u16,
    /// Skip the startup code-reading ingest pass. Useful for quickly bringing
    /// the daemon up to test conversations without waiting on (or perturbing
    /// the substrate with) the full per-file prefill sweep.
    pub skip_code_read: bool,
    /// Skip the startup repository scan. Useful for quickly bringing the daemon
    /// up to test conversations without waiting on (or perturbing the substrate
    /// with) the full repo scan.
    pub skip_repo_scan: bool,
    /// Whether startup redo-log compaction is enabled (default: `true`, opt-out
    /// via `--no-compact-substrate`). When enabled, compaction runs after the
    /// substrate reload (before serving) **only if** the loaded substrate has
    /// reclaimable markers — superseded turns, tombstoned timelines, distilled
    /// calibration content — reclaiming them and shrinking the log.
    pub compact_substrate: bool,
    /// Do not spawn the background summariser thread, and do not register new
    /// conversations for summarisation. Brings the engine up without the AVL
    /// summary forest running — useful for bulk corpus prefill. Opt-in.
    pub disable_summariser: bool,
}
