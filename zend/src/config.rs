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
    /// Force a one-shot redo-log compaction at startup (after the substrate
    /// reload, before serving). Reclaims dead records from superseded turns /
    /// tombstoned timelines, shrinking the log and future startup walks.
    /// Opt-in; compaction never runs otherwise.
    pub compact_substrate: bool,
}
