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
