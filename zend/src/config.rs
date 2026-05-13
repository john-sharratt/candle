use std::path::PathBuf;

/// Runtime configuration for the zend daemon.
#[derive(Clone, Debug)]
pub struct DaemonConfig {
    /// Absolute path to the root of the workspace being served.
    pub workspace: PathBuf,
    /// TCP port the HTTP server listens on.
    pub port: u16,
}
