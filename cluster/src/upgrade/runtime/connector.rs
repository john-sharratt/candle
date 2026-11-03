//! **The connector follows the daemon** (`docs/replicated_substrate.md` §3.9).
//! A machine's MCP connector (`mcp`) holds no substrate and is never certified
//! on its own: it moves with the daemon's release. Once a process is ready, if
//! keep's stage holds a connector of its own build and the connector keep runs
//! is of another, it asks keep to trial the staged one — keep proves it on its
//! proof line or puts the previous back, as for any trial.
//!
//! The staged connector is the release's: on the candidate's machine the
//! deploy built and staged it beside the daemon, and on every other machine
//! the install staged the one it fetched ([`super::fetch::connector_of`],
//! [`crate::upgrade::handoff::hand_off`]). One machine at a time, as the
//! daemon's install is: it runs as the process that took the install's leases
//! comes up.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::Upgrader;
use crate::frame::BuildId;
use crate::upgrade::artifacts::{binary, Probe};
use crate::upgrade::decisions::build_name;
use crate::upgrade::handoff::ask_keep;
use crate::version::info::MCP;

/// The folder keep runs its services from, which a process run by keep is in.
const STABLE: &str = "stable";
/// The folder keep is asked to try a file from, beside it.
const STAGE: &str = "stage";

/// The staged connector and the one keep runs, for a process whose executable
/// is `exe` — `None` when keep does not run it, so it has no connector to
/// follow.
pub fn paths(exe: &Path) -> Option<(PathBuf, PathBuf)> {
    let stable = exe.parent()?;
    if stable.file_name()? != STABLE {
        return None;
    }
    let keep = stable.parent()?;
    Some((binary(&keep.join(STAGE), MCP), binary(stable, MCP)))
}

/// Whether the connector staged as `staged` replaces the one running as
/// `running`, for a process of build `own`: it is of `own`, and the running
/// one is not.
pub fn due(staged: Option<&BuildId>, running: Option<&BuildId>, own: &BuildId) -> bool {
    staged == Some(own) && running != Some(own)
}

/// Follow the connector at `staged` over the one at `running`, if it is due.
fn follow(staged: &Path, running: &Path, own: &BuildId, probe: &dyn Probe) {
    if !staged.is_file() || !running.is_file() {
        return;
    }
    let build_of = |p: &Path| {
        probe
            .build_info(p)
            .filter(|i| i.kind == MCP)
            .map(|i| i.build())
    };
    let (s, r) = (build_of(staged), build_of(running));
    if !due(s.as_ref(), r.as_ref(), own) {
        return;
    }
    tracing::warn!(
        build = build_name(own),
        running = r.as_ref().map(build_name).unwrap_or_default(),
        "the connector follows the release: asking keep to try it"
    );
    match ask_keep(staged) {
        Ok(()) => tracing::warn!(
            build = build_name(own),
            "keep tried the connector: proven, or the previous put back"
        ),
        Err(e) => tracing::warn!(build = build_name(own), "the connector was not tried: {e}"),
    }
}

impl Upgrader {
    /// Once the process is ready, follow the connector, once.
    pub(super) fn connector_tick(self: &Arc<Self>) {
        if self.state().connector_followed || !self.ready() {
            return;
        }
        self.state().connector_followed = true;
        let Some((staged, running)) = paths(&self.setup.exe) else {
            return;
        };
        let own = self.setup.info.build();
        let probe = self.probe.clone();
        tokio::task::spawn_blocking(move || follow(&staged, &running, &own, &*probe));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(count: u32) -> BuildId {
        BuildId {
            count,
            hash: format!("h{count}"),
            local: String::new(),
        }
    }

    /// **A process keep runs finds its connector beside it**, and one keep
    /// does not run has none to follow.
    #[test]
    fn the_connectors_are_keeps_beside_the_daemon() {
        let exe = Path::new("D:/prog/candle/keep/stable").join(binary(Path::new(""), "motherd"));
        let (staged, running) = paths(&exe).unwrap();
        assert_eq!(staged, binary(Path::new("D:/prog/candle/keep/stage"), MCP));
        assert_eq!(
            running,
            binary(Path::new("D:/prog/candle/keep/stable"), MCP)
        );
        assert_eq!(
            paths(Path::new("D:/prog/candle/target/release/motherd.exe")),
            None
        );
    }

    /// **The staged connector replaces the running one only when it is the
    /// process's own build and the running one is not.**
    #[test]
    fn a_staged_connector_of_this_build_replaces_another() {
        let own = build(12);
        assert!(due(Some(&own), Some(&build(11)), &own));
        assert!(
            due(Some(&own), None, &own),
            "a running one that answers nothing is replaced"
        );
        assert!(!due(Some(&own), Some(&own), &own), "already this build");
        assert!(
            !due(Some(&build(13)), Some(&build(11)), &own),
            "staged for another build"
        );
        assert!(!due(None, Some(&build(11)), &own));
    }
}
