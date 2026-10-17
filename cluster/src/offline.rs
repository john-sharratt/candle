//! A process told the network is down: it dials no gateway that is not on this
//! machine, fetches nothing from the internet, and serves only loopback
//! callers — every member and agent in it carries on cut off, writing
//! detached, as it would with the gateway unreachable.
//!
//! **One switch for the process**, because what it stands for is the
//! machine's network, not one app's membership: the members of every app, the
//! agent and the model downloads all ask it. Set by a host's `offline` and
//! `online` commands, and kept in a marker file ([`init`]) — a supervisor that
//! starts the process again, or a machine that reboots, must not bring it back
//! online by forgetting. The marker is also how a second process on the machine
//! follows the switch: `keep` names the same file and [`refresh`]es from it
//! before each dial.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// The marker's file name, in the directory a host gives [`init`].
pub const MARKER_NAME: &str = "OFFLINE";

/// The switch and the file that keeps it. The process has one ([`global`]);
/// tests build their own.
#[derive(Debug, Default)]
pub struct Switch {
    offline: AtomicBool,
    marker: OnceLock<PathBuf>,
}

impl Switch {
    /// Keep the switch in `marker` from now on, reading it first. Answers
    /// whether the switch is on.
    pub fn init(&self, marker: PathBuf) -> bool {
        let _ = self.marker.set(marker);
        self.refresh()
    }

    /// Read the switch again from the marker, when there is one: what another
    /// process on the machine set is followed. Answers whether it is on.
    pub fn refresh(&self) -> bool {
        if let Some(marker) = self.marker.get() {
            self.offline.store(marker.exists(), Ordering::SeqCst);
        }
        self.is_offline()
    }

    /// Turn the switch, and keep it in the marker when there is one.
    pub fn set(&self, offline: bool) -> std::io::Result<()> {
        self.offline.store(offline, Ordering::SeqCst);
        let Some(marker) = self.marker.get() else {
            return Ok(());
        };
        if offline {
            std::fs::write(
                marker,
                "This machine's daemons are offline: they dial no gateway, fetch nothing \
                 from the internet and serve only loopback callers. motherd's `online` \
                 command lifts it, or deleting this file before they start.\n",
            )
        } else {
            match std::fs::remove_file(marker) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
                _ => Ok(()),
            }
        }
    }

    pub fn is_offline(&self) -> bool {
        self.offline.load(Ordering::SeqCst)
    }

    /// Where the switch is kept, if anywhere.
    pub fn marker(&self) -> Option<&Path> {
        self.marker.get().map(PathBuf::as_path)
    }
}

/// The process's switch.
pub fn global() -> &'static Switch {
    static SWITCH: OnceLock<Switch> = OnceLock::new();
    SWITCH.get_or_init(Switch::default)
}

/// [`Switch::init`] on the process's switch, with the marker in `dir`.
pub fn init(dir: &Path) -> bool {
    global().init(dir.join(MARKER_NAME))
}

/// [`Switch::refresh`] on the process's switch.
pub fn refresh() -> bool {
    global().refresh()
}

/// [`Switch::set`] on the process's switch.
pub fn set(offline: bool) -> std::io::Result<()> {
    global().set(offline)
}

/// Whether the process is offline.
pub fn is_offline() -> bool {
    global().is_offline()
}

/// Whether `host` — a name or an address — may be reached: anything while
/// online, only this machine while offline.
pub fn reaches(host: &str) -> bool {
    reaches_when(is_offline(), host)
}

/// [`reaches`] with the switch given.
pub fn reaches_when(offline: bool, host: &str) -> bool {
    !offline || is_local(host)
}

/// Whether a caller at `peer` is served: anyone while online, only this
/// machine while offline. A caller with no known address is not known to be
/// local.
pub fn serves_when(offline: bool, peer: Option<SocketAddr>) -> bool {
    !offline || peer.is_some_and(|p| p.ip().is_loopback())
}

/// Whether `host` names this machine: `localhost`, or a loopback address.
pub fn is_local(host: &str) -> bool {
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    bare.eq_ignore_ascii_case("localhost") || bare.parse::<IpAddr>().is_ok_and(|a| a.is_loopback())
}

/// The error a refused reach gives: says why, and how to lift it.
pub fn refused(what: &str) -> String {
    format!(
        "{what}: this machine is offline — only it is reached (motherd's `online` command \
         lifts it)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_this_machine_is_local() {
        for local in ["localhost", "127.0.0.1", "::1", "[::1]", "127.3.2.1"] {
            assert!(is_local(local), "{local}");
        }
        for remote in [
            "gateway.tokera.com",
            "192.168.0.5",
            "10.0.0.1",
            "example.localhost.com",
        ] {
            assert!(!is_local(remote), "{remote}");
        }
    }

    /// Offline, a gateway elsewhere is not dialled and a caller elsewhere is
    /// not served; this machine is, both ways. Online, everything is.
    #[test]
    fn offline_reaches_and_serves_only_this_machine() {
        assert!(!reaches_when(true, "gateway.tokera.com"));
        assert!(reaches_when(true, "127.0.0.1"));
        assert!(reaches_when(false, "gateway.tokera.com"));

        let lan: SocketAddr = "192.168.0.6:51000".parse().unwrap();
        let lo: SocketAddr = "127.0.0.1:51000".parse().unwrap();
        assert!(!serves_when(true, Some(lan)));
        assert!(serves_when(true, Some(lo)));
        assert!(
            !serves_when(true, None),
            "an unknown peer is not known to be local"
        );
        assert!(serves_when(false, Some(lan)));
    }

    /// **The switch outlives the process.** Set, the marker is written, and a
    /// process starting on it starts offline; lifted, the marker goes. A second
    /// process following the marker sees each change at its next refresh.
    #[test]
    fn the_switch_is_kept_in_its_marker_and_followed_from_it() {
        let dir = std::env::temp_dir().join(format!(
            "offline_switch_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join(MARKER_NAME);

        let host = Switch::default();
        assert!(!host.init(marker.clone()));
        let keep = Switch::default();
        assert!(!keep.init(marker.clone()));

        host.set(true).unwrap();
        assert!(marker.exists());
        assert!(keep.refresh(), "the other process follows the marker");
        let restarted = Switch::default();
        assert!(restarted.init(marker.clone()), "a restart starts offline");

        host.set(false).unwrap();
        assert!(!marker.exists());
        assert!(!keep.refresh());
        host.set(false).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
