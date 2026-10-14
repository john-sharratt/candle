//! The loopback switch `keep` answers, and the line a process prints once it
//! is serving.
//!
//! The gateway decides the canary and the rollout. A member that is installing
//! copies the new executable to [`STAGE_DIR`] and posts that path to
//! [`SWITCH_ADDR`]. It does not replace its own running file. `keep` starts
//! the posted path, and keeps it only if the process prints [`PROOF`] on stderr.

use std::fs;
use std::io;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

/// The exact line, on stderr, once the process is serving. The boot server
/// does not print it.
pub const PROOF: &str = "INFO serving";

/// `keep`'s switch. It binds this address and no other.
pub const SWITCH_PORT: u16 = 8092;

/// [`SWITCH_PORT`] on loopback.
pub const SWITCH_ADDR: &str = "127.0.0.1:8092";

/// Where a member leaves the executable it is asking `keep` to try, under the
/// process's working directory. Not the stable file.
pub const STAGE_DIR: &str = "keep/stage";

/// Where `keep` keeps the last executable that printed [`PROOF`].
pub const STABLE_DIR: &str = "keep/stable";

/// The socket `keep` binds: `127.0.0.1` and [`SWITCH_PORT`].
pub fn switch_addr() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, SWITCH_PORT))
}

/// `cwd/keep/stage/<exe_name>`.
pub fn staged_exe(cwd: &Path, exe_name: &str) -> PathBuf {
    cwd.join(STAGE_DIR).join(exe_name)
}

/// Copy `from` onto [`staged_exe`]. `live` is the executable this process is
/// running from; the copy is refused when the stage path is that file.
pub fn copy_to_stage(cwd: &Path, exe_name: &str, from: &Path, live: &Path) -> io::Result<PathBuf> {
    if !from.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no artifact at {}", from.display()),
        ));
    }
    let dest = staged_exe(cwd, exe_name);
    if same_path(&dest, live) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the stage path is the running executable",
        ));
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(from, &dest)?;
    Ok(dest)
}

fn same_path(a: &Path, b: &Path) -> bool {
    if let (Ok(a), Ok(b)) = (fs::canonicalize(a), fs::canonicalize(b)) {
        return a == b;
    }
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_proof_line_is_the_whole_line() {
        assert_eq!(PROOF, "INFO serving");
        assert!(!PROOF.contains('\n'));
    }

    #[test]
    fn the_switch_is_loopback_8092() {
        let addr = switch_addr();
        assert!(addr.ip().is_loopback());
        assert_eq!(addr.ip().to_string(), "127.0.0.1");
        assert_eq!(addr.port(), 8092);
        assert_eq!(addr.to_string(), SWITCH_ADDR);
    }

    #[test]
    fn a_staged_copy_leaves_the_running_file_alone() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("bin").join("web.exe");
        fs::create_dir_all(live.parent().unwrap()).unwrap();
        fs::write(&live, b"running").unwrap();
        let fetched = dir.path().join("fetched.exe");
        fs::write(&fetched, b"next").unwrap();
        let staged = copy_to_stage(dir.path(), "web.exe", &fetched, &live).unwrap();
        assert_eq!(staged, dir.path().join("keep/stage/web.exe"));
        assert_eq!(fs::read(&staged).unwrap(), b"next");
        assert_eq!(fs::read(&live).unwrap(), b"running");
    }

    #[test]
    fn the_stage_path_cannot_be_the_running_executable() {
        let dir = tempfile::tempdir().unwrap();
        let live = staged_exe(dir.path(), "web.exe");
        fs::create_dir_all(live.parent().unwrap()).unwrap();
        fs::write(&live, b"running").unwrap();
        let fetched = dir.path().join("fetched.exe");
        fs::write(&fetched, b"next").unwrap();
        let err = copy_to_stage(dir.path(), "web.exe", &fetched, &live).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(fs::read(&live).unwrap(), b"running");
    }
}
