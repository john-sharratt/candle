//! The old process's last act (§3.9): once it has drained, flushed, released
//! its substrate and closed its sockets, it copies the new build to
//! `keep/stage/` and posts that path to `keep`. The running executable is not
//! replaced. `keep` starts the staged file. This process then exits.

use std::fs;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::decisions::build_name;
use super::kept::builds_dir;
use super::runtime::{InstallJob, Setup};
use super::swap::{Source, Swapped};
use super::switch::{copy_to_stage, FORCE_HEADER, SWITCH_ADDR};
use super::watchdog::{write_plan, WatchPlan, PLAN, READY_WITHIN};
use crate::launch::LaunchSpec;

/// Where `source`'s executable bytes are.
fn source_file(launch: &LaunchSpec, source: &Source) -> io::Result<PathBuf> {
    match source {
        Source::Fetched(path) => Ok(path.clone()),
        Source::Kept(name) => {
            let file = launch.exe.file_name().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "the executable has no name")
            })?;
            Ok(builds_dir(&launch.exe).join(name).join(file))
        }
    }
}

/// Copy `source` to `keep/stage/<file name>` under `launch.cwd`. The file at
/// `launch.exe` is not moved or replaced.
pub fn stage(launch: &LaunchSpec, source: &Source) -> io::Result<PathBuf> {
    let name = launch
        .exe
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "the executable name is not utf-8",
            )
        })?;
    copy_to_stage(
        &launch.cwd,
        name,
        &source_file(launch, source)?,
        &launch.exe,
    )
}

/// Post `path` to `keep`. `Ok` when `keep` answered: it is running either the
/// trial or the stable file. `Err` when `keep` did not take the path, so the
/// caller relaunches itself.
fn ask_keep(path: &Path) -> io::Result<()> {
    let body = path.to_str().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "the staged path is not utf-8")
    })?;
    let mut stream = TcpStream::connect_timeout(
        &SWITCH_ADDR
            .parse()
            .expect("the switch address is 127.0.0.1:8092"),
        Duration::from_secs(5),
    )
    .map_err(|e| io::Error::new(e.kind(), format!("keep at {SWITCH_ADDR}: {e}")))?;
    stream.set_read_timeout(Some(Duration::from_secs(11 * 60)))?;
    let req = format!(
        "POST / HTTP/1.0\r\nHost: 127.0.0.1\r\n{FORCE_HEADER}: 1\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes())?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf)?;
    let text = String::from_utf8_lossy(&buf);
    let status = text.lines().next().unwrap_or("");
    if keep_ran_it(status) {
        return Ok(());
    }
    Err(io::Error::other(format!(
        "keep refused the trial: {status}"
    )))
}

/// Whether `status`, keep's answer's first line, says the trial ran: 200 proven,
/// 409 rolled back. Anything else started nothing. Read as run, a 423 for a
/// trial already under way kept the plan naming a build keep never started, and
/// the process restored afterwards recorded that build as failed: on WhiteWorld
/// (2026-10-31) a healthy 4107 was refused for good.
fn keep_ran_it(status: &str) -> bool {
    status.contains(" 200 ") || status.contains(" 409 ")
}

/// Stage `job`, write the plan the next process reads for its leases, and ask
/// `keep` to start the staged file. The executable at `launch.exe` stays where
/// it is.
pub fn hand_off(
    launch: &LaunchSpec,
    setup: &Setup,
    job: &InstallJob,
    log: Option<PathBuf>,
) -> io::Result<()> {
    let staged = stage(launch, &job.source)?;
    // `swapped.exe` is the staged file, not the running one. A restore that
    // read this plan would remove the staged file and would not remove the
    // executable this process is running from. `keep` does not restore from
    // the plan: the stable file is the way back.
    let plan = WatchPlan {
        old_pid: std::process::id(),
        launch: launch.clone(),
        swapped: Swapped {
            exe: staged.clone(),
            kept: setup.upgrade_dir.join("previous"),
            from_kept: None,
        },
        installing: build_name(&job.build),
        upgrade_dir: setup.upgrade_dir.clone(),
        log,
        ready_within_ms: READY_WITHIN.as_millis() as u64,
        keep_builds: setup.keep_builds,
        protect: job.protect.clone(),
        leases: job.leases.clone(),
    };
    write_plan(&plan)?;
    if let Err(e) = ask_keep(&staged) {
        let _ = fs::remove_file(setup.upgrade_dir.join(PLAN));
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::upgrade::watchdog::read_plan;

    #[test]
    fn only_a_proven_or_rolled_back_trial_counts_as_run() {
        assert!(keep_ran_it("HTTP/1.0 200 OK"));
        assert!(keep_ran_it("HTTP/1.0 409 Conflict"));
        assert!(!keep_ran_it("HTTP/1.0 423 Locked"));
        assert!(!keep_ran_it("HTTP/1.0 400 Bad Request"));
        assert!(!keep_ran_it(""));
    }

    fn launch(dir: &Path, exe: PathBuf) -> LaunchSpec {
        LaunchSpec {
            exe,
            args: Vec::new(),
            cwd: dir.to_path_buf(),
        }
    }

    /// The staged file is a copy. The executable the process is running from
    /// still has its old bytes.
    #[test]
    fn staging_copies_the_artifact_and_leaves_the_live_executable() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("web.exe");
        fs::write(&live, b"live").unwrap();
        let fetched = dir.path().join("fetched.exe");
        fs::write(&fetched, b"staged-bytes").unwrap();
        let staged = stage(&launch(dir.path(), live.clone()), &Source::Fetched(fetched)).unwrap();
        assert_eq!(staged, dir.path().join("keep/stage/web.exe"));
        assert_eq!(fs::read(&staged).unwrap(), b"staged-bytes");
        assert_eq!(fs::read(&live).unwrap(), b"live");
    }

    /// A kept build is staged from its own directory, and the live file stays.
    #[test]
    fn a_kept_build_is_staged_from_its_directory() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("motherd.exe");
        fs::write(&live, b"live").unwrap();
        let kept = builds_dir(&live).join("9-abcd");
        fs::create_dir_all(&kept).unwrap();
        fs::write(kept.join("motherd.exe"), b"kept-bytes").unwrap();
        let staged = stage(
            &launch(dir.path(), live.clone()),
            &Source::Kept("9-abcd".into()),
        )
        .unwrap();
        assert_eq!(fs::read(&staged).unwrap(), b"kept-bytes");
        assert_eq!(fs::read(&live).unwrap(), b"live");
    }

    /// The plan names the build being installed. The test stops before the
    /// loopback post: `keep` is not running here.
    #[test]
    fn the_plan_records_the_install_without_moving_the_executable() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("web.exe");
        fs::write(&live, b"live").unwrap();
        let fetched = dir.path().join("fetched.exe");
        fs::write(&fetched, b"new").unwrap();
        let launch = launch(dir.path(), live.clone());
        let staged = stage(&launch, &Source::Fetched(fetched)).unwrap();
        let upgrade_dir = dir.path().join("upgrade");
        let plan = WatchPlan {
            old_pid: 1,
            launch,
            swapped: Swapped {
                exe: staged,
                kept: upgrade_dir.join("previous"),
                from_kept: None,
            },
            installing: "3-abc".into(),
            upgrade_dir: upgrade_dir.clone(),
            log: None,
            ready_within_ms: 1,
            keep_builds: 5,
            protect: Vec::new(),
            leases: vec!["upgrade/zend".into()],
        };
        write_plan(&plan).unwrap();
        assert_eq!(fs::read(&live).unwrap(), b"live");
        let back = read_plan(&upgrade_dir.join("watch.json")).unwrap();
        assert_eq!(back.leases, vec!["upgrade/zend".to_string()]);
        assert_eq!(back.installing, "3-abc");
        assert_ne!(back.swapped.exe, live);
    }
}
