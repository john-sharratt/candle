//! How a process was launched, captured once at startup, so it can start an
//! identical one: the host's self-heal after a poisoned GPU context, the
//! `restart` command (§12.4), and an upgrade's relaunch of the new binary at the
//! old one's path (§3.9). Here rather than in the host because an agent, which
//! links no host, upgrades the same way.
//!
//! A flag that acts once at the start it was given to — a wipe, a forget, an
//! upgrade's own markers — is dropped from what is captured, so no relaunch
//! repeats it.

use std::env;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

/// The upgrade's one-shot flags (§3.9): the watchdog's plan, and the build an
/// install restored from.
pub const UPGRADE_WATCHDOG: OneShot = OneShot {
    flag: "--upgrade-watchdog",
    takes_value: true,
};
pub const UPGRADE_FAILED: OneShot = OneShot {
    flag: "--upgrade-failed",
    takes_value: true,
};
/// The successor waits for this pid to exit before it loads (§3.9).
pub const AFTER_PID: OneShot = OneShot {
    flag: "--after-pid",
    takes_value: true,
};

/// How a process was launched: its executable, its arguments exactly as given,
/// and its working directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchSpec {
    pub exe: PathBuf,
    pub args: Vec<OsString>,
    pub cwd: PathBuf,
}

/// A flag that acts once, at the start it was given to, and must not act
/// again when the process is relaunched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OneShot {
    /// The long form, with its dashes: `--wipe-substrate`.
    pub flag: &'static str,
    /// Whether it takes a value (`--wipe-layer NAME`), which goes with it.
    pub takes_value: bool,
}

impl LaunchSpec {
    /// Capture the current process's executable path, arguments (excluding
    /// `argv[0]`, exactly as the shell passed them — not reconstructed from
    /// parsed flags, so nothing about the original invocation is lost or
    /// normalised away) less every `one_shot` flag, and working directory.
    /// The upgrade's own one-shots are always dropped.
    pub fn capture(one_shot: &[OneShot]) -> io::Result<Self> {
        let mut dropped = one_shot.to_vec();
        dropped.extend([UPGRADE_WATCHDOG, UPGRADE_FAILED, AFTER_PID]);
        Ok(Self {
            exe: env::current_exe()?,
            args: without_one_shots(env::args_os().skip(1), &dropped),
            cwd: env::current_dir()?,
        })
    }

    /// Start an identical process. Its process id.
    pub fn relaunch(&self) -> io::Result<u32> {
        self.command().spawn().map(|child| child.id())
    }

    /// The command that starts an identical process, not spawned. A fresh
    /// stdin: the child must not inherit a handle whose other end goes away
    /// when this process exits.
    pub fn command(&self) -> Command {
        let mut cmd = Command::new(&self.exe);
        cmd.args(&self.args)
            .current_dir(&self.cwd)
            .stdin(Stdio::null());
        cmd
    }

    /// The same launch, with `extra` arguments after its own: a restored
    /// build's `--upgrade-failed <build>`.
    pub fn with_args(&self, extra: &[&str]) -> LaunchSpec {
        let mut spec = self.clone();
        spec.args.extend(extra.iter().map(OsString::from));
        spec
    }

    /// The same launch from another executable: the watchdog, started from
    /// the kept build's copy of the old one.
    pub fn from_exe(&self, exe: &Path) -> LaunchSpec {
        LaunchSpec {
            exe: exe.to_path_buf(),
            ..self.clone()
        }
    }
}

/// `args` without any of `one_shot`: `--flag`, `--flag=value`, and — for one
/// that takes a value — `--flag value`.
pub fn without_one_shots(
    args: impl IntoIterator<Item = OsString>,
    one_shot: &[OneShot],
) -> Vec<OsString> {
    let mut out = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let text = arg.to_string_lossy();
        let named = one_shot.iter().find(|o| {
            text == o.flag
                || text
                    .strip_prefix(o.flag)
                    .is_some_and(|rest| rest.starts_with('='))
        });
        match named {
            Some(o) if o.takes_value && text == o.flag => {
                args.next();
            }
            Some(_) => {}
            None => out.push(arg),
        }
    }
    out
}

/// The value `flag` was given in `args`, in either spelling.
pub fn value_of(args: &[OsString], flag: &str) -> Option<OsString> {
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let text = arg.to_string_lossy();
        if text == flag {
            return it.next().cloned();
        }
        if let Some(v) = text.strip_prefix(flag).and_then(|r| r.strip_prefix('=')) {
            return Some(OsString::from(v));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ffi::OsStr;

    fn strings(args: Vec<OsString>) -> Vec<String> {
        args.into_iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    /// A relaunch drops every one-shot flag in each spelling — and the value a
    /// valued one carries — and keeps everything else in order, including a
    /// flag that merely starts with a one-shot's name.
    #[test]
    fn a_relaunch_drops_the_one_shot_flags() {
        let one_shot = [
            OneShot {
                flag: "--wipe-substrate",
                takes_value: false,
            },
            OneShot {
                flag: "--wipe-layer",
                takes_value: true,
            },
            UPGRADE_FAILED,
        ];
        let args = [
            "D:\\prog",
            "--wipe-substrate",
            "--host",
            "192.168.0.5",
            "--wipe-layer",
            "repo_map",
            "--wipe-layer=code_reading",
            "--upgrade-failed",
            "4815-9f8e7d6",
            "--wipe-substrate-later",
            "-v",
        ]
        .map(OsString::from);
        assert_eq!(
            strings(without_one_shots(args, &one_shot)),
            [
                "D:\\prog",
                "--host",
                "192.168.0.5",
                "--wipe-substrate-later",
                "-v"
            ]
        );
    }

    /// `capture` reflects the real process, and the command it builds carries
    /// the captured exe, args and cwd through untouched.
    #[test]
    fn capture_round_trips_into_the_command() {
        let spec = LaunchSpec::capture(&[]).unwrap();
        assert_eq!(spec.exe, env::current_exe().unwrap());
        assert_eq!(spec.cwd, env::current_dir().unwrap());

        let cmd = spec.command();
        assert_eq!(cmd.get_program(), spec.exe.as_os_str());
        let got_args: Vec<&OsStr> = cmd.get_args().collect();
        let want_args: Vec<&OsStr> = spec.args.iter().map(OsString::as_os_str).collect();
        assert_eq!(got_args, want_args);
        assert_eq!(cmd.get_current_dir(), Some(spec.cwd.as_path()));
    }

    /// The restored build's marker goes after the launch's own arguments, and
    /// the watchdog runs the same launch from another executable.
    #[test]
    fn a_launch_can_take_extra_arguments_and_another_exe() {
        let spec = LaunchSpec {
            exe: PathBuf::from("/opt/zend"),
            args: vec!["--port".into(), "8080".into()],
            cwd: PathBuf::from("/srv"),
        };
        let failed = spec.with_args(&["--upgrade-failed", "4815-9f8e7d6"]);
        assert_eq!(
            strings(failed.args),
            ["--port", "8080", "--upgrade-failed", "4815-9f8e7d6"]
        );
        let watched = spec.from_exe(Path::new("/opt/builds/4810-a1b2c3d/zend"));
        assert_eq!(watched.exe, PathBuf::from("/opt/builds/4810-a1b2c3d/zend"));
        assert_eq!((watched.args, watched.cwd), (spec.args, spec.cwd));
    }

    #[test]
    fn a_flags_value_is_read_in_either_spelling() {
        let args = |a: &[&str]| a.iter().map(OsString::from).collect::<Vec<_>>();
        assert_eq!(
            value_of(&args(&["--upgrade-failed", "7-abc"]), "--upgrade-failed"),
            Some("7-abc".into())
        );
        assert_eq!(
            value_of(&args(&["-v", "--upgrade-failed=7-abc"]), "--upgrade-failed"),
            Some("7-abc".into())
        );
        assert_eq!(
            value_of(&args(&["--upgrade-failed"]), "--upgrade-failed"),
            None
        );
        assert_eq!(value_of(&args(&["--port", "1"]), "--upgrade-failed"), None);
    }
}
