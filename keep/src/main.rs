//! `keep` — the failsafe for `web` and `motherd`.
//!
//! It reads, from its state directory, the executable path that last became
//! ready. When that process is gone it starts that executable again. When a
//! replacement was swapped in and does not become ready, it restores the
//! previous executable (`cluster::upgrade::watchdog::recover`) and starts that.
//! It does not certify, roll out, route chunks, or open a substrate, and it
//! does not keep a copy of either binary.

use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::sleep;
use std::time::{Duration, Instant};

use cluster::launch::LaunchSpec;
use cluster::upgrade::watchdog::{read_plan, recover, WatchPlan, PLAN, READY_WITHIN};
use serde::Deserialize;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

use decide::{action, Action, Look};

mod decide;

/// How long after a start `keep` waits before starting again, so one death
/// does not become two processes.
const START_GAP: Duration = Duration::from_secs(10);

#[derive(Debug, Deserialize)]
struct Config {
    state: PathBuf,
    poll_ms: u64,
    services: Vec<Service>,
}

#[derive(Debug, Deserialize)]
struct Service {
    name: String,
    /// The executable's file name, `web.exe` or `motherd.exe`.
    exe: String,
    host: String,
    port: u16,
    path: String,
    /// Used when the running process's command line cannot be read.
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    /// `<data>/upgrade`, when this service writes an upgrade plan. Absent for
    /// `web`, which has no swap.
    #[serde(default)]
    upgrade_dir: Option<PathBuf>,
}

/// The launch that last became ready. Paths only.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Ready {
    exe: PathBuf,
    args: Vec<String>,
    cwd: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let config = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "keep/keep.yaml".into());
    let text = fs::read_to_string(&config).map_err(|e| anyhow::anyhow!("reading {config}: {e}"))?;
    let cfg: Config =
        serde_yaml::from_str(&text).map_err(|e| anyhow::anyhow!("in {config}: {e}"))?;
    fs::create_dir_all(&cfg.state)?;
    let poll = Duration::from_millis(cfg.poll_ms);
    let mut since_start: Vec<Option<Instant>> = vec![None; cfg.services.len()];
    let mut not_ready_since: Vec<Option<Instant>> = vec![None; cfg.services.len()];
    loop {
        for (i, service) in cfg.services.iter().enumerate() {
            if let Err(e) = look(
                &cfg.state,
                service,
                &mut since_start[i],
                &mut not_ready_since[i],
            ) {
                eprintln!("keep: {}: {e}", service.name);
            }
        }
        sleep(poll);
    }
}

fn look(
    state: &Path,
    service: &Service,
    since_start: &mut Option<Instant>,
    not_ready_since: &mut Option<Instant>,
) -> anyhow::Result<()> {
    let found = find(&service.exe);
    let ready = found.is_some() && http_ok(&service.host, service.port, &service.path);
    let last = read_ready(&state.join(format!("{}.json", service.name)))?;
    let plan_path = service.upgrade_dir.as_ref().map(|d| d.join(PLAN));
    let pending = plan_path.as_ref().and_then(|p| read_plan(p).ok());
    if ready {
        *not_ready_since = None;
        *since_start = None;
    } else if not_ready_since.is_none() {
        *not_ready_since = Some(Instant::now());
    }
    let waited = not_ready_since
        .map(|t| t.elapsed())
        .unwrap_or(Duration::ZERO);
    let limit = pending
        .as_ref()
        .map(|p| Duration::from_millis(p.ready_within_ms))
        .unwrap_or(READY_WITHIN);
    let just_started = since_start.is_some_and(|t| t.elapsed() < START_GAP);
    let act = action(Look {
        alive: found.is_some(),
        ready,
        has_last: last.is_some(),
        pending: pending.is_some(),
        waited,
        limit,
        just_started,
    });
    match act {
        Action::Hold => {}
        Action::Remember => {
            if let Some(mut live) = found {
                if live.args.is_empty() {
                    live.args = service.args.clone();
                }
                if live.cwd.as_os_str().is_empty() || live.cwd == PathBuf::from(".") {
                    if let Some(cwd) = &service.cwd {
                        live.cwd = cwd.clone();
                    }
                }
                write_ready(&state.join(format!("{}.json", service.name)), &live)?;
            }
            if let Some(path) = &plan_path {
                let _ = fs::remove_file(path);
            }
        }
        Action::Start => {
            let last = last.expect("Start is only chosen when a last-ready path is recorded");
            spawn(&last)?;
            *since_start = Some(Instant::now());
            *not_ready_since = Some(Instant::now());
            eprintln!(
                "keep: {} is down: started {}",
                service.name,
                last.exe.display()
            );
        }
        Action::Recover => {
            let plan = pending.expect("Recover is only chosen when a plan is on disk");
            restore_and_start(service, &plan)?;
            if let Some(path) = &plan_path {
                let _ = fs::remove_file(path);
            }
            *since_start = Some(Instant::now());
            *not_ready_since = Some(Instant::now());
        }
    }
    Ok(())
}

fn restore_and_start(service: &Service, plan: &WatchPlan) -> anyhow::Result<()> {
    for live in processes_named(&service.exe) {
        live.kill();
    }
    let (spec, reason) = recover(plan, format!("{} did not become ready", service.name));
    spawn_spec(&spec)?;
    eprintln!(
        "keep: {} did not become ready: restored {} ({reason})",
        service.name,
        spec.exe.display()
    );
    Ok(())
}

fn find(exe_name: &str) -> Option<Ready> {
    processes_named(exe_name).into_iter().next().map(|p| Ready {
        exe: p.exe.clone(),
        args: p.args,
        cwd: p.cwd,
    })
}

struct Live {
    exe: PathBuf,
    args: Vec<String>,
    cwd: PathBuf,
    kill: Box<dyn Fn()>,
}

fn processes_named(exe_name: &str) -> Vec<Live> {
    let mut sys = System::new();
    sys.refresh_processes_specifics(
        ProcessesToUpdate::All,
        ProcessRefreshKind::new()
            .with_exe(UpdateKind::Always)
            .with_cwd(UpdateKind::Always)
            .with_cmd(UpdateKind::Always),
    );
    let mut out = Vec::new();
    for proc in sys.processes().values() {
        let Some(exe) = proc.exe().map(PathBuf::from) else {
            continue;
        };
        let Some(name) = exe.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.eq_ignore_ascii_case(exe_name) {
            continue;
        }
        let cmd: Vec<String> = proc
            .cmd()
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        let args = cmd.into_iter().skip(1).collect();
        let cwd = proc
            .cwd()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let pid = proc.pid();
        out.push(Live {
            exe,
            args,
            cwd,
            kill: Box::new(move || {
                let mut sys = System::new();
                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]),
                    ProcessRefreshKind::new(),
                );
                if let Some(p) = sys.process(pid) {
                    p.kill();
                }
            }),
        });
    }
    out
}

impl Live {
    fn kill(&self) {
        (self.kill)();
    }
}

fn http_ok(host: &str, port: u16, path: &str) -> bool {
    let Ok(addr) = format!("{host}:{port}").parse::<SocketAddr>() else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_secs(2)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let req = format!("GET {path} HTTP/1.0\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    match stream.read(&mut buf) {
        Ok(n) if n > 0 => {
            let text = String::from_utf8_lossy(&buf[..n]);
            text.starts_with("HTTP/1.") && !text.contains(" 000")
        }
        _ => false,
    }
}

fn read_ready(path: &Path) -> anyhow::Result<Option<Ready>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    Ok(Some(serde_json::from_slice(&bytes)?))
}

fn write_ready(path: &Path, ready: &Ready) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(ready)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn spawn(ready: &Ready) -> anyhow::Result<()> {
    let spec = LaunchSpec {
        exe: ready.exe.clone(),
        args: ready.args.iter().map(std::ffi::OsString::from).collect(),
        cwd: ready.cwd.clone(),
    };
    spawn_spec(&spec)
}

fn spawn_spec(spec: &LaunchSpec) -> anyhow::Result<()> {
    let mut cmd = Command::new(&spec.exe);
    cmd.args(&spec.args).current_dir(&spec.cwd);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // Detach from keep so a restart of keep does not take the service with it.
        const DETACHED: u32 = 0x0000_0008;
        const NEW_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(DETACHED | NEW_GROUP);
    }
    cmd.spawn()?;
    Ok(())
}
