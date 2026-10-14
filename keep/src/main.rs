//! `keep` — the failsafe for `web` and `motherd`.
//!
//! On startup it runs the executable at each service's stable path, with the
//! arguments from its config. A switch is one request to `127.0.0.1:8092`
//! whose body is the path of a new executable. That file is started with the
//! same arguments. Its stderr must contain the line `INFO serving` within ten
//! minutes, which those processes print only once they are serving. The file
//! is then renamed onto the stable path, or copied there when a running
//! executable cannot be renamed, and that child is stopped and started again
//! from the stable path. The process that stays up was started from the
//! stable file. No line, or the process dying, stops the trial and starts
//! the stable file again.
//!
//! It does not certify, roll out, or choose a canary. It joins the gateway as
//! the app `keep` so the cluster page lists it.

use std::fs;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;

use cluster::upgrade::switch::SWITCH_ADDR;
use place::{pending_copy, record, same_file, How};
use serde::Deserialize;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
use trial::{drain_stderr, spawn, switch_to, Ended, Running};

mod place;
mod presence;
mod trial;

/// How long a trial may run before it is stopped and the stable file is
/// started. Not a config value.
const TRIAL: Duration = Duration::from_secs(10 * 60);

#[derive(Debug, Deserialize)]
struct ServiceCfg {
    name: String,
    stable: PathBuf,
    #[serde(default)]
    args: Vec<String>,
    cwd: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Config {
    state: PathBuf,
    services: Vec<ServiceCfg>,
    #[serde(default)]
    cluster: Option<presence::ClusterJoin>,
}

struct Service {
    name: String,
    /// `web.exe`, taken from the stable path.
    file: String,
    stable: PathBuf,
    args: Vec<std::ffi::OsString>,
    cwd: PathBuf,
    place: PathBuf,
}

struct Slot {
    file: String,
    stable: PathBuf,
    tx: Sender<Req>,
}

struct Req {
    path: PathBuf,
    reply: Sender<Answer>,
}

struct Answer {
    status: u16,
    reason: &'static str,
    body: String,
}

enum Live {
    Child(std::process::Child),
    Adopted(u32),
}

fn main() -> anyhow::Result<()> {
    let config = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "keep/keep.yaml".into());
    let text = fs::read_to_string(&config).map_err(|e| anyhow::anyhow!("reading {config}: {e}"))?;
    let cfg: Config =
        serde_yaml::from_str(&text).map_err(|e| anyhow::anyhow!("in {config}: {e}"))?;
    fs::create_dir_all(&cfg.state)?;
    if let Some(cluster) = cfg.cluster.clone() {
        presence::start(cluster, &cfg.state);
    }
    let mut slots = Vec::new();
    for svc in cfg.services {
        let svc = prepare(&cfg.state, svc)?;
        let (tx, rx) = mpsc::channel();
        slots.push(Slot {
            file: svc.file.clone(),
            stable: svc.stable.clone(),
            tx,
        });
        let name = svc.name.clone();
        thread::Builder::new()
            .name(format!("keep-{name}"))
            .spawn(move || supervise(svc, rx))
            .map_err(|e| anyhow::anyhow!("starting {name}: {e}"))?;
    }
    let listener = bind_switch()?;
    let addr = listener.local_addr()?;
    eprintln!("keep: listening on {addr}");
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let slots = slots.clone();
        thread::spawn(move || handle(stream, &slots));
    }
    Ok(())
}

impl Clone for Slot {
    fn clone(&self) -> Self {
        Slot {
            file: self.file.clone(),
            stable: self.stable.clone(),
            tx: self.tx.clone(),
        }
    }
}

fn prepare(state: &Path, cfg: ServiceCfg) -> anyhow::Result<Service> {
    let file = cfg
        .stable
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("{}: the stable path has no file name", cfg.name))?
        .to_string();
    let stable = if cfg.stable.is_absolute() {
        cfg.stable
    } else {
        std::env::current_dir()?.join(cfg.stable)
    };
    let cwd = if cfg.cwd.is_absolute() {
        cfg.cwd
    } else {
        std::env::current_dir()?.join(cfg.cwd)
    };
    Ok(Service {
        place: state.join(format!("{}.place", cfg.name)),
        name: cfg.name,
        file,
        stable,
        args: cfg.args.into_iter().map(std::ffi::OsString::from).collect(),
        cwd,
    })
}

fn bind_switch() -> io::Result<TcpListener> {
    let addr: std::net::SocketAddr = SWITCH_ADDR
        .parse()
        .expect("the switch address is 127.0.0.1:8092");
    if !addr.ip().is_loopback() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "keep listens on loopback only",
        ));
    }
    TcpListener::bind(addr)
}

fn supervise(svc: Service, rx: Receiver<Req>) {
    let mut leftover = startup_leftover(&svc);
    let mut live = ensure(&svc);
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(req) => {
                let (next, held) = apply(&svc, live, req);
                live = next;
                if let Some(path) = held {
                    leftover = Some(path);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                live = restart_if_gone(&svc, live, &mut leftover);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// A copy promotion leaves the trial file until the process exits. A keep
/// that starts again deletes it once the running image is the stable file.
fn startup_leftover(svc: &Service) -> Option<PathBuf> {
    let path = pending_copy(&svc.place)?;
    if let Some(found) = find(&svc.file) {
        if same_file(&found.exe, &path) {
            return Some(path);
        }
    }
    let _ = fs::remove_file(&path);
    None
}

fn apply(svc: &Service, live: Option<Live>, req: Req) -> (Option<Live>, Option<PathBuf>) {
    if let Some(mut live) = live {
        stop(&mut live);
    }
    let result = switch_to(&req.path, &svc.stable, &svc.args, &svc.cwd, TRIAL);
    let (answer, held) = match &result {
        Ok(Running {
            ended: Ended::Proven(how),
            ..
        }) => {
            if let Err(e) = record(&svc.place, *how, &req.path) {
                eprintln!("keep: {}: the promotion was not recorded: {e}", svc.name);
            }
            let held = (*how == How::Copy).then(|| req.path.clone());
            (
                Answer {
                    status: 200,
                    reason: "OK",
                    body: format!("proven {how}\n"),
                },
                held,
            )
        }
        Ok(Running {
            ended: Ended::RolledBack(why),
            ..
        }) => (
            Answer {
                status: 409,
                reason: "Conflict",
                body: format!("rolled back\n{why}\n"),
            },
            None,
        ),
        Err(e) => (
            Answer {
                status: 409,
                reason: "Conflict",
                body: format!("rolled back\n{e}\n"),
            },
            None,
        ),
    };
    eprintln!("keep: {} {}", svc.name, answer.body.trim());
    let _ = req.reply.send(answer);
    let live = match result {
        Ok(running) => Some(Live::Child(running.child)),
        Err(_) => ensure(svc),
    };
    (live, held)
}

fn restart_if_gone(
    svc: &Service,
    live: Option<Live>,
    leftover: &mut Option<PathBuf>,
) -> Option<Live> {
    match live {
        Some(Live::Adopted(pid)) if alive(pid) => Some(Live::Adopted(pid)),
        Some(Live::Child(mut child)) => match child.try_wait() {
            Ok(None) | Err(_) => Some(Live::Child(child)),
            Ok(Some(_)) => {
                let _ = child.wait();
                clear_leftover(leftover);
                eprintln!(
                    "keep: {} is down: starting {}",
                    svc.name,
                    svc.stable.display()
                );
                ensure(svc)
            }
        },
        gone => {
            if gone.is_some() {
                clear_leftover(leftover);
                eprintln!(
                    "keep: {} is down: starting {}",
                    svc.name,
                    svc.stable.display()
                );
            }
            ensure(svc)
        }
    }
}

fn clear_leftover(leftover: &mut Option<PathBuf>) {
    if let Some(path) = leftover.take() {
        let _ = fs::remove_file(path);
    }
}

fn ensure(svc: &Service) -> Option<Live> {
    if let Some(found) = find(&svc.file) {
        eprintln!("keep: {} is already running as pid {}", svc.name, found.pid);
        return Some(Live::Adopted(found.pid));
    }
    match spawn(&svc.stable, &svc.args, &svc.cwd) {
        Ok(mut child) => {
            drain_stderr(&mut child);
            eprintln!("keep: {} started {}", svc.name, svc.stable.display());
            Some(Live::Child(child))
        }
        Err(e) => {
            eprintln!("keep: {} did not start: {e}", svc.name);
            None
        }
    }
}

fn stop(live: &mut Live) {
    match live {
        Live::Child(child) => {
            let _ = child.kill();
            let _ = child.wait();
        }
        Live::Adopted(pid) => {
            kill_pid(*pid);
            let start = std::time::Instant::now();
            while alive(*pid) && start.elapsed() < Duration::from_secs(30) {
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
    // The listener is released with the process. A moment, so the trial can bind.
    thread::sleep(Duration::from_secs(1));
}

struct Found {
    pid: u32,
    exe: PathBuf,
}

fn find(exe_name: &str) -> Option<Found> {
    let mut sys = System::new();
    sys.refresh_processes_specifics(
        ProcessesToUpdate::All,
        ProcessRefreshKind::new().with_exe(UpdateKind::Always),
    );
    for proc in sys.processes().values() {
        let Some(exe) = proc.exe().map(PathBuf::from) else {
            continue;
        };
        let Some(name) = exe.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.eq_ignore_ascii_case(exe_name) {
            return Some(Found {
                pid: proc.pid().as_u32(),
                exe,
            });
        }
    }
    None
}

fn alive(pid: u32) -> bool {
    let mut sys = System::new();
    let pid = Pid::from_u32(pid);
    sys.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), ProcessRefreshKind::new());
    sys.process(pid).is_some()
}

fn kill_pid(pid: u32) {
    let mut sys = System::new();
    let pid = Pid::from_u32(pid);
    sys.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), ProcessRefreshKind::new());
    if let Some(proc) = sys.process(pid) {
        proc.kill();
    }
}

fn handle(mut stream: TcpStream, slots: &[Slot]) {
    let answer = match read_body(&mut stream) {
        Ok(body) => dispatch(body.trim(), slots),
        Err(e) => Answer {
            status: 400,
            reason: "Bad Request",
            body: format!("bad request: {e}\n"),
        },
    };
    let _ = write_answer(&mut stream, &answer);
}

fn dispatch(body: &str, slots: &[Slot]) -> Answer {
    if body.is_empty() {
        return Answer {
            status: 400,
            reason: "Bad Request",
            body: "the body is the path of the new executable\n".into(),
        };
    }
    let path = PathBuf::from(body);
    if !path.is_file() {
        return Answer {
            status: 400,
            reason: "Bad Request",
            body: format!("not a file: {}\n", path.display()),
        };
    }
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return Answer {
            status: 400,
            reason: "Bad Request",
            body: "the path has no file name\n".into(),
        };
    };
    let Some(slot) = slots.iter().find(|s| s.file.eq_ignore_ascii_case(name)) else {
        return Answer {
            status: 404,
            reason: "Not Found",
            body: "no such service\n".into(),
        };
    };
    if same_file(&path, &slot.stable) {
        return Answer {
            status: 400,
            reason: "Bad Request",
            body: "the stable file is not a trial\n".into(),
        };
    }
    let (tx, rx) = mpsc::channel();
    if slot.tx.send(Req { path, reply: tx }).is_err() {
        return Answer {
            status: 409,
            reason: "Conflict",
            body: "rolled back\nthe service is not running\n".into(),
        };
    }
    rx.recv().unwrap_or(Answer {
        status: 409,
        reason: "Conflict",
        body: "rolled back\nno answer\n".into(),
    })
}

fn read_body(stream: &mut TcpStream) -> io::Result<String> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    let header_end = loop {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the request ended before its headers",
            ));
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        if buf.len() > 64 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "the headers are too long",
            ));
        }
    };
    let header = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut request = header.lines();
    let start = request.next().unwrap_or("");
    if !start.starts_with("POST ") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the request is a POST",
        ));
    }
    let len = request
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "content-length is required"))?;
    if len > 4096 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the path is too long",
        ));
    }
    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < len {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(len);
    String::from_utf8(body)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "the path is not utf-8"))
}

fn write_answer(stream: &mut TcpStream, answer: &Answer) -> io::Result<()> {
    let bytes = answer.body.as_bytes();
    let head = format!(
        "HTTP/1.0 {} {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        answer.status,
        answer.reason,
        bytes.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use cluster::upgrade::switch::STABLE_DIR;

    #[test]
    fn the_shipped_config_joins_the_cluster_as_keep() {
        let cfg: Config = serde_yaml::from_str(include_str!("../keep.yaml")).unwrap();
        let cluster = cfg.cluster.expect("keep.yaml names its gateway");
        assert_eq!(cluster.url, "https://192.168.0.5:9443");
        assert_eq!(cluster.name, "tokera.com");
        assert_eq!(cluster.host, "blackworld");
        assert_eq!(cfg.services.len(), 2);
        assert_eq!(
            cfg.services[0].stable,
            PathBuf::from(STABLE_DIR).join("web.exe")
        );
        assert_eq!(
            cfg.services[1].stable,
            PathBuf::from(STABLE_DIR).join("motherd.exe")
        );
        assert_eq!(cfg.services[0].args, ["--config", "web/web.yaml"]);
        assert_eq!(cfg.services[1].args, ["--config", "motherd/motherd.yaml"]);
    }

    #[test]
    fn the_switch_address_is_loopback() {
        let addr: std::net::SocketAddr = SWITCH_ADDR.parse().unwrap();
        assert!(addr.ip().is_loopback());
        assert_eq!(addr.to_string(), "127.0.0.1:8092");
        let listener = TcpListener::bind(std::net::SocketAddr::from((
            std::net::Ipv4Addr::LOCALHOST,
            0,
        )))
        .unwrap();
        let bound = listener.local_addr().unwrap();
        assert_eq!(bound.ip().to_string(), "127.0.0.1");
    }
}
