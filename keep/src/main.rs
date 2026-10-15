//! `keep` — the failsafe for `web` and `motherd`.
//!
//! On startup it runs the executable at each service's stable path, with the
//! arguments from its config. That child is detached: stdout and stderr are
//! not inherited, and this process does not read or write them. A switch is
//! one request to `127.0.0.1:8092` whose body is the path of a new executable.
//! That file is the trial. The stable file is set aside, the trial's bytes are
//! installed on the stable path, and it is started from there — the path the
//! firewall allows — its stderr written to a file under keep's state, which
//! must contain the line `INFO serving` within ten minutes. Those processes
//! print it only once they are serving. The trial is then the service: the
//! set-aside file is let go and the process goes on running — a promotion
//! never restarts it. No line, or the process dying, stops the trial, puts the
//! set-aside file back and starts it.
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
use place::{previous_of, restore, same_file};
use serde::Deserialize;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
use trial::{spawn_detached, switch_to, Ended, Running};

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
    /// Where a trial's stderr is written, and read for the proof line.
    trial_log: PathBuf,
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
        trial_log: state.join(format!("{}.trial.log", cfg.name)),
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
    roll_back_interrupted_trial(&svc);
    let mut live = ensure(&svc);
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(req) => live = apply(&svc, live, req),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                live = restart_if_gone(&svc, live);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// A set-aside stable file exists only while a trial is unproven, so one
/// found at startup is a trial this keep's predecessor was stopped in the
/// middle of. It never proved itself: whatever runs from it is stopped and
/// the set-aside file put back.
fn roll_back_interrupted_trial(svc: &Service) {
    if !previous_of(&svc.stable).exists() {
        return;
    }
    eprintln!(
        "keep: {}: a trial was interrupted before it proved itself; rolling it back",
        svc.name
    );
    if let Some(found) = find(&svc.file) {
        stop(&mut Live::Adopted(found.pid));
    }
    if let Err(e) = restore(&svc.stable) {
        eprintln!(
            "keep: {}: the previous stable file was not put back: {e}",
            svc.name
        );
    }
}

fn apply(svc: &Service, live: Option<Live>, req: Req) -> Option<Live> {
    if let Some(mut live) = live {
        stop(&mut live);
    }
    let result = switch_to(
        &req.path,
        &svc.stable,
        &svc.trial_log,
        &svc.args,
        &svc.cwd,
        TRIAL,
    );
    let answer = match &result {
        Ok(Running {
            ended: Ended::Proven,
            ..
        }) => Answer {
            status: 200,
            reason: "OK",
            body: "proven\n".to_string(),
        },
        Ok(Running {
            ended: Ended::RolledBack(why),
            ..
        }) => Answer {
            status: 409,
            reason: "Conflict",
            body: format!("rolled back\n{why}\n"),
        },
        Err(e) => Answer {
            status: 409,
            reason: "Conflict",
            body: format!("rolled back\n{e}\n"),
        },
    };
    eprintln!("keep: {} {}", svc.name, answer.body.trim());
    let _ = req.reply.send(answer);
    match result {
        Ok(running) => Some(Live::Child(running.child)),
        Err(_) => ensure(svc),
    }
}

fn restart_if_gone(svc: &Service, live: Option<Live>) -> Option<Live> {
    match live {
        Some(Live::Adopted(pid)) if alive(pid) => Some(Live::Adopted(pid)),
        Some(Live::Child(mut child)) => match child.try_wait() {
            Ok(None) | Err(_) => Some(Live::Child(child)),
            Ok(Some(_)) => {
                let _ = child.wait();
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

fn ensure(svc: &Service) -> Option<Live> {
    if let Some(found) = find(&svc.file) {
        eprintln!("keep: {} is already running as pid {}", svc.name, found.pid);
        return Some(Live::Adopted(found.pid));
    }
    match spawn_detached(&svc.stable, &svc.args, &svc.cwd) {
        Ok(child) => {
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

/// A running instance of the service, found by its file name.
struct Found {
    pid: u32,
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
        assert_eq!(cluster.url, "https://gateway.tokera.com:9443");
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
