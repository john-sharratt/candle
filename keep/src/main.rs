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
//! `POST /restart/<service>` stops that service and starts its stable file
//! again — no trial, since the binary is the one already kept. `POST
//! /stop/<service>` stops it and holds it down until `POST /start/<service>`
//! (or a restart or switch) asks for it again. `POST /exit` makes keep itself
//! exit; its services keep running, and its supervisor starts keep again,
//! which adopts them — how a new keep binary is put in place.
//!
//! What keep starts is manageable by the account it runs as from any of its
//! sessions (`owner_access`), so a restart, a look at a service's threads or a
//! stack sample needs no elevation.
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

#[cfg(windows)]
mod owner_access;
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
    name: String,
    file: String,
    stable: PathBuf,
    tx: Sender<Req>,
}

/// What a service's supervisor is asked to do.
enum Req {
    /// Run `path` as a trial on the stable path (`POST /`, the path as body).
    Switch {
        path: PathBuf,
        reply: Sender<Answer>,
    },
    /// Stop the service and start its stable file again
    /// (`POST /restart/<name>`): no trial, the binary is the one already kept.
    Restart { reply: Sender<Answer> },
    /// Stop the service and hold it down (`POST /stop/<name>`): it is not
    /// started again until a start, restart or switch asks for it.
    Stop { reply: Sender<Answer> },
    /// Lift a hold and start the stable file (`POST /start/<name>`).
    Start { reply: Sender<Answer> },
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
    // Before any child starts: what they create is then manageable by this
    // account from any of its sessions, not only from keep's.
    #[cfg(windows)]
    owner_access::grant_owner()
        .map_err(|e| anyhow::anyhow!("granting the owner its children: {e}"))?;
    if let Some(cluster) = cfg.cluster.clone() {
        presence::start(cluster, &cfg.state);
    }
    let mut slots = Vec::new();
    for svc in cfg.services {
        let svc = prepare(&cfg.state, svc)?;
        let (tx, rx) = mpsc::channel();
        slots.push(Slot {
            name: svc.name.clone(),
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
            name: self.name.clone(),
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
    // Held down by a stop: the watchdog leaves the service stopped.
    let mut held = false;
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(Req::Switch { path, reply }) => {
                held = false;
                live = apply(&svc, live, &path, reply);
            }
            Ok(Req::Restart { reply }) => {
                held = false;
                live = restart(&svc, live, reply);
            }
            Ok(Req::Stop { reply }) => {
                held = true;
                live = hold(&svc, live, reply);
            }
            Ok(Req::Start { reply }) => {
                held = false;
                live = start(&svc, live, reply);
            }
            Err(mpsc::RecvTimeoutError::Timeout) if held => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {
                live = restart_if_gone(&svc, live);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// Stop the service and leave it stopped.
fn hold(svc: &Service, live: Option<Live>, reply: Sender<Answer>) -> Option<Live> {
    if let Some(mut live) = live {
        stop(&mut live);
    }
    // One found by name is the service too, whoever started it.
    if let Some(found) = find(&svc.file) {
        stop(&mut Live::Adopted(found.pid));
    }
    eprintln!("keep: {} stopped and held", svc.name);
    let _ = reply.send(Answer {
        status: 200,
        reason: "OK",
        body: "stopped; held until a start, restart or switch\n".to_string(),
    });
    None
}

/// Start the service if it is not running; a running one is left as it is.
fn start(svc: &Service, live: Option<Live>, reply: Sender<Answer>) -> Option<Live> {
    let live = restart_if_gone(svc, live);
    let answer = match &live {
        Some(_) => Answer {
            status: 200,
            reason: "OK",
            body: "running\n".to_string(),
        },
        None => Answer {
            status: 500,
            reason: "Internal Server Error",
            body: "the stable file did not start\n".to_string(),
        },
    };
    eprintln!("keep: {} {}", svc.name, answer.body.trim());
    let _ = reply.send(answer);
    live
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

/// Stop the service and start its stable file again. Answers once the new
/// process has started; it is not a trial, so nothing waits on its proof line.
fn restart(svc: &Service, live: Option<Live>, reply: Sender<Answer>) -> Option<Live> {
    if let Some(mut live) = live {
        stop(&mut live);
    }
    let started = ensure(svc);
    let answer = match &started {
        Some(_) => Answer {
            status: 200,
            reason: "OK",
            body: "restarted\n".to_string(),
        },
        None => Answer {
            status: 500,
            reason: "Internal Server Error",
            body: "stopped, and the stable file did not start\n".to_string(),
        },
    };
    eprintln!("keep: {} {}", svc.name, answer.body.trim());
    let _ = reply.send(answer);
    started
}

fn apply(svc: &Service, live: Option<Live>, path: &Path, reply: Sender<Answer>) -> Option<Live> {
    if let Some(mut live) = live {
        stop(&mut live);
    }
    let result = switch_to(
        path,
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
    let _ = reply.send(answer);
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
    let answer = match read_request(&mut stream) {
        Ok((target, body)) => match route(&target) {
            Route::Switch => dispatch(body.trim(), slots),
            Route::Service(action, name) => service_request(action, name, slots),
            Route::Exit => {
                let _ = write_answer(
                    &mut stream,
                    &Answer {
                        status: 200,
                        reason: "OK",
                        body: "exiting; the supervisor starts keep again\n".into(),
                    },
                );
                // The services are detached and keep running; the keep the
                // supervisor starts next finds them by name and adopts them.
                eprintln!("keep: exiting on request");
                std::process::exit(0);
            }
            Route::Unknown => Answer {
                status: 404,
                reason: "Not Found",
                body: "POST / with an executable's path, POST /restart|stop|start/<service>, \
                       or POST /exit\n"
                    .into(),
            },
        },
        Err(e) => Answer {
            status: 400,
            reason: "Bad Request",
            body: format!("bad request: {e}\n"),
        },
    };
    let _ = write_answer(&mut stream, &answer);
}

/// What a request's target asks for.
#[derive(Debug, PartialEq, Eq)]
enum Route<'a> {
    /// `/`: switch to the executable the body names.
    Switch,
    /// `/<action>/<service>`.
    Service(Action, &'a str),
    /// `/exit`: keep itself exits, for its supervisor to start it again.
    Exit,
    Unknown,
}

/// What a request asks of one service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Restart,
    Stop,
    Start,
}

fn route(target: &str) -> Route<'_> {
    if target == "/" {
        return Route::Switch;
    }
    if target == "/exit" {
        return Route::Exit;
    }
    for (prefix, action) in [
        ("/restart/", Action::Restart),
        ("/stop/", Action::Stop),
        ("/start/", Action::Start),
    ] {
        if let Some(name) = target.strip_prefix(prefix) {
            if !name.is_empty() && !name.contains('/') {
                return Route::Service(action, name);
            }
        }
    }
    Route::Unknown
}

fn service_request(action: Action, name: &str, slots: &[Slot]) -> Answer {
    let Some(slot) = slots.iter().find(|s| s.name.eq_ignore_ascii_case(name)) else {
        return Answer {
            status: 404,
            reason: "Not Found",
            body: "no such service\n".into(),
        };
    };
    let (tx, rx) = mpsc::channel();
    let req = match action {
        Action::Restart => Req::Restart { reply: tx },
        Action::Stop => Req::Stop { reply: tx },
        Action::Start => Req::Start { reply: tx },
    };
    if slot.tx.send(req).is_err() {
        return Answer {
            status: 409,
            reason: "Conflict",
            body: "the service's supervisor is gone\n".into(),
        };
    }
    rx.recv().unwrap_or(Answer {
        status: 409,
        reason: "Conflict",
        body: "no answer\n".into(),
    })
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
    if slot.tx.send(Req::Switch { path, reply: tx }).is_err() {
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

/// A request's target and body. A POST with no `Content-Length` has an empty
/// body — a restart carries none.
fn read_request(stream: &mut TcpStream) -> io::Result<(String, String)> {
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
    let Some(target) = start
        .strip_prefix("POST ")
        .and_then(|rest| rest.split_whitespace().next())
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the request is a POST",
        ));
    };
    let target = target.to_string();
    let len = request
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
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
    let body = String::from_utf8(body)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "the path is not utf-8"))?;
    Ok((target, body))
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

    /// `/` is a switch, `/<action>/<name>` that action on that service,
    /// `/exit` keep's own exit, and anything else is refused rather than read
    /// as one of them.
    #[test]
    fn a_target_routes_to_its_request() {
        assert_eq!(route("/"), Route::Switch);
        assert_eq!(
            route("/restart/motherd"),
            Route::Service(Action::Restart, "motherd")
        );
        assert_eq!(
            route("/stop/motherd"),
            Route::Service(Action::Stop, "motherd")
        );
        assert_eq!(route("/start/web"), Route::Service(Action::Start, "web"));
        assert_eq!(route("/exit"), Route::Exit);
        assert_eq!(route("/restart/"), Route::Unknown);
        assert_eq!(route("/stop/a/b"), Route::Unknown);
        assert_eq!(route("/exit/now"), Route::Unknown);
        assert_eq!(route("/switch"), Route::Unknown);
        assert_eq!(route(""), Route::Unknown);
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
