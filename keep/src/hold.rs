//! Holding a service's builds before it is stopped (`docs/build_agent.md`
//! §4, §5). A service whose entry names its `agent:` address runs a build
//! agent; before keep stops it — for a trial, an accept, a restart or a stop —
//! it asks that agent for a hold and waits for the answer, which comes once the
//! deploy build running there has handed off and the queue is parked. A forced
//! request asks for none.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// The route on the service's own port that holds its builds.
pub const HOLD_PATH: &str = "/v1/agent/hold";

/// The longest keep waits for a hold's answer: past the longest build hold an
/// agent file names, so the agent's own limit is what ends the wait.
pub const HOLD_WAIT: Duration = Duration::from_secs(30 * 60);

/// What came of asking for a hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Held {
    /// The builds are held: the service may be stopped.
    Drained,
    /// The service holds no builds, or could not be asked: it is stopped as
    /// it would be without a hold. Why, for keep's log.
    Not(String),
}

/// How often keep, waiting on a hold, reads the host's own status.
pub const STATUS_EVERY: Duration = Duration::from_secs(30);

/// A blocking pool that has started nothing for this long is a wedged host,
/// whose hold will never answer.
pub const WEDGED_AFTER: Duration = Duration::from_secs(60);

/// Ask the agent at `addr` for a hold, and wait for it to drain — reading the
/// host's `/v1/status` there every [`STATUS_EVERY`] while it waits.
///
/// **A wedged host is not waited out.** The hold's last step parks the queue
/// on keep through the host's blocking pool, so a pool whose every thread is
/// stuck never answers it; keep waited the whole [`HOLD_WAIT`] on WhiteWorld
/// (2026-11-20) before it would stop the process. The host's status stays
/// answerable through such a wedge and says so (`blocking_pool.waiting_ms`,
/// `host::blocking_canary`): past [`WEDGED_AFTER`], keep gives the hold up.
/// A deploy build handing off still has its whole wait.
pub fn hold_builds(addr: &str) -> Held {
    let (tx, rx) = std::sync::mpsc::channel();
    let asked = addr.to_string();
    std::thread::spawn(move || {
        let _ = tx.send(ask(&asked));
    });
    let started = std::time::Instant::now();
    loop {
        match rx.recv_timeout(STATUS_EVERY) {
            Ok(Ok(200)) => return Held::Drained,
            Ok(Ok(status)) => return Held::Not(format!("{addr} answered {status} to {HOLD_PATH}")),
            Ok(Err(e)) => return Held::Not(format!("{addr} was not asked for a hold: {e}")),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Held::Not(format!(
                    "{addr}: the hold's request ended without an answer"
                ))
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
        }
        if let Some(waiting) = blocking_pool_waiting(addr) {
            if waiting >= WEDGED_AFTER {
                return Held::Not(format!(
                    "{addr} is wedged: its blocking pool has started nothing for {} s, so its \
                     hold cannot answer",
                    waiting.as_secs()
                ));
            }
        }
        if started.elapsed() >= HOLD_WAIT {
            return Held::Not(format!("{addr} did not answer the hold in {HOLD_WAIT:?}"));
        }
    }
}

/// How long the host at `addr` says its blocking pool has gone without
/// starting a task, from its `/v1/status`; `None` when it does not say, or
/// cannot be read.
fn blocking_pool_waiting(addr: &str) -> Option<Duration> {
    let body = get(addr, "/v1/status").ok()?;
    pool_waiting_of(&body)
}

/// [`blocking_pool_waiting`] from a status body.
fn pool_waiting_of(body: &str) -> Option<Duration> {
    let status: serde_json::Value = serde_json::from_str(body).ok()?;
    status["blocking_pool"]["waiting_ms"]
        .as_u64()
        .map(Duration::from_millis)
}

/// The body of `GET path` at `addr`.
fn get(addr: &str, path: &str) -> io::Result<String> {
    let at = addr
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("{addr}: {e}")))?;
    let mut stream = TcpStream::connect_timeout(&at, Duration::from_secs(5))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let request = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes())?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf)?;
    let text = String::from_utf8_lossy(&buf).into_owned();
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    Ok(body)
}

/// The status of `POST /v1/agent/hold` at `addr`.
fn ask(addr: &str) -> io::Result<u16> {
    let at = addr
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("{addr}: {e}")))?;
    let mut stream = TcpStream::connect_timeout(&at, Duration::from_secs(5))?;
    stream.set_read_timeout(Some(HOLD_WAIT))?;
    let request = format!(
        "POST {HOLD_PATH} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes())?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf)?;
    String::from_utf8_lossy(&buf)
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "the answer has no status"))
}

/// Hold `name`'s builds, if it names an agent and the request is not forced,
/// saying what came of it on keep's log.
pub fn before_stop(name: &str, agent: Option<&str>, forced: bool) {
    let Some(addr) = agent else {
        return;
    };
    if forced {
        eprintln!("keep: {name}: forced, so its builds are not waited for");
        return;
    }
    eprintln!("keep: {name}: holding its builds before it stops");
    match hold_builds(addr) {
        Held::Drained => eprintln!("keep: {name}: its builds are held"),
        Held::Not(why) => eprintln!("keep: {name}: {why}; stopping it anyway"),
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::thread;

    use super::*;

    fn answering(status: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let n = conn.read(&mut buf).unwrap();
            assert!(String::from_utf8_lossy(&buf[..n]).starts_with("POST /v1/agent/hold "));
            conn.write_all(format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n").as_bytes())
                .unwrap();
        });
        addr
    }

    /// **A status saying the blocking pool has waited is read as that wait**;
    /// one that says nothing of it, or none waiting, is no wedge.
    #[test]
    fn a_status_names_how_long_its_pool_has_waited() {
        let wedged = r#"{"state":"ready","blocking_pool":{"last_start_ms":0,"waiting_ms":61000}}"#;
        assert_eq!(pool_waiting_of(wedged), Some(Duration::from_secs(61)));
        let free = r#"{"state":"ready","blocking_pool":{"last_start_ms":0,"waiting_ms":null}}"#;
        assert_eq!(pool_waiting_of(free), None);
        assert_eq!(
            pool_waiting_of(r#"{"state":"ready"}"#),
            None,
            "an older host"
        );
        assert_eq!(pool_waiting_of("not json"), None);
    }

    /// **A hold is drained on a 200, and anything else stops the service as it
    /// would be stopped without one.**
    #[test]
    fn a_hold_drains_on_a_200_and_nothing_else() {
        assert_eq!(hold_builds(&answering("200 OK")), Held::Drained);
        assert!(matches!(
            hold_builds(&answering("404 Not Found")),
            Held::Not(why) if why.contains("404")
        ));
        let closed = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = closed.local_addr().unwrap().to_string();
        drop(closed);
        assert!(matches!(hold_builds(&addr), Held::Not(_)));
    }
}
