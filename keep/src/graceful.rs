//! Stopping a service the way its own `shutdown` stops it
//! (`docs/replicated_substrate.md` §12.4), before killing it. A service whose
//! entry names its `status:` or `agent:` address is a host: keep asks it
//! `POST /v1/host/stop`, and it hands its live work on, drains, sends every
//! event it wrote, flushes and exits. keep waits [`STOP_WAIT`] for the exit and
//! kills only past it.
//!
//! Killing it outright left events written to the member's log but not yet on
//! any socket: a conversation's turns were on BlackWorld's disk and missing on
//! WhiteWorld until BlackWorld came back (2026-11-21).

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::thread;
use std::time::{Duration, Instant};

/// The route on the service's own port that stops it cleanly.
pub const STOP_PATH: &str = "/v1/host/stop";

/// How long keep waits for a service it asked to stop to exit, before it
/// kills it.
pub const STOP_WAIT: Duration = Duration::from_secs(30);

/// How often keep looks whether it has exited.
const POLL: Duration = Duration::from_millis(100);

/// Whether a stop request's answer means the service is stopping: it began
/// the stop (`202`), or one was already under way (`409`).
pub fn stopping(status: u16) -> bool {
    status == 202 || status == 409
}

/// The status of `POST /v1/host/stop` at `addr`.
fn ask(addr: &str) -> io::Result<u16> {
    let at = addr
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("{addr}: {e}")))?;
    let mut stream = TcpStream::connect_timeout(&at, Duration::from_secs(5))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let request = format!(
        "POST {STOP_PATH} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
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

/// Wait up to `limit` for `exited` to say so. Whether it did.
pub fn wait_for_exit(limit: Duration, mut exited: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    loop {
        if exited() {
            return true;
        }
        if start.elapsed() >= limit {
            return false;
        }
        thread::sleep(POLL);
    }
}

/// Ask `name`, at `addr`, to stop itself, and wait for `exited` up to
/// [`STOP_WAIT`]. Whether it exited: `false` — no address, a refusal, or no
/// exit in time — leaves the kill to the caller. Each outcome is on keep's log.
pub fn stop_cleanly(name: &str, addr: Option<&str>, exited: impl FnMut() -> bool) -> bool {
    let Some(addr) = addr else {
        return false;
    };
    match ask(addr) {
        Ok(status) if stopping(status) => {
            eprintln!("keep: {name}: asked to stop cleanly; waiting up to {STOP_WAIT:?}");
        }
        Ok(status) => {
            eprintln!("keep: {name}: {addr} answered {status} to {STOP_PATH}; killing it");
            return false;
        }
        Err(e) => {
            eprintln!("keep: {name}: {addr} was not asked to stop: {e}; killing it");
            return false;
        }
    }
    let t = Instant::now();
    if wait_for_exit(STOP_WAIT, exited) {
        eprintln!(
            "keep: {name}: exited cleanly in {:.1} s",
            t.elapsed().as_secs_f64()
        );
        true
    } else {
        eprintln!("keep: {name}: still running after {STOP_WAIT:?}; killing it");
        false
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use super::*;

    fn answering(status: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let n = conn.read(&mut buf).unwrap();
            assert!(String::from_utf8_lossy(&buf[..n]).starts_with("POST /v1/host/stop "));
            conn.write_all(format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n").as_bytes())
                .unwrap();
        });
        addr
    }

    /// **A stop under way, whoever began it, is waited for; anything else is
    /// killed.**
    #[test]
    fn a_stop_under_way_is_waited_for() {
        assert!(stopping(202) && stopping(409));
        assert!(!stopping(200) && !stopping(403) && !stopping(404));
    }

    /// **The exit is waited for only up to the limit.**
    #[test]
    fn the_exit_is_waited_for_up_to_the_limit() {
        let mut looks = 0;
        assert!(wait_for_exit(Duration::from_secs(5), || {
            looks += 1;
            looks == 3
        }));
        assert_eq!(looks, 3);
        let t = Instant::now();
        assert!(!wait_for_exit(Duration::from_millis(250), || false));
        assert!(t.elapsed() >= Duration::from_millis(250));
    }

    /// **A service that accepts the stop and exits is not killed**; one that
    /// refuses, or cannot be reached, or names no address, is left to the kill.
    #[test]
    fn a_service_that_accepts_and_exits_is_not_killed() {
        assert!(stop_cleanly(
            "motherd",
            Some(&answering("202 Accepted")),
            || true
        ));
        assert!(!stop_cleanly(
            "motherd",
            Some(&answering("403 Forbidden")),
            || true
        ));
        let closed = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = closed.local_addr().unwrap().to_string();
        drop(closed);
        assert!(!stop_cleanly("motherd", Some(&addr), || true));
        assert!(!stop_cleanly("web", None, || true));
    }
}
