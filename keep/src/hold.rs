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

/// Ask the agent at `addr` for a hold, and wait for it to drain.
pub fn hold_builds(addr: &str) -> Held {
    match ask(addr) {
        Ok(200) => Held::Drained,
        Ok(status) => Held::Not(format!("{addr} answered {status} to {HOLD_PATH}")),
        Err(e) => Held::Not(format!("{addr} was not asked for a hold: {e}")),
    }
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
