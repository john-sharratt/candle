//! The agent's calls to its machine's keep, on keep's local port
//! (`cluster::upgrade::switch::SWITCH_ADDR`): a deploy's trial, and the queue
//! parked while the process restarts (`docs/build_agent.md` §6, §7).

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use cluster::upgrade::switch::SWITCH_ADDR;
use serde_json::Value;
use tokio::task::spawn_blocking;

/// The longest keep's local port is waited on for one of these calls. Each
/// answers at once: a trial is put on keep's board and run on its own thread.
const ANSWER_WITHIN: Duration = Duration::from_secs(30);

/// What keep answered: its status and its body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answered {
    pub status: u16,
    pub body: String,
}

/// `POST <path>` with `body` as JSON to keep at `addr`.
pub fn post_to(addr: &str, path: &str, body: &Value) -> io::Result<Answered> {
    let text = body.to_string();
    let mut stream = TcpStream::connect_timeout(
        &addr
            .parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("{addr}: {e}")))?,
        Duration::from_secs(5),
    )?;
    stream.set_read_timeout(Some(ANSWER_WITHIN))?;
    let request = format!(
        "POST {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    );
    stream.write_all(request.as_bytes())?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf)?;
    parse_answer(&buf)
}

/// The status and body of an HTTP/1.0 answer.
fn parse_answer(buf: &[u8]) -> io::Result<Answered> {
    let text = String::from_utf8_lossy(buf);
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "keep's answer has no body"))?;
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "keep's answer has no status"))?;
    Ok(Answered {
        status,
        body: body.to_string(),
    })
}

/// [`post_to`] this machine's keep, off the runtime.
pub async fn post(path: &'static str, body: Value) -> Result<Answered, String> {
    spawn_blocking(move || post_to(SWITCH_ADDR, path, &body))
        .await
        .map_err(|e| format!("the call to keep stopped: {e}"))?
        .map_err(|e| format!("keep at {SWITCH_ADDR}: {e}"))
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::thread;

    use serde_json::json;

    use super::*;

    #[test]
    fn an_answer_reads_as_its_status_and_body() {
        let answer = parse_answer(
            b"HTTP/1.0 202 Accepted\r\nContent-Type: application/json\r\n\r\n{\"ok\":true}",
        )
        .unwrap();
        assert_eq!(
            answer,
            Answered {
                status: 202,
                body: "{\"ok\":true}".into()
            }
        );
        assert!(parse_answer(b"garbage").is_err());
    }

    /// **The request is one POST of the JSON, with its length**, as keep's
    /// local port reads it.
    #[test]
    fn a_call_is_one_post_of_its_json() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let server = thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let n = conn.read(&mut buf).unwrap();
            conn.write_all(b"HTTP/1.0 200 OK\r\n\r\n{}").unwrap();
            String::from_utf8_lossy(&buf[..n]).into_owned()
        });
        let answer = post_to(&addr, "/park", &json!({"node": "00000000000000ab"})).unwrap();
        assert_eq!(answer.status, 200);
        let request = server.join().unwrap();
        assert!(request.starts_with("POST /park HTTP/1.0\r\n"), "{request}");
        assert!(request.contains("Content-Length: 27\r\n"), "{request}");
        assert!(
            request.ends_with("{\"node\":\"00000000000000ab\"}"),
            "{request}"
        );
    }
}
