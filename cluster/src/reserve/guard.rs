//! A guard's status: its `/v1/status`, read with a bound, and the `refuse` it
//! carries.
//!
//! A status that cannot be read refuses nothing: a process not answering is
//! not doing anything a restart would break, and is reserving nothing.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use serde_json::Value;

/// How long a status is waited for.
const TIMEOUT: Duration = Duration::from_secs(3);

/// The `/v1/status` at `addr`, a `host:port`; `None` when it cannot be read.
pub fn status_at(addr: &str) -> Option<Value> {
    let addr: SocketAddr = addr.parse().ok()?;
    let read = || -> io::Result<String> {
        let mut stream = TcpStream::connect_timeout(&addr, TIMEOUT)?;
        stream.set_read_timeout(Some(TIMEOUT))?;
        write!(
            stream,
            "GET /v1/status HTTP/1.0\r\nHost: {addr}\r\nx-tokera-email: keep@test.invalid\r\n\r\n"
        )?;
        let mut answer = String::new();
        stream.read_to_string(&mut answer)?;
        Ok(answer)
    };
    let answer = read().ok()?;
    let (_, body) = answer.split_once("\r\n\r\n")?;
    serde_json::from_str(body).ok()
}

/// Why `status` refuses: its `refuse`, when it is a non-empty string.
pub fn reason(status: &Value) -> Option<String> {
    status["refuse"]
        .as_str()
        .map(str::trim)
        .filter(|why| !why.is_empty())
        .map(str::to_string)
}

/// The [`reason`] the process whose status is at `addr` gives, if it refuses.
pub fn reason_at(addr: &str) -> Option<String> {
    reason(&status_at(addr)?)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **A status's `refuse` is its reason**, trimmed; an empty or missing
    /// one, or one that is not text, refuses nothing.
    #[test]
    fn a_status_refuses_with_its_reason() {
        assert_eq!(
            reason(&json!({"refuse": " rekordbox is open "})).as_deref(),
            Some("rekordbox is open")
        );
        assert_eq!(reason(&json!({"refuse": "  "})), None);
        assert_eq!(reason(&json!({"refuse": null})), None);
        assert_eq!(reason(&json!({"refuse": true})), None);
        assert_eq!(reason(&json!({})), None);
    }

    #[test]
    fn a_status_that_cannot_be_read_refuses_nothing() {
        assert_eq!(reason_at("127.0.0.1:1"), None);
        assert_eq!(reason_at("not an address"), None);
    }
}
