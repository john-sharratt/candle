//! A line-at-a-time TCP control protocol: open a connection, send each command
//! as a CR/LF line, and read the device's short status lines back — the wire an
//! AV receiver speaks on port 8102.
//!
//! The parsing of the lines belongs to the profile; this is only the exchange.

use std::net::Ipv4Addr;
use std::time::Duration;

use anyhow::{bail, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

/// Open TCP to `address:port`, send each of `cmds` as a `\r\n` line, and read
/// lines until one begins with `want` — returning every non-empty line seen, in
/// order. `want` is the device's confirmation of the new state; many such
/// devices also stream their state unasked, so the lines before it are kept.
pub async fn exchange(
    address: Ipv4Addr,
    port: u16,
    cmds: &[&str],
    want: &str,
    answer: Duration,
) -> Result<Vec<String>> {
    let run = async {
        let mut conn = TcpStream::connect((address, port)).await?;
        for cmd in cmds {
            conn.write_all(format!("{cmd}\r\n").as_bytes()).await?;
        }
        let mut lines: Vec<String> = Vec::new();
        let mut buf = [0u8; 1024];
        let mut acc = String::new();
        loop {
            let n = conn.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            acc.push_str(&String::from_utf8_lossy(&buf[..n]));
            // Keep the last partial line in `acc`; emit every complete one.
            let mut rest = String::new();
            let mut done = false;
            let ends_clean = acc.ends_with('\r') || acc.ends_with('\n');
            let pieces: Vec<&str> = acc.split(['\r', '\n']).collect();
            let (complete, trailing) = pieces.split_at(pieces.len().saturating_sub(1));
            for line in complete {
                if line.is_empty() {
                    continue;
                }
                lines.push(line.to_string());
                if line.starts_with(want) {
                    done = true;
                }
            }
            if let Some(last) = trailing.first() {
                if ends_clean {
                    if !last.is_empty() {
                        lines.push(last.to_string());
                        if last.starts_with(want) {
                            done = true;
                        }
                    }
                } else {
                    rest.push_str(last);
                }
            }
            acc = rest;
            if done {
                break;
            }
        }
        Ok::<Vec<String>, anyhow::Error>(lines)
    };
    match timeout(answer, run).await {
        Ok(r) => r,
        Err(_) => bail!("it did not answer in time"),
    }
}
