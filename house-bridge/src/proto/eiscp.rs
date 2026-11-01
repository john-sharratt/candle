//! eISCP, the Integra Serial Control Protocol framed over TCP: the wire an
//! Onkyo, Integra or Pioneer network receiver speaks on port 60128.
//!
//! A message like `PWRQSTN` (query power) or `PWR01` (power on) is wrapped as
//! `!1<message>\r` and framed in a fixed 16-byte header; the device answers
//! with the same framing, its message trailed by the protocol's `EOF CR LF`.
//! The parsing of a message belongs to the profile; this is only the framing
//! and the exchange.

use std::net::Ipv4Addr;
use std::time::Duration;

use anyhow::{bail, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

/// The fixed eISCP header length: magic, header size, data size, version,
/// three reserved bytes.
const HEADER: usize = 16;

/// The packet carrying ISCP `message` to a receiver: the 16-byte header and the
/// `!1<message>\r` payload it frames (unit type `1`, the receiver).
pub fn pack(message: &str) -> Vec<u8> {
    let data = format!("!1{message}\r").into_bytes();
    let mut p = Vec::with_capacity(HEADER + data.len());
    p.extend_from_slice(b"ISCP");
    p.extend_from_slice(&(HEADER as u32).to_be_bytes());
    p.extend_from_slice(&(data.len() as u32).to_be_bytes());
    p.push(0x01); // version
    p.extend_from_slice(&[0, 0, 0]); // reserved
    p.extend_from_slice(&data);
    p
}

/// The ISCP message in one packet's data, stripped of its `!1` lead and its
/// trailing `EOF`/`CR`/`LF` — `!1PWR00\x1a\r\n` → `PWR00`. `None` if it carries
/// nothing.
fn clean(data: &[u8]) -> Option<String> {
    let s = String::from_utf8_lossy(data);
    let s = s.trim_end_matches(|c: char| matches!(c, '\u{1a}' | '\r' | '\n') || c.is_whitespace());
    let body = match s.strip_prefix('!') {
        // `rest` is the unit-type char followed by the message, e.g. `1PWR00`.
        Some(rest) => {
            let mut it = rest.chars();
            it.next();
            it.as_str()
        }
        None => s,
    };
    (!body.is_empty()).then(|| body.to_string())
}

/// Every complete packet's message in `buf`, in order, and how many bytes were
/// consumed — a trailing partial packet is left for the next read.
pub fn parse(buf: &[u8]) -> (Vec<String>, usize) {
    let mut msgs = Vec::new();
    let mut i = 0;
    while buf.len() - i >= HEADER {
        if &buf[i..i + 4] != b"ISCP" {
            break;
        }
        let header_size =
            u32::from_be_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) as usize;
        let data_size =
            u32::from_be_bytes([buf[i + 8], buf[i + 9], buf[i + 10], buf[i + 11]]) as usize;
        if header_size < HEADER || buf.len() - i < header_size + data_size {
            break;
        }
        let start = i + header_size;
        if let Some(m) = clean(&buf[start..start + data_size]) {
            msgs.push(m);
        }
        i = start + data_size;
    }
    (msgs, i)
}

/// Open TCP to `address:port`, send each of `cmds` as its own eISCP packet, and
/// read messages until one begins with `want` — the device's confirmation of
/// the new state — returning every message seen, in order. A receiver streams
/// state unasked, so the messages before the confirmation are kept.
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
            conn.write_all(&pack(cmd)).await?;
        }
        let mut got: Vec<String> = Vec::new();
        let mut buf = [0u8; 1024];
        let mut acc: Vec<u8> = Vec::new();
        loop {
            let n = conn.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            acc.extend_from_slice(&buf[..n]);
            let (msgs, consumed) = parse(&acc);
            acc.drain(..consumed);
            let mut done = false;
            for m in msgs {
                if m.starts_with(want) {
                    done = true;
                }
                got.push(m);
            }
            if done {
                break;
            }
        }
        Ok::<Vec<String>, anyhow::Error>(got)
    };
    match timeout(answer, run).await {
        Ok(r) => r,
        Err(_) => bail!("it did not answer in time"),
    }
}

/// Send each of `cmds` and collect whatever the receiver says over a short
/// `grace` — for a command it does not confirm with a known reply (transport
/// keys, the dimmer, a raw command). Silence is not an error: an empty answer
/// just means it acknowledged without a status.
pub async fn tell(
    address: Ipv4Addr,
    port: u16,
    cmds: &[&str],
    grace: Duration,
) -> Result<Vec<String>> {
    let mut conn = TcpStream::connect((address, port)).await?;
    for cmd in cmds {
        conn.write_all(&pack(cmd)).await?;
    }
    let mut got: Vec<String> = Vec::new();
    let mut buf = [0u8; 1024];
    let mut acc: Vec<u8> = Vec::new();
    loop {
        match timeout(grace, conn.read(&mut buf)).await {
            Ok(Ok(0)) => break,
            Ok(Ok(n)) => {
                acc.extend_from_slice(&buf[..n]);
                let (msgs, consumed) = parse(&acc);
                acc.drain(..consumed);
                got.extend(msgs);
            }
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => break, // the grace elapsed with nothing more to read
        }
    }
    Ok(got)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A query is the eISCP packet the receiver expects, byte for byte** — the
    /// exact bytes the Onkyo at `.61` answered a power query to.
    #[test]
    fn a_query_is_the_packet_the_receiver_expects() {
        assert_eq!(
            pack("PWRQSTN"),
            b"ISCP\x00\x00\x00\x10\x00\x00\x00\x0a\x01\x00\x00\x00!1PWRQSTN\r"
        );
    }

    /// **A reply is read out of its framing and cleaned** — the real `PWR00`
    /// answer, terminated by `EOF CR LF`, consumed whole.
    #[test]
    fn a_reply_is_read_out_of_its_framing() {
        let resp = b"ISCP\x00\x00\x00\x10\x00\x00\x00\x0a\x01\x00\x00\x00!1PWR00\x1a\r\n";
        let (msgs, consumed) = parse(resp);
        assert_eq!(msgs, vec!["PWR00"]);
        assert_eq!(consumed, resp.len());
    }

    /// **Back-to-back packets each parse** — the real volume and mute answers
    /// arriving in one read.
    #[test]
    fn back_to_back_packets_each_parse() {
        let two = b"ISCP\x00\x00\x00\x10\x00\x00\x00\x0a\x01\x00\x00\x00!1MVL4D\x1a\r\n\
                    ISCP\x00\x00\x00\x10\x00\x00\x00\x0a\x01\x00\x00\x00!1AMT00\x1a\r\n";
        let (msgs, consumed) = parse(two);
        assert_eq!(msgs, vec!["MVL4D", "AMT00"]);
        assert_eq!(consumed, two.len());
    }

    /// **A partial packet is left for the next read** — nothing is consumed
    /// until a whole packet is in hand.
    #[test]
    fn a_partial_packet_is_held_back() {
        let partial = b"ISCP\x00\x00\x00\x10\x00\x00\x00\x0a\x01\x00\x00\x00!1PWR0";
        let (msgs, consumed) = parse(partial);
        assert!(msgs.is_empty());
        assert_eq!(consumed, 0);
    }
}
