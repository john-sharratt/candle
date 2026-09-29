//! Git's pkt-line framing: every packet is its length, as four lowercase hex
//! digits counting themselves, then its bytes. Three lengths below four are
//! markers rather than packets: `0000` flush, `0001` delimiter, `0002` the
//! end of a response.

use super::ProbeError;

/// The flush packet, ending a message.
pub const FLUSH: &[u8] = b"0000";

/// The delimiter packet, separating a v2 request's capabilities from its
/// arguments.
pub const DELIM: &[u8] = b"0001";

/// The most bytes one packet carries: 65520 in all, less its length.
pub const MAX_DATA: usize = 65516;

/// One packet as read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packet<'a> {
    Data(&'a [u8]),
    Flush,
    Delim,
    ResponseEnd,
}

/// `data` framed as one packet.
///
/// # Panics
///
/// When `data` is longer than [`MAX_DATA`] — every packet this crate writes
/// is a short, fixed line.
pub fn encode(data: &[u8]) -> Vec<u8> {
    assert!(
        data.len() <= MAX_DATA,
        "a pkt-line holds at most {MAX_DATA} bytes"
    );
    let mut out = format!("{:04x}", data.len() + 4).into_bytes();
    out.extend_from_slice(data);
    out
}

/// Every packet in `bytes`, in order. A length that is not hex, that is 3,
/// or that runs past the end is malformed; so is anything after the last
/// whole packet.
pub fn decode(bytes: &[u8]) -> Result<Vec<Packet<'_>>, ProbeError> {
    let mut packets = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let head = bytes.get(at..at + 4).ok_or_else(|| {
            ProbeError::Malformed(format!("a packet length is cut off at byte {at}"))
        })?;
        let len = std::str::from_utf8(head)
            .ok()
            .filter(|h| h.bytes().all(|b| b.is_ascii_hexdigit()))
            .and_then(|h| usize::from_str_radix(h, 16).ok())
            .ok_or_else(|| {
                ProbeError::Malformed(format!(
                    "{:?} at byte {at} is not a packet length",
                    String::from_utf8_lossy(head)
                ))
            })?;
        let packet = match len {
            0 => Packet::Flush,
            1 => Packet::Delim,
            2 => Packet::ResponseEnd,
            3 => {
                return Err(ProbeError::Malformed(format!(
                    "a packet at byte {at} has length 3"
                )))
            }
            _ => Packet::Data(bytes.get(at + 4..at + len).ok_or_else(|| {
                ProbeError::Malformed(format!(
                    "a {len}-byte packet at byte {at} runs past the end"
                ))
            })?),
        };
        at += len.max(4);
        packets.push(packet);
    }
    Ok(packets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_packet_is_its_length_in_hex_then_its_bytes() {
        assert_eq!(encode(b"command=ls-refs\n"), b"0014command=ls-refs\n");
        assert_eq!(encode(b""), b"0004");
        assert_eq!(encode(b"a"), b"0005a");
    }

    #[test]
    fn packets_and_markers_decode_in_order() {
        let bytes = b"0005a0001000aab\ncd\n00020000";
        assert_eq!(
            decode(bytes).unwrap(),
            vec![
                Packet::Data(b"a"),
                Packet::Delim,
                Packet::Data(b"ab\ncd\n"),
                Packet::ResponseEnd,
                Packet::Flush,
            ]
        );
        assert_eq!(decode(b"").unwrap(), vec![]);
    }

    #[test]
    fn what_round_trips_is_what_was_framed() {
        let framed = [encode(b"one\n"), encode(b"two\n"), FLUSH.to_vec()].concat();
        assert_eq!(
            decode(&framed).unwrap(),
            vec![
                Packet::Data(b"one\n"),
                Packet::Data(b"two\n"),
                Packet::Flush
            ]
        );
    }

    /// Git reads a length's hex in either case, so a server writing capitals
    /// is still understood.
    #[test]
    fn a_length_in_capitals_is_read() {
        assert_eq!(
            decode(b"000Aabcdef").unwrap(),
            vec![Packet::Data(b"abcdef")]
        );
    }

    /// A length of 3, a packet running past the end, a length cut off part
    /// way and a length that is not hex are each malformed — never read as a
    /// shorter packet.
    #[test]
    fn a_bad_length_is_malformed() {
        for bad in [&b"0003"[..], b"0009abc", b"00", b"zzzz", b"+004"] {
            assert!(
                matches!(decode(bad), Err(ProbeError::Malformed(_))),
                "{:?}",
                String::from_utf8_lossy(bad)
            );
        }
    }
}
