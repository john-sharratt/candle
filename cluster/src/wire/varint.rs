//! Unsigned LEB128: seven bits a byte, low first, the top bit set on every byte
//! but the last. Used where most values are small — the applied set's packing
//! (§7.5) sends about ten bytes per caught-up origin.

use super::WireError;

pub fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// The value and the bytes it took. More than ten bytes, a tenth byte above
/// `1`, or a non-minimal encoding is refused, so one value has one spelling.
pub fn read_varint(buf: &[u8], what: &'static str) -> Result<(u64, usize), WireError> {
    let mut v: u64 = 0;
    for (i, byte) in buf.iter().enumerate() {
        if i == 9 && *byte > 1 {
            return Err(WireError::Invalid {
                what,
                value: *byte as u64,
            });
        }
        v |= ((byte & 0x7f) as u64) << (7 * i);
        if byte & 0x80 == 0 {
            if i > 0 && *byte == 0 {
                return Err(WireError::Invalid { what, value: v });
            }
            return Ok((v, i + 1));
        }
    }
    Err(WireError::Truncated { what, need: 1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        write_varint(&mut out, v);
        out
    }

    #[test]
    fn values_encode_to_exact_bytes() {
        assert_eq!(enc(0), vec![0x00]);
        assert_eq!(enc(1), vec![0x01]);
        assert_eq!(enc(127), vec![0x7f]);
        assert_eq!(enc(128), vec![0x80, 0x01]);
        assert_eq!(enc(300), vec![0xac, 0x02]);
        assert_eq!(
            enc(u64::MAX),
            vec![0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]
        );
        for v in [0, 1, 127, 128, 300, 1 << 35, u64::MAX] {
            assert_eq!(read_varint(&enc(v), "t").unwrap(), (v, enc(v).len()));
        }
    }

    #[test]
    fn a_truncated_overlong_or_padded_varint_is_refused() {
        assert!(matches!(
            read_varint(&[0x80], "t"),
            Err(WireError::Truncated { .. })
        ));
        assert!(matches!(
            read_varint(
                &[0xff; 9].iter().copied().chain([0x02]).collect::<Vec<_>>(),
                "t"
            ),
            Err(WireError::Invalid { .. })
        ));
        assert!(matches!(
            read_varint(&[0x81, 0x00], "t"),
            Err(WireError::Invalid { .. })
        ));
    }
}
