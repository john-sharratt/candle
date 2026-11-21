//! Tests the CRC-32 check value and the byte reader and writer.
//!
//! The check value pins the same function the log has always used, and a split
//! update must equal a single pass over the same bytes. The byte tests check that
//! a round trip is exact and that a short read is an error.

use super::*;

#[cfg(test)]
pub(super) mod crc32_tests {
    use super::{crc32, crc32_finish, crc32_init, crc32_update};

    /// The standard CRC-32 check value — the same function the log has always
    /// carried, so every existing record still verifies.
    #[test]
    fn crc32_is_the_ieee_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    /// A running accumulator fed in pieces lands where one pass does — the form
    /// the chunked writers use.
    #[test]
    fn a_split_update_equals_one_pass() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i * 31 % 251) as u8).collect();
        let (a, b) = data.split_at(3_777);
        let split = crc32_finish(crc32_update(crc32_update(crc32_init(), a), b));
        assert_eq!(split, crc32(&data));
    }
}

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn byte_primitives_roundtrip() {
        let mut w = ByteWriter::new();
        w.put_u8(0xAB);
        w.put_u16(0x1234);
        w.put_u32(0xDEAD_BEEF);
        w.put_u64(0x0123_4567_89AB_CDEF);
        w.put_i32(-12345);
        w.put_f32(3.5);
        w.put_blob(&[9, 8, 7]);
        w.put_str("héllo");
        let bytes = w.into_bytes();

        let mut r = ByteReader::new(&bytes);
        assert_eq!(r.get_u8().unwrap(), 0xAB);
        assert_eq!(r.get_u16().unwrap(), 0x1234);
        assert_eq!(r.get_u32().unwrap(), 0xDEAD_BEEF);
        assert_eq!(r.get_u64().unwrap(), 0x0123_4567_89AB_CDEF);
        assert_eq!(r.get_i32().unwrap(), -12345);
        assert_eq!(r.get_f32().unwrap(), 3.5);
        assert_eq!(r.get_blob().unwrap(), &[9, 8, 7]);
        assert_eq!(r.get_str().unwrap(), "héllo");
        assert!(r.is_done());
    }

    #[test]
    fn byte_reader_truncation_is_an_error() {
        let mut r = ByteReader::new(&[1, 2, 3]);
        assert!(r.get_u64().is_err());
    }
}
