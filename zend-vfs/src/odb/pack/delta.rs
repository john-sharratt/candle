//! Git's delta format: an object written as instructions over a base.
//!
//! A delta opens with two sizes, the base's and the result's, each seven bits a
//! byte, low bits first. Then instructions: a byte with its top bit set copies
//! from the base — its low four bits say which offset bytes follow, the next
//! three which size bytes, a size of zero meaning 0x10000 — and any other
//! non-zero byte inserts that many of the bytes after it. A zero byte is
//! reserved, and refused.

use crate::error::GitError;

fn bad(detail: &str) -> GitError {
    GitError::malformed("pack", format!("a delta: {detail}"))
}

/// One size from the front of `delta` at `at`; the size and where it ends.
fn size_at(delta: &[u8], mut at: usize) -> Result<(u64, usize), GitError> {
    let mut size = 0u64;
    let mut shift = 0;
    loop {
        let byte = *delta.get(at).ok_or_else(|| bad("its sizes cut short"))?;
        at += 1;
        if shift > 63 {
            return Err(bad("a size over 64 bits"));
        }
        size |= u64::from(byte & 0x7f) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Ok((size, at));
        }
    }
}

/// The base's size and the result's, from the front of `delta`, and where the
/// instructions start.
pub(crate) fn sizes(delta: &[u8]) -> Result<(u64, u64, usize), GitError> {
    let (base, at) = size_at(delta, 0)?;
    let (result, at) = size_at(delta, at)?;
    Ok((base, result, at))
}

/// `delta` applied to `base`.
pub(crate) fn apply(base: &[u8], delta: &[u8]) -> Result<Vec<u8>, GitError> {
    let (base_size, result_size, mut at) = sizes(delta)?;
    if base_size != base.len() as u64 {
        return Err(bad(&format!(
            "its base is {} bytes, it says {base_size}",
            base.len()
        )));
    }
    let result_size =
        usize::try_from(result_size).map_err(|_| bad("a result too large to hold"))?;
    let mut out = Vec::with_capacity(result_size);
    while at < delta.len() {
        let op = delta[at];
        at += 1;
        if op & 0x80 != 0 {
            let mut offset = 0usize;
            for i in 0..4 {
                if op & (1 << i) != 0 {
                    let b = *delta.get(at).ok_or_else(|| bad("a copy cut short"))?;
                    at += 1;
                    offset |= usize::from(b) << (8 * i);
                }
            }
            let mut size = 0usize;
            for i in 0..3 {
                if op & (0x10 << i) != 0 {
                    let b = *delta.get(at).ok_or_else(|| bad("a copy cut short"))?;
                    at += 1;
                    size |= usize::from(b) << (8 * i);
                }
            }
            if size == 0 {
                size = 0x10000;
            }
            let end = offset
                .checked_add(size)
                .filter(|&end| end <= base.len())
                .ok_or_else(|| bad("a copy past its base's end"))?;
            out.extend_from_slice(&base[offset..end]);
        } else if op != 0 {
            let n = usize::from(op);
            let bytes = delta
                .get(at..at + n)
                .ok_or_else(|| bad("an insert cut short"))?;
            out.extend_from_slice(bytes);
            at += n;
        } else {
            return Err(bad("the reserved instruction 0"));
        }
        if out.len() > result_size {
            return Err(bad("a result longer than it says"));
        }
    }
    if out.len() != result_size {
        return Err(bad(&format!(
            "a result of {} bytes, it says {result_size}",
            out.len()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A copy and an insert make the result**: base `hello world`, the
    /// result `hello there` — copy 6 bytes from 0, insert `there`.
    #[test]
    fn a_copy_and_an_insert_make_the_result() {
        let base = b"hello world";
        // sizes 11, 11; copy (0x80 | size byte 0 present 0x10) size 6 → 0x90 0x06;
        // insert 5 → 0x05 "there".
        let delta = [11, 11, 0x90, 6, 5, b't', b'h', b'e', b'r', b'e'];
        assert_eq!(apply(base, &delta).unwrap(), b"hello there");
        assert_eq!(sizes(&delta).unwrap(), (11, 11, 2));
    }

    /// **A copy from an offset**: offset byte 0 present (0x01), size byte 0
    /// present (0x10): copy 5 bytes from 6 → `world`.
    #[test]
    fn a_copy_reads_its_offset() {
        let delta = [11, 5, 0x91, 6, 5];
        assert_eq!(apply(b"hello world", &delta).unwrap(), b"world");
    }

    /// **A copy of size zero copies 0x10000 bytes.**
    #[test]
    fn a_copy_of_size_zero_is_sixty_four_kibibytes() {
        let base = vec![9u8; 0x10000];
        // base size 65536 = 0x80 0x80 0x04; result the same.
        let delta = [0x80, 0x80, 0x04, 0x80, 0x80, 0x04, 0x80];
        assert_eq!(apply(&base, &delta).unwrap(), base);
    }

    #[test]
    fn a_delta_that_disagrees_with_its_base_or_itself_is_refused() {
        assert!(
            apply(b"short", &[11, 5, 0x90, 5]).is_err(),
            "wrong base size"
        );
        assert!(apply(b"hello", &[5, 9, 0x90, 5]).is_err(), "result shorter");
        assert!(apply(b"hello", &[5, 3, 0x90, 5]).is_err(), "result longer");
        assert!(
            apply(b"hello", &[5, 5, 0x91, 3, 5]).is_err(),
            "copy past the end"
        );
        assert!(apply(b"hello", &[5, 1, 0x00]).is_err(), "instruction 0");
        assert!(
            apply(b"hello", &[5, 3, 4, b'a']).is_err(),
            "insert cut short"
        );
    }
}
