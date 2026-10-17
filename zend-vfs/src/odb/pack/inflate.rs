//! An entry's zlib stream, inflated.
//!
//! The header says how large the stream inflates to, and nothing says how long
//! it is compressed: the stream ends where zlib says it does, which is where
//! the next entry begins.

use flate2::{Decompress, FlushDecompress, Status};

use crate::error::GitError;

/// The stream at `at` in `pack`, inflated to exactly `size` bytes, and how
/// many compressed bytes it took.
pub(crate) fn inflate(pack: &[u8], at: usize, size: u64) -> Result<(Vec<u8>, usize), GitError> {
    let size = usize::try_from(size)
        .map_err(|_| GitError::malformed("pack", "an entry too large to hold"))?;
    let input = pack
        .get(at..)
        .ok_or_else(|| GitError::malformed("pack", "an entry's data past the pack's end"))?;
    let mut z = Decompress::new(true);
    let mut out = Vec::with_capacity(size);
    loop {
        let before = out.len();
        let status = z
            .decompress_vec(
                &input[z.total_in() as usize..],
                &mut out,
                FlushDecompress::Finish,
            )
            .map_err(|e| GitError::malformed("pack", format!("an entry's zlib stream: {e}")))?;
        match status {
            Status::StreamEnd => break,
            // Room for the stream's last bytes: a stream that holds more than
            // its header said is caught below.
            Status::Ok | Status::BufError if out.len() == out.capacity() => {
                out.reserve(64);
            }
            Status::Ok | Status::BufError if out.len() == before => {
                return Err(GitError::malformed(
                    "pack",
                    "an entry's zlib stream cut short",
                ));
            }
            Status::Ok | Status::BufError => {}
        }
    }
    if out.len() != size {
        return Err(GitError::malformed(
            "pack",
            format!(
                "an entry inflated to {} bytes, its header says {size}",
                out.len()
            ),
        ));
    }
    Ok((out, z.total_in() as usize))
}

/// The first bytes of the stream at `at`, at most `max` of them: enough of a
/// delta to read the sizes at its front without inflating the rest.
pub(crate) fn inflate_prefix(pack: &[u8], at: usize, max: usize) -> Result<Vec<u8>, GitError> {
    let input = pack
        .get(at..)
        .ok_or_else(|| GitError::malformed("pack", "an entry's data past the pack's end"))?;
    let mut z = Decompress::new(true);
    let mut out = Vec::with_capacity(max);
    while out.len() < max {
        let before = out.len();
        let status = z
            .decompress_vec(
                &input[z.total_in() as usize..],
                &mut out,
                FlushDecompress::None,
            )
            .map_err(|e| GitError::malformed("pack", format!("an entry's zlib stream: {e}")))?;
        if status == Status::StreamEnd || out.len() == before {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn deflate(bytes: &[u8]) -> Vec<u8> {
        let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
        e.write_all(bytes).unwrap();
        e.finish().unwrap()
    }

    /// **A stream inflates to its bytes and says where it ended**, so the next
    /// entry is found right after it.
    #[test]
    fn a_stream_inflates_and_reports_its_length() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let packed = deflate(&data);
        let mut pack = b"xx".to_vec();
        pack.extend_from_slice(&packed);
        pack.extend_from_slice(b"NEXT");
        let (out, used) = inflate(&pack, 2, data.len() as u64).unwrap();
        assert_eq!(out, data);
        assert_eq!(used, packed.len());
        assert_eq!(&pack[2 + used..], b"NEXT");
    }

    #[test]
    fn an_empty_object_inflates_to_nothing() {
        let packed = deflate(b"");
        assert_eq!(inflate(&packed, 0, 0).unwrap(), (Vec::new(), packed.len()));
    }

    /// **A size that disagrees with the stream is refused**, either way.
    #[test]
    fn a_size_that_disagrees_is_refused() {
        let packed = deflate(b"hello");
        assert!(inflate(&packed, 0, 4).is_err());
        assert!(inflate(&packed, 0, 6).is_err());
        assert!(inflate(&packed[..packed.len() - 3], 0, 5).is_err());
    }

    #[test]
    fn a_prefix_reads_only_the_front() {
        let data: Vec<u8> = (0..100_000u32).map(|i| (i * 7 % 256) as u8).collect();
        let packed = deflate(&data);
        let front = inflate_prefix(&packed, 0, 16).unwrap();
        assert!(front.len() >= 16);
        assert_eq!(&front[..16], &data[..16]);
    }
}
