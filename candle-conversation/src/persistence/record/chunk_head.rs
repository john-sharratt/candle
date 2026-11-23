//! A KV chunk record's head: what a `ChunkPayload` says before its arena
//! bytes — the format of every K and V sub-band, and how long the metadata and
//! the arena bytes are — decoded from the payload's first bytes alone, so a
//! census reads one sector of a chunk rather than all of it.

use super::*;

/// The formats and lengths a chunk payload's metadata prefix states.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkHead {
    /// `KvFormat` tag of every K sub-band, as `ChunkPayload::k_formats`.
    pub k_formats: Vec<u8>,
    /// The same for V.
    pub v_formats: Vec<u8>,
    /// Bytes before the arena blob: the formats, palette maps, scales and
    /// every length prefix.
    pub side_len: usize,
    /// The arena blob's length.
    pub kv_len: usize,
}

/// The head of the chunk payload that `head` begins: every field up to the
/// arena blob's length prefix, which must lie within `head`.
pub fn decode_chunk_head(head: &[u8]) -> Result<ChunkHead> {
    let mut r = ByteReader::new(head);
    let _offset = r.get_u16()?;
    let k_formats = r.get_blob()?.to_vec();
    let v_formats = r.get_blob()?.to_vec();
    let _k_pal = r.get_blob()?;
    let _v_pal = r.get_blob()?;
    for _ in 0..2 {
        let n = r.get_u32()? as usize;
        for _ in 0..n {
            r.get_f32()?;
        }
    }
    let kv_len = r.get_u32()? as usize;
    Ok(ChunkHead {
        k_formats,
        v_formats,
        side_len: r.position(),
        kv_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload() -> ChunkPayload {
        ChunkPayload {
            offset: 3,
            k_formats: vec![15, 15],
            v_formats: vec![8, 8],
            k_pal: vec![0xAA],
            v_pal: vec![0xBB, 0xCC],
            k_scale: vec![1.0],
            v_scale: vec![],
            kv_bytes: vec![7; 100],
        }
    }

    /// **The head reads from the prefix alone**: offset 2 bytes, two
    /// 4-byte-prefixed format blobs of 2 (12), palette blobs of 1 and 2 (11),
    /// one scale (8), none (4), then the arena's length prefix (4) — 41 bytes
    /// before 100 bytes of arena.
    #[test]
    fn a_head_is_read_from_the_payloads_first_bytes() {
        let encoded = payload().encode();
        assert_eq!(encoded.len(), 141);
        let head = decode_chunk_head(&encoded[..41]).unwrap();
        assert_eq!(
            head,
            ChunkHead {
                k_formats: vec![15, 15],
                v_formats: vec![8, 8],
                side_len: 41,
                kv_len: 100,
            }
        );
        assert_eq!(decode_chunk_head(&encoded).unwrap(), head);
    }

    #[test]
    fn a_head_cut_before_the_arena_length_is_refused() {
        let encoded = payload().encode();
        assert!(decode_chunk_head(&encoded[..40]).is_err());
    }
}
