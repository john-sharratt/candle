//! Shared reader for the KV-cache binary dump files produced by
//! `quantized_llama::tests::test_dump_kv_cache_data`.
//!
//! Supports format versions:
//!   v1 – header only (no token data)
//!   v2 – header + token sequence
//!   v3 – v2 + per-chunk token_start (first sequence position in each chunk)
//!   v4 – v3 + per-chunk Q data (from R16 Q-capture arenas)
//!
//! Binary layout (little-endian):
//!
//!   magic        : [u8; 8]  = b"KVDUMP\0\0"
//!   version      : u32      (1, 2, 3 or 4)
//!   num_layers   : u32
//!   n_kv_head    : u32
//!   chunk_size   : u32
//!   head_dim     : u32
//!   [v2+ only]
//!   num_tokens   : u32
//!   tokens       : [u32; num_tokens]
//!
//!   Per layer (num_layers total):
//!     num_chunks : u32
//!     Per chunk:
//!       block_idx   : u32
//!       [v3+ only]
//!       token_start : u32   (first sequence position covered by this chunk)
//!       k_data      : [f32; n_kv_head * chunk_size * head_dim]
//!       v_data      : [f32; n_kv_head * chunk_size * head_dim]
//!       [v4 only]
//!       q_data      : [f32; n_kv_head * chunk_size * head_dim]

use std::path::Path;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// Header decoded from the dump file.
#[derive(Debug, Clone)]
pub struct DumpHeader {
    pub num_layers: usize,
    pub n_kv_head: usize,
    pub chunk_size: usize,
    pub head_dim: usize,
    /// Token IDs for the session (empty for v1 dumps that pre-date token recording).
    pub tokens: Vec<u32>,
}

/// One chunk's worth of float KV data.
#[derive(Debug, Clone)]
pub struct ChunkData {
    pub layer_idx: usize,
    pub block_idx: usize,
    /// First sequence position (token index) covered by this chunk.
    /// For a chunk of size S at sequential position N, token_start = N * S.
    pub token_start: usize,
    /// Flat f32, logical shape [n_kv_head, chunk_size, head_dim].
    pub k: Vec<f32>,
    /// Flat f32, logical shape [n_kv_head, chunk_size, head_dim].
    pub v: Vec<f32>,
    /// Flat f32, logical shape [n_kv_head, chunk_size, head_dim].  Only present in v4 dumps.
    pub q: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Primitive readers
// ---------------------------------------------------------------------------

pub fn read_u32_le(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    if *pos + 4 > bytes.len() {
        return None;
    }
    let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().ok()?);
    *pos += 4;
    Some(v)
}

pub fn read_f32_le(bytes: &[u8], pos: &mut usize) -> Option<f32> {
    if *pos + 4 > bytes.len() {
        return None;
    }
    let v = f32::from_le_bytes(bytes[*pos..*pos + 4].try_into().ok()?);
    *pos += 4;
    Some(v)
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load the binary dump file.  Returns `None` if the file does not exist or
/// the magic / version check fails.
pub fn load_dump(path: &Path) -> Option<(DumpHeader, Vec<ChunkData>)> {
    let bytes = std::fs::read(path).ok()?;
    let mut pos: usize;

    // Magic
    if bytes.len() < 8 || &bytes[0..8] != b"KVDUMP\0\0" {
        eprintln!("dump_reader: bad magic in {:?}", path);
        return None;
    }
    pos = 8;

    let version = read_u32_le(&bytes, &mut pos)?;
    if version != 1 && version != 2 && version != 3 && version != 4 {
        eprintln!("dump_reader: unsupported dump version {} in {:?}", version, path);
        return None;
    }

    let num_layers = read_u32_le(&bytes, &mut pos)? as usize;
    let n_kv_head  = read_u32_le(&bytes, &mut pos)? as usize;
    let chunk_size = read_u32_le(&bytes, &mut pos)? as usize;
    let head_dim   = read_u32_le(&bytes, &mut pos)? as usize;

    // v2: read token sequence
    let tokens = if version >= 2 {
        let num_tokens = read_u32_le(&bytes, &mut pos)? as usize;
        let mut ts = Vec::with_capacity(num_tokens);
        for _ in 0..num_tokens {
            ts.push(read_u32_le(&bytes, &mut pos)?);
        }
        ts
    } else {
        Vec::new()
    };

    let header = DumpHeader { num_layers, n_kv_head, chunk_size, head_dim, tokens };
    let elems_per_chunk = n_kv_head * chunk_size * head_dim;

    let mut chunks = Vec::new();
    for layer_idx in 0..num_layers {
        let num_chunks = read_u32_le(&bytes, &mut pos)? as usize;
        for chunk_seq_idx in 0..num_chunks {
            let block_idx = read_u32_le(&bytes, &mut pos)? as usize;
            // v3+ stores token_start explicitly; earlier versions derive it from
            // sequential chunk position (valid for single linear sequences).
            let token_start = if version >= 3 {
                read_u32_le(&bytes, &mut pos)? as usize
            } else {
                chunk_seq_idx * chunk_size
            };
            let mut k = Vec::with_capacity(elems_per_chunk);
            for _ in 0..elems_per_chunk {
                k.push(read_f32_le(&bytes, &mut pos)?);
            }
            let mut v = Vec::with_capacity(elems_per_chunk);
            for _ in 0..elems_per_chunk {
                v.push(read_f32_le(&bytes, &mut pos)?);
            }
            let q = if version >= 4 {
                let mut q = Vec::with_capacity(elems_per_chunk);
                for _ in 0..elems_per_chunk {
                    q.push(read_f32_le(&bytes, &mut pos)?);
                }
                Some(q)
            } else {
                None
            };
            chunks.push(ChunkData { layer_idx, block_idx, token_start, k, v, q });
        }
    }

    Some((header, chunks))
}
