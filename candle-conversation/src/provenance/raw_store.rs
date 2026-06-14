//! Mmap-backed flat file storing raw K and Q float vectors across a layer
//! band for every token in a sealed turn.
//!
//! Unlike [`super::store`] which stores binarised sign-bit signatures, this
//! file keeps full-precision f32 values so the test harness can sweep all
//! signature strategies (sign(Q), sign(K), Q·K dot-product, float SimHash,
//! multi-head variants, etc.) without re-running the model.
//!
//! # File layout
//!
//! ```text
//! [Header: 64 bytes]
//! [Entry 0 data blob]
//! [Entry 1 data blob]
//! ...
//! ```
//!
//! Each entry blob is `token_count × bytes_per_token` bytes where:
//! ```text
//! bytes_per_token = 3 × n_layers_per_band × n_kv_heads × 2 × head_dim × 4
//!                  ↑               ↑              ↑      ↑        ↑       ↑
//!                 bands       layers/band       heads  K+Q    head_dim  f32
//! ```
//!
//! Within one token's slice, data is ordered:
//! `band → layer → head → {K[head_dim], Q[head_dim]}` — all f32 LE.
//!
//! K layout (as extracted from `dump_r16_kv_for_provenance`):
//!   per block: `[head][CHUNK_SIZE tokens][head_dim]` → stored as `[head_dim]` per (t, h).
//!
//! Q layout (de-interleaved from R16 `[head][palette][CHUNK_SIZE][sub_dim]`):
//!   → stored as `[head_dim]` per (t, h) in natural dim order.

use std::fs::File;
use std::io::Write;
use std::sync::Mutex;

use memmap2::Mmap;
use tempfile::tempfile;

use crate::error::ConversationError;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Magic bytes at byte 0 of every raw provenance file.
pub const RAW_PROV_MAGIC: [u8; 8] = *b"RAWPROV1";

/// Fixed header size in bytes.
pub const RAW_HEADER_BYTES: usize = 64;

/// Number of depth bands (syntactic / semantic / pragmatic).
pub const RAW_NUM_BANDS: usize = 3;

// ── RawFileHeader ─────────────────────────────────────────────────────────────

/// Fixed-size 64-byte header stored at byte 0 of a [`RawProvenanceFile`].
///
/// All numeric fields are little-endian.
///
/// ```text
/// [0..8]   magic           = b"RAWPROV1"
/// [8..12]  n_kv_heads      (u32 LE)
/// [12..16] head_dim        (u32 LE)
/// [16..20] n_layers_per_band (u32 LE)
/// [20..24] band_half_width (u32 LE)  — informational
/// [24..28] band_centers[0] (u32 LE)  — syntactic center layer
/// [28..32] band_centers[1] (u32 LE)  — semantic center layer
/// [32..36] band_centers[2] (u32 LE)  — pragmatic center layer
/// [36..40] n_total_layers  (u32 LE)  — total model layers
/// [40..64] reserved zeros
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawFileHeader {
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Number of layers extracted per band (all three bands have the same count).
    pub n_layers_per_band: u32,
    /// Half-width used when computing bands: actual layer list = center ± half.
    pub band_half_width: u32,
    /// Center layer indices for [syntactic, semantic, pragmatic] bands.
    pub band_centers: [u32; 3],
    pub n_total_layers: u32,
}

impl RawFileHeader {
    /// Bytes contributed by one token to the per-entry data blob.
    #[inline]
    pub fn bytes_per_token(&self) -> usize {
        RAW_NUM_BANDS
            * self.n_layers_per_band as usize
            * self.n_kv_heads as usize
            * 2 // K and Q
            * self.head_dim as usize
            * 4 // f32
    }

    /// Byte offset within a token's slice for `(band, layer, head, is_q)`.
    ///
    /// The returned offset points to the first of `head_dim` consecutive f32
    /// values.  `is_q = false` → K vector, `is_q = true` → Q vector.
    #[inline]
    pub fn slot_offset(&self, band: usize, layer: usize, head: usize, is_q: bool) -> usize {
        let nl = self.n_layers_per_band as usize;
        let nh = self.n_kv_heads as usize;
        let hd = self.head_dim as usize;
        let kq: usize = if is_q { 1 } else { 0 };
        ((band * nl + layer) * nh * 2 + head * 2 + kq) * hd * 4
    }

    /// Byte offset within the entry blob for token `t`'s `(band, layer, head, is_q)` slot.
    #[inline]
    pub fn entry_offset(
        &self,
        t: usize,
        band: usize,
        layer: usize,
        head: usize,
        is_q: bool,
    ) -> usize {
        t * self.bytes_per_token() + self.slot_offset(band, layer, head, is_q)
    }

    pub(crate) fn to_bytes(self) -> [u8; RAW_HEADER_BYTES] {
        let mut buf = [0u8; RAW_HEADER_BYTES];
        buf[0..8].copy_from_slice(&RAW_PROV_MAGIC);
        buf[8..12].copy_from_slice(&self.n_kv_heads.to_le_bytes());
        buf[12..16].copy_from_slice(&self.head_dim.to_le_bytes());
        buf[16..20].copy_from_slice(&self.n_layers_per_band.to_le_bytes());
        buf[20..24].copy_from_slice(&self.band_half_width.to_le_bytes());
        buf[24..28].copy_from_slice(&self.band_centers[0].to_le_bytes());
        buf[28..32].copy_from_slice(&self.band_centers[1].to_le_bytes());
        buf[32..36].copy_from_slice(&self.band_centers[2].to_le_bytes());
        buf[36..40].copy_from_slice(&self.n_total_layers.to_le_bytes());
        buf
    }

    pub(crate) fn from_bytes(buf: &[u8; RAW_HEADER_BYTES]) -> crate::Result<Self> {
        if buf[0..8] != RAW_PROV_MAGIC {
            return Err(ConversationError::Other(
                "RawProvenanceFile: bad magic bytes".into(),
            ));
        }
        Ok(Self {
            n_kv_heads: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            head_dim: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
            n_layers_per_band: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            band_half_width: u32::from_le_bytes(buf[20..24].try_into().unwrap()),
            band_centers: [
                u32::from_le_bytes(buf[24..28].try_into().unwrap()),
                u32::from_le_bytes(buf[28..32].try_into().unwrap()),
                u32::from_le_bytes(buf[32..36].try_into().unwrap()),
            ],
            n_total_layers: u32::from_le_bytes(buf[36..40].try_into().unwrap()),
        })
    }
}

// ── RawSigEntry ───────────────────────────────────────────────────────────────

/// Location and token count of one scenario's raw KVQ blob in a [`RawProvenanceFile`].
#[derive(Clone, Copy, Debug)]
pub struct RawSigEntry {
    /// Byte offset of the blob within the file.
    pub byte_offset: u64,
    /// Number of tokens stored.
    pub token_count: u16,
}

impl RawSigEntry {
    pub fn byte_len(&self, header: &RawFileHeader) -> usize {
        self.token_count as usize * header.bytes_per_token()
    }
}

// ── RawProvenanceFile ─────────────────────────────────────────────────────────

struct State {
    file: File,
    write_pos: u64,
}

/// Append-only mmap-backed file storing raw f32 K and Q vectors.
///
/// One file per generator run.  Not shared across multiple processes.
///
/// When opened with [`RawProvenanceFile::open`] (read-only mode), the mmap is
/// created once and cached, so repeated `read_kq_vector` calls avoid the
/// `MapViewOfFile`/`UnmapViewOfFile` kernel overhead that would otherwise
/// dominate harness scan loops.  Write-mode constructors (`new`, `create`)
/// leave the cache empty and fall back to per-call mapping.
pub struct RawProvenanceFile {
    header: RawFileHeader,
    state: Mutex<State>,
    /// Persistent mmap created by `open()`; `None` in write mode.
    mmap: Option<Mmap>,
}

impl RawProvenanceFile {
    /// Create a new anonymous (tempfile-backed) raw provenance file.
    pub fn new(header: RawFileHeader) -> crate::Result<Self> {
        let mut file = tempfile()?;
        file.write_all(&header.to_bytes())?;
        file.flush()?;
        Ok(Self {
            header,
            state: Mutex::new(State {
                file,
                write_pos: RAW_HEADER_BYTES as u64,
            }),
            mmap: None,
        })
    }

    /// Create a new persistent raw provenance file at `path`.
    ///
    /// Truncates any existing file.
    pub fn create(path: impl AsRef<std::path::Path>, header: RawFileHeader) -> crate::Result<Self> {
        let path = path.as_ref();
        let mut file = File::create(path)?;
        file.write_all(&header.to_bytes())?;
        file.flush()?;
        Ok(Self {
            header,
            state: Mutex::new(State {
                file,
                write_pos: RAW_HEADER_BYTES as u64,
            }),
            mmap: None,
        })
    }

    /// Open an existing raw provenance file for reading (and optional appending).
    pub fn open(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        use std::fs::OpenOptions;
        use std::io::Read;
        let path = path.as_ref();
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut header_bytes = [0u8; RAW_HEADER_BYTES];
        file.read_exact(&mut header_bytes)?;
        let header = RawFileHeader::from_bytes(&header_bytes)?;
        let write_pos = file.metadata()?.len();
        // Cache the mmap once for all subsequent reads — avoids per-call
        // MapViewOfFile/UnmapViewOfFile overhead in read-only harness loops.
        let mmap = unsafe { Mmap::map(&file) }.ok();
        Ok(Self {
            header,
            state: Mutex::new(State { file, write_pos }),
            mmap,
        })
    }

    pub fn header(&self) -> &RawFileHeader {
        &self.header
    }

    /// Append one scenario's raw token data blob.
    ///
    /// `token_data` must be a byte slice of exactly `token_count ×
    /// header.bytes_per_token()` bytes, with layout as described in the
    /// module docs.  Returns a [`RawSigEntry`] recording the byte offset and
    /// token count for later reading.
    pub fn append(&self, token_data: &[u8]) -> crate::Result<RawSigEntry> {
        let bpt = self.header.bytes_per_token();
        if bpt == 0 {
            return Ok(RawSigEntry {
                byte_offset: 0,
                token_count: 0,
            });
        }
        assert_eq!(
            token_data.len() % bpt,
            0,
            "token_data length {} not a multiple of bytes_per_token {}",
            token_data.len(),
            bpt
        );
        let token_count = token_data.len() / bpt;
        assert!(
            token_count <= u16::MAX as usize,
            "token_count exceeds u16::MAX"
        );

        if token_count == 0 {
            let state = self.state.lock().unwrap();
            return Ok(RawSigEntry {
                byte_offset: state.write_pos,
                token_count: 0,
            });
        }

        let mut state = self.state.lock().unwrap();
        let byte_offset = state.write_pos;

        use std::io::Seek;
        state.file.seek(std::io::SeekFrom::Start(byte_offset))?;
        state.file.write_all(token_data)?;
        state.file.flush()?;
        state.write_pos += token_data.len() as u64;

        Ok(RawSigEntry {
            byte_offset,
            token_count: token_count as u16,
        })
    }

    /// Read the raw token blob for `entry` as a byte Vec.
    ///
    /// Caller can parse individual vectors using [`RawFileHeader::entry_offset`].
    pub fn read_entry_bytes(&self, entry: RawSigEntry) -> crate::Result<Vec<u8>> {
        if entry.token_count == 0 {
            return Ok(Vec::new());
        }
        let total = entry.byte_len(&self.header);
        self.with_mmap(|mmap| {
            let offset = entry.byte_offset as usize;
            if offset + total > mmap.len() {
                Vec::new()
            } else {
                mmap[offset..offset + total].to_vec()
            }
        })
    }

    /// Read one K or Q vector for token `t`, band `b`, layer `l`, head `h`.
    ///
    /// Returns `head_dim` f32 values.  Returns a zero vector on out-of-range access.
    pub fn read_kq_vector(
        &self,
        entry: RawSigEntry,
        t: usize,
        band: usize,
        layer: usize,
        head: usize,
        is_q: bool,
    ) -> crate::Result<Vec<f32>> {
        let hd = self.header.head_dim as usize;
        let inner_offset = self.header.entry_offset(t, band, layer, head, is_q);
        let file_offset = entry.byte_offset as usize + inner_offset;
        self.with_mmap(|mmap| {
            if file_offset + hd * 4 > mmap.len() {
                return vec![0.0f32; hd];
            }
            (0..hd)
                .map(|d| {
                    let b = file_offset + d * 4;
                    f32::from_le_bytes(mmap[b..b + 4].try_into().unwrap())
                })
                .collect()
        })
    }

    /// Read all K vectors for token `t` across all bands, layers, and heads.
    ///
    /// Returns a flat `Vec<f32>` of length `3 × n_layers × n_heads × head_dim`
    /// ordered as `(band, layer, head, dim)`.
    pub fn read_k_all(&self, entry: RawSigEntry, t: usize) -> crate::Result<Vec<f32>> {
        let h = &self.header;
        let nb = RAW_NUM_BANDS;
        let nl = h.n_layers_per_band as usize;
        let nh = h.n_kv_heads as usize;
        let hd = h.head_dim as usize;
        let total = nb * nl * nh * hd;
        let blob = self.read_entry_bytes(entry)?;
        if blob.is_empty() {
            return Ok(vec![0.0f32; total]);
        }
        let mut out = Vec::with_capacity(total);
        for band in 0..nb {
            for layer in 0..nl {
                for head in 0..nh {
                    let off = h.entry_offset(t, band, layer, head, false);
                    for d in 0..hd {
                        let b = off + d * 4;
                        out.push(f32::from_le_bytes(blob[b..b + 4].try_into().unwrap()));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Read all Q vectors for token `t` across all bands, layers, and heads.
    pub fn read_q_all(&self, entry: RawSigEntry, t: usize) -> crate::Result<Vec<f32>> {
        let h = &self.header;
        let nb = RAW_NUM_BANDS;
        let nl = h.n_layers_per_band as usize;
        let nh = h.n_kv_heads as usize;
        let hd = h.head_dim as usize;
        let total = nb * nl * nh * hd;
        let blob = self.read_entry_bytes(entry)?;
        if blob.is_empty() {
            return Ok(vec![0.0f32; total]);
        }
        let mut out = Vec::with_capacity(total);
        for band in 0..nb {
            for layer in 0..nl {
                for head in 0..nh {
                    let off = h.entry_offset(t, band, layer, head, true);
                    for d in 0..hd {
                        let b = off + d * 4;
                        out.push(f32::from_le_bytes(blob[b..b + 4].try_into().unwrap()));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Return a direct zero-copy slice of the cached mmap for `entry`'s raw
    /// data blob.  Returns `None` when no persistent mmap is cached (write
    /// mode) or when the entry would fall outside the mapped region.
    pub fn entry_slice(&self, entry: RawSigEntry) -> Option<&[u8]> {
        let mmap = self.mmap.as_ref()?;
        if entry.token_count == 0 {
            return Some(&[]);
        }
        let start = entry.byte_offset as usize;
        let len = entry.byte_len(&self.header);
        if start + len > mmap.len() {
            return None;
        }
        Some(&mmap[start..start + len])
    }

    fn with_mmap<F, R>(&self, f: F) -> crate::Result<R>
    where
        F: FnOnce(&[u8]) -> R,
    {
        // Fast path: use the persistent cached mmap (set by open()).
        if let Some(mmap) = &self.mmap {
            return Ok(f(&mmap[..]));
        }
        // Fallback for write mode (new / create): map transiently.
        let mmap = {
            let state = self.state.lock().unwrap();
            unsafe { Mmap::map(&state.file)? }
        };
        Ok(f(&mmap[..]))
    }
}

// ── Extraction helpers (called by the generator) ──────────────────────────────

/// De-interleave one Q vector from R16 format for token `t_in_block`, head `h`.
///
/// R16 q_flat layout: `[head][palette][CHUNK_SIZE tokens][sub_dim]`
/// where `N_PALETTE = 4`, `sub_dim = head_dim / N_PALETTE`.
///
/// Returns `head_dim` f32 values in natural dimension order.
pub fn extract_q_vector_r16(
    q_flat: &[f32],
    t_in_block: usize,
    head: usize,
    _n_kv_heads: usize,
    head_dim: usize,
    chunk_size: usize,
) -> Vec<f32> {
    const N_PALETTE: usize = 4;
    let sub_dim = head_dim / N_PALETTE;
    let per_subband = chunk_size * sub_dim;
    let per_head = N_PALETTE * per_subband;

    let mut q_vec = Vec::with_capacity(head_dim);
    let head_base = head * per_head;
    for p in 0..N_PALETTE {
        let pal_base = head_base + p * per_subband;
        let tok_base = pal_base + t_in_block * sub_dim;
        let end = (tok_base + sub_dim).min(q_flat.len());
        if tok_base < q_flat.len() {
            q_vec.extend_from_slice(&q_flat[tok_base..end]);
            // zero-fill if sub_dim was clipped
            if end - tok_base < sub_dim {
                q_vec.extend(std::iter::repeat_n(0.0f32, sub_dim - (end - tok_base)));
            }
        } else {
            q_vec.extend(std::iter::repeat_n(0.0f32, sub_dim));
        }
    }
    // ensure length
    q_vec.resize(head_dim, 0.0);
    q_vec
}

/// Extract K vector for token `t_in_block`, head `h` from k_flat.
///
/// k_flat layout: `[head][CHUNK_SIZE tokens][head_dim]` (natural order).
pub fn extract_k_vector(
    k_flat: &[f32],
    t_in_block: usize,
    head: usize,
    head_dim: usize,
    chunk_size: usize,
) -> Vec<f32> {
    let start = head * chunk_size * head_dim + t_in_block * head_dim;
    let end = start + head_dim;
    if end <= k_flat.len() {
        k_flat[start..end].to_vec()
    } else if start < k_flat.len() {
        let mut v = k_flat[start..].to_vec();
        v.resize(head_dim, 0.0);
        v
    } else {
        vec![0.0f32; head_dim]
    }
}

/// Build the per-token blob for `append` from raw block data.
///
/// `band_layer_data`: for each of 3 bands, a slice of `n_layers_per_band`
/// tuples `(k_flat, q_flat)` as returned by `dump_r16_kv_for_provenance`.
/// Each `k_flat`/`q_flat` corresponds to one 32-token block (blocks are
/// processed one at a time so the caller loops over blocks).
///
/// Returns a byte slice of length `actual_tokens × header.bytes_per_token()`
/// ready for [`RawProvenanceFile::append`].  If `band_layer_data` is ragged
/// (a band has fewer layers than `header.n_layers_per_band`), missing slots
/// are zero-filled.
pub fn build_token_blob(
    header: &RawFileHeader,
    actual_tokens: usize,
    band_layer_data: &[Vec<(&[f32], &[f32])>; 3], // [band][layer] -> (k_flat, q_flat)
) -> Vec<u8> {
    let bpt = header.bytes_per_token();
    let nl = header.n_layers_per_band as usize;
    let nh = header.n_kv_heads as usize;
    let hd = header.head_dim as usize;
    let chunk_size = candle_nn::CHUNK_SIZE;

    let mut blob = vec![0u8; actual_tokens * bpt];

    for t in 0..actual_tokens {
        let t_base = t * bpt;
        for (band, layer_data) in band_layer_data.iter().enumerate().take(3) {
            for layer in 0..nl {
                let (k_flat, q_flat) = layer_data.get(layer).copied().unwrap_or((&[], &[]));
                for head in 0..nh {
                    // K
                    let k_off = t_base + header.slot_offset(band, layer, head, false);
                    let k_vec = extract_k_vector(k_flat, t, head, hd, chunk_size);
                    for (d, &v) in k_vec.iter().enumerate() {
                        let b = k_off + d * 4;
                        if b + 4 <= blob.len() {
                            blob[b..b + 4].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                    // Q
                    let q_off = t_base + header.slot_offset(band, layer, head, true);
                    let q_vec = extract_q_vector_r16(q_flat, t, head, nh, hd, chunk_size);
                    for (d, &v) in q_vec.iter().enumerate() {
                        let b = q_off + d * 4;
                        if b + 4 <= blob.len() {
                            blob[b..b + 4].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                }
            }
        }
    }
    blob
}

// ── Band index computation ────────────────────────────────────────────────────

/// Compute the `n_layers_per_band` layer indices for a band centered at
/// `center` with half-width `half`, clamped to `[0, n_total_layers)`.
///
/// Returns exactly `n_layers_per_band` indices (some may be duplicated at
/// boundaries if the band would extend past the model's layer range).
pub fn band_layer_indices(
    center: usize,
    half: usize,
    n_total_layers: usize,
    n_layers_per_band: usize,
) -> Vec<usize> {
    if n_layers_per_band == 0 || n_total_layers == 0 {
        return Vec::new();
    }
    let start = center.saturating_sub(half);
    let end = (start + n_layers_per_band).min(n_total_layers);
    // If end was clamped, shift start back
    let start = end.saturating_sub(n_layers_per_band);
    (start..end)
        .chain(std::iter::repeat_n(
            end.saturating_sub(1),
            n_layers_per_band.saturating_sub(end - start),
        ))
        .take(n_layers_per_band)
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_header() -> RawFileHeader {
        RawFileHeader {
            n_kv_heads: 2,
            head_dim: 8,
            n_layers_per_band: 2,
            band_half_width: 1,
            band_centers: [2, 10, 18],
            n_total_layers: 20,
        }
    }

    #[test]
    fn header_roundtrip() {
        let h = test_header();
        let bytes = h.to_bytes();
        let h2 = RawFileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h, h2);
    }

    #[test]
    fn bytes_per_token_correct() {
        let h = test_header();
        // 3 × 2 × 2 × 2 × 8 × 4 = 768
        assert_eq!(h.bytes_per_token(), 3 * 2 * 2 * 2 * 8 * 4);
    }

    #[test]
    fn append_and_read_roundtrip() {
        let h = test_header();
        let pf = RawProvenanceFile::new(h).unwrap();

        let n_tokens = 3usize;
        let bpt = h.bytes_per_token();
        let mut data = vec![0u8; n_tokens * bpt];

        // Write a known pattern: token t, band b, layer l, head h, K[0] = t+1 as f32
        for t in 0..n_tokens {
            for band in 0..3 {
                for layer in 0..2usize {
                    for head in 0..2usize {
                        let k_off = t * bpt + h.slot_offset(band, layer, head, false);
                        let val = (t * 100 + band * 10 + layer * 4 + head + 1) as f32;
                        data[k_off..k_off + 4].copy_from_slice(&val.to_le_bytes());
                    }
                }
            }
        }

        let entry = pf.append(&data).unwrap();
        assert_eq!(entry.token_count, n_tokens as u16);

        // Read back and verify
        for t in 0..n_tokens {
            for band in 0..3 {
                for layer in 0..2usize {
                    for head in 0..2usize {
                        let vec = pf
                            .read_kq_vector(entry, t, band, layer, head, false)
                            .unwrap();
                        let expected = (t * 100 + band * 10 + layer * 4 + head + 1) as f32;
                        assert_eq!(vec[0], expected, "t={t} b={band} l={layer} h={head}");
                    }
                }
            }
        }
    }

    #[test]
    fn band_layer_indices_basic() {
        let indices = band_layer_indices(6, 2, 12, 5);
        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 4); // 6 - 2 = 4
        assert_eq!(indices[4], 8); // 4 + 4 = 8
    }

    #[test]
    fn band_layer_indices_clamped_at_start() {
        let indices = band_layer_indices(1, 4, 20, 5);
        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0); // clamped
    }

    #[test]
    fn extract_q_r16_basic() {
        let n_kv_heads = 1;
        let head_dim = 8;
        let chunk_size = 4;
        let sub_dim = head_dim / 4; // = 2
                                    // q_flat layout: [head][palette][token][sub_dim]
                                    // head=0, palette=0..4, token=0..4, sub_dim=0..2
        let n = n_kv_heads * 4 * chunk_size * sub_dim;
        let mut q_flat = vec![0.0f32; n];
        // For token t=0, palette p=0, sub_dim d=0: index = 0*4*4*2 + 0*4*2 + 0*2 + 0 = 0
        // set a unique value per (palette, sub_dim)
        for p in 0..4usize {
            for d in 0..sub_dim {
                // Token 0 — token offset is 0 * sub_dim, elided.
                let idx = p * chunk_size * sub_dim + d;
                q_flat[idx] = (p * sub_dim + d + 1) as f32;
            }
        }
        let v = extract_q_vector_r16(&q_flat, 0, 0, n_kv_heads, head_dim, chunk_size);
        assert_eq!(v.len(), head_dim);
        // palette 0 contributes dims 0,1 → 1.0, 2.0
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        // palette 1 contributes dims 2,3 → 3.0, 4.0
        assert_eq!(v[2], 3.0);
        assert_eq!(v[3], 4.0);
    }
}
