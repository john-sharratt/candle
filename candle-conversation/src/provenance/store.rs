//! Mmap-backed flat file for provenance signatures.
//!
//! All conversations share a single `ProvenanceFile` (one fd).  Each sealed
//! 32-token KV block contributes one chunk-group entry: three contiguous
//! `TokenSignature` arrays — syntactic, semantic, pragmatic — written in order.
//! Per-`(group, turn)` `SigEntry` lists live on the workspace substrate;
//! each entry records the byte offset and token count for one chunk group
//! so the scanner can seek directly.

use std::fs::File;
use std::io::Write;
use std::sync::Mutex;

use memmap2::Mmap;
use tempfile::tempfile;

use super::signature::TokenSignature;

// ── Layout constants ──────────────────────────────────────────────────────────

/// Number of BDP signature depths captured per token: syntactic (~15%
/// transformer layer), semantic (~50%), pragmatic (~85%).
pub const NUM_DEPTHS: usize = 3;

/// On-disk indices for the three depth slices within a chunk-group entry.
pub const DEPTH_SYNTACTIC: usize = 0;
pub const DEPTH_SEMANTIC: usize = 1;
pub const DEPTH_PRAGMATIC: usize = 2;

/// Bytes one token contributes to a chunk-group entry: `NUM_DEPTHS`
/// signatures, each `TokenSignature::BYTE_LEN` bytes.
pub const ENTRY_BYTES_PER_TOKEN: usize = NUM_DEPTHS * TokenSignature::BYTE_LEN;

/// Byte range `[start, end)` of `depth`'s slice within an `n`-token entry.
#[inline]
pub fn depth_byte_range(depth: usize, n: usize) -> std::ops::Range<usize> {
    debug_assert!(depth < NUM_DEPTHS);
    let start = depth * n * TokenSignature::BYTE_LEN;
    let end = (depth + 1) * n * TokenSignature::BYTE_LEN;
    start..end
}

// ── ProbeSignatures ───────────────────────────────────────────────────────────

/// One representative `TokenSignature` per semantic depth.
///
/// Used as the query side of provenance scanning — each field is matched
/// against the corresponding depth's stored signatures in the `ProvenanceFile`.
/// Tested end-to-end via `provenance_tests`; not currently constructed from
/// production code (the in-decode reprojection path uses live per-block
/// signatures rather than folded probe signatures).
#[derive(Clone, Copy, Debug, Default)]
pub struct ProbeSignatures {
    /// Signature from the syntactic (~15%) layer.
    pub syntactic: TokenSignature,
    /// Signature from the semantic (~50%) layer.
    pub semantic: TokenSignature,
    /// Signature from the pragmatic (~85%) layer.
    pub pragmatic: TokenSignature,
}

// ── TurnChunkRank ─────────────────────────────────────────────────────────────

/// One ranked result from [`ProvenanceFile::scan_entries`].
#[derive(Clone, Debug)]
pub struct TurnChunkRank {
    /// Turn ID for the matching chunk group, packed as
    /// `(layer << 48) | (group << 32) | index`.
    pub turn_id: u64,
    /// Combined density score: sum of `hit² / token_count` across all depths.
    pub score: f64,
    /// Per-depth hit counts: `[syntactic, semantic, pragmatic]`.
    pub hit_counts: [usize; 3],
}

// ── SigEntry ──────────────────────────────────────────────────────────────────

/// Location and token count of one chunk-group entry in [`ProvenanceFile`].
///
/// Three depth slices are stored contiguously:
/// `[syntactic_0..syntactic_N][semantic_0..semantic_N][pragmatic_0..pragmatic_N]`
/// each `token_count × 16` bytes, so the total entry size is `token_count × 48`
/// bytes.
#[derive(Clone, Copy, Debug)]
pub struct SigEntry {
    /// Byte offset of this chunk group in the backing file.
    pub byte_offset: u64,
    /// Number of tokens in each depth slice.  All three slices are the same length.
    pub token_count: u16,
}

impl SigEntry {
    /// Total byte size of this chunk group: `token_count × ENTRY_BYTES_PER_TOKEN`.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.token_count as usize * ENTRY_BYTES_PER_TOKEN
    }
}

// ── ProvenanceFile ────────────────────────────────────────────────────────────

struct State {
    file: File,
    write_pos: u64,
}

/// Single mmap-backed file that stores all provenance signatures for all
/// conversations in a process.
///
/// One file descriptor total — shared via `Arc<ProvenanceFile>`.
///
/// # On-disk layout (per chunk group)
///
/// ```text
/// [syntactic_0..syntactic_N | semantic_0..semantic_N | pragmatic_0..pragmatic_N]
/// ```
///
/// Each `TokenSignature` is 16 bytes (128 sign bits).  `N = SigEntry::token_count`.
///
/// The file is append-only.  Entries are never removed or compacted.
pub struct ProvenanceFile {
    state: Mutex<State>,
}

impl ProvenanceFile {
    /// Create a new provenance file backed by an anonymous temporary file.
    ///
    /// The OS deletes the file when the last handle is closed (on Unix it is
    /// unlinked immediately; on Windows it is deleted on close).  Use this
    /// for single-process sessions where durability is not required.
    pub fn new() -> crate::Result<Self> {
        let file = tempfile()?;
        Ok(Self {
            state: Mutex::new(State { file, write_pos: 0 }),
        })
    }

    /// Open or create a persistent provenance file at `path`.
    ///
    /// Restores `write_pos` from the file length so reopening an existing file
    /// appends after the last valid entry without a directory scan.
    pub fn open(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        use std::fs::OpenOptions;
        let path = path.as_ref();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        let write_pos = file.metadata()?.len();
        Ok(Self {
            state: Mutex::new(State { file, write_pos }),
        })
    }

    /// Append one chunk-group triplet to the file.
    ///
    /// All three slices must have the same length; that length becomes
    /// `SigEntry::token_count`.  Returns a `SigEntry` recording the byte
    /// offset and token count for later scanning.
    ///
    /// # Panics
    ///
    /// Panics if the slices have different lengths or if `len > u16::MAX`.
    pub fn append(
        &self,
        syntactic: &[TokenSignature],
        semantic: &[TokenSignature],
        pragmatic: &[TokenSignature],
    ) -> crate::Result<SigEntry> {
        assert_eq!(syntactic.len(), semantic.len(), "depth slice length mismatch");
        assert_eq!(syntactic.len(), pragmatic.len(), "depth slice length mismatch");
        let token_count = syntactic.len();
        assert!(token_count <= u16::MAX as usize, "token_count exceeds u16::MAX");

        if token_count == 0 {
            let state = self.state.lock().unwrap();
            return Ok(SigEntry {
                byte_offset: state.write_pos,
                token_count: 0,
            });
        }

        let mut state = self.state.lock().unwrap();
        let byte_offset = state.write_pos;

        for sig in syntactic.iter().chain(semantic.iter()).chain(pragmatic.iter()) {
            state.file.write_all(sig.as_bytes())?;
        }
        state.file.flush()?;
        state.write_pos += (token_count * ENTRY_BYTES_PER_TOKEN) as u64;

        Ok(SigEntry {
            byte_offset,
            token_count: token_count as u16,
        })
    }

    /// Read the per-depth `TokenSignature` slices for `entry`.  Returns three
    /// vectors of equal length (`entry.token_count`), in syn/sem/prag order.
    /// Returns three empty vectors if the entry is empty or if the byte
    /// range is truncated/out of bounds.
    pub fn read_entry(
        &self,
        entry: SigEntry,
    ) -> crate::Result<(Vec<TokenSignature>, Vec<TokenSignature>, Vec<TokenSignature>)> {
        if entry.token_count == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let n = entry.token_count as usize;
        self.with_mmap(|mmap| {
            let offset = entry.byte_offset as usize;
            let total = n * ENTRY_BYTES_PER_TOKEN;
            if offset + total > mmap.len() {
                return (Vec::new(), Vec::new(), Vec::new());
            }
            let chunk = &mmap[offset..offset + total];
            let read_depth = |slice: &[u8]| -> Vec<TokenSignature> {
                slice
                    .chunks_exact(TokenSignature::BYTE_LEN)
                    .map(|c| {
                        let arr: [u8; TokenSignature::BYTE_LEN] = c.try_into().unwrap();
                        TokenSignature::from_bytes(&arr)
                    })
                    .collect()
            };
            (
                read_depth(&chunk[depth_byte_range(DEPTH_SYNTACTIC, n)]),
                read_depth(&chunk[depth_byte_range(DEPTH_SEMANTIC, n)]),
                read_depth(&chunk[depth_byte_range(DEPTH_PRAGMATIC, n)]),
            )
        })
    }

    /// Map the file read-only and hand the byte slice to `f`.  Releases the
    /// write lock immediately after mapping so concurrent appends are not
    /// blocked while `f` runs.  The mmap is dropped automatically when `f`
    /// returns.
    pub fn with_mmap<F, R>(&self, f: F) -> crate::Result<R>
    where
        F: FnOnce(&[u8]) -> R,
    {
        let mmap = {
            let state = self.state.lock().unwrap();
            // SAFETY: the file is valid; we're creating a read-only mapping.
            unsafe { Mmap::map(&state.file)? }
        };
        // MADV_SEQUENTIAL biases the OS prefetcher for our offset-sorted
        // scan walk.  Unix-only; on Windows the OS handles read-ahead heuristically.
        #[cfg(unix)]
        {
            let _ = mmap.advise(memmap2::Advice::Sequential);
        }
        Ok(f(&mmap[..]))
    }

    /// Scan `entries`, score each chunk group against `probe`, and return the
    /// top-`top_k` results sorted descending by score.
    ///
    /// Score per chunk group = Σ over depths of `hit² / token_count`.
    /// A "hit" is a token whose agreement with the probe exceeds `hit_threshold`
    /// (range 0–128; random baseline is 64).
    ///
    /// Creates one read-only `Mmap` for the scan and releases the write lock
    /// immediately after mapping so append calls are not blocked.
    pub fn scan_entries(
        &self,
        entries: &[(u64, SigEntry)],
        probe: &ProbeSignatures,
        hit_threshold: u32,
        top_k: usize,
    ) -> crate::Result<Vec<TurnChunkRank>> {
        if entries.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        // Map the file read-only and immediately release the write lock so
        // concurrent appends are not blocked by the scan.
        let mmap = {
            let state = self.state.lock().unwrap();
            // SAFETY: the file is valid; we're creating a read-only mapping.
            unsafe { Mmap::map(&state.file)? }
        };

        let mut ranks: Vec<TurnChunkRank> = Vec::new();

        for &(turn_id, entry) in entries {
            let offset = entry.byte_offset as usize;
            let n = entry.token_count as usize;
            if n == 0 {
                continue;
            }
            let total = n * ENTRY_BYTES_PER_TOKEN;
            if offset + total > mmap.len() {
                continue; // truncated or corrupt entry
            }

            let chunk = &mmap[offset..offset + total];
            let depth_slices: [(&[u8], &TokenSignature); NUM_DEPTHS] = [
                (&chunk[depth_byte_range(DEPTH_SYNTACTIC, n)], &probe.syntactic),
                (&chunk[depth_byte_range(DEPTH_SEMANTIC, n)],  &probe.semantic),
                (&chunk[depth_byte_range(DEPTH_PRAGMATIC, n)], &probe.pragmatic),
            ];

            let mut hit_counts = [0usize; 3];
            let mut score = 0.0f64;
            for (di, (bytes, probe_sig)) in depth_slices.iter().enumerate() {
                let hits = count_hits(bytes, probe_sig, hit_threshold);
                hit_counts[di] = hits;
                score += (hits * hits) as f64 / n as f64;
            }

            if score > 0.0 {
                ranks.push(TurnChunkRank { turn_id, score, hit_counts });
            }
        }

        ranks.sort_unstable_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranks.truncate(top_k);
        Ok(ranks)
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Count how many `TokenSignature::BYTE_LEN`-byte signatures in `data` have
/// agreement with `probe` at or above `threshold`.
fn count_hits(data: &[u8], probe: &TokenSignature, threshold: u32) -> usize {
    let probe_bytes = probe.as_bytes();
    data.chunks_exact(TokenSignature::BYTE_LEN)
        .filter(|chunk| {
            let agree: u32 = chunk
                .iter()
                .zip(probe_bytes.iter())
                .map(|(&a, &b)| (!(a ^ b)).count_ones())
                .sum();
            agree >= threshold
        })
        .count()
}
