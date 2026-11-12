//! Where the host reads a sequence's hit counters without waiting for them
//! (`docs/hit_driven_projection.md` §6.4, AGENTS hot-path invariant 3).
//!
//! The counters live on the device and are added to by every sampled layer's
//! selection. After a wave's last sampled layer one launch publishes each
//! counted sequence's counters into a block of mapped pinned memory, the way
//! MoE routing publishes its summary: the device writes the block through its
//! device address, and the projection reads it through its host address at the
//! next boundary. Nothing on either side waits on the other. A read sees the
//! newest publish the device has finished — one wave behind the host at most —
//! and a later read picks up what it missed, because the counts are cumulative.
//!
//! # The block
//!
//! ```text
//! word 0   begin  — the publish's sequence word, stored first
//! word 1   token  — which `SeqHits` the block belongs to
//! word 2   generation — which of its declarations
//! word 3   end    — the sequence word again, stored last
//! word 4   queries counted, cumulative
//! word 5.. hits per piece, cumulative
//! ```
//!
//! A read takes `end`, then the header and the counts, then `begin` again: a
//! publish under way has stored a newer `begin`, so a torn read is seen and
//! taken again. `token` and `generation` keep a block that changed hands — a
//! publish queued for its previous holder lands before any of the new
//! holder's, on the one stream both use — from being read as the new holder's.
//!
//! Blocks are never freed while the process runs: a block given back goes to
//! a pool and is handed to the next holder that fits, so the mapped memory is
//! the peak of sequences counted at once, and no free can race a publish still
//! queued against it.

use std::ffi::c_void;
use std::ptr::{null_mut, read_volatile, write_bytes};
use std::sync::atomic::{fence, AtomicU32, Ordering};
use std::sync::Mutex;

use candle::cuda_backend::cudarc::driver::sys;
use candle::{CudaDevice, Result};

/// Header words ahead of the counts (`HIT_SUMMARY_HEADER` in `qsa_hits.cu`).
pub const HEADER_WORDS: usize = 5;

/// The smallest block handed out, in words: room for a few hundred pieces.
const MIN_BLOCK_WORDS: usize = 512;

/// How often a read retries one torn by a publish landing under it. A publish
/// writes a few thousand words at most, so a second try almost always holds.
const TORN_RETRIES: usize = 8;

/// One block of mapped pinned memory.
#[derive(Debug)]
pub struct MappedBlock {
    host: *mut u32,
    dev: u64,
    words: usize,
    ordinal: usize,
}

// SAFETY: the block is written by the device and read by whichever thread
// holds the `SeqHits` it belongs to, under that holder's lock.
unsafe impl Send for MappedBlock {}

impl MappedBlock {
    /// The device address the publish writes through.
    pub fn dev_ptr(&self) -> u64 {
        self.dev
    }

    /// Words the block holds.
    pub fn words(&self) -> usize {
        self.words
    }

    fn alloc(dev: &CudaDevice, words: usize) -> Result<Self> {
        dev.bind_to_thread()?;
        let mut raw: *mut c_void = null_mut();
        // SAFETY: a page-locked, device-mapped allocation, kept in the pool for
        // the life of the process.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, words * 4, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("qsa hits: mapped summary allocation of {words} words failed: {r:?}");
        }
        let mut dev_ptr: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe {
                sys::cuMemFreeHost(raw);
            }
            candle::bail!("qsa hits: mapped summary has no device address: {r:?}");
        }
        // SAFETY: `words` u32s just allocated. A zero `end` is never a publish's
        // sequence word, so a fresh block reads as unpublished.
        unsafe { write_bytes(raw as *mut u32, 0, words) };
        Ok(Self {
            host: raw as *mut u32,
            dev: dev_ptr,
            words,
            ordinal: ordinal_of(dev),
        })
    }

    /// The newest finished publish for `token`'s declaration `generation` of
    /// `n_pieces` pieces: its sequence word, the queries and the hits per
    /// piece, all cumulative. `None` while nothing of that declaration has
    /// landed.
    pub fn read(&self, token: u32, generation: u32, n_pieces: usize) -> Option<Snapshot> {
        if HEADER_WORDS + n_pieces > self.words {
            return None;
        }
        for _ in 0..TORN_RETRIES {
            // SAFETY: every index below is inside the block (checked above);
            // the device writes it, so each load is volatile.
            let end = unsafe { read_volatile(self.host.add(3)) };
            if end == 0 {
                return None;
            }
            fence(Ordering::Acquire);
            let (begin, tok, gen) = unsafe {
                (
                    read_volatile(self.host),
                    read_volatile(self.host.add(1)),
                    read_volatile(self.host.add(2)),
                )
            };
            if tok != token || gen != generation {
                return None;
            }
            if begin != end {
                continue;
            }
            let queries = unsafe { read_volatile(self.host.add(4)) };
            let hits: Vec<u32> = (0..n_pieces)
                .map(|i| unsafe { read_volatile(self.host.add(HEADER_WORDS + i)) })
                .collect();
            fence(Ordering::Acquire);
            if unsafe { read_volatile(self.host) } != end {
                continue;
            }
            return Some(Snapshot {
                seq: end,
                queries,
                hits,
            });
        }
        None
    }
}

/// One finished publish, cumulative over its declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// The publish's sequence word.
    pub seq: u32,
    /// Query rows counted, summed over the sampled layers.
    pub queries: u32,
    /// Hits per piece.
    pub hits: Vec<u32>,
}

/// Blocks given back, waiting for their next holder.
static POOL: Mutex<Vec<MappedBlock>> = Mutex::new(Vec::new());

/// Tokens name holders; zero is never one, so a fresh block matches none.
static NEXT_TOKEN: AtomicU32 = AtomicU32::new(1);

/// The device a block is mapped into: its context's ordinal.
fn ordinal_of(dev: &CudaDevice) -> usize {
    dev.cuda_stream().context().ordinal()
}

/// A fresh holder token.
pub fn next_token() -> u32 {
    loop {
        let t = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
        if t != 0 {
            return t;
        }
    }
}

/// A block of at least `words` on `dev`: the smallest pooled one that fits,
/// or a new one rounded up to a power of two.
pub fn acquire(dev: &CudaDevice, words: usize) -> Result<MappedBlock> {
    let want = words.max(MIN_BLOCK_WORDS);
    {
        let mut pool = POOL
            .lock()
            .map_err(|_| candle::Error::Msg("qsa hits: summary pool lock poisoned".into()))?;
        let best = pool
            .iter()
            .enumerate()
            .filter(|(_, b)| b.ordinal == ordinal_of(dev) && b.words >= want)
            .min_by_key(|(_, b)| b.words)
            .map(|(i, _)| i);
        if let Some(i) = best {
            return Ok(pool.swap_remove(i));
        }
    }
    MappedBlock::alloc(dev, want.next_power_of_two())
}

/// Give `block` back for the next holder.
pub fn release(block: MappedBlock) {
    if let Ok(mut pool) = POOL.lock() {
        pool.push(block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(words: usize) -> (Vec<u32>, MappedBlock) {
        let mut mem = vec![0u32; words];
        let b = MappedBlock {
            host: mem.as_mut_ptr(),
            dev: 0,
            words,
            ordinal: 0,
        };
        (mem, b)
    }

    /// A block nothing has published to reads as nothing; a finished publish
    /// reads whole; another holder's, or another declaration's, reads as
    /// nothing; a publish under way (`begin` ahead of `end`) is not read.
    #[test]
    fn a_read_takes_only_a_finished_publish_of_its_own_declaration() {
        let (mut mem, b) = block(16);
        assert_eq!(b.read(7, 1, 3), None, "fresh");
        mem[..8].copy_from_slice(&[4, 7, 1, 4, 30, 5, 0, 9]);
        let snap = b.read(7, 1, 3);
        assert_eq!(
            snap,
            Some(Snapshot {
                seq: 4,
                queries: 30,
                hits: vec![5, 0, 9]
            })
        );
        assert_eq!(b.read(8, 1, 3), None, "another holder's");
        assert_eq!(b.read(7, 2, 3), None, "another declaration's");
        mem[0] = 5;
        assert_eq!(b.read(7, 1, 3), None, "a publish under way");
        assert_eq!(b.read(7, 1, 12), None, "more pieces than the block holds");
    }

    #[test]
    fn tokens_are_never_zero_and_never_repeat_in_a_row() {
        let a = next_token();
        let b = next_token();
        assert_ne!(a, 0);
        assert_ne!(b, 0);
        assert_ne!(a, b);
    }
}
