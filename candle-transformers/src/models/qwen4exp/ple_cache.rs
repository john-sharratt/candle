//! The PLE row cache: a bounded, **non-pinned** RAM cache of quantized table
//! rows in front of the NVMe-resident 320M-row n-gram table — §0.1's module.
//!
//! Rows are cached in their **on-disk quantized form** (Q8_0: five 34-byte
//! blocks per 160-wide row, 170 bytes), never dequantized here. That is the
//! §0.2-style capacity argument applied to the cache: 2 GB holds ~12.6M
//! quantized rows against ~1.6M dequantized ones, and the row is consumed by
//! an upload whose cheapest form *is* the quantized record — the engine
//! transfers the gathered records and dequantizes on the card
//! (`simple/ple_gather_dequant.cu`), so caching wider would be a pessimum on
//! both axes. Non-pinned deliberately: pinning 2 GB would compete with the
//! expert cache's warm tier for the same pool, and the per-token transfer is
//! ~2.7 KB — pageable is fine (§0.1).
//!
//! Structure: a slot arena of fixed-stride records plus a row-id → slot map,
//! with CLOCK (second-chance) eviction — an LRU approximation whose metadata
//! is one byte per slot, because at ~12.6M entries a linked LRU's pointers
//! would cost more than the records. The counters exist to answer §8 item 4
//! (hit rate against real traffic) — they report, they do not steer.

use std::sync::Mutex;

use candle::Result;

/// Hit/miss/eviction counters — the §8-item-4 instrument.
#[derive(Debug, Clone, Copy, Default)]
pub struct PleCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl PleCacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Where table rows come from on a miss: a positioned read of the row's
/// quantized record. The disk implementation wraps the GGUF slab; tests use
/// an in-memory table.
pub trait PleRowFetch: Send + Sync {
    /// Read row `id`'s record into `dst` (exactly `record_bytes` long).
    fn fetch(&self, id: u32, dst: &mut [u8]) -> Result<()>;
}

struct CacheState {
    /// Slot arena, `slots × stride` bytes.
    arena: Vec<u8>,
    /// row id → slot index.
    map: ahash::HashMap<u32, u32>,
    /// slot index → row id (for eviction's map removal). `u32::MAX` = empty.
    owner: Vec<u32>,
    /// CLOCK reference bits.
    referenced: Vec<u8>,
    /// CLOCK hand.
    hand: usize,
    /// Next never-used slot (fill before the clock starts evicting).
    frontier: usize,
    stats: PleCacheStats,
}

/// The bounded row cache. Interior-mutable behind one mutex: gathers are
/// 16 rows per token on the session thread, microseconds against the forward
/// they precede, and a sharded map would buy contention headroom no measured
/// caller has asked for.
pub struct PleRowCache<F: PleRowFetch> {
    fetch: F,
    record_bytes: usize,
    state: Mutex<CacheState>,
}

impl<F: PleRowFetch> PleRowCache<F> {
    /// A cache of at most `capacity_bytes` of row records (arena bytes; the
    /// map/metadata overhead is ~13 bytes/slot on top and is deliberately not
    /// counted against the budget — the budget names the figure §0.1 states).
    pub fn new(fetch: F, record_bytes: usize, capacity_bytes: usize) -> Result<Self> {
        if record_bytes == 0 {
            candle::bail!("ple cache: zero record size");
        }
        let slots = capacity_bytes / record_bytes;
        if slots == 0 {
            candle::bail!(
                "ple cache: capacity {capacity_bytes} below one {record_bytes}-byte record"
            );
        }
        Ok(Self {
            fetch,
            record_bytes,
            state: Mutex::new(CacheState {
                arena: vec![0u8; slots * record_bytes],
                map: ahash::HashMap::default(),
                owner: vec![u32::MAX; slots],
                referenced: vec![0u8; slots],
                hand: 0,
                frontier: 0,
                stats: PleCacheStats::default(),
            }),
        })
    }

    /// Gather `ids`' records into one contiguous buffer (`ids.len() ×
    /// record_bytes`), reading through the cache. This is the buffer the
    /// engine uploads verbatim; the oracle dequantizes it host-side.
    pub fn gather(&self, ids: &[u32], out: &mut Vec<u8>) -> Result<()> {
        out.clear();
        out.reserve(ids.len() * self.record_bytes);
        let mut st = self.state.lock().expect("ple cache poisoned");
        let st = &mut *st;
        for &id in ids {
            let slot = match st.map.get(&id) {
                Some(&s) => {
                    st.stats.hits += 1;
                    st.referenced[s as usize] = 1;
                    s as usize
                }
                None => {
                    st.stats.misses += 1;
                    let slot = if st.frontier < st.owner.len() {
                        let s = st.frontier;
                        st.frontier += 1;
                        s
                    } else {
                        // CLOCK: sweep until an unreferenced slot turns up,
                        // clearing reference bits as the hand passes.
                        loop {
                            let s = st.hand;
                            st.hand = (st.hand + 1) % st.owner.len();
                            if st.referenced[s] == 0 {
                                break s;
                            }
                            st.referenced[s] = 0;
                        }
                    };
                    if st.owner[slot] != u32::MAX {
                        st.map.remove(&st.owner[slot]);
                        st.stats.evictions += 1;
                    }
                    let dst =
                        &mut st.arena[slot * self.record_bytes..(slot + 1) * self.record_bytes];
                    self.fetch.fetch(id, dst)?;
                    st.owner[slot] = id;
                    st.referenced[slot] = 1;
                    st.map.insert(id, slot as u32);
                    slot
                }
            };
            out.extend_from_slice(
                &st.arena[slot * self.record_bytes..(slot + 1) * self.record_bytes],
            );
        }
        Ok(())
    }

    pub fn stats(&self) -> PleCacheStats {
        self.state.lock().expect("ple cache poisoned").stats
    }

    pub fn record_bytes(&self) -> usize {
        self.record_bytes
    }
}

/// Upload gathered Q8_0 records and dequantize them **on the card**
/// (`simple/ple_gather_dequant.cu`), returning the widened rows to the host.
///
/// The transfer half of §0.1 as the engine runs it: ~2.7 KB per token crosses
/// PCIe in quantized form and widens device-side. This helper returns the
/// result to the host so the parity test can pin the kernel against the CPU
/// Q8_0 dequant bit-for-bit; the engine's forward keeps the device buffer.
pub fn gpu_dequant_rows_to_host(
    device: &candle::Device,
    records: &[u8],
    n_rows: usize,
) -> Result<Vec<f32>> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::WrapErr;
    use candle_kernels::simple::ple_gather_dequant::run_ple_dequant_q8;

    const ROW_BYTES: usize = 170;
    const ROW_ELEMS: usize = 160;
    let candle::Device::Cuda(dev) = device else {
        candle::bail!("ple gpu dequant: CUDA device required");
    };
    if records.len() != n_rows * ROW_BYTES {
        candle::bail!(
            "ple gpu dequant: {} bytes for {n_rows} rows of {ROW_BYTES}",
            records.len()
        );
    }
    let stream = dev.cuda_stream();
    let rec_gpu = stream.memcpy_stod(records).w()?;
    // Fully overwritten by the kernel — allocate uninitialised.
    let out_gpu = unsafe { dev.alloc::<f32>(n_rows * ROW_ELEMS)? };
    {
        let (rp, _gr) = rec_gpu.device_ptr(&stream);
        let (op, _go) = out_gpu.device_ptr(&stream);
        unsafe {
            run_ple_dequant_q8(
                rp as *const u8,
                op as *mut f32,
                n_rows as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    let mut out = vec![0f32; n_rows * ROW_ELEMS];
    stream.memcpy_dtoh(&out_gpu, &mut out[..]).w()?;
    stream.synchronize().w()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A synthetic table whose row `id`'s record is a recognisable pattern,
    /// counting fetches so tests can see the disk traffic the cache saved.
    struct MemTable {
        fetches: AtomicU64,
    }

    impl PleRowFetch for MemTable {
        fn fetch(&self, id: u32, dst: &mut [u8]) -> Result<()> {
            self.fetches.fetch_add(1, Ordering::Relaxed);
            for (i, b) in dst.iter_mut().enumerate() {
                *b = (id as usize + i) as u8;
            }
            Ok(())
        }
    }

    fn cache(slots: usize) -> PleRowCache<MemTable> {
        let rb = 16usize;
        PleRowCache::new(
            MemTable {
                fetches: AtomicU64::new(0),
            },
            rb,
            slots * rb,
        )
        .unwrap()
    }

    fn record(id: u32, rb: usize) -> Vec<u8> {
        (0..rb).map(|i| (id as usize + i) as u8).collect()
    }

    #[test]
    fn gather_returns_the_source_bytes_and_caches_repeats() {
        let c = cache(8);
        let mut out = Vec::new();
        c.gather(&[3, 5, 3, 9, 5], &mut out).unwrap();
        let rb = c.record_bytes();
        for (i, &id) in [3u32, 5, 3, 9, 5].iter().enumerate() {
            assert_eq!(&out[i * rb..(i + 1) * rb], record(id, rb), "row {id}");
        }
        let s = c.stats();
        assert_eq!(s.misses, 3, "three distinct rows");
        assert_eq!(s.hits, 2, "two repeats served from cache");
        assert_eq!(c.fetch.fetches.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn capacity_is_respected_and_eviction_recycles_slots() {
        let c = cache(4);
        let mut out = Vec::new();
        // Fill past capacity: 8 distinct rows through 4 slots.
        c.gather(&[0, 1, 2, 3, 4, 5, 6, 7], &mut out).unwrap();
        let s = c.stats();
        assert_eq!(s.misses, 8);
        assert_eq!(s.evictions, 4, "four slots recycled");
        // The arena never grew.
        assert_eq!(
            c.state.lock().unwrap().arena.len(),
            4 * c.record_bytes(),
            "cache grew past its budget"
        );
        // Whatever is resident still reads back correctly.
        c.gather(&[7], &mut out).unwrap();
        assert_eq!(&out[..c.record_bytes()], record(7, c.record_bytes()));
    }

    /// The on-card dequant must reproduce the CPU Q8_0 dequant bit-for-bit
    /// over real quantized records — the parity that lets the engine transfer
    /// rows quantized without a numerics question. Needs a GPU; the suite
    /// runs under `--features cuda` on machines that have one.
    #[test]
    fn gpu_dequant_matches_cpu_dequant_bit_for_bit() {
        use candle::quantized::{GgmlDType, QTensor};
        use candle::{Device, Tensor};

        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("ple gpu parity: no CUDA device visible; nothing proven here");
            return;
        };
        let n_rows = 64usize;
        let cpu = Device::Cpu;
        let mut lcg = 99u64;
        let vals: Vec<f32> = (0..n_rows * 160)
            .map(|_| {
                lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((lcg >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        let t = Tensor::from_vec(vals, (n_rows, 160), &cpu).unwrap();
        let q = QTensor::quantize(&t, GgmlDType::Q8_0).unwrap();
        let records = q.data().unwrap().to_vec();
        assert_eq!(records.len(), n_rows * 170);

        let want: Vec<f32> = q
            .dequantize(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got = gpu_dequant_rows_to_host(&device, &records, n_rows).unwrap();
        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(g == w, "elem {i}: gpu {g} != cpu {w}");
        }
    }

    #[test]
    fn clock_keeps_a_row_touched_since_the_hand_last_passed() {
        // Second-chance semantics: a reference bit protects a row from the
        // sweep that follows it — not from every future sweep. So establish a
        // hand position first (one eviction clears the bits behind it), THEN
        // touch a row, THEN evict: the untouched row must go first.
        let c = cache(4);
        let mut out = Vec::new();
        c.gather(&[1, 2, 3, 4], &mut out).unwrap();
        // First eviction: all bits set → the hand clears them all, wraps, and
        // recycles slot 0 (row 1). Hand now sits past slot 0; bits are clear.
        c.gather(&[5], &mut out).unwrap();
        // Touch row 3: its bit is set again.
        c.gather(&[3], &mut out).unwrap();
        // Next eviction must take an untouched row (row 2, at the hand), not
        // the freshly touched row 3.
        c.gather(&[6], &mut out).unwrap();
        let before = c.stats();
        c.gather(&[3], &mut out).unwrap();
        let after = c.stats();
        assert_eq!(
            after.hits,
            before.hits + 1,
            "the touched row was evicted ahead of untouched ones"
        );
    }
}
