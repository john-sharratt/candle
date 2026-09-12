//! Walk the KV a paged decode kernel dereferenced, for the rows it got wrong.
//!
//! The decode attention reads no tensor for its history. It follows pointers:
//! a per-row `SlotHeader` in the wave's header table names a `TokenSlice`
//! array, each slice names a `KvHead[n_kv_head]` record, and each record names
//! the bands the K and V live in. Every assert upstream of the kernel watches a
//! tensor — the new token's Q, K and V — and none of them sees this chain.
//!
//! So when the decode context comes out non-finite for a row whose queries were
//! finite, this reads the chain back from the device exactly as the kernel
//! walked it, and asks, in order of what each answer would mean:
//!
//! 1. **Does every band pointer agree with the host's block table?** A record
//!    naming a slot the host does not place this chunk in means the kernel read
//!    through a stale or half-uploaded record — the chain changed under it.
//! 2. **Is anything inside the read window allocation poison?** Every claimed
//!    slot is stamped `0xFF` under `tensor-assert`, so poison the kernel read
//!    is a value that was never written: a slot read before its bytes landed.
//! 3. **Is anything inside the read window otherwise non-finite?** Written, and
//!    written bad — the fault is upstream of the store.
//!
//! None of the three: the history was clean, and the fault is in the kernel's
//! arithmetic over it.

mod band;
mod device;
mod headers;
mod record;
mod rows;

pub(crate) use device::read_device;
pub(crate) use rows::nonfinite_rows;

pub(crate) use band::{band_payload_bytes, scan_band};
pub(crate) use headers::{parse_slices, parse_slot_header, RECORD_BYTES};
use record::{expected_ptr, head_record_bytes, parse_record};

use candle::cuda_backend::CudaDevice;
use candle::tensor_assert::Dump;
use candle::Result;
use candle_nn::kv_cache::{KvCache, N_PALETTE};

/// Tokens per chunk; a slice window reaching past it is a corrupt header.
const CHUNK_TOKENS: usize = 32;
/// Raw band dumps written per capture. Each is one band's payload (8 KiB at
/// head_dim 256) and a bad row can hold hundreds; the tallies cover them all.
const MAX_BAND_DUMPS: usize = 64;

/// What the walk found across the rows it followed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Tally {
    /// Slices whose own header is impossible: a window past the chunk, or no record.
    header_corrupt: usize,
    /// Places where the device chain and the host's block table disagree.
    ptr_mismatch: usize,
    /// Values inside a read window that are exactly the allocation poison.
    poison: usize,
    /// Values inside a read window that are NaN or inf (poison included).
    nonfinite: usize,
    /// Bands in a format this walk cannot decode.
    unscanned: usize,
    /// Values examined.
    elems: usize,
    /// Device reads that failed.
    read_errors: usize,
}

impl Tally {
    fn verdict(&self) -> &'static str {
        if self.header_corrupt > 0 {
            "HEADER_CORRUPT"
        } else if self.ptr_mismatch > 0 {
            "POINTER_MISMATCH"
        } else if self.poison > 0 {
            "POISON_IN_READ_WINDOW"
        } else if self.nonfinite > 0 {
            "NONFINITE_IN_READ_WINDOW"
        } else if self.read_errors > 0 {
            "INCOMPLETE"
        } else {
            "KV_CLEAN"
        }
    }

    fn add(&mut self, o: &Tally) {
        self.header_corrupt += o.header_corrupt;
        self.ptr_mismatch += o.ptr_mismatch;
        self.poison += o.poison;
        self.nonfinite += o.nonfinite;
        self.unscanned += o.unscanned;
        self.elems += o.elems;
        self.read_errors += o.read_errors;
    }
}

/// Walk `bad_rows` of the decode batch whose header table is `headers_ptr`,
/// writing what each row's chain holds into `d` and the overall answer as
/// `kvwalk.verdict`.
///
/// A row that cannot be walked is recorded as such and the rest still are: a
/// capture happens once, and losing every row to one unreadable record would
/// throw away the answer for the others.
///
/// # Safety
///
/// `headers_ptr` must be the header table the decode kernel just read — one
/// 16-byte `SlotHeader` per entry of `caches`, in the same order — and the
/// wave that launched the kernel must still be open, so every pointer behind
/// it names what the kernel read.
pub(crate) unsafe fn walk(
    d: &mut Dump,
    dev: &CudaDevice,
    headers_ptr: u64,
    caches: &[&mut KvCache],
    bad_rows: &[usize],
    n_kv_head: usize,
    head_dim: usize,
) -> Result<()> {
    if headers_ptr == 0 {
        d.note("kvwalk.verdict", "NO_HEADER_TABLE");
        return Ok(());
    }
    let mut total = Tally::default();
    let mut dumps = 0usize;
    for &r in bad_rows {
        // SAFETY: the caller's contract, row by row.
        let row = unsafe { walk_row(d, dev, headers_ptr, caches, r, n_kv_head, head_dim, &mut dumps) };
        match row {
            Ok(t) => {
                d.note(&format!("kvwalk.row{r}.tally"), format!("{t:?}"));
                d.note(&format!("kvwalk.row{r}.verdict"), t.verdict());
                total.add(&t);
            }
            Err(e) => {
                d.note(&format!("kvwalk.row{r}.error"), e);
                total.read_errors += 1;
            }
        }
    }
    d.note("kvwalk.rows", format!("{bad_rows:?}"));
    d.note("kvwalk.tally", format!("{total:?}"));
    d.note("kvwalk.verdict", total.verdict());
    tracing::error!(
        target: "candle_transformers::nan_capture",
        verdict = total.verdict(),
        rows = ?bad_rows,
        tally = ?total,
        "decode KV walk: the history the attention kernel read for its bad rows"
    );
    Ok(())
}

/// One row of [`walk`].
#[allow(clippy::too_many_arguments)]
unsafe fn walk_row(
    d: &mut Dump,
    dev: &CudaDevice,
    headers_ptr: u64,
    caches: &[&mut KvCache],
    r: usize,
    n_kv_head: usize,
    head_dim: usize,
    dumps: &mut usize,
) -> Result<Tally> {
    let key = |s: &str| format!("kvwalk.row{r}.{s}");
    let mut t = Tally::default();
    let cache = caches.get(r).ok_or_else(|| {
        candle::Error::Msg(format!("kv walk: row {r} is past the {} caches", caches.len()))
    })?;

    // SAFETY: the caller's contract — one header per cache.
    let hdr_bytes =
        unsafe { read_device(dev, headers_ptr + (r * RECORD_BYTES) as u64, RECORD_BYTES)? };
    d.bytes(&key("header"), &hdr_bytes)?;
    let hdr = parse_slot_header(&hdr_bytes)?;
    d.note(&key("header.n_slices"), hdr.n_slices);
    d.note(&key("header.write_slice"), hdr.write_slice);
    d.note(&key("header.slices_ptr"), format!("{:#x}", hdr.slices_ptr));

    // The host's view of the same row: which chunk each block is, and where the
    // arena table says those chunks live.
    let kc = cache.k_cache();
    let (Some(backing), Some(batch)) = (kc.chunked_backing(), kc.chunked_batch_idx()) else {
        candle::bail!("kv walk: row {r} is not a chunked cache");
    };
    let host_blocks = backing.slot_chunk_ids(batch)?;
    let info = kc
        .chunked_resolve_arena_info()
        .ok_or_else(|| candle::Error::Msg(format!("kv walk: row {r} has no arena table")))??;
    d.note(&key("host.batch_idx"), batch);
    d.note(&key("host.blocks"), host_blocks.len());
    if host_blocks.len() != hdr.n_slices as usize {
        t.ptr_mismatch += 1;
        d.note(
            &key("block_count"),
            format!("MISMATCH device {} host {}", hdr.n_slices, host_blocks.len()),
        );
    }
    if hdr.n_slices == 0 {
        return Ok(t);
    }

    // SAFETY: `slices_ptr` came from the header the kernel read, and names
    // `n_slices` 16-byte slices.
    let slice_bytes =
        unsafe { read_device(dev, hdr.slices_ptr, hdr.n_slices as usize * RECORD_BYTES)? };
    d.bytes(&key("slices"), &slice_bytes)?;
    let slices = parse_slices(&slice_bytes)?;

    // The kernel instantiates its record walk at N_PALETTE bands per head
    // (`get_head<HEAD_DIM>`), whatever the host serialised.
    let rec_len = head_record_bytes(head_dim, N_PALETTE) * n_kv_head;
    let sub = head_dim / N_PALETTE;
    for (s, sl) in slices.iter().enumerate() {
        let window = sl.offset as usize..sl.offset as usize + sl.len as usize;
        let slice_line = format!(
            "offset {} len {} rope {} kvheads {:#x}",
            sl.offset, sl.len, sl.rope, sl.kvheads_ptr
        );
        if window.end > CHUNK_TOKENS || sl.kvheads_ptr == 0 {
            t.header_corrupt += 1;
            d.note(&key(&format!("slice{s}")), format!("CORRUPT {slice_line}"));
            continue;
        }
        // SAFETY: `kvheads_ptr` is the record this slice points the kernel at.
        let rec = match unsafe { read_device(dev, sl.kvheads_ptr, rec_len) } {
            Ok(b) => b,
            Err(e) => {
                t.read_errors += 1;
                d.note(&key(&format!("slice{s}.record.error")), e);
                continue;
            }
        };
        let heads = parse_record(&rec, n_kv_head, head_dim, N_PALETTE)?;
        let host = host_blocks.get(s).map(|g| g.as_slice());
        let mut slice_dirty = false;
        for (h, head) in heads.iter().enumerate() {
            for p in 0..N_PALETTE {
                let sides = [
                    ("K", 0usize, head.k_ptr[p], head.k_fmt[p]),
                    ("V", 1usize, head.v_ptr[p], head.v_fmt[p]),
                ];
                for (side, is_v, dev_ptr, fmt) in sides {
                    let band = format!("slice{s}.h{h}.p{p}.{side}");
                    // The host serialises each head's gids at its own palette
                    // stride, K then V per band (`serialize_kv_heads`).
                    let want = host.and_then(|g| {
                        let np = g.len() / (n_kv_head * 2).max(1);
                        (p < np)
                            .then(|| g.get(h * np * 2 + p * 2 + is_v))
                            .flatten()
                            .map(|gid| expected_ptr(&info, gid.arena_idx(), gid.chunk_idx()))
                    });
                    if want != Some(dev_ptr) {
                        t.ptr_mismatch += 1;
                        slice_dirty = true;
                        let host_ptr = want.map_or_else(|| "none".to_string(), |w| format!("{w:#x}"));
                        d.note(
                            &key(&format!("{band}.ptr")),
                            format!("device {dev_ptr:#x} host {host_ptr}"),
                        );
                    }
                    // The kernel skips a null band, and so does the walk.
                    if dev_ptr == 0 {
                        continue;
                    }
                    let Some(len) = band_payload_bytes(fmt, sub) else {
                        t.unscanned += 1;
                        continue;
                    };
                    // SAFETY: `dev_ptr` is the band this record points the
                    // kernel at, `len` bytes of its format.
                    let bytes = match unsafe { read_device(dev, dev_ptr, len) } {
                        Ok(b) => b,
                        Err(e) => {
                            t.read_errors += 1;
                            d.note(&key(&format!("{band}.error")), e);
                            continue;
                        }
                    };
                    let scan = scan_band(fmt, &bytes, sub, window.clone())?;
                    t.elems += scan.elems;
                    t.nonfinite += scan.nonfinite;
                    t.poison += scan.poison;
                    if scan.nonfinite > 0 {
                        slice_dirty = true;
                        d.note(
                            &key(&format!("{band}.scan")),
                            format!(
                                "fmt {fmt} ptr {dev_ptr:#x} window {window:?} elems {} \
                                 nonfinite {} poison {}",
                                scan.elems, scan.nonfinite, scan.poison
                            ),
                        );
                        if *dumps < MAX_BAND_DUMPS {
                            d.bytes(&key(&band), &bytes)?;
                            *dumps += 1;
                        }
                    }
                }
            }
        }
        if slice_dirty {
            d.bytes(&key(&format!("slice{s}.record")), &rec)?;
            d.note(&key(&format!("slice{s}")), slice_line);
        }
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::Tally;

    #[test]
    fn the_verdict_names_the_most_damning_finding() {
        let clean = Tally { elems: 10, ..Tally::default() };
        assert_eq!(clean.verdict(), "KV_CLEAN");
        let nonfinite = Tally { nonfinite: 1, ..clean };
        assert_eq!(nonfinite.verdict(), "NONFINITE_IN_READ_WINDOW");
        let poison = Tally { poison: 1, ..nonfinite };
        assert_eq!(poison.verdict(), "POISON_IN_READ_WINDOW");
        let mismatch = Tally { ptr_mismatch: 1, ..poison };
        assert_eq!(mismatch.verdict(), "POINTER_MISMATCH");
        let corrupt = Tally { header_corrupt: 1, ..mismatch };
        assert_eq!(corrupt.verdict(), "HEADER_CORRUPT");
    }

    /// A walk that could not read everything must not claim the history clean.
    #[test]
    fn a_read_error_is_never_reported_clean() {
        let t = Tally { read_errors: 1, elems: 10, ..Tally::default() };
        assert_eq!(t.verdict(), "INCOMPLETE");
    }

    #[test]
    fn tallies_add_field_by_field() {
        let mut a = Tally { poison: 1, elems: 3, ..Tally::default() };
        a.add(&Tally { poison: 2, nonfinite: 4, elems: 5, read_errors: 1, ..Tally::default() });
        assert_eq!(
            a,
            Tally { poison: 3, nonfinite: 4, elems: 8, read_errors: 1, ..Tally::default() }
        );
    }
}
