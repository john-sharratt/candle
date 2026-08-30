//! Fletcher-32 fingerprints of every resident expert weight, taken once after
//! load and re-checked only when something has already gone wrong.
//!
//! ## What this is for
//!
//! The expert grid on this configuration is **fully resident**: 10,496 slots
//! for 41 layers × 256 experts, `free_slots=0`, `warm_slots=0`, one permanent
//! slot per expert. Nothing evicts, nothing is re-uploaded, and the dispatch
//! tables hold a fixed address per expert for the life of the process. So an
//! expert's bytes are written once, during the startup fill, and should never
//! change again.
//!
//! They do. A captured MoE GEMM was replayed offline and cleared — deterministic
//! output, finite operand, in-range tile tables — and the fault resolved to one
//! expert weight reading `[-2.0e6, 2.0e6]` with 25,472 NaN out of 1,048,576,
//! against a load-time audit of all 33.0e9 weights that found them finite in
//! `[-0.77, 0.71]`. One slot of 256, the other 255 clean.
//!
//! A fingerprint answers the question that magnitude cannot: **did these bytes
//! change since load, or were they always like this?** Those point at opposite
//! causes — a write that lands on a resident weight, versus a fill that got it
//! wrong and an audit that missed it.
//!
//! ## Why a kernel
//!
//! The alternative is 23 GiB across PCIe. `run_fletcher32` reads the weights
//! where they already live and returns four bytes per slot, and it takes the
//! (pointer, length) plan model that the dispatch tables already are — so the
//! pointers the GEMM dereferences are literally the pointers that get
//! fingerprinted, with no separate accounting to drift.
//!
//! ## The three checks, and why there are three
//!
//! They differ in *when* they can speak, which is the only axis that matters
//! when the thing being hunted is a write with no other symptom.
//!
//! 1. [`SlotIntegrity::verify`] — the whole grid, ~21 GiB, tens of milliseconds.
//!    Says *which* weight changed, but only affordable once a wave has already
//!    failed. By then the write is long past.
//! 2. [`SlotIntegrity::scan_shard`] — one 1/512th of the grid, ~40 MiB, tens of
//!    microseconds. Called at every layer boundary with a rotating index, so the
//!    grid is covered end to end roughly every twelve waves at about a percent
//!    of the wave. **This is the one that can speak before the damage shows up
//!    in an activation**, which is what makes it the instrument rather than the
//!    audit. It re-baselines on report, so a drifted slot is named once.
//! 3. [`watch`] / [`check_watch`] — a single weight, ~688 KB. Once a scan has
//!    named a slot, this re-checks that one slot at every layer boundary, which
//!    turns "changed within the last twelve waves" into "was intact at layer
//!    N-1 and corrupt at layer N".
//!
//! Each narrows the window the one above it leaves: run → twelve waves → one
//! layer. The last is small enough to name a writer.
//!
//! ## Danger
//!
//! Every check here **fences the stream**. That is unavoidable — a fingerprint
//! of bytes a kernel is still writing means nothing — but it also means this
//! module perturbs exactly the timing of the race it exists to find. Keep the
//! per-layer costs where they are (shard ≈ 40 MiB, watch ≈ 688 KB): the measured
//! throughput with both armed is ~11.4k t/s against ~10k uninstrumented, and a
//! heavier check has already been observed to suppress the fault entirely for
//! over an hour. An instrument that stops the bug reproducing is not an
//! instrument.

use candle::cuda_backend::CudaDevice;
use candle::Result;
use candle_kernels::simple::fletcher32::run_fletcher32;
use cudarc::driver::{CudaSlice, DevicePtr};
use std::ops::Range;
use std::sync::Mutex;

/// Which projection family a slot belongs to, for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proj {
    Gate,
    Up,
    Down,
}

impl Proj {
    pub fn name(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

/// One slot whose bytes differ from what they were after the fill.
#[derive(Debug, Clone, Copy)]
pub struct Drifted {
    pub proj: Proj,
    /// Index into the projection's flat `[n_layers × n_experts]` table.
    pub index: usize,
    pub layer_row: usize,
    pub expert: usize,
    pub at_load: u32,
    pub now: u32,
    /// Where the weight actually lives, and how long it is.
    ///
    /// The address is the whole point of reporting one. A weight's *identity*
    /// (layer 39, expert 92) says nothing about who could have written it, but
    /// its *position in the span* says a great deal: a drifted slot sitting
    /// immediately above `weight_floor` is what an overrun from the tier below
    /// looks like, while one in the middle of the zone is not.
    pub ptr: u64,
    pub len: usize,
}

/// Fingerprints of the whole grid, taken once.
pub struct SlotIntegrity {
    /// Expected fingerprint per weight, indexed `[Gate, Up, Down]`.
    ///
    /// Behind a lock and mutable because [`SlotIntegrity::scan_shard`]
    /// **re-baselines a slot it has reported**. Without that, one drifted weight
    /// would be re-reported on every pass over its shard forever, burying the
    /// second distinct write — which is the event that distinguishes a one-shot
    /// writer from a recurring one — under repeats of the first.
    base: Mutex<[Vec<u32>; 3]>,
    n_experts: usize,
    /// Total bytes fingerprinted, so a report can say it looked at something.
    bytes: u64,
    /// Device-resident byte-length plans, one entry per weight.
    ///
    /// **Per weight, not per family.** A uniform length is wrong the moment a
    /// checkpoint varies dtype by layer — this one carries `Q5_KO` down-
    /// projections through layer 33 and `Q6_KO` from 34 — and reading a weight
    /// at another layer's length is how the first version of this faulted the
    /// driver outright. The lengths come from each tensor's own allocation, so
    /// there is nothing to re-derive and nothing to drift.
    gate_lens: CudaSlice<i64>,
    up_lens: CudaSlice<i64>,
    down_lens: CudaSlice<i64>,
    /// The pointer tables the fingerprints were taken through, copied.
    ///
    /// Owned rather than borrowed so the whole check is self-contained and can
    /// be registered as a process-global probe that any capture site reaches —
    /// the alternative is threading a handle to the expert cache through every
    /// checkpoint in the forward, including the ones nowhere near the MoE.
    /// Three tables of `n` `u64` is ~250 KiB, which is nothing against the
    /// 23 GiB they address.
    gate_ptrs: CudaSlice<u64>,
    up_ptrs: CudaSlice<u64>,
    down_ptrs: CudaSlice<u64>,
    /// The same plans on the host, indexed `[Gate, Up, Down]`.
    ///
    /// A drift report states a weight's address and extent without a readback,
    /// and `scan_shard` builds each shard's small device plan from these rather
    /// than sub-slicing the full tables.
    host_ptrs: [Vec<u64>; 3],
    host_lens: [Vec<i64>; 3],
}

/// How many shards the rotating scan splits the grid into.
///
/// Sized against the **fence**, not the bandwidth. One shard at this count is
/// ~330 MiB, well under a millisecond of read — but every fingerprint has to
/// synchronise the stream before it can trust the bytes it read, and a fence
/// drains the pipeline the race needs full to reproduce. So the scan runs once
/// per sweep rather than once per layer, and the shard is sized to cover the
/// grid in about a minute of running at production rates while costing one
/// fence per wave. Making the shards smaller and the scan more frequent trades
/// coverage latency for exactly the serialisation that hides the bug.
pub const SHARD_COUNT: usize = 64;

/// The half-open index range shard `shard` of `n` weights covers.
///
/// Split by proportion rather than by a fixed stride so the shards **partition**
/// `0..n` — every index in exactly one, no gaps and no overlaps — for any `n`,
/// including one not divisible by [`SHARD_COUNT`] and one smaller than it. A
/// gap here is a weight the sweep silently never looks at, which would show up
/// as "the scan found nothing" rather than as a bug in the scan.
pub fn shard_range(shard: usize, n: usize) -> Range<usize> {
    let s = shard % SHARD_COUNT;
    (s * n / SHARD_COUNT)..((s + 1) * n / SHARD_COUNT)
}

/// Fingerprint `n` weights addressed by `ptrs`, each `bytes_each` long.
fn fingerprint(
    dev: &CudaDevice,
    ptrs: &CudaSlice<u64>,
    lens: &CudaSlice<i64>,
    n: usize,
) -> Result<Vec<u32>> {
    let out = unsafe { dev.alloc::<u32>(n) }
        .map_err(|e| candle::Error::Msg(format!("slot_integrity: out alloc: {e}")))?;
    let stream = dev.cuda_stream();
    let (pp, _pg) = ptrs.device_ptr(&stream);
    let (lp, _lg) = lens.device_ptr(&stream);
    let (op, _og) = out.device_ptr(&stream);
    unsafe {
        candle::set_kernel_breadcrumb("run_fletcher32", file!(), line!());
        run_fletcher32(
            pp as *const i64,
            lp as *const i64,
            op as *mut u32,
            n as i32,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    stream
        .synchronize()
        .map_err(|e| candle::Error::Msg(format!("slot_integrity: fence: {e}")))?;
    dev.memcpy_dtov(&out)
        .map_err(|e| candle::Error::Msg(format!("slot_integrity: readback: {e}")))
}

impl SlotIntegrity {
    /// Take the grid's fingerprint. Call once, immediately after the fill, with
    /// the same pointer tables the GEMM will dereference.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        dev: &CudaDevice,
        gate_ptrs: &CudaSlice<u64>,
        up_ptrs: &CudaSlice<u64>,
        down_ptrs: &CudaSlice<u64>,
        gate_bytes: &[i64],
        up_bytes: &[i64],
        down_bytes: &[i64],
        n_experts: usize,
    ) -> Result<Self> {
        let n = gate_bytes.len();
        if up_bytes.len() != n || down_bytes.len() != n {
            candle::bail!(
                "slot_integrity: length plans disagree ({n} gate, {} up, {} down)",
                up_bytes.len(),
                down_bytes.len()
            );
        }
        let plan = |b: &[i64]| -> Result<CudaSlice<i64>> {
            dev.memcpy_stod(b)
                .map_err(|e| candle::Error::Msg(format!("slot_integrity: len plan: {e}")))
        };
        let gate_lens = plan(gate_bytes)?;
        let up_lens = plan(up_bytes)?;
        let down_lens = plan(down_bytes)?;
        let bytes = gate_bytes
            .iter()
            .chain(up_bytes)
            .chain(down_bytes)
            .map(|b| *b as u64)
            .sum();
        // Declare every weight immutable now that the fill is complete.
        //
        // The fingerprint says *whether* a weight changed; this says *who*
        // changed it, at the moment they try. They answer different halves of
        // the same question and neither replaces the other — a fingerprint
        // catches a write that has already happened, including one from a path
        // that never goes through the allocator, while this catches the pool
        // handing out a block it had already given away, before a byte lands.
        let host_ptrs = |s: &CudaSlice<u64>| -> Result<Vec<u64>> {
            dev.memcpy_dtov(s)
                .map_err(|e| candle::Error::Msg(format!("slot_integrity: ptr read: {e}")))
        };
        // Merged, not one per weight: 31,488 slots are carved out of a handful
        // of pool blocks and sit end to end, so declaring them individually
        // would overflow the table AND turn every allocation into a 31,488-entry
        // scan. Merged, they collapse to the spans actually distinct in the
        // address space.
        let mut spans: Vec<(u64, usize)> = Vec::with_capacity(3 * gate_bytes.len());
        for (ptrs, lens) in [
            (host_ptrs(gate_ptrs)?, gate_bytes),
            (host_ptrs(up_ptrs)?, up_bytes),
            (host_ptrs(down_ptrs)?, down_bytes),
        ] {
            spans.extend(ptrs.iter().zip(lens).map(|(&p, &l)| (p, l as usize)));
        }
        let merged = candle::readonly_regions::declare_merged("expert.weights", &mut spans);
        let (n_regions, region_bytes) = candle::readonly_regions::coverage();
        // The pool's idea of where the weights start, printed beside where they
        // actually are.
        //
        // `tier_fits` refuses to place the wave transient tier above
        // `weight_floor`, so a tier that lands on a weight means the two
        // disagree — and which way they disagree says whether the floor was
        // published too high or the weights reach lower than the zone's
        // capacity implies. Cheap to print once and it settles the question
        // without another instrumented run.
        if let Some(l) = candle_nn::kv_cache::span_layout(dev.cuda_stream().context().ordinal()) {
            let lo = spans.iter().map(|(b, _)| *b).min().unwrap_or(0);
            let hi = spans.iter().map(|(b, n)| b + *n as u64).max().unwrap_or(0);
            tracing::info!(
                weights_lo = format!("{lo:#x}"),
                weights_hi = format!("{hi:#x}"),
                weight_floor = format!("{:#x}", l.weight_floor),
                span_base = format!("{:#x}", l.span_base),
                span_end = format!("{:#x}", l.span_end),
                floor_vs_weights = (l.weight_floor as i128 - lo as i128) as i64,
                "readonly_regions: pool weight_floor vs the weights' real extent — a POSITIVE \
                 floor_vs_weights means the pool believes the weights start higher than they \
                 do, and the tier may legally be placed on top of them"
            );
        }
        tracing::info!(
            weights = spans.len(),
            merged,
            "readonly_regions: expert weight spans merged"
        );
        tracing::info!(
            regions = n_regions,
            bytes = region_bytes,
            "readonly_regions: expert weights declared immutable — an allocation or kernel \
             write that lands on one now panics naming the writer"
        );

        let keep = |s: &CudaSlice<u64>| -> Result<CudaSlice<u64>> {
            let host = dev
                .memcpy_dtov(s)
                .map_err(|e| candle::Error::Msg(format!("slot_integrity: ptr copy: {e}")))?;
            dev.memcpy_stod(&host)
                .map_err(|e| candle::Error::Msg(format!("slot_integrity: ptr keep: {e}")))
        };
        Ok(Self {
            base: Mutex::new([
                fingerprint(dev, gate_ptrs, &gate_lens, n)?,
                fingerprint(dev, up_ptrs, &up_lens, n)?,
                fingerprint(dev, down_ptrs, &down_lens, n)?,
            ]),
            n_experts,
            bytes,
            gate_lens,
            up_lens,
            down_lens,
            host_ptrs: [
                host_ptrs(gate_ptrs)?,
                host_ptrs(up_ptrs)?,
                host_ptrs(down_ptrs)?,
            ],
            host_lens: [gate_bytes.to_vec(), up_bytes.to_vec(), down_bytes.to_vec()],
            gate_ptrs: keep(gate_ptrs)?,
            up_ptrs: keep(up_ptrs)?,
            down_ptrs: keep(down_ptrs)?,
        })
    }

    /// How many weights were fingerprinted, for a report that has to be able to
    /// say it actually looked at something.
    pub fn covered(&self) -> usize {
        3 * self.host_lens[0].len()
    }

    /// How many bytes those weights span.
    pub fn bytes_covered(&self) -> u64 {
        self.bytes
    }

    /// Re-fingerprint and return every slot whose bytes have changed.
    ///
    /// Empty means the resident weights are byte-identical to what the fill
    /// left, which clears them and moves the search elsewhere. Non-empty names
    /// the slots that were written to after load — and since nothing in a
    /// fully-resident grid should ever write to one, each entry is a bug on its
    /// own.
    pub fn verify(&self, dev: &CudaDevice) -> Result<Vec<Drifted>> {
        let n = self.host_lens[0].len();
        let now = [
            fingerprint(dev, &self.gate_ptrs, &self.gate_lens, n)?,
            fingerprint(dev, &self.up_ptrs, &self.up_lens, n)?,
            fingerprint(dev, &self.down_ptrs, &self.down_lens, n)?,
        ];
        let base = self.base.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        for family in 0..3 {
            for i in 0..n {
                if base[family][i] != now[family][i] {
                    out.push(self.drifted(family, i, base[family][i], now[family][i]));
                }
            }
        }
        Ok(out)
    }

    /// Re-check one shard of the grid, re-baselining anything that drifted.
    ///
    /// The rotating counterpart to [`verify`](Self::verify): `shard` selects a
    /// contiguous `1/SHARD_COUNT` slice of each projection's table, so calling
    /// this once per layer boundary with an incrementing index sweeps the whole
    /// grid continuously for a fraction of a wave. That is what lets a resident
    /// weight's corruption be seen *while the run is still healthy*, instead of
    /// being reconstructed after an activation has already gone non-finite.
    ///
    /// Reported slots are re-baselined to their current bytes, so the next pass
    /// over this shard reports only a *further* change. A slot that appears
    /// twice therefore means a writer that keeps running, which is a materially
    /// different bug from one that fired once during startup.
    pub fn scan_shard(&self, dev: &CudaDevice, shard: usize) -> Result<Vec<Drifted>> {
        let n = self.host_lens[0].len();
        let Range { start: lo, end: hi } = shard_range(shard, n);
        if lo >= hi {
            return Ok(Vec::new());
        }
        // All three families in ONE plan, so the scan costs one fence rather
        // than three. The fence is the expensive part — see `SHARD_COUNT`.
        let width = hi - lo;
        let mut ptrs = Vec::with_capacity(3 * width);
        let mut lens = Vec::with_capacity(3 * width);
        for family in 0..3 {
            ptrs.extend_from_slice(&self.host_ptrs[family][lo..hi]);
            lens.extend_from_slice(&self.host_lens[family][lo..hi]);
        }
        let plan_p = dev
            .memcpy_stod(&ptrs)
            .map_err(|e| candle::Error::Msg(format!("scan_shard: ptr plan: {e}")))?;
        let plan_l = dev
            .memcpy_stod(&lens)
            .map_err(|e| candle::Error::Msg(format!("scan_shard: len plan: {e}")))?;
        let now = fingerprint(dev, &plan_p, &plan_l, 3 * width)?;

        let mut base = self.base.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        for (k, &b) in now.iter().enumerate() {
            let (family, i) = (k / width, lo + k % width);
            if base[family][i] == b {
                continue;
            }
            out.push(self.drifted(family, i, base[family][i], b));
            base[family][i] = b;
        }
        Ok(out)
    }

    /// Describe one drifted weight, including where it lives.
    fn drifted(&self, family: usize, i: usize, at_load: u32, now: u32) -> Drifted {
        Drifted {
            proj: [Proj::Gate, Proj::Up, Proj::Down][family],
            index: i,
            layer_row: i / self.n_experts,
            expert: i % self.n_experts,
            at_load,
            now,
            ptr: self.host_ptrs[family].get(i).copied().unwrap_or(0),
            len: self.host_lens[family].get(i).copied().unwrap_or(0) as usize,
        }
    }
}

/// The one slot a narrowed hunt is watching, and its fingerprint at load.
///
/// Once a drain has named a single drifted weight, the question stops being
/// *which* weight and becomes *when* it changes. Re-fingerprinting the whole
/// grid answers that at 23 GiB a look, which is far too slow to do more than
/// once; re-fingerprinting ONE weight is ~688 KB, cheap enough to do at every
/// layer boundary of every wave. That turns "it changed sometime this run" into
/// "it changed during layer N of this wave", which is a small enough window to
/// name the writer.
static WATCH: Mutex<Option<Watch>> = Mutex::new(None);

/// Fletcher-32 over one resident weight, planned from a raw device address.
///
/// The grid check plans over the cache's own pointer tables; the watch cannot,
/// because it deliberately holds no handle to the cache — it is a bare address
/// and a length, so it stays valid even while the writer that is being hunted
/// is rewriting the tables around it.
fn fingerprint_one(dev: &CudaDevice, ptr: u64, len: usize) -> Result<u32> {
    let plan_p = dev
        .memcpy_stod(&[ptr])
        .map_err(|e| candle::Error::Msg(format!("watch: ptr plan: {e}")))?;
    let plan_l = dev
        .memcpy_stod(&[len as i64])
        .map_err(|e| candle::Error::Msg(format!("watch: len plan: {e}")))?;
    Ok(fingerprint(dev, &plan_p, &plan_l, 1)?[0])
}

struct Watch {
    ptr: u64,
    len: usize,
    at_load: u32,
    proj: &'static str,
    layer_row: usize,
    expert: usize,
}

/// Whether a slot is being watched. One lock-free-ish check per call site; the
/// mutex is uncontended and only taken when the hunt has already narrowed.
pub fn watching() -> bool {
    WATCH.lock().map(|w| w.is_some()).unwrap_or(false)
}

/// A drifted weight's identity, address, and position in the live span.
///
/// The address is the half that names a writer. "Layer 39, expert 92" is an
/// identity and constrains nothing; `above_floor` and `tier_overlap` say whether
/// the bytes sit where an overrun out of the transient tier below would land, or
/// inside a tier that was placed on top of them — two different bugs with two
/// different fixes, told apart by two numbers.
pub fn describe(dev: &CudaDevice, d: &Drifted) -> String {
    let end = d.ptr + d.len as u64;
    let place = match candle_nn::kv_cache::span_layout(dev.cuda_stream().context().ordinal()) {
        Some(l) => {
            let tier = match (l.transient_base, l.transient_bytes) {
                (Some(b), n) if n > 0 => format!(
                    " tier=[{b:#x},{:#x}) tier_overlap={}",
                    b + n as u64,
                    d.ptr < b + n as u64 && end > b
                ),
                _ => " tier=none".to_string(),
            };
            format!(
                " above_floor={} floor={:#x}{tier}",
                d.ptr as i128 - l.weight_floor as i128,
                l.weight_floor
            )
        }
        None => String::new(),
    };
    format!(
        "{} layer_row={} expert={} at={:#x}..{end:#x} at_load={:#010x} now={:#010x}{place}",
        d.proj.name(),
        d.layer_row,
        d.expert,
        d.ptr,
        d.at_load,
        d.now
    )
}

/// Begin watching a drifted weight, if nothing is being watched yet.
///
/// First one wins deliberately. The scan reports in table order, not in the
/// order the writes happened, so there is no "best" slot to pick among several —
/// but there is a real cost to picking again, because re-arming resets the
/// baseline and discards a crossing that may already be halfway to being caught.
pub fn arm_watch(dev: &CudaDevice, d: &Drifted, n_experts: usize) {
    if watching() {
        return;
    }
    let at_load = match fingerprint_one(dev, d.ptr, d.len) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(error = %e, "could not watch the drifted slot");
            return;
        }
    };
    tracing::error!(
        target: "candle_transformers::slot_integrity",
        proj = d.proj.name(), layer_row = d.layer_row, expert = d.expert,
        ptr = format!("{:#x}", d.ptr), len = d.len,
        at_load = format!("{at_load:#010x}"), n_experts,
        "watching a single weight — every layer boundary now re-checks it"
    );
    *WATCH.lock().unwrap_or_else(|e| e.into_inner()) = Some(Watch {
        ptr: d.ptr,
        len: d.len,
        at_load,
        proj: d.proj.name(),
        layer_row: d.layer_row,
        expert: d.expert,
    });
}

/// Re-check the watched weight. Returns its description if it has changed.
///
/// Clears the watch on the first change, so the report is the *first* crossing
/// rather than one line per layer thereafter.
pub fn check_watch(dev: &CudaDevice, where_: &str) -> Option<String> {
    let mut guard = WATCH.lock().unwrap_or_else(|e| e.into_inner());
    let w = guard.as_ref()?;
    let now = fingerprint_one(dev, w.ptr, w.len).ok()?;
    if now == w.at_load {
        return None;
    }
    let msg = format!(
        "{} layer_row={} expert={} at {:#x} changed {:#010x} -> {:#010x}, first seen at {where_}",
        w.proj, w.layer_row, w.expert, w.ptr, w.at_load, now
    );
    *guard = None;
    Some(msg)
}

#[cfg(test)]
mod tests {
    use super::{shard_range, Proj, SHARD_COUNT};

    #[test]
    fn the_projection_families_name_themselves_distinctly() {
        let names: Vec<&str> = [Proj::Gate, Proj::Up, Proj::Down]
            .iter()
            .map(|p| p.name())
            .collect();
        assert_eq!(names, ["gate", "up", "down"]);
    }

    #[test]
    fn a_flat_index_splits_into_layer_row_and_expert() {
        // The tables are `[n_layers × n_experts]` row-major, and a report that
        // got this backwards would blame the wrong layer for every finding.
        let n_experts = 256usize;
        for (idx, want_row, want_expert) in [
            (0usize, 0usize, 0usize),
            (255, 0, 255),
            (256, 1, 0),
            (9728 + 17, 38, 17),
        ] {
            assert_eq!(idx / n_experts, want_row, "row of {idx}");
            assert_eq!(idx % n_experts, want_expert, "expert of {idx}");
        }
    }

    #[test]
    fn the_shards_partition_the_grid_with_no_gap_and_no_overlap() {
        // A gap is the failure that hides itself: the sweep would report
        // "nothing changed" for weights it never read. Check the real grid
        // (41 layers × 256 experts) and sizes that stress the rounding —
        // indivisible, smaller than SHARD_COUNT, and empty.
        for n in [10_496usize, 10_501, 1, 7, 0, SHARD_COUNT, SHARD_COUNT + 1] {
            let mut seen = vec![0u8; n];
            for s in 0..SHARD_COUNT {
                let r = shard_range(s, n);
                assert!(r.end <= n, "shard {s} of {n} reaches past the grid");
                for i in r {
                    seen[i] += 1;
                }
            }
            assert!(
                seen.iter().all(|&c| c == 1),
                "n={n}: every weight must be covered exactly once"
            );
        }
    }

    #[test]
    fn the_shard_index_rotates_rather_than_running_off_the_end() {
        // The caller passes a monotonically increasing counter for the life of
        // the process, so wrapping is the normal case, not an edge case.
        let n = 10_496;
        assert_eq!(shard_range(SHARD_COUNT, n), shard_range(0, n));
        assert_eq!(shard_range(3 * SHARD_COUNT + 5, n), shard_range(5, n));
    }
}
