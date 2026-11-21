//! Builds, in one launch, every device table the grouped expert pipeline reads
//! from a router's expert indices.
//!
//! The `[n_tokens, k]` indices become expert-grouped gather lists, grouped-GEMM
//! tile tables and the deterministic scatter's segment tables, with no readback
//! to the host, so the pipeline issues without waiting on the router. A workspace
//! holds the buffers across layers: it is sized for the largest forward, grown
//! when a larger one arrives and never shrunk, so each MoE layer costs one launch
//! and no allocation. When a live expert table is given, each routed expert is
//! classified as VRAM, pinned or cold on the device, and the layer's routing
//! summary is written to mapped memory for the host threads to read. The pipeline
//! launches at the `n_tokens × k` bound, so the kernels read the header's counts
//! and the host never has to.

use super::*;

/// Reusable device workspace for [`moe_bucketize`]. All output/scratch buffers
/// live in VRAM and are sized to a maximum `n_tokens × k`; one workspace is
/// allocated per forward pipeline and reused across every MoE layer, so the
/// per-layer bucketize costs a single kernel launch — no allocations, no
/// host↔device traffic.
pub struct MoeBucketizeWorkspace {
    pub(super) cap_assign: usize,
    pub(super) cap_starts: usize,
    /// Expert-grouped token ids (gather rows); padding rows are `!0`.
    pub tok_ids: CudaSlice<u32>,
    /// Expert-grouped `widx` into the flattened routing weights; padding `!0`.
    pub weight_ids: CudaSlice<u32>,
    /// RAW owning expert id per grouped-GEMM tile; padding tiles are expert 0.
    pub tile_expert: CudaSlice<i32>,
    /// Stacked-batch start row per tile.
    pub tile_b_start: CudaSlice<i32>,
    /// Tokens per tile; 0 marks a padding tile the grouped kernel skips.
    pub tile_b_cnt: CudaSlice<i32>,
    /// Token-major valid assignment → its expert-grouped row (scatter `perm`).
    pub perm: CudaSlice<u32>,
    /// Token-major valid assignment → its `widx` (scatter `reordered_weight_ids`).
    pub rw_ids: CudaSlice<u32>,
    /// Per-token scatter segment boundaries, `[n_tokens + 1]`.
    pub token_starts: CudaSlice<i32>,
    /// Device header `{n_active, total_valid, num_tiles, remote experts,
    /// remote tiles, CPU rows, CPU experts, _}`. The live grouped GEMM's blocks
    /// read the remote words, and the CPU export and the down launch's CPU item
    /// the CPU rows; the host never reads it — the pipeline launches at the
    /// `n_tokens × k` upper bound.
    pub header: CudaSlice<i32>,
    pub(super) inv: CudaSlice<u32>,
    pub(super) scan: CudaSlice<i32>,
}

impl MoeBucketizeWorkspace {
    /// Allocate for up to `max_tokens × k` routing assignments.
    pub fn new(device: &CudaDevice, max_tokens: usize, k: usize) -> Result<Self> {
        let cap_assign = max_tokens.max(1) * k.max(1);
        let cap_starts = max_tokens.max(1) + 1;
        Ok(Self {
            cap_assign,
            cap_starts,
            tok_ids: unsafe { device.alloc::<u32>(cap_assign)? },
            weight_ids: unsafe { device.alloc::<u32>(cap_assign)? },
            tile_expert: unsafe { device.alloc::<i32>(cap_assign)? },
            tile_b_start: unsafe { device.alloc::<i32>(cap_assign)? },
            tile_b_cnt: unsafe { device.alloc::<i32>(cap_assign)? },
            perm: unsafe { device.alloc::<u32>(cap_assign)? },
            rw_ids: unsafe { device.alloc::<u32>(cap_assign)? },
            token_starts: unsafe { device.alloc::<i32>(cap_starts)? },
            header: unsafe {
                device.alloc::<i32>(candle_kernels::simple::moe_bucketize::HEADER_WORDS)?
            },
            inv: unsafe { device.alloc::<u32>(cap_assign)? },
            scan: unsafe { device.alloc::<i32>(cap_assign)? },
        })
    }

    /// Grow (never shrink) to cover `n_tokens × k`.
    pub(super) fn ensure(&mut self, device: &CudaDevice, n_tokens: usize, k: usize) -> Result<()> {
        if n_tokens * k > self.cap_assign || n_tokens + 1 > self.cap_starts {
            // Eager for the whole swap: the old tables are freed here, behind
            // every launch that read them — including any this thread has
            // recorded and not yet issued.
            let _eager = device.pause_capture()?;
            *self = Self::new(device, n_tokens, k)?;
        }
        Ok(())
    }
}

/// GPU-native expert bucketize: turn the router's `[n_tokens, k]` u32 index
/// tensor into every device table the grouped expert pipeline consumes —
/// expert-grouped gather lists, grouped-GEMM tile tables (padded to the
/// `n_tokens × k` launch bound with `b_cnt = 0`), and the deterministic
/// scatter's token-major segment tables — in ONE launch on the compute stream,
/// with no GPU→CPU readback. The row grouping is stable in (token, slot) order
/// and ascending by expert id; an index `≥ n_experts` is the router's
/// empty-slot sentinel and is skipped. See
/// `candle-kernels/src/simple/moe_bucketize.cu` for the padding contract.
///
/// - `live` — a layer over a live expert table ([`BucketizeLive`]): its routed
///   experts are classified VRAM / pinned / cold from their entries, which are
///   snapshotted into the layer's GEMM weight tables, the remote (pinned, then
///   cold) experts' tiles come first and are listed, and the routing summary is
///   written. `None` treats every expert as in VRAM (plain
///   ascending tile order, no summary, no list).
/// - `decode` — the tokens scored as decode rows ([`DecodeRows`]).
pub fn moe_bucketize(
    indices: &LiveTensor<'_>,
    n_experts: usize,
    tile_w: usize,
    ws: &mut MoeBucketizeWorkspace,
    live: Option<&BucketizeLive>,
    decode: &DecodeRows,
) -> Result<()> {
    use crate::cuda_backend::CudaStorageSlice;

    use candle_kernels::simple::moe_bucketize::{
        BUCKETIZE_EARLIER_FAILURE, BUCKETIZE_LAUNCHED, BUCKETIZE_REFUSED, MAX_EXPERTS, MAX_TOPK,
    };
    let (decode_lo, decode_hi) = decode.ranges();

    let (n_tokens, k) = indices.dims2()?;
    // These bounds mirror the launcher's own guards (via the shared constants),
    // so an invalid call errors HERE instead of the launcher silently skipping
    // the launch and leaving the workspace holding the previous layer's tables.
    if n_tokens == 0 || k == 0 {
        crate::bail!("moe_bucketize: empty indices [{n_tokens}, {k}]");
    }
    if n_experts == 0 || n_experts > MAX_EXPERTS {
        crate::bail!("moe_bucketize: n_experts={n_experts} must be in 1..={MAX_EXPERTS}");
    }
    if k > MAX_TOPK {
        crate::bail!("moe_bucketize: k={k} exceeds {MAX_TOPK} (kernel per-token sort bound)");
    }
    if tile_w == 0 {
        crate::bail!("moe_bucketize: tile_w must be > 0");
    }
    let device = match indices.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("moe_bucketize: expected a CUDA tensor"),
    };
    if indices.dtype() != crate::DType::U32 {
        crate::bail!(
            "moe_bucketize: indices must be u32, got {:?}",
            indices.dtype()
        );
    }
    if live.is_some_and(|l| l.gate_row != 0 && l.snap == 0) {
        crate::bail!("moe_bucketize: a live layer needs a snapshot buffer");
    }
    let promo = live.and_then(|l| l.promo);
    if live.is_some_and(|l| {
        l.promo.is_some_and(|p| {
            p.cap == 0 || p.sweep == 0 || p.victims == 0 || p.retarget == 0 || l.gate_row == 0
        }) || (l.promo.is_some() && l.remote_dst == 0)
    }) {
        crate::bail!(
            "moe_bucketize: a promotion ring needs a capacity, its sweep word, its victim \
             arrays, a live table and a remote_dst list"
        );
    }
    let owner = live.map_or(OwnerCheck::default(), |l| l.owner);
    if owner.owners != 0 && (owner.slot_bytes == 0 || live.is_none_or(|l| l.gate_row == 0)) {
        crate::bail!("moe_bucketize: the owner check needs a slot size and a live table");
    }
    let cpu = live.and_then(|l| l.cpu);
    if cpu.is_some_and(|c| {
        c.split.cap_rows != 0 && (c.dst_row == 0 || live.is_none_or(|l| l.gate_row == 0))
    }) {
        crate::bail!("moe_bucketize: the CPU class needs a ring-row map and a live table");
    }
    let ahead = live.and_then(|l| l.ahead);
    if let (Some(a), Some(l)) = (ahead, live) {
        let ring_ok = l.promo.is_some_and(|p| {
            p.window != 0
                && p.depth != 0
                && p.ahead_n != 0
                && p.ahead_list != 0
                && p.ahead_src != 0
                && p.ahead_cap != 0
        });
        if !ring_ok
            || l.remote == 0
            || a.row_layout == 0
            || a.items == 0
            || a.done == 0
            || a.rows <= 0
            || l.row >= a.rows
        {
            crate::bail!(
                "moe_bucketize: read-ahead needs the promotion ring with its read-ahead words, \
                 a remote list, the row layout, the item buffers and a row inside the table"
            );
        }
    }
    ws.ensure(&device, n_tokens, k)?;

    let (storage, layout) = indices.storage_and_layout();
    let (o1, o2) = layout.contiguous_offsets().ok_or_else(|| {
        crate::Error::RequiresContiguous {
            op: "moe_bucketize",
        }
        .bt()
    })?;
    let cuda = match &*storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("moe_bucketize: expected CUDA storage"),
    };
    let ids = match &cuda.slice {
        CudaStorageSlice::U32(s) => s.slice(o1..o2),
        _ => crate::bail!("moe_bucketize: expected u32 storage"),
    };

    let stream = device.cuda_stream();
    let (idp, _g0) = ids.device_ptr(&stream);
    let (tok, _g1) = ws.tok_ids.device_ptr(&stream);
    let (wid, _g2) = ws.weight_ids.device_ptr(&stream);
    let (te, _g3) = ws.tile_expert.device_ptr(&stream);
    let (tbs, _g4) = ws.tile_b_start.device_ptr(&stream);
    let (tbc, _g5) = ws.tile_b_cnt.device_ptr(&stream);
    let (pm, _g6) = ws.perm.device_ptr(&stream);
    let (rw, _g7) = ws.rw_ids.device_ptr(&stream);
    let (ts, _g8) = ws.token_starts.device_ptr(&stream);
    let (hd, _g9) = ws.header.device_ptr(&stream);
    let (iv, _g10) = ws.inv.device_ptr(&stream);
    let (sc, _g11) = ws.scan.device_ptr(&stream);
    let status = unsafe {
        candle_kernels::simple::moe_bucketize::run_moe_bucketize(
            idp as *const std::ffi::c_void,
            n_tokens as i32,
            k as i32,
            n_experts as i32,
            tile_w as i32,
            tok as *mut std::ffi::c_void,
            wid as *mut std::ffi::c_void,
            te as *mut std::ffi::c_void,
            tbs as *mut std::ffi::c_void,
            tbc as *mut std::ffi::c_void,
            pm as *mut std::ffi::c_void,
            rw as *mut std::ffi::c_void,
            ts as *mut std::ffi::c_void,
            hd as *mut std::ffi::c_void,
            iv as *mut std::ffi::c_void,
            sc as *mut std::ffi::c_void,
            live.map_or(0, |l| l.gate_row) as *const std::ffi::c_void,
            live.map_or(0, |l| l.table_plane),
            live.map_or(0, |l| l.snap) as *mut std::ffi::c_void,
            live.map_or(0, |l| l.pinned[0].0),
            live.map_or(0, |l| l.pinned[0].1),
            live.map_or(0, |l| l.pinned[1].0),
            live.map_or(0, |l| l.pinned[1].1),
            decode_lo.as_ptr(),
            decode_hi.as_ptr(),
            decode_lo.len() as i32,
            live.map_or(0, |l| l.summary) as *mut std::ffi::c_void,
            live.map_or(0, |l| l.summary_seq),
            live.map_or(0, |l| l.remote) as *mut std::ffi::c_void,
            live.map_or(0, |l| l.counters) as *mut std::ffi::c_void,
            promo.map_or(0, |p| p.slots) as *const std::ffi::c_void,
            promo.map_or(0, |p| p.log) as *mut std::ffi::c_void,
            promo.map_or(0, |p| p.head) as *mut std::ffi::c_void,
            promo.map_or(0, |p| p.tail) as *const std::ffi::c_void,
            promo.map_or(0, |p| p.cap),
            promo.map_or(0, |p| p.marks) as *mut std::ffi::c_void,
            promo.map_or(0, |p| p.reserve) as *const std::ffi::c_void,
            promo.map_or(0, |p| p.sweep) as *const std::ffi::c_void,
            promo.map_or(0, |p| p.victims) as *const std::ffi::c_void,
            promo.map_or(0, |p| p.retarget) as *const std::ffi::c_void,
            owner.owners as *const std::ffi::c_void,
            owner.zone_end,
            owner.slot_bytes,
            owner.slots,
            live.map_or(0, |l| l.row),
            live.map_or(0, |l| l.remote_dst) as *mut std::ffi::c_void,
            live.map_or(0, |l| l.started_rows) as *mut std::ffi::c_void,
            live.map_or(0, |l| l.ticket),
            promo.map_or(0, |p| p.window) as *const std::ffi::c_void,
            promo.map_or(0, |p| p.depth) as *const std::ffi::c_void,
            promo.map_or(0, |p| p.ahead_n) as *const std::ffi::c_void,
            promo.map_or(0, |p| p.ahead_list) as *const std::ffi::c_void,
            promo.map_or(0, |p| p.ahead_src) as *const std::ffi::c_void,
            promo.map_or(0, |p| p.ahead_cap),
            ahead.map_or(0, |a| a.rows),
            ahead.map_or(0, |a| a.row_layout) as *const std::ffi::c_void,
            ahead.map_or(0, |a| a.items) as *mut std::ffi::c_void,
            ahead.map_or(0, |a| a.done) as *mut std::ffi::c_void,
            ahead.map_or(0, |a| i32::from(a.predict)),
            cpu.as_ref()
                .map_or(std::ptr::null(), |c| &c.split as *const _),
            cpu.map_or(0, |c| c.work) as *mut std::ffi::c_void,
            cpu.map_or(0, |c| c.dst_row) as *mut std::ffi::c_void,
            cpu.map_or(0, |c| c.carry) as *mut std::ffi::c_void,
            cpu.map_or(0, |c| c.tok_first) as *mut std::ffi::c_void,
            cpu.map_or(0, |c| c.in_tok) as *mut std::ffi::c_void,
            cpu.map_or(0, |c| c.input_map) as *mut std::ffi::c_void,
            stream.cu_stream() as *mut std::ffi::c_void,
        )
    };
    match status {
        BUCKETIZE_LAUNCHED => Ok(()),
        BUCKETIZE_REFUSED => crate::bail!(
            "moe_bucketize: the launcher refused its arguments — the workspace still holds \
             the previous layer's tables"
        ),
        BUCKETIZE_EARLIER_FAILURE => crate::bail!(
            "moe_bucketize: an earlier launch on this thread failed, so a kernel before this \
             layer's bucketize did not run (the launcher logged the error)"
        ),
        other => crate::bail!("moe_bucketize: the kernel launch failed (status {other})"),
    }
}

/// One routed layer's live-table inputs and outputs for [`moe_bucketize`].
/// Every address is a device address (mapped host memory where noted).
#[derive(Debug, Clone, Copy)]
pub struct BucketizeLive {
    /// The layer's row of the live gate table (mapped): an entry in VRAM is a
    /// VRAM expert, one inside a `pinned` range is pinned, 0 is cold. The up
    /// and down rows are `table_plane` and `2 · table_plane` entries after it,
    /// and an expert with any of its three entries 0 is cold.
    pub gate_row: u64,
    pub table_plane: i64,
    /// `u64[3][n_experts]` snapshot of the routed experts' entries (gate, up,
    /// down) — the weight tables of the layer's three grouped GEMMs.
    pub snap: u64,
    /// The two pinned host ranges `[lo, hi)` — the warm tier and the pad.
    pub pinned: [(u64, u64); 2],
    /// `u32[n_experts + 1]` routing summary (mapped): `count | pinned << 29 |
    /// cold << 30 | decode << 31` per expert, then `summary_seq`, stored last
    /// once every count is visible system-wide.
    pub summary: u64,
    pub summary_seq: u32,
    /// `i32[n_experts][4]` remote list, `{expert, first_tile, n_tiles, cold}`.
    pub remote: u64,
    /// `i32[3]` work counters of the layer's gate, up and down launches, zeroed.
    pub counters: u64,
    /// The layer's row, recorded in the promotion log.
    pub row: i32,
    /// The promotion ring, or `None`: each remote expert takes the next free
    /// VRAM slot image from it into `remote_dst` (see [`PromoRing`]).
    pub promo: Option<PromoRing>,
    /// `u64[n_experts]` promotion slot per remote expert (0 = none).
    pub remote_dst: u64,
    /// `u64[rows]` started words (mapped), or 0: the kernel stores `ticket`
    /// into `started_rows[row]`, behind a system fence, before it reads any
    /// live entry — the host's reclaim rule keys slot reuse on it.
    pub started_rows: u64,
    /// This invocation's ticket.
    pub ticket: u64,
    /// The slot-tenancy check ([`OwnerCheck`]); all zero runs none.
    pub owner: OwnerCheck,
    /// Read-ahead into the promotion ring ([`ReadAhead`]), or `None`. Requires
    /// `promo`.
    pub ahead: Option<ReadAhead>,
    /// The CPU class ([`BucketizeCpu`]), or `None`.
    pub cpu: Option<BucketizeCpu>,
}

/// The CPU class for one [`moe_bucketize`] (`moe_bucketize_cpu.cuh`): the
/// split's measured costs, and where the chosen experts are published.
#[derive(Debug, Clone, Copy)]
pub struct BucketizeCpu {
    pub split: candle_kernels::simple::moe_bucketize::CpuSplit,
    /// This invocation's work slot (mapped, `CPU_WORK_SLOT_BYTES`).
    pub work: u64,
    /// Device `u32[n_tokens · k]`: ring row → expert-grouped row.
    pub dst_row: u64,
    /// Device `u32[CPU_CARRY_WORDS]`, or 0: the CPU experts one launch hands
    /// the next to promote behind the pool (`split.promote`).
    pub carry: u64,
    /// The inputs, each distinct token once (`moe_bucketize_cpu.cuh`,
    /// `cpu_inputs`): device `u32[n_tokens]` scratch, device `u32[cap]` input
    /// row → token, and the mapped `u32[1 + cap]` input-row count then ring
    /// row → input row. `input_map` 0 writes none.
    pub tok_first: u64,
    pub in_tok: u64,
    pub input_map: u64,
}

/// The device buffers [`moe_bucketize`]'s read-ahead walk writes and the gate
/// launch's workers consume (`candle-kernels/src/moe_read_ahead.cuh`). The
/// predictions it walks, its budget and its depth are mapped words in the
/// promotion ring ([`PromoRing`]).
#[derive(Debug, Clone, Copy)]
pub struct ReadAhead {
    /// `u64[rows][4]`: each row's gate, up and down offsets inside a slot image,
    /// and the image's bytes — what a read-ahead copy moves.
    pub row_layout: u64,
    /// Rows in the live table: targets wrap at it.
    pub rows: i32,
    /// `u64[1 + AHEAD_MAX · AHEAD_ITEM_WORDS]`: the item count, then the items.
    pub items: u64,
    /// `u32[AHEAD_MAX]`: pieces of each item copied so far, zeroed per item.
    pub done: u64,
    /// Whether the predictions are read ahead. Without it the items carry only
    /// the CPU pool's promotions ([`BucketizeCpu::carry`]).
    pub predict: bool,
}

/// Where [`moe_bucketize`] checks every VRAM entry it snapshots against the
/// expert last installed in that slot. `owners` is a mapped `u32[slots]` of
/// tags `(row + 1) << 16 | expert`; slot `s` spans `[zone_end - (s + 1) ·
/// slot_bytes, zone_end - s · slot_bytes)`, the weight zone's downward layout.
/// A mismatch — a GEMM about to read one expert's weights under another's
/// name — traps with the slot named. `owners == 0` checks nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct OwnerCheck {
    pub owners: u64,
    pub zone_end: u64,
    pub slot_bytes: u64,
    pub slots: u32,
}

/// The promotion ring `moe_bucketize` takes slots from — device addresses into
/// mapped host memory. The host offers VRAM slot images at `slots[tail % cap]`
/// (`u64`) and advances `tail` (`u32`); the kernel takes them from `head`
/// (`u32`), records `summary_seq << 32 | row << 16 | expert` in `log` (`u64`)
/// at the same index, and advances `head`. `marks` (`u32[rows][n_experts]`) is
/// set by the kernel for each expert it gives a slot and cleared by the host
/// once the promotion lands; a marked expert is not given another. `reserve`
/// is the address of a `u32`: the stock kept for decode-scored experts, so a
/// prompt-only expert takes a slot only while more than that value is stocked
/// (a value of 0 lets it take any). A null address means a prompt-only expert
/// never takes one.
///
/// `sweep` is the address of a `u32`: a launch with more claiming experts than
/// it holds is a sweep and claims nothing. `victims` (`u64[cap]`) names the
/// resident expert behind each offer — `row · n_experts + expert`, the index of
/// its gate entry — or `PROMO_EMPTY` for an empty slot; a claimed victim's
/// three entries are retargeted to `retarget` (`u64[cap][3]`: gate, up, down)
/// before the slot is written, and a victim the launch itself routes is
/// skipped and logged with the expert `PROMO_SKIP`.
///
/// The read-ahead words ([`ReadAhead`]): `window` (`u32`) is the slot images
/// the link moves in one layer — a launch reads ahead with what its own remote
/// experts leave of it; `depth` (`u32`) the last hop, so a launch at row `r`
/// reads ahead for rows `r + 2 ..= r + depth`; and the host's predictions,
/// `ahead_n` (`u32[rows]`) experts listed per row in `ahead_list`
/// (`u32[rows][ahead_cap]`), each with the slot image the host vetted as its
/// source in `ahead_src` (`u64[rows][ahead_cap]`) — a warm slot, or a pad slot
/// it holds pinned. An expert is read ahead only while its entries still point
/// at its vetted image.
#[derive(Debug, Clone, Copy)]
pub struct PromoRing {
    pub slots: u64,
    pub log: u64,
    pub head: u64,
    pub tail: u64,
    pub cap: u32,
    pub marks: u64,
    pub reserve: u64,
    pub sweep: u64,
    pub victims: u64,
    pub retarget: u64,
    pub window: u64,
    pub depth: u64,
    pub ahead_n: u64,
    pub ahead_list: u64,
    pub ahead_src: u64,
    pub ahead_cap: u32,
}
