//! The CPU reference that every GPU bucketize test asserts against, bit for bit.
//!
//! The reference mirrors the production counting sort's single-pass cursor
//! style, so the tests also show that the kernel's per-expert scan reproduces
//! the grouping production uses. Every value is an integer, so equality is
//! exact and no tolerance is involved. The live reference fixes the tile order
//! as pinned experts, then cold, then VRAM, each ascending. The row layout stays
//! in ascending expert id, because that order decides the summation order of
//! the scatter.

use super::*;

// moe_bucketize: bit-exact GPU vs CPU-reference tests
// The GPU bucketize replaces the CPU counting-sort in the grouped expert
// compute path, so its outputs must be BIT-IDENTICAL to the CPU grouping —
// asserted here with exact equality (never tolerances; the computation is pure
// integer). The reference below deliberately mirrors the production sort's
// single-pass cursor style (`forward_with_indices`), so these tests also prove
// the kernel's per-expert-scan formulation reproduces the production grouping.

/// CPU reference of the `moe_bucketize.cu` contract, padding included.
pub(super) struct BucketizeRef {
    pub(super) tok_ids: Vec<u32>,
    pub(super) weight_ids: Vec<u32>,
    pub(super) tile_expert: Vec<i32>,
    pub(super) tile_b_start: Vec<i32>,
    pub(super) tile_b_cnt: Vec<i32>,
    pub(super) perm: Vec<u32>,
    pub(super) rw_ids: Vec<u32>,
    pub(super) token_starts: Vec<i32>,
    pub(super) header: [i32; 5],
    /// `count | pinned << 29 | cold << 30 | decode << 31` per expert.
    pub(super) summary: Vec<u32>,
    /// `{expert, first_tile, n_tiles, cold}` per routed remote expert.
    pub(super) remote: Vec<[i32; 4]>,
}

/// Where an expert's gate entry points, as `moe_bucketize` classifies it.
pub(super) const CLS_VRAM: u8 = 0;

pub(super) const CLS_PINNED: u8 = 1;

pub(super) const CLS_COLD: u8 = 2;

/// The reference with every expert in VRAM and every token a decode row.
pub(super) fn bucketize_ref(
    ids: &[u32],
    n_tokens: usize,
    k: usize,
    n_experts: usize,
    tile_w: usize,
) -> BucketizeRef {
    bucketize_ref_live(
        ids,
        n_tokens,
        k,
        n_experts,
        tile_w,
        &vec![CLS_VRAM; n_experts],
        n_tokens,
    )
}

/// The reference over a live table: `class[e]` is where expert `e`'s gate entry
/// points. The ROW layout ascends by expert id whatever the classes; the TILE
/// order is pinned experts, then cold, then VRAM, each ascending; the routed
/// pinned and cold experts are listed in that order.
pub(super) fn bucketize_ref_live(
    ids: &[u32],
    n_tokens: usize,
    k: usize,
    n_experts: usize,
    tile_w: usize,
    class: &[u8],
    decode_tokens: usize,
) -> BucketizeRef {
    let a_ub = n_tokens * k;
    assert_eq!(ids.len(), a_ub);
    assert_eq!(class.len(), n_experts);
    let valid = |e: u32| (e as usize) < n_experts;

    // Counts → offsets → tile prefix (kernel phase 2).
    let mut counts = vec![0i32; n_experts];
    let mut decode = vec![false; n_experts];
    for (i, &e) in ids.iter().enumerate() {
        if valid(e) {
            counts[e as usize] += 1;
            if i / k < decode_tokens {
                decode[e as usize] = true;
            }
        }
    }
    // Only a routed expert is classified: an unrouted one's word is 0.
    let summary: Vec<u32> = (0..n_experts)
        .map(|e| {
            let routed = counts[e] > 0;
            counts[e] as u32
                | (u32::from(routed && class[e] == CLS_PINNED) << 29)
                | (u32::from(routed && class[e] == CLS_COLD) << 30)
                | (u32::from(decode[e]) << 31)
        })
        .collect();
    let mut offsets = vec![0i32; n_experts + 1];
    let mut n_active = 0i32;
    for e in 0..n_experts {
        offsets[e + 1] = offsets[e] + counts[e];
        if counts[e] > 0 {
            n_active += 1;
        }
    }
    let tiles_of = |e: usize| (counts[e] + tile_w as i32 - 1) / tile_w as i32;
    let mut tile_pref = vec![0i32; n_experts];
    let mut num_tiles = 0i32;
    let mut remote = Vec::new();
    let mut remote_tiles = 0i32;
    for want in [CLS_PINNED, CLS_COLD, CLS_VRAM] {
        for e in 0..n_experts {
            if class[e] == want {
                tile_pref[e] = num_tiles;
                if want != CLS_VRAM && tiles_of(e) > 0 {
                    remote.push([
                        e as i32,
                        num_tiles,
                        tiles_of(e),
                        i32::from(want == CLS_COLD),
                    ]);
                }
                num_tiles += tiles_of(e);
            }
        }
        if want == CLS_COLD {
            remote_tiles = num_tiles;
        }
    }
    let total_valid = offsets[n_experts];

    // Stable bucket write — single-pass cursor, the `forward_with_indices` style.
    let mut tok_ids = vec![u32::MAX; a_ub];
    let mut weight_ids = vec![u32::MAX; a_ub];
    let mut inv = vec![0u32; a_ub];
    let mut cursors: Vec<i32> = offsets[..n_experts].to_vec();
    for (i, &e) in ids.iter().enumerate() {
        if valid(e) {
            let row = cursors[e as usize];
            cursors[e as usize] += 1;
            tok_ids[row as usize] = (i / k) as u32;
            weight_ids[row as usize] = i as u32;
            inv[i] = row as u32;
        }
    }

    // Tile tables + padding.
    let mut tile_expert = vec![0i32; a_ub];
    let mut tile_b_start = vec![0i32; a_ub];
    let mut tile_b_cnt = vec![0i32; a_ub];
    let tw = tile_w as i32;
    for e in 0..n_experts {
        let cnt = counts[e];
        let base = tile_pref[e];
        let mut t = 0i32;
        while t * tw < cnt {
            tile_expert[(base + t) as usize] = e as i32;
            tile_b_start[(base + t) as usize] = offsets[e] + t * tw;
            tile_b_cnt[(base + t) as usize] = (cnt - t * tw).min(tw);
            t += 1;
        }
    }

    // Token-major compaction + segment boundaries. Within a token the pairs are
    // ordered by ascending expert-grouped row — the production scatter's
    // `sort_by_key((token_id, row))` accumulation order, so the float-summation
    // order downstream is bit-identical to the CPU-built tables.
    let mut perm = vec![0u32; a_ub];
    let mut rw_ids = vec![0u32; a_ub];
    let mut token_starts = vec![0i32; n_tokens + 1];
    let mut j = 0usize;
    for t in 0..n_tokens {
        token_starts[t] = j as i32;
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for s in 0..k {
            let i = t * k + s;
            if valid(ids[i]) {
                pairs.push((inv[i], i as u32));
            }
        }
        pairs.sort_unstable();
        for (row, widx) in pairs {
            perm[j] = row;
            rw_ids[j] = widx;
            j += 1;
        }
    }
    token_starts[n_tokens] = total_valid;

    BucketizeRef {
        tok_ids,
        weight_ids,
        tile_expert,
        tile_b_start,
        tile_b_cnt,
        perm,
        rw_ids,
        token_starts,
        header: [
            n_active,
            total_valid,
            num_tiles,
            remote.len() as i32,
            remote_tiles,
        ],
        summary,
        remote,
    }
}

/// Run the GPU bucketize for `ids` and assert every output buffer is
/// bit-identical to the CPU reference.
#[allow(clippy::too_many_arguments)]
pub(super) fn assert_bucketize_case(
    device: &crate::Device,
    ids: Vec<u32>,
    n_tokens: usize,
    k: usize,
    n_experts: usize,
    tile_w: usize,
    label: &str,
) -> Result<()> {
    let reference = bucketize_ref(&ids, n_tokens, k, n_experts, tile_w);
    let t = crate::Tensor::from_vec(ids, (n_tokens, k), device)?;
    let cuda_dev = match device {
        crate::Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let mut ws = MoeBucketizeWorkspace::new(&cuda_dev, n_tokens, k)?;
    moe_bucketize(
        &t,
        n_experts,
        tile_w,
        &mut ws,
        None,
        &DecodeRows::prefix(n_tokens),
    )?;

    let a_ub = n_tokens * k;
    let tok = cuda_dev.memcpy_dtov(&ws.tok_ids.slice(..a_ub))?;
    let wid = cuda_dev.memcpy_dtov(&ws.weight_ids.slice(..a_ub))?;
    let te = cuda_dev.memcpy_dtov(&ws.tile_expert.slice(..a_ub))?;
    let tbs = cuda_dev.memcpy_dtov(&ws.tile_b_start.slice(..a_ub))?;
    let tbc = cuda_dev.memcpy_dtov(&ws.tile_b_cnt.slice(..a_ub))?;
    let pm = cuda_dev.memcpy_dtov(&ws.perm.slice(..a_ub))?;
    let rw = cuda_dev.memcpy_dtov(&ws.rw_ids.slice(..a_ub))?;
    let ts = cuda_dev.memcpy_dtov(&ws.token_starts.slice(..n_tokens + 1))?;
    let hd = cuda_dev.memcpy_dtov(&ws.header.slice(..5))?;

    assert_eq!(hd, reference.header.to_vec(), "{label}: header");
    assert_eq!(tok, reference.tok_ids, "{label}: tok_ids");
    assert_eq!(wid, reference.weight_ids, "{label}: weight_ids");
    assert_eq!(te, reference.tile_expert, "{label}: tile_expert");
    assert_eq!(tbs, reference.tile_b_start, "{label}: tile_b_start");
    assert_eq!(tbc, reference.tile_b_cnt, "{label}: tile_b_cnt");
    assert_eq!(pm, reference.perm, "{label}: perm");
    assert_eq!(rw, reference.rw_ids, "{label}: rw_ids");
    assert_eq!(ts, reference.token_starts, "{label}: token_starts");
    Ok(())
}

/// Assert every `moe_bucketize` output table is bit-identical to the CPU
/// reference for a workspace already run on `ids` — the gate the micro-bench
/// runs before timing each config.
#[cfg(test)]
pub(super) fn assert_bucketize_ws(
    dev: &CudaDevice,
    ws: &MoeBucketizeWorkspace,
    ids: &[u32],
    n_tokens: usize,
    k: usize,
    n_experts: usize,
    tile_w: usize,
    label: &str,
) -> Result<()> {
    let reference = bucketize_ref(ids, n_tokens, k, n_experts, tile_w);
    let a_ub = n_tokens * k;
    assert_eq!(
        dev.memcpy_dtov(&ws.header.slice(..5))?,
        reference.header.to_vec(),
        "{label}: header"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.tok_ids.slice(..a_ub))?,
        reference.tok_ids,
        "{label}: tok_ids"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.weight_ids.slice(..a_ub))?,
        reference.weight_ids,
        "{label}: weight_ids"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.tile_expert.slice(..a_ub))?,
        reference.tile_expert,
        "{label}: tile_expert"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.tile_b_start.slice(..a_ub))?,
        reference.tile_b_start,
        "{label}: tile_b_start"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.tile_b_cnt.slice(..a_ub))?,
        reference.tile_b_cnt,
        "{label}: tile_b_cnt"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.perm.slice(..a_ub))?,
        reference.perm,
        "{label}: perm"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.rw_ids.slice(..a_ub))?,
        reference.rw_ids,
        "{label}: rw_ids"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.token_starts.slice(..n_tokens + 1))?,
        reference.token_starts,
        "{label}: token_starts"
    );
    Ok(())
}

/// The two pinned ranges the live bucketize tests classify against: an entry
/// inside either is a pinned expert.
pub(super) const TEST_PINNED: [(u64, u64); 2] =
    [(0x7000_0000, 0x7100_0000), (0x9000_0000, 0x9010_0000)];

/// The `[3][n_experts]` live table the bucketize tests classify: VRAM gate
/// entries at `0x10_0000 + e·0x100`, pinned ones inside `TEST_PINNED`, cold 0;
/// up and down at `+0x10` and `+0x20` of the gate entry; the down entry of every
/// expert in `zero_down` cleared.
pub(super) fn test_live_table(class: &[u8], zero_down: &[usize]) -> Vec<u64> {
    let n_experts = class.len();
    let gate: Vec<u64> = class
        .iter()
        .enumerate()
        .map(|(e, &c)| match c {
            CLS_VRAM => 0x10_0000 + e as u64 * 0x100,
            CLS_PINNED if e % 2 == 0 => TEST_PINNED[0].0 + e as u64 * 0x1000,
            CLS_PINNED => TEST_PINNED[1].0 + e as u64 * 0x100,
            _ => 0,
        })
        .collect();
    let mut table: Vec<u64> = gate.clone();
    table.extend(gate.iter().map(|&g| if g == 0 { 0 } else { g + 0x10 }));
    table.extend(gate.iter().map(|&g| if g == 0 { 0 } else { g + 0x20 }));
    for &e in zero_down {
        table[2 * n_experts + e] = 0;
    }
    table
}

/// `bytes` of zeroed mapped pinned host memory: (host pointer, device address).
pub(super) fn mapped_host(bytes: usize) -> (*mut c_void, u64) {
    use cudarc::driver::sys;
    let mut raw: *mut c_void = std::ptr::null_mut();
    let mut d: sys::CUdeviceptr = 0;
    unsafe {
        assert_eq!(
            sys::cuMemHostAlloc(&mut raw, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP),
            sys::CUresult::CUDA_SUCCESS
        );
        assert_eq!(
            sys::cuMemHostGetDevicePointer_v2(&mut d, raw, 0),
            sys::CUresult::CUDA_SUCCESS
        );
        std::ptr::write_bytes(raw as *mut u8, 0, bytes);
    }
    (raw, d)
}

/// Run the GPU bucketize over a live table built from `class` (VRAM gate
/// entries at `0x10_0000 + e·0x100`, pinned ones inside `TEST_PINNED`, cold 0;
/// up and down at `+0x10` and `+0x20` of the gate entry), with the down entry of
/// every expert in `zero_down` cleared, a summary buffer and a remote list.
/// Checks the snapshot of every routed expert against the table, and returns
/// every other output.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_bucketize_live(
    dev: &CudaDevice,
    ids: &[u32],
    n_tokens: usize,
    k: usize,
    n_experts: usize,
    tile_w: usize,
    class: &[u8],
    zero_down: &[usize],
    decode: &DecodeRows,
) -> Result<BucketizeRef> {
    let device = crate::Device::Cuda(dev.clone());
    let t = crate::Tensor::from_vec(ids.to_vec(), (n_tokens, k), &device)?;
    let table = test_live_table(class, zero_down);
    let table_dev = dev.memcpy_stod(&table)?;
    let snap_dev = dev.memcpy_stod(&vec![0xffff_ffffu64; 3 * n_experts])?;
    // One word past the counts for the sequence number, stored last.
    let summary_dev = dev.memcpy_stod(&vec![0xdead_beefu32; n_experts + 1])?;
    let remote_dev = dev.memcpy_stod(&vec![-1i32; n_experts * 4])?;
    let counters_dev = dev.memcpy_stod(&[7i32, 7, 7])?;
    let mut ws = MoeBucketizeWorkspace::new(dev, n_tokens, k)?;
    let stream = dev.cuda_stream();
    {
        let (gp, _g0) = table_dev.device_ptr(&stream);
        let (sp, _g1) = summary_dev.device_ptr(&stream);
        let (rp, _g2) = remote_dev.device_ptr(&stream);
        let (cp, _g3) = counters_dev.device_ptr(&stream);
        let (np, _g4) = snap_dev.device_ptr(&stream);
        let live = BucketizeLive {
            gate_row: gp,
            table_plane: n_experts as i64,
            snap: np,
            pinned: TEST_PINNED,
            summary: sp,
            summary_seq: 77,
            remote: rp,
            counters: cp,
            row: 0,
            promo: None,
            remote_dst: 0,
            started_rows: 0,
            ticket: 0,
            owner: OwnerCheck::default(),
            ahead: None,
            cpu: None,
        };
        moe_bucketize(&t, n_experts, tile_w, &mut ws, Some(&live), decode)?;
    }
    let a_ub = n_tokens * k;
    let hd = dev.memcpy_dtov(&ws.header.slice(..5))?;
    let mut summary = dev.memcpy_dtov(&summary_dev)?;
    assert_eq!(summary.pop(), Some(77), "summary sequence word");
    assert_eq!(
        dev.memcpy_dtov(&counters_dev)?,
        vec![0, 0, 0],
        "launch counters zeroed"
    );
    // The snapshot holds every routed expert's three entries as read, and
    // nothing else is written.
    let snap = dev.memcpy_dtov(&snap_dev)?;
    for e in 0..n_experts {
        let routed = summary[e] & SUMMARY_COUNT_MASK != 0;
        for p in 0..3 {
            let want = if routed {
                table[p * n_experts + e]
            } else {
                0xffff_ffff
            };
            assert_eq!(
                snap[p * n_experts + e],
                want,
                "snapshot of expert {e}, projection {p}"
            );
        }
    }
    let remote_all = dev.memcpy_dtov(&remote_dev)?;
    let remote = remote_all
        .chunks(4)
        .take(hd[3] as usize)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    Ok(BucketizeRef {
        tok_ids: dev.memcpy_dtov(&ws.tok_ids.slice(..a_ub))?,
        weight_ids: dev.memcpy_dtov(&ws.weight_ids.slice(..a_ub))?,
        tile_expert: dev.memcpy_dtov(&ws.tile_expert.slice(..a_ub))?,
        tile_b_start: dev.memcpy_dtov(&ws.tile_b_start.slice(..a_ub))?,
        tile_b_cnt: dev.memcpy_dtov(&ws.tile_b_cnt.slice(..a_ub))?,
        perm: dev.memcpy_dtov(&ws.perm.slice(..a_ub))?,
        rw_ids: dev.memcpy_dtov(&ws.rw_ids.slice(..a_ub))?,
        token_starts: dev.memcpy_dtov(&ws.token_starts.slice(..n_tokens + 1))?,
        header: [hd[0], hd[1], hd[2], hd[3], hd[4]],
        summary,
        remote,
    })
}
