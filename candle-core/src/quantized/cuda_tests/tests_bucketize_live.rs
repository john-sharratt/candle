//! `moe_bucketize` over a live table: promotion slots, read-ahead, victims and
//! sweeps.
//!
//! A live table classifies each routed expert as VRAM, pinned or cold, and the
//! launch decides what to do about each from the promotion ring the host keeps.
//! These tests pin that decision in raw words. Remote experts take promotion
//! slots cold first, a prompt-only expert takes only an empty offer, and a miss
//! claims a victim behind its offer after retargeting the victim's entries. A
//! sweep claims nothing. Each row's started word records the invocation the
//! device is inside, and remote experts are ordered ahead of the VRAM tiles.

use super::*;

/// A promotion ring in mapped memory: `u64 slots[cap] | u64 log[cap] | u32 head
/// | u32 tail | u32 marks[rows × n_experts] | u32 reserve | u32 sweep | u64
/// victims[cap] | u64 retarget[cap][3] | u32 window | u32 depth | u32
/// ahead_n[rows] | u32 ahead_list[rows][TEST_AHEAD_CAP] | u64
/// ahead_src[rows][TEST_AHEAD_CAP]` (the victims and the sources 8-aligned).
/// The reserve starts at `u32::MAX` — no prompt-only expert takes a slot — and
/// so does the sweep word — no launch is a sweep; the window starts at 0 —
/// nothing is read ahead. Freed on drop.
pub(super) struct TestRing {
    pub(super) host: *mut c_void,
    pub(super) dev: u64,
    pub(super) cap: usize,
    pub(super) n_experts: usize,
    pub(super) reserve_at: usize,
    pub(super) victims_at: usize,
    pub(super) ahead_at: usize,
    pub(super) src_at: usize,
    pub(super) rows: usize,
}

/// Predictions per row the test ring's read-ahead lists hold.
pub(super) const TEST_AHEAD_CAP: usize = 8;

impl TestRing {
    pub(super) fn new(cap: usize, rows: usize, n_experts: usize) -> Self {
        let reserve_at = cap * 16 + 8 + rows * n_experts * 4;
        let victims_at = (reserve_at + 8).next_multiple_of(8);
        let ahead_at = victims_at + cap * 32;
        let src_at = (ahead_at + 8 + rows * 4 + rows * TEST_AHEAD_CAP * 4).next_multiple_of(8);
        let (host, dev) = mapped_host(src_at + rows * TEST_AHEAD_CAP * 8);
        let ring = Self {
            host,
            dev,
            cap,
            n_experts,
            reserve_at,
            victims_at,
            ahead_at,
            src_at,
            rows,
        };
        ring.set_reserve(u32::MAX);
        ring.set_sweep(u32::MAX);
        ring
    }

    pub(super) fn ring(&self) -> PromoRing {
        PromoRing {
            slots: self.dev,
            log: self.dev + (self.cap * 8) as u64,
            head: self.dev + (self.cap * 16) as u64,
            tail: self.dev + (self.cap * 16 + 4) as u64,
            cap: self.cap as u32,
            marks: self.dev + (self.cap * 16 + 8) as u64,
            reserve: self.dev + self.reserve_at as u64,
            sweep: self.dev + (self.reserve_at + 4) as u64,
            victims: self.dev + self.victims_at as u64,
            retarget: self.dev + (self.victims_at + self.cap * 8) as u64,
            window: self.dev + self.ahead_at as u64,
            depth: self.dev + (self.ahead_at + 4) as u64,
            ahead_n: self.dev + (self.ahead_at + 8) as u64,
            ahead_list: self.dev + (self.ahead_at + 8 + self.rows * 4) as u64,
            ahead_src: self.dev + self.src_at as u64,
            ahead_cap: TEST_AHEAD_CAP as u32,
        }
    }

    pub(super) fn ahead_word(&self, at: usize) -> *mut u32 {
        unsafe { (self.host as *mut u8).add(self.ahead_at + at) as *mut u32 }
    }

    /// Read ahead with up to `window` slot images per layer, over rows
    /// `row + 2 ..= row + depth`.
    pub(super) fn set_read_ahead(&self, window: u32, depth: u32) {
        unsafe {
            std::ptr::write_volatile(self.ahead_word(0), window);
            std::ptr::write_volatile(self.ahead_word(4), depth);
        }
    }

    /// `experts` predicted for `row`, in order, each with its vetted source
    /// image.
    pub(super) fn predict(&self, row: usize, experts: &[(u32, u64)]) {
        assert!(experts.len() <= TEST_AHEAD_CAP);
        unsafe {
            for (q, &(e, src)) in experts.iter().enumerate() {
                std::ptr::write_volatile(
                    self.ahead_word(8 + self.rows * 4 + (row * TEST_AHEAD_CAP + q) * 4),
                    e,
                );
                std::ptr::write_volatile(
                    (self.host as *mut u8).add(self.src_at + (row * TEST_AHEAD_CAP + q) * 8)
                        as *mut u64,
                    src,
                );
            }
            std::ptr::write_volatile(self.ahead_word(8 + row * 4), experts.len() as u32);
        }
    }

    pub(super) fn set_reserve(&self, n: u32) {
        unsafe {
            std::ptr::write_volatile((self.host as *mut u8).add(self.reserve_at) as *mut u32, n)
        }
    }

    pub(super) fn set_sweep(&self, n: u32) {
        unsafe {
            std::ptr::write_volatile(
                (self.host as *mut u8).add(self.reserve_at + 4) as *mut u32,
                n,
            )
        }
    }

    pub(super) fn victims(&self) -> *mut u64 {
        unsafe { (self.host as *mut u8).add(self.victims_at) as *mut u64 }
    }

    pub(super) fn retarget(&self) -> *mut u64 {
        unsafe { (self.host as *mut u8).add(self.victims_at + self.cap * 8) as *mut u64 }
    }

    pub(super) fn mark_ptr(&self, row: usize, e: usize) -> *mut u32 {
        unsafe {
            ((self.host as *mut u8).add(self.cap * 16 + 8) as *mut u32)
                .add(row * self.n_experts + e)
        }
    }

    pub(super) fn mark(&self, row: usize, e: usize) -> u32 {
        unsafe { std::ptr::read_volatile(self.mark_ptr(row, e)) }
    }

    pub(super) fn clear_mark(&self, row: usize, e: usize) {
        unsafe { std::ptr::write_volatile(self.mark_ptr(row, e), 0) }
    }

    pub(super) fn slots(&self) -> *mut u64 {
        self.host as *mut u64
    }

    pub(super) fn log(&self, i: usize) -> u64 {
        unsafe {
            std::ptr::read_volatile((self.host as *const u8).add(self.cap * 8 + 8 * i) as *const u64)
        }
    }

    pub(super) fn head(&self) -> u32 {
        unsafe {
            std::ptr::read_volatile((self.host as *const u8).add(self.cap * 16) as *const u32)
        }
    }

    /// Set head and tail, an empty ring positioned at `at`.
    pub(super) fn start_at(&self, at: u32) {
        unsafe {
            std::ptr::write_volatile((self.host as *mut u8).add(self.cap * 16) as *mut u32, at);
            std::ptr::write_volatile(
                (self.host as *mut u8).add(self.cap * 16 + 4) as *mut u32,
                at,
            );
        }
    }

    /// Publish empty `slots` at `tail`, then advance `tail` past them.
    pub(super) fn push(&self, slots: &[u64]) {
        let offers: Vec<(u64, Option<(u64, [u64; 3])>)> =
            slots.iter().map(|&s| (s, None)).collect();
        self.offer(&offers);
    }

    /// Publish `offers` at `tail` — each a slot image and, for a slot still
    /// holding an expert, its gate-plane index and the three entries it is
    /// retargeted to — then advance `tail` past them.
    pub(super) fn offer(&self, offers: &[(u64, Option<(u64, [u64; 3])>)]) {
        unsafe {
            let tail_p = (self.host as *mut u8).add(self.cap * 16 + 4) as *mut u32;
            let tail = std::ptr::read_volatile(tail_p);
            for (i, &(s, victim)) in offers.iter().enumerate() {
                let at = (tail as usize + i) % self.cap;
                std::ptr::write_volatile(self.slots().add(at), s);
                let (index, entries) = victim.unwrap_or((PROMO_EMPTY, [0; 3]));
                std::ptr::write_volatile(self.victims().add(at), index);
                for (p, &v) in entries.iter().enumerate() {
                    std::ptr::write_volatile(self.retarget().add(3 * at + p), v);
                }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(tail_p, tail + offers.len() as u32);
        }
    }
}

impl Drop for TestRing {
    fn drop(&mut self) {
        unsafe {
            cuMemFreeHost(self.host);
        }
    }
}

/// **Bucketize gives remote experts promotion slots from the ring, cold ones
/// first, while it has them.** Raw expected values for the hand case of
/// `cuda_moe_bucketize_live_table_orders_remote_first` (e1 pinned, e5 cold —
/// listed in that order, walked e5 first, `promotion_walk`): one slot free —
/// e5 takes it, e1 gets 0; then two more — both take one. The log names
/// `row << 16 | expert` at each taken index, and `head` advances past exactly
/// the slots taken.
#[test]
pub(super) fn cuda_moe_bucketize_promotes_remote_experts_from_the_ring() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let ids = [5u32, 1, 1, 3, 5, 6];
    let mut class = vec![CLS_VRAM; 8];
    class[1] = CLS_PINNED;
    class[5] = CLS_COLD;
    let t = crate::Tensor::from_vec(ids.to_vec(), (3, 2), &device)?;
    let table_dev = dev.memcpy_stod(&test_live_table(&class, &[]))?;
    let snap = dev.memcpy_stod(&[0u64; 24])?;
    let summary = dev.memcpy_stod(&[0u32; 9])?;
    let remote = dev.memcpy_stod(&[0i32; 32])?;
    let counters = dev.memcpy_stod(&[0i32; 3])?;
    let remote_dst = dev.memcpy_stod(&[7u64; 8])?;
    let ring = TestRing::new(4, 10, 8);
    // Head and tail at 3: the first slot published lands at index 3.
    ring.start_at(3);
    let mut ws = MoeBucketizeWorkspace::new(&dev, 3, 2)?;
    let stream = dev.cuda_stream();
    let run_ring = |ws: &mut MoeBucketizeWorkspace,
                    seq: u32,
                    decode: &DecodeRows,
                    promo: PromoRing|
     -> Result<Vec<u64>> {
        let live = BucketizeLive {
            gate_row: table_dev.device_ptr(&stream).0,
            table_plane: 8,
            snap: snap.device_ptr(&stream).0,
            pinned: TEST_PINNED,
            summary: summary.device_ptr(&stream).0,
            summary_seq: seq,
            remote: remote.device_ptr(&stream).0,
            counters: counters.device_ptr(&stream).0,
            row: 9,
            promo: Some(promo),
            remote_dst: remote_dst.device_ptr(&stream).0,
            started_rows: 0,
            ticket: 0,
            owner: OwnerCheck::default(),
            ahead: None,
            cpu: None,
        };
        moe_bucketize(&t, 8, 2, ws, Some(&live), decode)?;
        Ok(dev.memcpy_dtov(&remote_dst.slice(..2))?)
    };
    let run_with = |ws: &mut MoeBucketizeWorkspace, seq: u32, decode: &DecodeRows| {
        run_ring(ws, seq, decode, ring.ring())
    };
    // Token 0, the decode row, routes both remote experts (e5 and e1).
    let run = |ws: &mut MoeBucketizeWorkspace, seq: u32| run_with(ws, seq, &DecodeRows::prefix(1));

    ring.push(&[0xd000]);
    assert_eq!(
        run(&mut ws, 1)?,
        vec![0, 0xd000],
        "one slot: the cold expert"
    );
    assert_eq!(ring.head(), 4);
    assert_eq!(ring.log(3), (1 << 32) | (9 << 16) | 5);
    assert_eq!(
        (ring.mark(9, 1), ring.mark(9, 5)),
        (0, 1),
        "e5 marked in flight"
    );

    // e5 is still in flight: only e1 takes a slot.
    ring.push(&[0xa000, 0xb000]);
    assert_eq!(
        run(&mut ws, 2)?,
        vec![0xa000, 0],
        "the marked expert is not promoted twice"
    );
    assert_eq!(ring.head(), 5);
    assert_eq!(ring.log(0), (2 << 32) | (9 << 16) | 1);
    assert_eq!(ring.mark(9, 1), 2);

    // Both landed: the host cleared the marks, so both take slots again.
    ring.clear_mark(9, 1);
    ring.clear_mark(9, 5);
    ring.push(&[0xc000]);
    assert_eq!(run(&mut ws, 3)?, vec![0xc000, 0xb000], "cold first");
    assert_eq!(ring.head(), 7);
    assert_eq!(
        (ring.log(1), ring.log(2)),
        ((3 << 32) | (9 << 16) | 5, (3 << 32) | (9 << 16) | 1)
    );

    // An empty ring promotes nothing and moves nothing.
    ring.clear_mark(9, 1);
    ring.clear_mark(9, 5);
    assert_eq!(run(&mut ws, 4)?, vec![0, 0]);
    assert_eq!(ring.head(), 7);

    // A prompt-only expert takes no slot, whatever the ring holds: its workers
    // compute it from pinned memory and the zone is left to decode. Token 2
    // alone is decode-scored, routing e5 (remote) and e6 (VRAM) — e1, routed
    // only by prompt tokens, gets nothing.
    ring.push(&[0xe000, 0xf000]);
    let mut last = DecodeRows::none();
    last.push(2, 3);
    assert_eq!(
        run_with(&mut ws, 5, &last)?,
        vec![0, 0xe000],
        "only the decode expert"
    );
    assert_eq!(ring.head(), 8);
    assert_eq!((ring.mark(9, 1), ring.mark(9, 5)), (0, 5));
    ring.clear_mark(9, 5);
    assert_eq!(
        run_with(&mut ws, 6, &DecodeRows::none())?,
        vec![0, 0],
        "a prompt-only launch"
    );
    assert_eq!(ring.head(), 8, "takes nothing");

    // With a reserve of one, prompt-only experts take slots only while more
    // than one is stocked: of the two stocked (0xf000 left from above, and
    // 0x9000), e5 — cold, so walked first — takes the first and e1 is left
    // the reserve.
    ring.set_reserve(1);
    ring.push(&[0x9000]);
    assert_eq!(
        run_with(&mut ws, 7, &DecodeRows::none())?,
        vec![0, 0xf000],
        "above the reserve"
    );
    assert_eq!(ring.head(), 9);
    // A decode expert still takes the reserved slot.
    ring.clear_mark(9, 5);
    assert_eq!(
        run_with(&mut ws, 8, &last)?,
        vec![0, 0x9000],
        "the reserve is decode's"
    );
    assert_eq!(ring.head(), 10);

    // A null reserve address: a prompt-only expert never takes a slot, however
    // much is stocked, while a decode expert still does.
    ring.clear_mark(9, 5);
    ring.push(&[0x1000, 0x2000, 0x3000]);
    let unreserved = PromoRing {
        reserve: 0,
        ..ring.ring()
    };
    assert_eq!(
        run_ring(&mut ws, 9, &DecodeRows::none(), unreserved)?,
        vec![0, 0],
        "no reserve word: prompt-only never"
    );
    assert_eq!(ring.head(), 10);
    assert_eq!(
        run_ring(&mut ws, 10, &last, unreserved)?,
        vec![0, 0x1000],
        "no reserve word: decode still takes one"
    );
    assert_eq!(ring.head(), 11);
    Ok(())
}

/// A `[3][rows][n_experts]` live table with every gate entry in VRAM at
/// `0x10_0000 + (row · n_experts + e) · 0x100` — slot `rows · n_experts - 1 -
/// (row · n_experts + e)` of a zone of `0x100`-byte slots ending at
/// `0x10_0000 + rows · n_experts · 0x100` — except the `(row, e, gate)` in
/// `pinned`; up and down at `+0x10` and `+0x20` of the gate entry.
pub(super) fn rows_live_table(
    rows: usize,
    n_experts: usize,
    pinned: &[(usize, usize, u64)],
) -> Vec<u64> {
    let mut gate: Vec<u64> = (0..rows * n_experts)
        .map(|i| 0x10_0000 + i as u64 * 0x100)
        .collect();
    for &(r, e, g) in pinned {
        gate[r * n_experts + e] = g;
    }
    let mut table = gate.clone();
    table.extend(gate.iter().map(|&g| g + 0x10));
    table.extend(gate.iter().map(|&g| g + 0x20));
    table
}

/// **Read-ahead claims offers for the vetted experts predicted for the rows after
/// the next, within the window its own misses leave.** Five rows of 8 experts;
/// the launch is row 1, routing only e3 (in VRAM), so it has no misses and the
/// whole window of 2 is read-ahead. Depth 3 reads rows 3 and 4 — not row 2,
/// whose prediction (e0, warm) is the next launch's own. Row 3's list is e5 (in
/// VRAM), e4 (in the pad, vetted — the host holds it pinned), e6 (cold), e2
/// (warm, but listed with a source it no longer reads from — a stale vetting):
/// only e4 is read ahead. Row 4's is e1 (warm but marked — a promotion of it is
/// in flight) and e7 (warm, vetted). e4 takes the empty offer `0xa000`; e7
/// finds `0x10_0b00` holding (1, 3), which this launch
/// routes, skips it, and claims `0x10_0500` from (0, 5), whose entries are
/// retargeted first. The budget is spent, so the offer behind it stays. Every
/// word of both items is checked raw, the owner tag address included: `0xa000`
/// lies outside the zone and has none. Then the reserve: with the one offer
/// left standing and a reserve of 1 nothing is read ahead; at 0, e4 takes it.
/// Last, an expert the CPU class takes spends the window as a copy does.
#[test]
pub(super) fn cuda_moe_bucketize_reads_ahead_into_offers_within_the_window() -> Result<()> {
    use candle_kernels::simple::moe_bucketize::{AHEAD_FLAG, AHEAD_ITEM_WORDS, AHEAD_MAX};
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (rows, n) = (5usize, 8usize);
    let warm = TEST_PINNED[0].0;
    let mut table = rows_live_table(
        rows,
        n,
        &[
            (2, 0, warm + 0x1000),
            (3, 2, warm + 0x2000),
            (3, 4, TEST_PINNED[1].0 + 0x4000),
            (4, 1, warm + 0x8000),
            (4, 7, warm + 0x9000),
            (1, 2, warm + 0xa000),
        ],
    );
    let plane = rows * n;
    for p in 0..3 {
        table[p * plane + 3 * n + 6] = 0;
    }
    let table_dev = dev.memcpy_stod(&table)?;
    let snap = dev.memcpy_stod(&[0u64; 24])?;
    let summary = dev.memcpy_stod(&[0u32; 9])?;
    let remote = dev.memcpy_stod(&[0i32; 32])?;
    let counters = dev.memcpy_stod(&[0i32; 3])?;
    let remote_dst = dev.memcpy_stod(&[7u64; 8])?;
    // Row 4's slot images are laid out differently from the rest.
    let mut layout = vec![0u64; rows * 4];
    for r in 0..rows {
        let l: [u64; 4] = if r == 4 {
            [0x80, 0x880, 0x1080, 0x1800]
        } else {
            [0x40, 0x1040, 0x2040, 0x3000]
        };
        layout[4 * r..4 * r + 4].copy_from_slice(&l);
    }
    let row_layout = dev.memcpy_stod(&layout)?;
    let items = dev.memcpy_stod(&vec![u64::MAX; 1 + AHEAD_MAX * AHEAD_ITEM_WORDS])?;
    let done = dev.memcpy_stod(&vec![u32::MAX; AHEAD_MAX])?;
    let (owners_h, owners_d) = mapped_host(rows * n * 4);
    for i in 0..rows * n {
        let (r, e) = (i / n, i % n);
        unsafe {
            std::ptr::write_volatile(
                (owners_h as *mut u32).add(rows * n - 1 - i),
                ((r as u32 + 1) << 16) | e as u32,
            );
        }
    }
    let zone_end = 0x10_0000 + (rows * n) as u64 * 0x100;
    let owner = OwnerCheck {
        owners: owners_d,
        zone_end,
        slot_bytes: 0x100,
        slots: (rows * n) as u32,
    };
    let ring = TestRing::new(8, rows, n);
    ring.start_at(5);
    let retarget = |base: u64| [base, base + 0x10, base + 0x20];
    ring.offer(&[
        (0xa000, None),
        (0x10_0b00, Some(((n + 3) as u64, retarget(0x9000_0300)))),
        (0x10_0500, Some((5u64, retarget(warm + 0x5500)))),
        (0xf000, None),
    ]);
    // Vetted sources are slot image bases: the gate entry less the row's gate
    // offset (0x40, or 0x80 on row 4).
    let pad_e4 = TEST_PINNED[1].0 + 0x4000;
    ring.predict(2, &[(0, warm + 0x1000 - 0x40)]);
    ring.predict(
        3,
        &[
            (5, 0x10_0000 + (3 * n + 5) as u64 * 0x100 - 0x40),
            (4, pad_e4 - 0x40),
            (6, 0),
            (2, warm + 0x7700),
        ],
    );
    ring.predict(4, &[(1, warm + 0x8000 - 0x80), (7, warm + 0x9000 - 0x80)]);
    unsafe { std::ptr::write_volatile(ring.mark_ptr(4, 1), 99) };
    ring.set_read_ahead(2, 3);
    // Nothing held back for the next rows' decode misses: the whole stock is
    // the walk's to spend.
    ring.set_reserve(0);
    let stream = dev.cuda_stream();
    let work = dev.memcpy_stod(&vec![0u8; CPU_WORK_SLOT_BYTES])?;
    let dst_row = dev.memcpy_stod(&[u32::MAX; 2])?;
    let carry = dev.memcpy_stod(&[0u32; CPU_CARRY_WORDS])?;
    // Row 1 routes e3 (in VRAM) twice, or e3 and the warm e2; `cpu` the CPU
    // class's split, if on; `predict` whether the predictions are read ahead.
    let launch_with =
        |seq: u32, route_e2: bool, cpu: Option<CpuSplit>, predict: bool| -> Result<()> {
            let ids = if route_e2 {
                vec![3u32, 2]
            } else {
                vec![3u32, 3]
            };
            let t = crate::Tensor::from_vec(ids, (1, 2), &device)?;
            let mut ws = MoeBucketizeWorkspace::new(&dev, 1, 2)?;
            let live = BucketizeLive {
                gate_row: table_dev.device_ptr(&stream).0 + (n * 8) as u64,
                table_plane: plane as i64,
                snap: snap.device_ptr(&stream).0,
                pinned: TEST_PINNED,
                summary: summary.device_ptr(&stream).0,
                summary_seq: seq,
                remote: remote.device_ptr(&stream).0,
                counters: counters.device_ptr(&stream).0,
                row: 1,
                promo: Some(ring.ring()),
                remote_dst: remote_dst.device_ptr(&stream).0,
                started_rows: 0,
                ticket: 0,
                owner,
                ahead: Some(ReadAhead {
                    row_layout: row_layout.device_ptr(&stream).0,
                    rows: rows as i32,
                    items: items.device_ptr(&stream).0,
                    done: done.device_ptr(&stream).0,
                    predict,
                }),
                cpu: cpu.map(|split| BucketizeCpu {
                    split,
                    work: work.device_ptr(&stream).0,
                    dst_row: dst_row.device_ptr(&stream).0,
                    carry: carry.device_ptr(&stream).0,
                    tok_first: 0,
                    in_tok: 0,
                    input_map: 0,
                }),
            };
            moe_bucketize(&t, n, 2, &mut ws, Some(&live), &DecodeRows::prefix(1))
        };
    let launch = |seq: u32| launch_with(seq, false, None, true);

    launch(7)?;
    let t0 = table_dev.device_ptr(&stream).0;
    let entry = |p: usize, r: usize, e: usize| t0 + ((p * plane + r * n + e) * 8) as u64;
    let got = dev.memcpy_dtov(&items.slice(..1 + 2 * AHEAD_ITEM_WORDS))?;
    let (e4_base, e7_base) = (pad_e4 - 0x40, warm + 0x9000 - 0x80);
    let e7_slot = (zone_end - 1 - 0x10_0500) / 0x100;
    assert_eq!(
        got,
        vec![
            2,
            // (3, 4), from its pinned pad slot → the empty offer, outside the
            // zone.
            e4_base,
            0xa000,
            0x3000,
            entry(0, 3, 4),
            entry(1, 3, 4),
            entry(2, 3, 4),
            0xa000 + 0x40,
            0xa000 + 0x1040,
            0xa000 + 0x2040,
            0,
            (4 << 16) | 4,
            // (4, 7) → (0, 5)'s slot, in row 4's layout.
            e7_base,
            0x10_0500,
            0x1800,
            entry(0, 4, 7),
            entry(1, 4, 7),
            entry(2, 4, 7),
            0x10_0500 + 0x80,
            0x10_0500 + 0x880,
            0x10_0500 + 0x1080,
            owners_d + e7_slot * 4,
            (5 << 16) | 7,
        ]
    );
    assert_eq!(
        dev.memcpy_dtov(&done.slice(..3))?,
        vec![0, 0, u32::MAX],
        "zeroed per item"
    );
    assert_eq!(
        ring.head(),
        8,
        "two claims and a skip; the last offer stands"
    );
    let logged = |r: u64, x: u64| (7u64 << 32) | (r << 16) | x;
    assert_eq!(
        (ring.log(5), ring.log(6), ring.log(7)),
        (
            logged(3, AHEAD_FLAG as u64 | 4),
            logged(1, 0xffff),
            logged(4, AHEAD_FLAG as u64 | 7)
        )
    );
    assert_eq!(
        (
            ring.mark(3, 4),
            ring.mark(4, 7),
            ring.mark(4, 1),
            ring.mark(3, 2)
        ),
        (7, 7, 99, 0),
        "the stale vetting is not read"
    );
    let after = dev.memcpy_dtov(&table_dev)?;
    assert_eq!(
        (0..3).map(|p| after[p * plane + 5]).collect::<Vec<_>>(),
        retarget(warm + 0x5500).to_vec(),
        "the claimed victim, retargeted"
    );
    assert_eq!(
        after[n + 3],
        0x10_0000 + (n + 3) as u64 * 0x100,
        "the skipped victim untouched"
    );
    assert_eq!(
        dev.memcpy_dtov(&remote_dst.slice(..1))?,
        vec![7],
        "no misses, so no demand slots written"
    );

    // A window no wider than the launch's misses — here none, but a window of
    // 0 — reads nothing ahead, and the count says so.
    ring.clear_mark(3, 4);
    ring.set_read_ahead(0, 3);
    launch(8)?;
    assert_eq!(dev.memcpy_dtov(&items.slice(..1))?, vec![0]);
    assert_eq!(ring.head(), 8, "nothing taken");

    // The reserve is the next rows' decode misses': read-ahead spends only the
    // stock above it. One offer stands (`0xf000`) and e4 is listed again.
    ring.set_read_ahead(2, 3);
    ring.set_reserve(1);
    launch(9)?;
    assert_eq!(dev.memcpy_dtov(&items.slice(..1))?, vec![0]);
    assert_eq!(ring.head(), 8, "the last offer is the reserve's");
    ring.set_reserve(0);
    launch(10)?;
    assert_eq!(
        dev.memcpy_dtov(&items.slice(..3))?,
        vec![1, e4_base, 0xf000]
    );
    assert_eq!(
        ring.head(),
        9,
        "above a reserve of 0, the offer is read-ahead's"
    );
    // Ring index 8 wraps to log slot 0 of the 8.
    assert_eq!(
        ring.log(0),
        (10u64 << 32) | (3 << 16) | AHEAD_FLAG as u64 | 4
    );

    // An expert the CPU class takes is charged against the window as a copy
    // is. Row 1 now also routes the warm e2, which the split gives the pool:
    // no remote expert, one CPU expert, so of the window of 2 one read-ahead
    // claim is left — e4, not e4 and e7 — and the second offer stands.
    ring.clear_mark(3, 4);
    ring.clear_mark(4, 7);
    ring.offer(&[(0xb000, None), (0xc000, None)]);
    launch_with(11, true, Some(test_cpu_split(0.0)), true)?;
    assert_eq!(
        dev.memcpy_dtov(&summary.slice(2..4))?
            .iter()
            .map(|w| w >> SUMMARY_CPU_BIT & 1)
            .collect::<Vec<_>>(),
        vec![1, 0],
        "e2 on the CPU, e3 not"
    );
    assert_eq!(
        dev.memcpy_dtov(&items.slice(..3))?,
        vec![1, e4_base, 0xb000]
    );
    assert_eq!(ring.head(), 10, "one claim; the second offer stands");
    assert_eq!(
        dev.memcpy_dtov(&carry.slice(..2))?,
        vec![11, 0],
        "promotion off: the carry is written empty"
    );

    // THE CARRY. With promotion on, launch 12 — the predictions not read
    // ahead — leaves the CPU's e2 for the next launch, and promotes nothing
    // itself.
    let mut promote = test_cpu_split(0.0);
    promote.promote = CPU_PROMOTE_ALL;
    launch_with(12, true, Some(promote), false)?;
    assert_eq!(dev.memcpy_dtov(&items.slice(..1))?, vec![0]);
    assert_eq!(dev.memcpy_dtov(&carry.slice(..4))?, vec![12, 1, 1, 2]);
    // Launch 13 routes only e3 and takes nothing for the CPU, but promotes
    // the carried (1, 2) from its warm image into the standing offer, under
    // row 1's layout.
    let mut off = CpuSplit::OFF;
    off.promote = CPU_PROMOTE_ALL;
    launch_with(13, false, Some(off), false)?;
    let e2_base = warm + 0xa000 - 0x40;
    assert_eq!(
        dev.memcpy_dtov(&items.slice(..1 + AHEAD_ITEM_WORDS))?,
        vec![
            1,
            e2_base,
            0xc000,
            0x3000,
            entry(0, 1, 2),
            entry(1, 1, 2),
            entry(2, 1, 2),
            0xc000 + 0x40,
            0xc000 + 0x1040,
            0xc000 + 0x2040,
            0,
            (2 << 16) | 2,
        ]
    );
    assert_eq!(ring.head(), 11);
    assert_eq!(
        ring.log(10 % 8),
        (13u64 << 32) | (1 << 16) | (AHEAD_FLAG | CARRY_FLAG) as u64 | 2,
        "the carry's claim names the expert's own row"
    );
    assert_eq!(ring.mark(1, 2), 13);
    assert_eq!(dev.memcpy_dtov(&carry.slice(..2))?, vec![13, 0]);
    // A carry is the next launch's only: launch 14 leaves one, and launch 16
    // finds it stale and promotes nothing, though an offer stands.
    ring.clear_mark(1, 2);
    launch_with(14, true, Some(promote), false)?;
    assert_eq!(dev.memcpy_dtov(&carry.slice(..4))?, vec![14, 1, 1, 2]);
    ring.offer(&[(0xd000, None)]);
    launch_with(16, false, Some(off), false)?;
    assert_eq!(dev.memcpy_dtov(&items.slice(..1))?, vec![0]);
    assert_eq!(ring.head(), 11, "a stale carry claims nothing");
    unsafe { cudarc::driver::sys::cuMemFreeHost(owners_h) };
    Ok(())
}

/// **A miss claims the victim behind an offer, retargeting its entries before
/// the slot is written, and skips a victim its own launch routes.** Row 1 of
/// three, 8 experts; e1 and e5 pinned, e6 in VRAM. Three offers: `0xd000`
/// holding (2, 3) — another row, claimable — `0xe000` holding (1, 6) — this
/// launch routes e6, so skipped — and `0xf000` holding (1, 7), claimable. e1
/// takes `0xd000`, e5 skips `0xe000` and takes `0xf000`; (2, 3) and (1, 7) read
/// their retarget values in all three planes and (1, 6) is untouched. A later
/// launch of row 2 then finds (2, 3) pinned. The owner check runs throughout,
/// over tags matching the table, and passes.
#[test]
pub(super) fn cuda_moe_bucketize_claims_victims_and_skips_its_own_routed_ones() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (rows, n) = (3usize, 8usize);
    let (e1_gate, e5_gate) = (TEST_PINNED[1].0 + 0x100, TEST_PINNED[1].0 + 0x500);
    let table = rows_live_table(rows, n, &[(1, 1, e1_gate), (1, 5, e5_gate)]);
    let table_dev = dev.memcpy_stod(&table)?;
    let plane = (rows * n) as i64;
    let snap = dev.memcpy_stod(&[0u64; 24])?;
    let summary = dev.memcpy_stod(&[0u32; 9])?;
    let remote = dev.memcpy_stod(&[0i32; 32])?;
    let counters = dev.memcpy_stod(&[0i32; 3])?;
    let remote_dst = dev.memcpy_stod(&[7u64; 8])?;
    // Every VRAM slot tagged with the expert the table says is there.
    let (owners_h, owners_d) = mapped_host(rows * n * 4);
    for i in 0..rows * n {
        let (r, e) = (i / n, i % n);
        unsafe {
            std::ptr::write_volatile(
                (owners_h as *mut u32).add(rows * n - 1 - i),
                ((r as u32 + 1) << 16) | e as u32,
            );
        }
    }
    let owner = OwnerCheck {
        owners: owners_d,
        zone_end: 0x10_0000 + (rows * n) as u64 * 0x100,
        slot_bytes: 0x100,
        slots: (rows * n) as u32,
    };
    let ring = TestRing::new(4, rows, n);
    let retarget = |base: u64| [base, base + 0x10, base + 0x20];
    let (r23, r17) = (retarget(0x9000_0300), retarget(0x9000_0700));
    ring.offer(&[
        (0xd000, Some(((2 * n + 3) as u64, r23))),
        (0xe000, Some(((n + 6) as u64, retarget(0x9000_0600)))),
        (0xf000, Some(((n + 7) as u64, r17))),
    ]);
    let stream = dev.cuda_stream();
    let launch = |row: usize, ids: &[u32], seq: u32| -> Result<()> {
        let t = crate::Tensor::from_vec(ids.to_vec(), (ids.len() / 2, 2), &device)?;
        let mut ws = MoeBucketizeWorkspace::new(&dev, ids.len() / 2, 2)?;
        let live = BucketizeLive {
            gate_row: table_dev.device_ptr(&stream).0 + (row * n * 8) as u64,
            table_plane: plane,
            snap: snap.device_ptr(&stream).0,
            pinned: TEST_PINNED,
            summary: summary.device_ptr(&stream).0,
            summary_seq: seq,
            remote: remote.device_ptr(&stream).0,
            counters: counters.device_ptr(&stream).0,
            row: row as i32,
            promo: Some(ring.ring()),
            remote_dst: remote_dst.device_ptr(&stream).0,
            started_rows: 0,
            ticket: 0,
            owner,
            ahead: None,
            cpu: None,
        };
        moe_bucketize(
            &t,
            n,
            2,
            &mut ws,
            Some(&live),
            &DecodeRows::prefix(ids.len() / 2),
        )
    };

    launch(1, &[5, 1, 1, 6, 5, 6], 7)?;
    assert_eq!(
        dev.memcpy_dtov(&remote_dst.slice(..2))?,
        vec![0xd000, 0xf000]
    );
    assert_eq!(ring.head(), 3, "two claims and a skip");
    let logged = |e: u64| (7u64 << 32) | (1 << 16) | e;
    assert_eq!(
        (ring.log(0), ring.log(1), ring.log(2)),
        (logged(1), logged(PROMO_SKIP as u64), logged(5))
    );
    assert_eq!((ring.mark(1, 1), ring.mark(1, 5)), (7, 7));
    let after = dev.memcpy_dtov(&table_dev)?;
    let mut want = table.clone();
    for (p, &v) in r23.iter().enumerate() {
        want[p * rows * n + 2 * n + 3] = v;
    }
    for (p, &v) in r17.iter().enumerate() {
        want[p * rows * n + n + 7] = v;
    }
    assert_eq!(
        after, want,
        "only the two claimed victims retargeted, in all three planes"
    );

    // Row 2 now routes its displaced expert: pinned, from the entry it reads.
    ring.clear_mark(1, 1);
    ring.clear_mark(1, 5);
    launch(2, &[3, 0], 8)?;
    let s = dev.memcpy_dtov(&summary)?;
    assert_eq!(s[3] & (1 << 29), 1 << 29, "(2, 3) is pinned");
    assert_eq!(s[0] & (3 << 29), 0, "(2, 0) is still in VRAM");
    assert_eq!(
        dev.memcpy_dtov(&remote_dst.slice(..1))?,
        vec![0],
        "nothing left to claim"
    );
    unsafe {
        cuMemFreeHost(owners_h);
    }
    Ok(())
}

/// **A prompt-only expert takes only an empty offer.** One prompt-only remote
/// expert (e1, no decode row) with the reserve at 0: in front of it a victim
/// offer, so it takes nothing and `head` stays; once an empty offer stands in
/// front, it takes that. A prompt never evicts.
#[test]
pub(super) fn cuda_moe_bucketize_a_prompt_only_expert_takes_only_an_empty_offer() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let mut class = vec![CLS_VRAM; 8];
    class[1] = CLS_PINNED;
    let t = crate::Tensor::from_vec(vec![1u32, 3], (1, 2), &device)?;
    let table_dev = dev.memcpy_stod(&test_live_table(&class, &[]))?;
    let snap = dev.memcpy_stod(&[0u64; 24])?;
    let remote = dev.memcpy_stod(&[0i32; 32])?;
    let remote_dst = dev.memcpy_stod(&[7u64; 8])?;
    let ring = TestRing::new(4, 1, 8);
    ring.set_reserve(0);
    ring.offer(&[(0xa000, Some((6, [0x9000_0600, 0x9000_0610, 0x9000_0620])))]);
    let mut ws = MoeBucketizeWorkspace::new(&dev, 1, 2)?;
    let stream = dev.cuda_stream();
    let mut run = |seq: u32| -> Result<u64> {
        let live = BucketizeLive {
            gate_row: table_dev.device_ptr(&stream).0,
            table_plane: 8,
            snap: snap.device_ptr(&stream).0,
            pinned: TEST_PINNED,
            summary: 0,
            summary_seq: seq,
            remote: remote.device_ptr(&stream).0,
            counters: 0,
            row: 0,
            promo: Some(ring.ring()),
            remote_dst: remote_dst.device_ptr(&stream).0,
            started_rows: 0,
            ticket: 0,
            owner: OwnerCheck::default(),
            ahead: None,
            cpu: None,
        };
        moe_bucketize(&t, 8, 2, &mut ws, Some(&live), &DecodeRows::none())?;
        Ok(dev.memcpy_dtov(&remote_dst.slice(..1))?[0])
    };
    assert_eq!(run(1)?, 0, "a victim in front: the prompt evicts nothing");
    assert_eq!(ring.head(), 0);
    assert_eq!(
        dev.memcpy_dtov(&table_dev)?[6],
        0x10_0000 + 6 * 0x100,
        "the victim's entry is untouched"
    );

    // A fresh ring with an empty offer in front.
    let ring2 = TestRing::new(4, 1, 8);
    ring2.set_reserve(0);
    ring2.offer(&[(0xb000, None), (0xa000, Some((6, [0; 3])))]);
    let live = BucketizeLive {
        gate_row: table_dev.device_ptr(&stream).0,
        table_plane: 8,
        snap: snap.device_ptr(&stream).0,
        pinned: TEST_PINNED,
        summary: 0,
        summary_seq: 2,
        remote: remote.device_ptr(&stream).0,
        counters: 0,
        row: 0,
        promo: Some(ring2.ring()),
        remote_dst: remote_dst.device_ptr(&stream).0,
        started_rows: 0,
        ticket: 0,
        owner: OwnerCheck::default(),
        ahead: None,
        cpu: None,
    };
    moe_bucketize(&t, 8, 2, &mut ws, Some(&live), &DecodeRows::none())?;
    assert_eq!(
        dev.memcpy_dtov(&remote_dst.slice(..1))?[0],
        0xb000,
        "an empty offer it takes"
    );
    assert_eq!(ring2.head(), 1, "and only that one");
    Ok(())
}

/// **A sweep claims nothing.** Two decode-scored remote experts against four
/// empty offers: with the sweep word at 1 the launch has more claiming experts
/// than it allows, so neither takes a slot and `head` stays; at 2 both do.
#[test]
pub(super) fn cuda_moe_bucketize_claims_nothing_on_a_sweep() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let mut class = vec![CLS_VRAM; 8];
    class[1] = CLS_PINNED;
    class[5] = CLS_COLD;
    let t = crate::Tensor::from_vec(vec![5u32, 1, 1, 3, 5, 6], (3, 2), &device)?;
    let table_dev = dev.memcpy_stod(&test_live_table(&class, &[]))?;
    let snap = dev.memcpy_stod(&[0u64; 24])?;
    let remote = dev.memcpy_stod(&[0i32; 32])?;
    let remote_dst = dev.memcpy_stod(&[7u64; 8])?;
    let ring = TestRing::new(4, 1, 8);
    ring.push(&[0xa000, 0xb000, 0xc000, 0xd000]);
    let mut ws = MoeBucketizeWorkspace::new(&dev, 3, 2)?;
    let stream = dev.cuda_stream();
    let mut run = |seq: u32| -> Result<Vec<u64>> {
        let live = BucketizeLive {
            gate_row: table_dev.device_ptr(&stream).0,
            table_plane: 8,
            snap: snap.device_ptr(&stream).0,
            pinned: TEST_PINNED,
            summary: 0,
            summary_seq: seq,
            remote: remote.device_ptr(&stream).0,
            counters: 0,
            row: 0,
            promo: Some(ring.ring()),
            remote_dst: remote_dst.device_ptr(&stream).0,
            started_rows: 0,
            ticket: 0,
            owner: OwnerCheck::default(),
            ahead: None,
            cpu: None,
        };
        moe_bucketize(&t, 8, 2, &mut ws, Some(&live), &DecodeRows::prefix(3))?;
        Ok(dev.memcpy_dtov(&remote_dst.slice(..2))?)
    };
    ring.set_sweep(1);
    assert_eq!(
        run(1)?,
        vec![0, 0],
        "two claiming experts over a sweep word of 1"
    );
    assert_eq!(ring.head(), 0, "a sweep takes nothing");
    ring.set_sweep(2);
    // Listed e1 then e5; walked cold first, so e5 takes the first offer.
    assert_eq!(run(2)?, vec![0xb000, 0xa000], "within it, both claim");
    assert_eq!(ring.head(), 2);
    Ok(())
}

/// **Bucketize marks its row begun with its ticket**, in the row's word and no
/// other, so the host's reclaim rule can key on the invocation the device is
/// inside. Raw expected words after two invocations of row 2 and one of row 0.
#[test]
pub(super) fn cuda_moe_bucketize_stores_its_ticket_in_its_rows_started_word() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let t = crate::Tensor::from_vec(vec![1u32, 3, 3, 5], (2, 2), &device)?;
    let table_dev = dev.memcpy_stod(&test_live_table(&[CLS_VRAM; 8], &[]))?;
    let snap = dev.memcpy_stod(&[0u64; 24])?;
    let started = dev.memcpy_stod(&[0u64; 4])?;
    let mut ws = MoeBucketizeWorkspace::new(&dev, 2, 2)?;
    let stream = dev.cuda_stream();
    let mut run = |row: i32, ticket: u64| -> Result<()> {
        let live = BucketizeLive {
            gate_row: table_dev.device_ptr(&stream).0,
            table_plane: 8,
            snap: snap.device_ptr(&stream).0,
            pinned: TEST_PINNED,
            summary: 0,
            summary_seq: 0,
            remote: 0,
            counters: 0,
            row,
            promo: None,
            remote_dst: 0,
            started_rows: started.device_ptr(&stream).0,
            ticket,
            owner: OwnerCheck::default(),
            ahead: None,
            cpu: None,
        };
        moe_bucketize(&t, 8, 2, &mut ws, Some(&live), &DecodeRows::prefix(2))
    };
    run(2, 7)?;
    run(0, 8)?;
    run(2, 11)?;
    assert_eq!(dev.memcpy_dtov(&started)?, vec![8, 0, 11, 0]);

    // With no live table the kernel reads no entries, so it has nothing to
    // order the store against: the started word is left as it was.
    let live = BucketizeLive {
        gate_row: 0,
        table_plane: 0,
        snap: 0,
        pinned: TEST_PINNED,
        summary: 0,
        summary_seq: 0,
        remote: 0,
        counters: 0,
        row: 1,
        promo: None,
        remote_dst: 0,
        started_rows: started.device_ptr(&stream).0,
        ticket: 13,
        owner: OwnerCheck::default(),
        ahead: None,
        cpu: None,
    };
    moe_bucketize(&t, 8, 2, &mut ws, Some(&live), &DecodeRows::prefix(2))?;
    assert_eq!(dev.memcpy_dtov(&started)?, vec![8, 0, 11, 0]);
    Ok(())
}

/// Over a live table, bucketize puts the remote experts' tiles first — pinned,
/// then cold — then the VRAM experts', lists the remote experts, and writes the
/// routing summary, while the ROW layout — which decides the scatter's
/// summation order — stays ascending by expert id.
///
/// Raw expected values for a case small enough to check by hand: 3 tokens,
/// top-2, 8 experts, 2-token tiles, routing `[5,1 | 1,3 | 5,6]`, expert 1
/// pinned, expert 5 cold, the rest in VRAM, token 0 the only decode row.
#[test]
pub(super) fn cuda_moe_bucketize_live_table_orders_remote_first() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let dev = CudaDevice::new(0)?;
    let ids = [5u32, 1, 1, 3, 5, 6];
    let mut class = vec![CLS_VRAM; 8];
    class[1] = CLS_PINNED;
    class[5] = CLS_COLD;
    let got = run_bucketize_live(&dev, &ids, 3, 2, 8, 2, &class, &[], &DecodeRows::prefix(1))?;

    // Pinned e1, cold e5, then VRAM e3 and e6; two remote experts own two tiles.
    assert_eq!(got.header, [4, 6, 4, 2, 2], "header");
    assert_eq!(got.remote, vec![[1, 0, 1, 0], [5, 1, 1, 1]], "remote list");
    assert_eq!(got.tile_expert, vec![1, 5, 3, 6, 0, 0], "tile_expert");
    assert_eq!(got.tile_b_start, vec![0, 3, 2, 5, 0, 0], "tile_b_start");
    assert_eq!(got.tile_b_cnt, vec![2, 2, 1, 1, 0, 0], "tile_b_cnt");
    // Rows ascend by expert id: e1 {i1, i2}, e3 {i3}, e5 {i0, i4}, e6 {i5}.
    assert_eq!(got.tok_ids, vec![0, 1, 1, 0, 2, 2], "tok_ids");
    assert_eq!(got.weight_ids, vec![1, 2, 3, 0, 4, 5], "weight_ids");
    // Token-major, each token's pairs by ascending row: t0 {(0,i1),(3,i0)},
    // t1 {(1,i2),(2,i3)}, t2 {(4,i4),(5,i5)}.
    assert_eq!(got.perm, vec![0, 3, 1, 2, 4, 5], "perm");
    assert_eq!(got.rw_ids, vec![1, 0, 2, 3, 4, 5], "rw_ids");
    assert_eq!(got.token_starts, vec![0, 2, 4, 6], "token_starts");
    // Count, bit 29 pinned, bit 30 cold, bit 31 for the experts token 0 (the
    // decode row) routed to.
    let (p, c, d) = (1u32 << 29, 1u32 << 30, 1u32 << 31);
    assert_eq!(
        got.summary,
        vec![0, 2 | p | d, 0, 1, 0, 2 | c | d, 1, 0],
        "summary"
    );

    // The decode rows are any set of ranges, not only a prefix: token 2 alone
    // (the last token of a prompt, say) marks e5 and e6, and nothing else
    // moves.
    let mut last = DecodeRows::none();
    last.push(2, 3);
    let got = run_bucketize_live(&dev, &ids, 3, 2, 8, 2, &class, &[], &last)?;
    assert_eq!(
        got.summary,
        vec![0, 2 | p, 0, 1, 0, 2 | c | d, 1 | d, 0],
        "summary, token 2 decode"
    );
    assert_eq!(
        got.tile_expert,
        vec![1, 5, 3, 6, 0, 0],
        "tile_expert, token 2 decode"
    );

    // An expert whose gate entry names VRAM but whose down entry is 0 is cold:
    // a host retarget caught half-way must never reach a block as an address.
    let got = run_bucketize_live(&dev, &ids, 3, 2, 8, 2, &class, &[3], &DecodeRows::prefix(1))?;
    assert_eq!(got.header, [4, 6, 4, 3, 3], "header, e3 cold");
    assert_eq!(
        got.tile_b_start,
        vec![0, 2, 3, 5, 0, 0],
        "tile_b_start, e3 cold"
    );
    assert_eq!(
        got.remote,
        vec![[1, 0, 1, 0], [3, 1, 1, 1], [5, 2, 1, 1]],
        "remote list, e3 cold"
    );
    assert_eq!(
        got.tile_expert,
        vec![1, 3, 5, 6, 0, 0],
        "tile_expert, e3 cold"
    );
    assert_eq!(
        got.summary,
        vec![0, 2 | p | d, 0, 1 | c, 0, 2 | c | d, 1, 0],
        "summary, e3 cold"
    );

    // Against the reference across shapes, class mixes and decode splits — the
    // row tables must not move with the classes.
    let mut rng = StdRng::seed_from_u64(0x11fe_7ab1);
    for &(n_tokens, k, n_experts) in &[
        (1usize, 8usize, 128usize),
        (64, 8, 128),
        (333, 8, 256),
        (16, 10, 512),
        (713, 10, 512),
    ] {
        for &dense in &[0u32, 1, 2, 4] {
            let ids: Vec<u32> = (0..n_tokens * k)
                .map(|_| {
                    if rng.random_ratio(1, 50) {
                        n_experts as u32
                    } else {
                        rng.random_range(0..n_experts as u32)
                    }
                })
                .collect();
            // dense 0: nothing in VRAM; else 1 in `dense` remote, split
            // between pinned and cold.
            let class: Vec<u8> = (0..n_experts)
                .map(|_| {
                    if dense != 0 && !rng.random_ratio(1, dense) {
                        CLS_VRAM
                    } else if rng.random_ratio(1, 2) {
                        CLS_PINNED
                    } else {
                        CLS_COLD
                    }
                })
                .collect();
            let decode_tokens = rng.random_range(0..=n_tokens);
            let want = bucketize_ref_live(&ids, n_tokens, k, n_experts, 32, &class, decode_tokens);
            let got = run_bucketize_live(
                &dev,
                &ids,
                n_tokens,
                k,
                n_experts,
                32,
                &class,
                &[],
                &DecodeRows::prefix(decode_tokens),
            )?;
            let label = format!("{n_tokens}x{k}-e{n_experts}-dense{dense}");
            assert_eq!(got.header, want.header, "{label}: header");
            assert_eq!(got.remote, want.remote, "{label}: remote list");
            assert_eq!(got.tile_expert, want.tile_expert, "{label}: tile_expert");
            assert_eq!(got.tile_b_start, want.tile_b_start, "{label}: tile_b_start");
            assert_eq!(got.tile_b_cnt, want.tile_b_cnt, "{label}: tile_b_cnt");
            assert_eq!(got.tok_ids, want.tok_ids, "{label}: tok_ids");
            assert_eq!(got.weight_ids, want.weight_ids, "{label}: weight_ids");
            assert_eq!(got.perm, want.perm, "{label}: perm");
            assert_eq!(got.rw_ids, want.rw_ids, "{label}: rw_ids");
            assert_eq!(got.token_starts, want.token_starts, "{label}: token_starts");
            assert_eq!(got.summary, want.summary, "{label}: summary");
            // The row tables are those of the all-VRAM order.
            let base = bucketize_ref(&ids, n_tokens, k, n_experts, 32);
            assert_eq!(got.perm, base.perm, "{label}: perm moved with the classes");
            assert_eq!(
                got.tok_ids, base.tok_ids,
                "{label}: rows moved with the classes"
            );
        }
    }
    Ok(())
}
