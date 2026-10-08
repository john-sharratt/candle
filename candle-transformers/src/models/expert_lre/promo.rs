//! The promotion ring — VRAM slots the GPU fills with the experts it misses.
//!
//! A routed expert that is not in VRAM is copied over the link by its layer's
//! GEMM workers anyway (`dispatch`), slice by slice. If the slices land in a
//! weight-zone slot, the expert is in VRAM once the layer is done — a promotion
//! that costs no crossing of the link of its own. This ring is how the slots
//! get there:
//!
//! - the pipeline thread offers slots at `tail`: empty ones, and **lazy
//!   victims** — slots whose expert is still resident and hittable, named with
//!   the entries it is retargeted to when the slot is claimed;
//! - `moe_bucketize` gives each remote expert the next offer from `head`; a
//!   victim is evicted only then, on the device, by retargeting its entries
//!   before the slot is written. It logs `summary_word << 32 | row << 16 |
//!   expert` at the offer's index, or the expert `PROMO_SKIP` for a victim its
//!   own launch routes, which it passes over;
//! - the pipeline thread collects each logged offer — a claimed victim's
//!   eviction at once, the promotion once the invocation that took it has
//!   completed (its ticket is below the observed one) — and takes a skipped
//!   victim's offer back.
//!
//! The ring also carries **read-ahead**: the host's predictions for each row
//! (`predict`), each with the slot image it vetted as the expert's source, the
//! layer window the link has for them (`set_window`) and how many rows ahead to
//! read (`set_depth`). A launch at row `r` claims offers for
//! the predicted warm experts of rows `r + 2 ..= r + depth` with what its own
//! misses leave of the window, and its gate launch's workers copy and publish
//! them (`moe_bucketize.cu`, READ-AHEAD); the log names such a claim's target
//! row and sets `AHEAD_FLAG` in its expert field.
//!
//! In mapped pinned memory: `u64 slots[cap] | u64 log[cap] | u32 head | u32
//! tail | u32 marks[rows][n_experts] | u32 reserve | u32 sweep | u64
//! victims[cap] | u64 retarget[cap][3] | u32 window | u32 depth | u32
//! ahead_n[rows] | u32 ahead_list[rows][AHEAD_CAP] | u64
//! ahead_src[rows][AHEAD_CAP]` (the victims and the sources 8-aligned).
//! `tail`, `reserve`, `sweep`, the victims, the retarget entries and the
//! read-ahead words are written only by the host, `head` only by the device. `reserve` is the stock kept for
//! decode-scored experts: a prompt-only expert takes a slot only while more
//! than it is stocked. `sweep` is the most claiming experts a launch may have
//! and still claim: past it the launch is a prompt passing over the table and
//! claims nothing (`moe_bucketize.cu`, PROMOTION). An expert's mark is set by
//! the device when it gives the expert a slot and cleared by the host when the
//! promotion lands (or is dropped): until then a later invocation that still
//! finds the expert remote — a prefill visiting the row again before the host
//! has caught up — does not spend a second slot on it.

use candle::quantized::cuda::PromoRing;
use candle::Result;
use candle_kernels::simple::moe_bucketize::{AHEAD_FLAG, PROMO_EMPTY, PROMO_SKIP};
use cudarc::driver::sys;
use std::sync::atomic::{fence, Ordering};

/// Predictions the ring holds per row for read-ahead.
pub(crate) const AHEAD_CAP: usize = 32;

pub(crate) struct PromotionRing {
    host: *mut u8,
    dev: u64,
    cap: usize,
    n_experts: usize,
    rows: usize,
    /// Byte offset of the `reserve` word, after the marks; `sweep` follows it.
    reserve_at: usize,
    /// Byte offset of the victims, 8-aligned; the retarget entries follow.
    victims_at: usize,
    /// Byte offset of the read-ahead words: window, depth, then the lists.
    ahead_at: usize,
    /// Byte offset of the lists' vetted sources, 8-aligned.
    src_at: usize,
}

/// The resident expert behind an offer: its index `row · n_experts + expert`
/// in the gate plane, and the entries (gate, up, down) it is retargeted to
/// when a miss claims its slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Victim {
    pub(crate) index: u64,
    pub(crate) retarget: [u64; 3],
}

// SAFETY: `tail` and the slots are written only by the pipeline thread (the
// one owner of the host side), `head` and the log only by the device; every
// access is volatile and fenced.
unsafe impl Send for PromotionRing {}
unsafe impl Sync for PromotionRing {}

/// One logged promotion: which invocation's word, which row and expert, and
/// whether it was read ahead — claimed for a later row's predicted expert
/// rather than for a miss of the launch's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Logged {
    pub(crate) word: u32,
    pub(crate) row: usize,
    pub(crate) expert: usize,
    pub(crate) ahead: bool,
}

impl Logged {
    fn decode(v: u64) -> Self {
        let x = (v & 0xffff) as u32;
        let ahead = x != PROMO_SKIP && x & AHEAD_FLAG != 0;
        Self {
            word: (v >> 32) as u32,
            row: ((v >> 16) & 0xffff) as usize,
            expert: if ahead { x & !AHEAD_FLAG } else { x } as usize,
            ahead,
        }
    }

    /// A victim the launch routed and passed over: the offer comes back.
    pub(crate) fn skipped(&self) -> bool {
        self.expert == PROMO_SKIP as usize
    }
}

/// The ticket whose summary word is `word`, nearest `near`. Words are the
/// tickets' low 32 bits, and a logged slot is never more than a ring's worth of
/// invocations from the one being served.
pub(crate) fn ticket_from_word(word: u32, near: u64) -> u64 {
    let base = near & !0xffff_ffff;
    let candidates = [base.wrapping_sub(1 << 32), base, base + (1 << 32)];
    candidates
        .into_iter()
        .map(|b| b | word as u64)
        .min_by_key(|&t| t.abs_diff(near))
        .expect("three candidates")
}

impl PromotionRing {
    pub(crate) fn new(cap: usize, rows: usize, n_experts: usize) -> Result<Self> {
        let reserve_at = cap * 16 + 8 + rows * n_experts * 4;
        let victims_at = (reserve_at + 8).next_multiple_of(8);
        let ahead_at = victims_at + cap * 32;
        let src_at = (ahead_at + 8 + rows * 4 + rows * AHEAD_CAP * 4).next_multiple_of(8);
        let bytes = src_at + rows * AHEAD_CAP * 8;
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("expert promotion ring: mapped allocation failed: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe {
                sys::cuMemFreeHost(raw);
            }
            candle::bail!("expert promotion ring has no device address: {r:?}");
        }
        // SAFETY: `bytes` just allocated, not yet visible to the device.
        unsafe { std::ptr::write_bytes(raw as *mut u8, 0, bytes) };
        let ring = Self {
            host: raw as *mut u8,
            dev,
            cap,
            n_experts,
            rows,
            reserve_at,
            victims_at,
            ahead_at,
            src_at,
        };
        // No prompt-only expert takes a slot until the host says how much is
        // decode's, no launch claims until it says how many may, and nothing is
        // read ahead until it says how much the link has room for (the zeroed
        // window).
        ring.set_reserve(u32::MAX);
        ring.set_sweep(0);
        Ok(ring)
    }

    fn ahead_word(&self, at: usize) -> *mut u32 {
        // SAFETY: callers pass offsets inside the read-ahead words.
        unsafe { self.host.add(self.ahead_at + at) as *mut u32 }
    }

    /// The slot images the link moves in one layer: a launch reads ahead with
    /// what its own misses leave of it.
    pub(crate) fn set_window(&self, slots: u32) {
        // SAFETY: the window word, first of the read-ahead words.
        unsafe { std::ptr::write_volatile(self.ahead_word(0), slots) };
        fence(Ordering::SeqCst);
    }

    /// A launch at row `r` reads ahead for rows `r + 2 ..= r + depth`.
    pub(crate) fn set_depth(&self, depth: u32) {
        // SAFETY: the depth word, after the window.
        unsafe { std::ptr::write_volatile(self.ahead_word(4), depth) };
        fence(Ordering::SeqCst);
    }

    /// Predict `experts` for `row`, best first, at most [`AHEAD_CAP`], each
    /// with the slot image vetted as its source: the list's entries, then its
    /// count. A bucketize reading the list while it changes may pair an
    /// expert with the source listed beside the one it replaced; it acts only
    /// when the expert's entry points at that source, and a source still
    /// readable through any list is a warm slot or a pinned pad slot — neither
    /// of which can come to hold another expert — so a mismatched pair is
    /// passed over.
    pub(crate) fn predict(&self, row: usize, experts: &[(usize, u64)]) {
        let n = experts.len().min(AHEAD_CAP);
        let list = 8 + self.rows * 4 + row * AHEAD_CAP * 4;
        let src = self.src_at + row * AHEAD_CAP * 8;
        for (q, &(e, image)) in experts[..n].iter().enumerate() {
            // SAFETY: inside row `row`'s list and sources of `AHEAD_CAP`
            // entries each.
            unsafe {
                std::ptr::write_volatile(self.ahead_word(list + q * 4), e as u32);
                std::ptr::write_volatile(self.host.add(src + q * 8) as *mut u64, image);
            }
        }
        fence(Ordering::Release);
        // SAFETY: row `row`'s count, inside `ahead_n`.
        unsafe { std::ptr::write_volatile(self.ahead_word(8 + row * 4), n as u32) };
    }

    /// Keep `n` stocked slots for decode-scored experts: a prompt-only expert
    /// takes one only while more than `n` are stocked.
    pub(crate) fn set_reserve(&self, n: u32) {
        // SAFETY: the reserve word this struct owns, after the marks.
        unsafe { std::ptr::write_volatile(self.host.add(self.reserve_at) as *mut u32, n) };
        fence(Ordering::SeqCst);
    }

    /// A launch with more than `n` claiming experts is a sweep and claims
    /// nothing.
    pub(crate) fn set_sweep(&self, n: u32) {
        // SAFETY: the sweep word this struct owns, after the reserve.
        unsafe { std::ptr::write_volatile(self.host.add(self.reserve_at + 4) as *mut u32, n) };
        fence(Ordering::SeqCst);
    }

    /// The expert's promotion has landed or been dropped: it may be given a
    /// slot again.
    pub(crate) fn clear_mark(&self, row: usize, expert: usize) {
        // SAFETY: inside the marks array, `rows × n_experts` u32s after the
        // counters; the device writes a mark only before handing out a slot.
        unsafe {
            let marks = self.host.add(self.cap * 16 + 8) as *mut u32;
            std::ptr::write_volatile(marks.add(row * self.n_experts + expert), 0);
        }
    }

    /// The device addresses `moe_bucketize` takes.
    pub(crate) fn ring(&self) -> PromoRing {
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
            ahead_cap: AHEAD_CAP as u32,
        }
    }

    pub(crate) fn cap(&self) -> usize {
        self.cap
    }

    fn word(&self, at: usize) -> *mut u32 {
        // SAFETY: `at` is one of the two counter offsets inside the allocation.
        unsafe { self.host.add(at) as *mut u32 }
    }

    /// Slots the device has taken, ever (wrapping).
    pub(crate) fn head(&self) -> u32 {
        // SAFETY: the counter word this struct owns.
        let h = unsafe { std::ptr::read_volatile(self.word(self.cap * 16)) };
        // The log entries the device wrote before it.
        fence(Ordering::Acquire);
        h
    }

    /// Slots the host has published, ever (wrapping).
    pub(crate) fn tail(&self) -> u32 {
        // SAFETY: as `head`.
        unsafe { std::ptr::read_volatile(self.word(self.cap * 16 + 4)) }
    }

    /// Offer `slot_base` at `tail` and advance it — empty, or holding `victim`.
    /// The caller must have room (`tail - taken < cap`, where `taken` is what it
    /// has collected).
    pub(crate) fn offer(&self, slot_base: u64, victim: Option<Victim>) {
        let tail = self.tail();
        let i = tail as usize % self.cap;
        let (index, retarget) = victim.map_or((PROMO_EMPTY, [0; 3]), |v| (v.index, v.retarget));
        // SAFETY: `i < cap`, inside the slots, victims and retarget arrays.
        unsafe {
            std::ptr::write_volatile((self.host as *mut u64).add(i), slot_base);
            let victims = self.host.add(self.victims_at) as *mut u64;
            std::ptr::write_volatile(victims.add(i), index);
            let entries = self.host.add(self.victims_at + self.cap * 8) as *mut u64;
            for (p, &v) in retarget.iter().enumerate() {
                std::ptr::write_volatile(entries.add(3 * i + p), v);
            }
        }
        fence(Ordering::SeqCst);
        // SAFETY: the counter word this struct owns.
        unsafe { std::ptr::write_volatile(self.word(self.cap * 16 + 4), tail.wrapping_add(1)) };
        fence(Ordering::SeqCst);
    }

    /// The log entry at ring index `i` (mod `cap`).
    pub(crate) fn log(&self, i: u32) -> Logged {
        let i = i as usize % self.cap;
        // SAFETY: inside the log array.
        let v =
            unsafe { std::ptr::read_volatile((self.host.add(self.cap * 8) as *const u64).add(i)) };
        Logged::decode(v)
    }

    /// Withdraw every untaken offer: `tail = head`. Only with no bucketize in
    /// flight and none able to begin.
    pub(crate) fn withdraw(&self) {
        self.withdraw_to(self.head());
    }

    /// Withdraw the untaken offers past `tail`: set `tail` back to it (`head <=
    /// tail <= self.tail()`). Only with no bucketize in flight and none able to
    /// begin — a bucketize reads `tail` once and takes up to it.
    pub(crate) fn withdraw_to(&self, tail: u32) {
        // SAFETY: the counter word this struct owns.
        unsafe { std::ptr::write_volatile(self.word(self.cap * 16 + 4), tail) };
        fence(Ordering::SeqCst);
    }
}

impl Drop for PromotionRing {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` in `new`; the device and the
        // pipeline thread are done with it before the cache drops.
        unsafe {
            sys::cuMemFreeHost(self.host as *mut std::ffi::c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A log entry carries the word, the row and the expert; a read-ahead
    /// claim's expert field carries `AHEAD_FLAG` over the expert, and a skip's
    /// `PROMO_SKIP` — whose bits include the flag's — is no read-ahead.
    #[test]
    fn a_log_entry_decodes() {
        let v = (0xdead_beefu64 << 32) | (39 << 16) | 255;
        assert_eq!(
            Logged::decode(v),
            Logged {
                word: 0xdead_beef,
                row: 39,
                expert: 255,
                ahead: false
            }
        );
        let a = (7u64 << 32) | (41 << 16) | AHEAD_FLAG as u64 | 300;
        assert_eq!(
            Logged::decode(a),
            Logged {
                word: 7,
                row: 41,
                expert: 300,
                ahead: true
            }
        );
        let skip = Logged::decode((7u64 << 32) | (2 << 16) | PROMO_SKIP as u64);
        assert!(skip.skipped() && !skip.ahead);
    }

    /// The read-ahead words follow the retarget entries: window, depth, the
    /// per-row counts, the per-row lists of `AHEAD_CAP`, then their vetted
    /// sources, 8-aligned. A prediction writes its experts, their sources and
    /// its count, truncated at the cap; a shorter one later leaves the old tail
    /// behind its count.
    #[test]
    fn read_ahead_words_sit_after_the_retarget_entries() {
        let Ok(_device) = candle::Device::new_cuda(0) else {
            return;
        };
        let ring = PromotionRing::new(4, 2, 8).unwrap();
        let word = |at: usize| unsafe { std::ptr::read_volatile(ring.host.add(at) as *const u32) };
        // victims at 144, retarget 144 + 32 = 176 for 96 bytes: words at 272.
        assert_eq!(ring.ahead_at, 272);
        let long = |at: usize| unsafe { std::ptr::read_volatile(ring.host.add(at) as *const u64) };
        let r = ring.ring();
        // Lists at 288 for 2 rows × 32 × 4 bytes: sources at 544.
        assert_eq!(
            (
                r.window - ring.dev,
                r.depth - ring.dev,
                r.ahead_n - ring.dev,
                r.ahead_list - ring.dev,
                r.ahead_src - ring.dev,
                r.ahead_cap
            ),
            (272, 276, 280, 288, 544, AHEAD_CAP as u32)
        );
        assert_eq!(
            (word(272), word(276)),
            (0, 0),
            "nothing read ahead until told"
        );
        ring.set_window(12);
        ring.set_depth(4);
        assert_eq!((word(272), word(276)), (12, 4));

        let many: Vec<(usize, u64)> = (100..100 + AHEAD_CAP + 3)
            .map(|e| (e, 0x7000_0000 + e as u64 * 0x1000))
            .collect();
        ring.predict(1, &many);
        let (row1, src1) = (288 + AHEAD_CAP * 4, 544 + AHEAD_CAP * 8);
        assert_eq!(word(284), AHEAD_CAP as u32, "truncated at the cap");
        assert_eq!(
            (word(row1), word(row1 + 4 * (AHEAD_CAP - 1))),
            (100, 100 + AHEAD_CAP as u32 - 1)
        );
        assert_eq!(
            (long(src1), long(src1 + 8 * (AHEAD_CAP - 1))),
            (
                0x7006_4000,
                0x7000_0000 + (100 + AHEAD_CAP as u64 - 1) * 0x1000
            )
        );
        ring.predict(1, &[(7, 0x9000_0000), (9, 0x9000_1000)]);
        assert_eq!(
            (word(284), word(row1), word(row1 + 4), word(row1 + 8)),
            (2, 7, 9, 102)
        );
        assert_eq!((long(src1), long(src1 + 8)), (0x9000_0000, 0x9000_1000));
        assert_eq!(word(280), 0, "row 0 untouched");
    }

    /// Published slots land at `tail` in order and wrap at `cap`; withdrawing
    /// sets `tail` back so the slots past it are no longer offered, and the
    /// next push reuses the first withdrawn index.
    #[test]
    fn pushes_publish_in_order_and_a_withdraw_sets_the_tail_back() {
        let Ok(_device) = candle::Device::new_cuda(0) else {
            return;
        };
        let ring = PromotionRing::new(4, 2, 8).unwrap();
        assert_eq!((ring.head(), ring.tail()), (0, 0));
        for base in [0x1000u64, 0x2000, 0x3000] {
            ring.offer(base, None);
        }
        assert_eq!(ring.tail(), 3);
        // SAFETY: the slots array is the allocation's first `cap` u64s.
        let slot = |i: usize| unsafe { std::ptr::read_volatile((ring.host as *const u64).add(i)) };
        assert_eq!([slot(0), slot(1), slot(2)], [0x1000, 0x2000, 0x3000]);

        ring.withdraw_to(1);
        assert_eq!(ring.tail(), 1);
        ring.offer(0x4000, None);
        assert_eq!((ring.tail(), slot(1)), (2, 0x4000));

        ring.withdraw();
        assert_eq!(ring.tail(), 0);
        for base in [0x5000u64, 0x6000, 0x7000, 0x8000, 0x9000] {
            ring.offer(base, None);
        }
        assert_eq!((ring.tail(), slot(0)), (5, 0x9000));
    }

    /// An offer writes its victim's index and retarget entries at its own ring
    /// index, and an empty one writes `PROMO_EMPTY` there; the sweep word starts
    /// at 0 — nothing claims until the host says how many may — and sits right
    /// after the reserve.
    #[test]
    fn an_offer_names_its_victim_and_the_sweep_word_follows_the_reserve() {
        let Ok(_device) = candle::Device::new_cuda(0) else {
            return;
        };
        let ring = PromotionRing::new(4, 2, 8).unwrap();
        let word = |at: usize| unsafe { std::ptr::read_volatile(ring.host.add(at) as *const u32) };
        let long = |at: usize| unsafe { std::ptr::read_volatile(ring.host.add(at) as *const u64) };
        // reserve at 4·16 + 8 + 2·8·4 = 136, sweep at 140, victims at 144.
        assert_eq!((ring.reserve_at, ring.victims_at), (136, 144));
        assert_eq!((word(136), word(140)), (u32::MAX, 0));
        ring.set_sweep(48);
        assert_eq!(word(140), 48);

        ring.offer(0x1000, None);
        ring.offer(
            0x2000,
            Some(Victim {
                index: 13,
                retarget: [0x9000, 0x9010, 0x9020],
            }),
        );
        assert_eq!((long(144), long(152)), (PROMO_EMPTY, 13));
        // retarget at 144 + 4·8 = 176, three u64s per ring index.
        assert_eq!(
            (long(176 + 24), long(176 + 32), long(176 + 40)),
            (0x9000, 0x9010, 0x9020)
        );
        let r = ring.ring();
        assert_eq!(
            (
                r.sweep - r.reserve,
                r.victims - ring.dev,
                r.retarget - ring.dev
            ),
            (4, 144, 176)
        );
    }

    /// A log entry naming `PROMO_SKIP` is a victim passed over.
    #[test]
    fn a_skip_entry_decodes_as_one() {
        let skip = Logged::decode((5u64 << 32) | (3 << 16) | PROMO_SKIP as u64);
        assert!(skip.skipped());
        assert!(!Logged::decode((5u64 << 32) | (3 << 16) | 7).skipped());
    }

    /// A cleared mark is zero in the device's marks array, at `row × E + e`.
    #[test]
    fn clearing_a_mark_zeroes_its_word() {
        let Ok(_device) = candle::Device::new_cuda(0) else {
            return;
        };
        let ring = PromotionRing::new(4, 2, 8).unwrap();
        // SAFETY: the marks array follows the slots, the log and the counters.
        let marks = unsafe { ring.host.add(4 * 16 + 8) as *mut u32 };
        // SAFETY: row 1, expert 3 is inside the 2 × 8 marks.
        unsafe { std::ptr::write_volatile(marks.add(11), 7) };
        ring.clear_mark(1, 3);
        // SAFETY: as above.
        assert_eq!(unsafe { std::ptr::read_volatile(marks.add(11)) }, 0);
    }

    /// The ticket nearest the one being served, across a 32-bit wrap either
    /// way.
    #[test]
    fn a_word_maps_to_the_nearest_ticket() {
        assert_eq!(ticket_from_word(7, 9), 7);
        assert_eq!(ticket_from_word(12, 9), 12);
        let near = (5u64 << 32) | 3;
        assert_eq!(
            ticket_from_word(0xffff_fffe, near),
            (4u64 << 32) | 0xffff_fffe
        );
        let near = (5u64 << 32) | 0xffff_fff0;
        assert_eq!(ticket_from_word(4, near), (6u64 << 32) | 4);
    }
}
