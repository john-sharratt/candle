//! The gap's destination claim (`candle-kernels/src/simple/kv_gap_claim.cu`,
//! `docs/kv_quantize_in_the_gap.md` §3.2–§3.4), alone: its outputs against a
//! host reference, byte for byte, and its cost.
//!
//! The selection's outputs (tags, outer scales, per-channel palette slots) are
//! synthetic and uploaded once, so a launch here is exactly the claim's work:
//! every band's index in its class, each job's deferral, and every head's
//! destination record into the convert buffer and the summary.
//!
//! ```text
//! cargo test --release --features cuda -p candle-core --test kv_gap_claim_bench
//! cargo test --release --features cuda -p candle-core --test kv_gap_claim_bench \
//!     -- --ignored bench --nocapture --test-threads=1
//! ncu --kernel-name regex:gap_ --set full <test exe> profile_gap_claim \
//!     --ignored --exact --nocapture
//! ```
#![cfg(feature = "cuda")]
// The reference walks (job, head, palette, side) COORDINATES and computes
// offsets from all of them at once; the indices are the subject.
#![allow(clippy::needless_range_loop)]

use std::time::Instant;

use candle_core::backend::BackendDevice;
use candle_core::cuda_backend::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};
use candle_core::cuda_backend::kernels::simple::kv_gap_claim::{
    claim_status_words, run_kv_gap_claim, summary_head_bytes, BANDS_PER_HEAD, MAX_CLASS,
    MAX_KV_HEAD, NO_OVERRIDE, N_CODES,
};
use candle_core::cuda_backend::CudaDevice;
use candle_core::{Device, Result};

const N_PAL: usize = 4;
const SINK: u64 = 0x0000_dead_0000_0000;
/// The codes the synthetic selection draws from, and the class each falls in:
/// several codes share a class, as formats of one size do.
const CODES: [i32; 20] = [
    1, 2, 3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
];
const N_CLASS: usize = 15;

fn class_of(i: usize) -> i8 {
    (i % N_CLASS) as i8
}

fn record_bytes(hd: usize) -> usize {
    hd / 2 + 104
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 11) % n
    }
}

/// One policy group's claim inputs: the selection's outputs plus the gap's
/// tables, lease and running class counts.
#[derive(Clone)]
struct Case {
    n_jobs: usize,
    nkh: usize,
    hd: usize,
    k_override: i32,
    v_override: i32,
    k_tags: Vec<i32>,
    v_tags: Vec<i32>,
    k_scale: Vec<f32>,
    v_scale: Vec<f32>,
    k_map: Vec<i32>,
    v_map: Vec<i32>,
    code_class: Vec<i8>,
    code_fmt: Vec<u8>,
    lease_len: Vec<u32>,
    stack_off: Vec<u32>,
    stack: Vec<u64>,
    taken: Vec<u32>,
}

impl Case {
    fn heads(&self) -> usize {
        self.n_jobs * self.nkh
    }

    /// `lease_frac`: each class's lease as a fraction of the bands that would
    /// claim it, so below 1.0 the tail of the table defers.
    fn new(n_jobs: usize, nkh: usize, hd: usize, lease_frac: f64, seed: u64) -> Case {
        let mut rng = Rng(seed | 1);
        let heads = n_jobs * nkh;
        // Skewed like the selection: most palettes take the level's first
        // passing format, the rest spread.
        let tag = |rng: &mut Rng| -> i32 {
            if rng.below(10) < 6 {
                CODES[3]
            } else {
                CODES[rng.below(CODES.len() as u64) as usize]
            }
        };
        let k_tags: Vec<i32> = (0..heads * N_PAL).map(|_| tag(&mut rng)).collect();
        let v_tags: Vec<i32> = (0..heads * N_PAL).map(|_| tag(&mut rng)).collect();
        let scale = |rng: &mut Rng| (rng.below(1 << 20) as f32) / (1 << 16) as f32 + 0.5;
        let k_scale = (0..heads * N_PAL).map(|_| scale(&mut rng)).collect();
        let v_scale = (0..heads * N_PAL).map(|_| scale(&mut rng)).collect();
        // Mostly the selection's 0..3, with a few out-of-range values the
        // packing must clamp exactly as the host readback does.
        let slot = |rng: &mut Rng| match rng.below(64) {
            0 => -1 - rng.below(5) as i32,
            1 => 4 + rng.below(5) as i32,
            _ => rng.below(4) as i32,
        };
        let k_map = (0..heads * hd).map(|_| slot(&mut rng)).collect();
        let v_map = (0..heads * hd).map(|_| slot(&mut rng)).collect();
        let mut code_class = vec![-1i8; N_CODES];
        let mut code_fmt = vec![0u8; N_CODES];
        for (i, &c) in CODES.iter().enumerate() {
            code_class[c as usize] = class_of(i);
            code_fmt[c as usize] = (c as u8).wrapping_mul(3).wrapping_add(1);
        }
        let mut case = Case {
            n_jobs,
            nkh,
            hd,
            k_override: NO_OVERRIDE,
            v_override: NO_OVERRIDE,
            k_tags,
            v_tags,
            k_scale,
            v_scale,
            k_map,
            v_map,
            code_class,
            code_fmt,
            lease_len: vec![0; N_CLASS],
            stack_off: vec![0; N_CLASS],
            stack: Vec::new(),
            taken: vec![0; N_CLASS],
        };
        let mut demand = [0u64; N_CLASS];
        for h in 0..heads {
            for p in 0..N_PAL {
                for s in 0..2 {
                    let c = case.class(case.code(h, p, s));
                    demand[c as usize] += 1;
                }
            }
        }
        case.set_lease(|c| (demand[c] as f64 * lease_frac) as u32);
        case
    }

    fn set_lease(&mut self, len: impl Fn(usize) -> u32) {
        self.stack.clear();
        for c in 0..N_CLASS {
            self.lease_len[c] = len(c);
            self.stack_off[c] = self.stack.len() as u32;
            for i in 0..self.lease_len[c] as u64 {
                self.stack
                    .push(0x1000_0000_0000 + ((c as u64) << 36) + i * 0x140);
            }
        }
    }

    fn code(&self, head: usize, p: usize, side: usize) -> i32 {
        match side {
            0 if self.k_override >= 0 => self.k_override,
            1 if self.v_override >= 0 => self.v_override,
            0 => self.k_tags[head * N_PAL + p],
            _ => self.v_tags[head * N_PAL + p],
        }
    }

    fn class(&self, code: i32) -> i32 {
        if (0..N_CODES as i32).contains(&code) {
            let c = self.code_class[code as usize] as i32;
            if c < N_CLASS as i32 {
                return c;
            }
        }
        -1
    }
}

/// What the claim leaves behind, as the host sees it.
#[derive(PartialEq, Eq, Debug)]
struct Out {
    dst_records: Vec<u8>,
    summary_heads: Vec<u8>,
    summary_flags: Vec<u32>,
    taken: Vec<u32>,
    error_word: u32,
}

/// The claim, sequentially: bands in table order (job, head, palette, K
/// before V), each taking the next index of its class.
fn reference(case: &Case) -> Out {
    let hd = case.hd;
    let rec = record_bytes(hd);
    let srow = summary_head_bytes(hd);
    let heads = case.heads();
    let mut out = Out {
        dst_records: vec![0; heads * rec],
        summary_heads: vec![0; heads * srow],
        summary_flags: vec![0; case.n_jobs],
        taken: case.taken.clone(),
        error_word: 0,
    };
    let mut idx = vec![[0u32; BANDS_PER_HEAD]; case.nkh];
    for job in 0..case.n_jobs {
        let mut deferred = false;
        for h in 0..case.nkh {
            let head = job * case.nkh + h;
            for p in 0..N_PAL {
                for s in 0..2 {
                    let c = case.class(case.code(head, p, s));
                    let i = if c < 0 {
                        out.error_word = 1;
                        deferred = true;
                        u32::MAX
                    } else {
                        let i = out.taken[c as usize];
                        out.taken[c as usize] += 1;
                        if i >= case.lease_len[c as usize] {
                            deferred = true;
                        }
                        i
                    };
                    idx[h][p * 2 + s] = i;
                }
            }
        }
        out.summary_flags[job] = deferred as u32;
        for h in 0..case.nkh {
            let head = job * case.nkh + h;
            let mut r = vec![0u8; rec];
            for side in 0..2 {
                let (ov, map) = if side == 0 {
                    (case.k_override, &case.k_map)
                } else {
                    (case.v_override, &case.v_map)
                };
                for byte in 0..hd / 4 {
                    let mut b = 0u8;
                    for q in 0..4 {
                        let dim = byte * 4 + q;
                        let slot = if ov >= 0 {
                            dim / (hd / 4)
                        } else {
                            map[head * hd + dim].clamp(0, 3) as usize
                        };
                        b |= (slot as u8) << (2 * q);
                    }
                    r[side * hd / 4 + byte] = b;
                }
                for p in 0..N_PAL {
                    let code = case.code(head, p, side);
                    let c = case.class(code);
                    let i = idx[h][p * 2 + side];
                    let addr = if deferred || c < 0 {
                        SINK
                    } else {
                        case.stack[(case.stack_off[c as usize] + i) as usize]
                    };
                    let fmt = if (0..N_CODES as i32).contains(&code) {
                        case.code_fmt[code as usize]
                    } else {
                        0
                    };
                    let scale = if ov >= 0 {
                        1.0f32
                    } else if side == 0 {
                        case.k_scale[head * N_PAL + p]
                    } else {
                        case.v_scale[head * N_PAL + p]
                    };
                    let po = hd / 2 + side * 32 + p * 8;
                    r[po..po + 8].copy_from_slice(&addr.to_le_bytes());
                    r[hd / 2 + 64 + side * 4 + p] = fmt;
                    let so = hd / 2 + 72 + side * 16 + p * 4;
                    r[so..so + 4].copy_from_slice(&scale.to_le_bytes());
                }
            }
            out.dst_records[head * rec..(head + 1) * rec].copy_from_slice(&r);
            let s = &mut out.summary_heads[head * srow..(head + 1) * srow];
            s[..rec].copy_from_slice(&r);
            for (b, i) in idx[h].iter().enumerate() {
                s[rec + b * 4..rec + b * 4 + 4].copy_from_slice(&i.to_le_bytes());
            }
        }
    }
    out
}

/// A case on the device: inputs uploaded once, outputs poisoned so a byte the
/// kernel fails to write shows.
struct OnDevice {
    k_tags: CudaSlice<i32>,
    v_tags: CudaSlice<i32>,
    k_scale: CudaSlice<f32>,
    v_scale: CudaSlice<f32>,
    k_map: CudaSlice<i32>,
    v_map: CudaSlice<i32>,
    code_class: CudaSlice<i8>,
    code_fmt: CudaSlice<u8>,
    lease_len: CudaSlice<u32>,
    stack_off: CudaSlice<u32>,
    stack: CudaSlice<u64>,
    taken: CudaSlice<u32>,
    dst_records: CudaSlice<u8>,
    summary_heads: CudaSlice<u8>,
    summary_flags: CudaSlice<u32>,
    error_word: CudaSlice<u32>,
    status: CudaSlice<u64>,
    epoch: std::cell::Cell<u32>,
}

const POISON: u8 = 0xAB;

impl OnDevice {
    fn new(dev: &CudaDevice, case: &Case) -> Result<OnDevice> {
        let heads = case.heads();
        let stack = if case.stack.is_empty() {
            vec![0u64]
        } else {
            case.stack.clone()
        };
        Ok(OnDevice {
            k_tags: dev.memcpy_stod(&case.k_tags)?,
            v_tags: dev.memcpy_stod(&case.v_tags)?,
            k_scale: dev.memcpy_stod(&case.k_scale)?,
            v_scale: dev.memcpy_stod(&case.v_scale)?,
            k_map: dev.memcpy_stod(&case.k_map)?,
            v_map: dev.memcpy_stod(&case.v_map)?,
            code_class: dev.memcpy_stod(&case.code_class)?,
            code_fmt: dev.memcpy_stod(&case.code_fmt)?,
            lease_len: dev.memcpy_stod(&case.lease_len)?,
            stack_off: dev.memcpy_stod(&case.stack_off)?,
            stack: dev.memcpy_stod(&stack)?,
            taken: dev.memcpy_stod(&case.taken)?,
            dst_records: dev.memcpy_stod(&vec![POISON; heads * record_bytes(case.hd)])?,
            summary_heads: dev.memcpy_stod(&vec![POISON; heads * summary_head_bytes(case.hd)])?,
            summary_flags: dev.memcpy_stod(&vec![0xABAB_ABABu32; case.n_jobs])?,
            error_word: dev.memcpy_stod(&[0u32])?,
            status: dev.alloc_zeros::<u64>(claim_status_words(heads))?,
            epoch: std::cell::Cell::new(0),
        })
    }

    fn read(&self, dev: &CudaDevice) -> Result<Out> {
        Ok(Out {
            dst_records: dev.memcpy_dtov(&self.dst_records)?,
            summary_heads: dev.memcpy_dtov(&self.summary_heads)?,
            summary_flags: dev.memcpy_dtov(&self.summary_flags)?,
            taken: dev.memcpy_dtov(&self.taken)?,
            error_word: dev.memcpy_dtov(&self.error_word)?[0],
        })
    }
}

/// One claim launch over `case` on `on`.
fn launch(dev: &CudaDevice, case: &Case, on: &OnDevice) -> Result<()> {
    let stream = dev.cuda_stream();
    let p = |s: &CudaSlice<u8>| s.device_ptr(&stream).0;
    macro_rules! ptr {
        ($s:expr, $t:ty) => {
            $s.device_ptr(&stream).0 as $t
        };
    }
    let rc = unsafe {
        run_kv_gap_claim(
            ptr!(on.k_tags, *const i32),
            ptr!(on.v_tags, *const i32),
            ptr!(on.k_scale, *const f32),
            ptr!(on.v_scale, *const f32),
            ptr!(on.k_map, *const i32),
            ptr!(on.v_map, *const i32),
            case.k_override,
            case.v_override,
            ptr!(on.code_class, *const i8),
            ptr!(on.code_fmt, *const u8),
            ptr!(on.lease_len, *const u32),
            ptr!(on.stack, *const u64),
            ptr!(on.stack_off, *const u32),
            ptr!(on.taken, *mut u32),
            SINK,
            p(&on.dst_records) as *mut u8,
            p(&on.summary_heads) as *mut u8,
            ptr!(on.summary_flags, *mut u32),
            ptr!(on.error_word, *mut u32),
            ptr!(on.status, *mut u64),
            {
                on.epoch.set(on.epoch.get().wrapping_add(1));
                on.epoch.get()
            },
            case.n_jobs as i32,
            case.nkh as i32,
            case.hd as i32,
            N_CLASS as i32,
            stream.cu_stream() as *mut std::ffi::c_void,
        )
    };
    assert_eq!(rc, 0, "run_kv_gap_claim refused the launch");
    Ok(())
}

fn cuda() -> Result<CudaDevice> {
    match Device::new_cuda(0)? {
        Device::Cuda(d) => Ok(d),
        _ => unreachable!(),
    }
}

/// Launch once and compare every output byte with the reference.
fn check(dev: &CudaDevice, case: &Case) -> Result<Out> {
    check_on(dev, case, &OnDevice::new(dev, case)?)
}

/// As [`check`], on device state the caller made — a status buffer and epoch
/// carried over from earlier launches.
fn check_on(dev: &CudaDevice, case: &Case, on: &OnDevice) -> Result<Out> {
    launch(dev, case, on)?;
    dev.synchronize()?;
    let got = on.read(dev)?;
    let want = reference(case);
    assert_eq!(got.taken, want.taken, "taken");
    assert_eq!(got.error_word, want.error_word, "error word");
    assert_eq!(got.summary_flags, want.summary_flags, "deferred flags");
    let rec = record_bytes(case.hd);
    let srow = summary_head_bytes(case.hd);
    for head in 0..case.heads() {
        assert_eq!(
            got.dst_records[head * rec..(head + 1) * rec],
            want.dst_records[head * rec..(head + 1) * rec],
            "dst record of head {head}"
        );
        assert_eq!(
            got.summary_heads[head * srow..(head + 1) * srow],
            want.summary_heads[head * srow..(head + 1) * srow],
            "summary row of head {head}"
        );
    }
    Ok(want)
}

#[test]
fn claim_matches_reference_with_ample_leases() -> Result<()> {
    let dev = cuda()?;
    for &(jobs, nkh, seed) in &[
        (1usize, 4usize, 1u64),
        (7, 4, 2),
        (64, 4, 3),
        (65, 4, 4),
        (1000, 4, 5),
        (3072, 4, 6),
        (777, 8, 7),
        (513, 1, 8),
        (300, 2, 9),
        (129, 16, 10),
        (20_000, 4, 11),
        (95, 64, 12),
        (130, 33, 13),
        (61, 5, 14),
    ] {
        let want = check(&dev, &Case::new(jobs, nkh, 128, 1.0, seed))?;
        assert!(want.summary_flags.iter().all(|&f| f == 0), "nothing defers");
    }
    Ok(())
}

#[test]
fn claim_defers_the_tail_past_each_lease() -> Result<()> {
    let dev = cuda()?;
    for &(jobs, nkh, frac, seed) in &[
        (1000usize, 4usize, 0.5f64, 21u64),
        (3072, 4, 0.9, 22),
        (777, 8, 0.1, 23),
        (20_000, 4, 0.75, 24),
        (100, 4, 0.0, 25),
    ] {
        let want = check(&dev, &Case::new(jobs, nkh, 128, frac, seed))?;
        assert!(want.summary_flags.contains(&1), "the tail defers");
    }
    Ok(())
}

#[test]
fn claim_continues_from_earlier_groups() -> Result<()> {
    let dev = cuda()?;
    let mut case = Case::new(2000, 4, 128, 1.0, 31);
    // Earlier groups of the gap took part of each class's lease.
    case.taken = (0..N_CLASS as u32).map(|c| c * 37 + 5).collect();
    case.set_lease(|c| 2000 * 8 + c as u32 * 50);
    check(&dev, &case)?;
    Ok(())
}

#[test]
fn claim_under_overrides_uses_identity_maps_and_unit_scales() -> Result<()> {
    let dev = cuda()?;
    for &(ko, vo) in &[
        (CODES[3], NO_OVERRIDE),
        (NO_OVERRIDE, CODES[5]),
        (CODES[0], CODES[9]),
    ] {
        let mut case = Case::new(1500, 4, 128, 1.0, 41);
        case.k_override = ko;
        case.v_override = vo;
        case.set_lease(|_| 1500 * 4 * 8);
        check(&dev, &case)?;
    }
    Ok(())
}

#[test]
fn claim_at_the_other_head_dim() -> Result<()> {
    // 128 and 256 are the head dims the convert has a record for, and so the
    // only ones the claim is built at.
    let dev = cuda()?;
    check(&dev, &Case::new(900, 4, 256, 0.8, 307))?;
    check(&dev, &Case::new(900, 4, 256, 1.0, 308))?;
    Ok(())
}

#[test]
fn claim_reports_an_uncovered_code_and_defers_its_job() -> Result<()> {
    let dev = cuda()?;
    let mut case = Case::new(3000, 4, 128, 1.0, 61);
    // Codes no class covers, in three jobs.
    case.k_tags[5 * 4 * N_PAL + 2] = 4;
    case.v_tags[1700 * 4 * N_PAL + 9] = 63;
    case.k_tags[2999 * 4 * N_PAL + 15] = 99;
    let want = check(&dev, &case)?;
    assert_eq!(want.error_word, 1);
    assert_eq!(
        [
            want.summary_flags[5],
            want.summary_flags[1700],
            want.summary_flags[2999]
        ],
        [1, 1, 1]
    );
    Ok(())
}

/// Shapes the bench and the profile run: (jobs, kv heads). A job is one
/// layer's block of one sequence: a decode gap is a few sequences × 48 layers,
/// a prefill gap an 8K prompt's 256 blocks × 48 layers, and 65,535 is the most
/// one convert launch takes.
const SHAPES: [(usize, usize); 6] = [
    (48, 4),
    (768, 4),
    (3072, 4),
    (12_288, 4),
    (65_535, 4),
    (12_288, 8),
];

/// Per launch: the GPU's own time (event pair around one launch; median and
/// best of `iters`), and the wall time of `iters` launches issued back to
/// back, which adds the host's launch cost.
struct Timing {
    gpu_med_us: f64,
    gpu_min_us: f64,
    wall_us: f64,
}

fn time_claim(dev: &CudaDevice, case: &Case, iters: usize) -> Result<Timing> {
    let on = OnDevice::new(dev, case)?;
    for _ in 0..10 {
        launch(dev, case, &on)?;
    }
    dev.synchronize()?;
    let event = || {
        dev.cuda_stream()
            .record_event(Some(CU_EVENT_DEFAULT))
            .map_err(|e| candle_core::Error::Msg(format!("event: {e}")))
    };
    let mut us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = event()?;
        launch(dev, case, &on)?;
        let stop = event()?;
        dev.synchronize()?;
        let ms = start
            .elapsed_ms(&stop)
            .map_err(|e| candle_core::Error::Msg(format!("elapsed: {e}")))?;
        us.push(ms as f64 * 1e3);
    }
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t = Instant::now();
    for _ in 0..iters {
        launch(dev, case, &on)?;
    }
    dev.synchronize()?;
    Ok(Timing {
        gpu_med_us: us[us.len() / 2],
        gpu_min_us: us[0],
        wall_us: t.elapsed().as_secs_f64() * 1e6 / iters as f64,
    })
}

/// The ceiling on this card as it stands (shared or not): a device-to-device
/// copy moving the same bytes (half read, half written), timed the same way.
fn time_copy(dev: &CudaDevice, bytes: usize, iters: usize) -> Result<Timing> {
    let src = dev.alloc_zeros::<u8>(bytes / 2)?;
    let mut dst = dev.alloc_zeros::<u8>(bytes / 2)?;
    let mut us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let stream = dev.cuda_stream();
        let start = stream
            .record_event(Some(CU_EVENT_DEFAULT))
            .map_err(|e| candle_core::Error::Msg(format!("event: {e}")))?;
        stream
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| candle_core::Error::Msg(format!("copy: {e}")))?;
        let stop = stream
            .record_event(Some(CU_EVENT_DEFAULT))
            .map_err(|e| candle_core::Error::Msg(format!("event: {e}")))?;
        drop(stream);
        dev.synchronize()?;
        let ms = start
            .elapsed_ms(&stop)
            .map_err(|e| candle_core::Error::Msg(format!("elapsed: {e}")))?;
        us.push(ms as f64 * 1e3);
    }
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(Timing {
        gpu_med_us: us[us.len() / 2],
        gpu_min_us: us[0],
        wall_us: 0.0,
    })
}

#[test]
#[ignore = "microbenchmark; run with --ignored --nocapture"]
fn bench_gap_claim() -> Result<()> {
    let dev = cuda()?;
    println!(
        "{:>7} {:>4} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "jobs", "nkh", "gpu med", "gpu min", "wall", "MB", "GB/s min", "copy min"
    );
    for &(jobs, nkh) in &SHAPES {
        let case = Case::new(jobs, nkh, 128, 1.0, 0x5EED);
        let heads = jobs * nkh;
        // Bytes the claim must touch: tags, scales and maps in; records,
        // summary rows and flags out.
        let bytes = heads * (2 * N_PAL * 4 * 2 + 2 * 128 * 4)
            + heads * (record_bytes(128) + summary_head_bytes(128))
            + jobs * 4;
        let iters = (400_000_000 / bytes).clamp(50, 2000);
        let t = time_claim(&dev, &case, iters)?;
        let copy = time_copy(&dev, bytes, iters)?;
        println!(
            "{jobs:>7} {nkh:>4} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.1} {:>9.2}",
            t.gpu_med_us,
            t.gpu_min_us,
            t.wall_us,
            bytes as f64 / 1e6,
            bytes as f64 / t.gpu_min_us / 1e3,
            copy.gpu_min_us
        );
    }
    Ok(())
}

/// One launch per shape and nothing else: the target for `ncu`.
#[test]
#[ignore = "profiling target; run under ncu"]
fn profile_gap_claim() -> Result<()> {
    let dev = cuda()?;
    for &(jobs, nkh) in &SHAPES {
        let case = Case::new(jobs, nkh, 128, 1.0, 0x5EED);
        let on = OnDevice::new(&dev, &case)?;
        launch(&dev, &case, &on)?;
        dev.synchronize()?;
    }
    Ok(())
}

// The reference itself, on hand-checked values.
#[test]
fn reference_takes_indices_in_band_order() {
    let mut case = Case::new(2, 1, 128, 1.0, 71);
    for t in case.k_tags.iter_mut().chain(case.v_tags.iter_mut()) {
        *t = CODES[3];
    }
    case.set_lease(|_| 100);
    let out = reference(&case);
    let rec = record_bytes(128);
    let srow = summary_head_bytes(128);
    // Job 1's head starts after job 0's eight bands of the one class.
    let idx = |head: usize, b: usize| {
        let o = head * srow + rec + b * 4;
        u32::from_le_bytes(out.summary_heads[o..o + 4].try_into().unwrap())
    };
    assert_eq!(
        (0..8).map(|b| idx(0, b)).collect::<Vec<_>>(),
        (0..8).collect::<Vec<_>>()
    );
    assert_eq!(idx(1, 0), 8);
    assert_eq!(out.taken[class_of(3) as usize], 16);
}

// ── Edge cases ──

/// Launch `cases` in order on ONE status buffer, its epoch starting at
/// `epoch`, as a long-lived gap device does: each launch must match its
/// reference whatever the earlier launches left in the status words.
fn check_sequence(dev: &CudaDevice, cases: &[Case], epoch: u32) -> Result<Vec<Out>> {
    let widest = cases.iter().map(Case::heads).max().unwrap_or(0);
    let mut status = dev.alloc_zeros::<u64>(claim_status_words(widest))?;
    let mut outs = Vec::new();
    let mut next_epoch = epoch;
    for case in cases {
        let mut on = OnDevice::new(dev, case)?;
        std::mem::swap(&mut on.status, &mut status);
        on.epoch.set(next_epoch);
        outs.push(check_on(dev, case, &on)?);
        next_epoch = on.epoch.get();
        std::mem::swap(&mut on.status, &mut status);
    }
    Ok(outs)
}

#[test]
fn claim_reuses_one_status_buffer_across_launches_of_any_width() -> Result<()> {
    let dev = cuda()?;
    // Wide, narrow, wide again: the narrow launch meets only its own tiles'
    // words, and the wide one after it meets words two epochs stale past them.
    check_sequence(
        &dev,
        &[
            Case::new(2000, 4, 128, 1.0, 81),
            Case::new(100, 4, 128, 0.7, 82),
            Case::new(2000, 4, 128, 0.9, 83),
            Case::new(3, 64, 128, 1.0, 84),
            Case::new(2001, 4, 128, 1.0, 85),
        ],
        0,
    )?;
    Ok(())
}

#[test]
fn claim_survives_the_epoch_wrapping_through_zero() -> Result<()> {
    let dev = cuda()?;
    // The launches run at epochs u32::MAX - 1, u32::MAX, 0, 1. Epoch 0 is the
    // value a freshly zeroed status word carries, flagged nothing.
    let cases: Vec<Case> = (0..4)
        .map(|i| Case::new(700 + 300 * i, 4, 128, 0.8, 91 + i as u64))
        .collect();
    check_sequence(&dev, &cases, u32::MAX - 2)?;
    // And the very first launch on a zeroed buffer at epoch 0.
    check_sequence(&dev, &cases[..1], u32::MAX)?;
    Ok(())
}

#[test]
fn claim_at_every_tile_boundary() -> Result<()> {
    let dev = cuda()?;
    // A tile is 64 heads of whole jobs: 16 jobs at 4 heads, 21 at 3 (one head
    // idle), one at 33..=64. Launch at, just under and just over 1 and 2 tiles.
    for &nkh in &[3usize, 4, 7, 32, 33, 63, 64] {
        let per_tile = 64 / nkh;
        for jobs in [
            1,
            per_tile - 1,
            per_tile,
            per_tile + 1,
            2 * per_tile,
            2 * per_tile + 1,
        ] {
            if jobs == 0 {
                continue;
            }
            check(
                &dev,
                &Case::new(jobs, nkh, 128, 0.9, (nkh * 1000 + jobs) as u64),
            )?;
        }
    }
    Ok(())
}

#[test]
fn claim_defers_exactly_the_job_holding_each_class_last_band() -> Result<()> {
    let dev = cuda()?;
    let mut case = Case::new(500, 4, 128, 1.0, 101);
    // Each class one slot short of its demand: only the band past the end
    // defers, and with it only its own job.
    let full: Vec<u32> = case.lease_len.clone();
    case.set_lease(|c| full[c].saturating_sub(1));
    let want = check(&dev, &case)?;
    let deferred = want.summary_flags.iter().filter(|&&f| f == 1).count();
    let short_classes = full.iter().filter(|&&n| n > 0).count();
    assert!(
        (1..=short_classes).contains(&deferred),
        "{deferred} jobs deferred"
    );
    Ok(())
}

#[test]
fn claim_with_no_lease_left_defers_every_job() -> Result<()> {
    let dev = cuda()?;
    let mut case = Case::new(300, 4, 128, 1.0, 111);
    case.taken = case.lease_len.clone();
    let want = check(&dev, &case)?;
    assert!(want.summary_flags.iter().all(|&f| f == 1));
    assert_eq!(
        want.error_word, 0,
        "running dry is a deferral, not an error"
    );
    Ok(())
}

#[test]
fn claim_with_every_band_in_one_class() -> Result<()> {
    let dev = cuda()?;
    let mut case = Case::new(4000, 4, 128, 1.0, 121);
    for t in case.k_tags.iter_mut().chain(case.v_tags.iter_mut()) {
        *t = CODES[7];
    }
    case.set_lease(|c| {
        if c == class_of(7) as usize {
            4000 * 32
        } else {
            0
        }
    });
    let want = check(&dev, &case)?;
    assert_eq!(want.taken[class_of(7) as usize], 4000 * 32);
    Ok(())
}

#[test]
fn claim_under_an_uncovered_override_defers_everything_and_reports_it() -> Result<()> {
    let dev = cuda()?;
    let mut case = Case::new(200, 4, 128, 1.0, 131);
    case.k_override = 4; // no class covers it
    let want = check(&dev, &case)?;
    assert_eq!(want.error_word, 1);
    assert!(want.summary_flags.iter().all(|&f| f == 1));
    Ok(())
}

#[test]
fn claim_treats_a_class_past_n_class_as_uncovered() -> Result<()> {
    let dev = cuda()?;
    let mut case = Case::new(200, 4, 128, 1.0, 141);
    // A table entry naming a class the launch does not count.
    case.code_class[CODES[3] as usize] = N_CLASS as i8;
    let want = check(&dev, &case)?;
    assert_eq!(want.error_word, 1);
    Ok(())
}

#[test]
fn claim_refuses_arguments_outside_its_contract_without_launching() -> Result<()> {
    let dev = cuda()?;
    let stream = dev.cuda_stream();
    let null = std::ptr::null_mut::<u8>();
    let call = |n_jobs: i32, nkh: i32, hd: i32, n_class: i32| unsafe {
        run_kv_gap_claim(
            null as *const i32,
            null as *const i32,
            null as *const f32,
            null as *const f32,
            null as *const i32,
            null as *const i32,
            NO_OVERRIDE,
            NO_OVERRIDE,
            null as *const i8,
            null as *const u8,
            null as *const u32,
            null as *const u64,
            null as *const u32,
            null as *mut u32,
            SINK,
            null,
            null,
            null as *mut u32,
            null as *mut u32,
            null as *mut u64,
            1,
            n_jobs,
            nkh,
            hd,
            n_class,
            stream.cu_stream() as *mut std::ffi::c_void,
        )
    };
    for (args, what) in [
        ((0, 4, 128, 4), "no jobs"),
        ((-1, 4, 128, 4), "negative jobs"),
        ((10, 0, 128, 4), "no heads"),
        ((10, MAX_KV_HEAD as i32 + 1, 128, 4), "too many heads"),
        ((10, 4, 96, 4), "a head_dim with no kernel"),
        ((10, 4, 64, 4), "a head_dim the convert has no record for"),
        ((10, 4, 512, 4), "a head_dim the convert has no record for"),
        ((10, 4, 128, 0), "no classes"),
        ((10, 4, 128, MAX_CLASS as i32 + 1), "too many classes"),
    ] {
        let (j, k, h, c) = args;
        assert_eq!(call(j, k, h, c), -1, "{what}");
    }
    Ok(())
}

// The synthetic ladder fits the kernel's class counters.
const _: () = assert!(N_CLASS <= MAX_CLASS);
