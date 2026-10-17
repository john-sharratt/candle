//! The service end to end on the host: mapped rings, experts in pinned memory,
//! a work slot written as bucketize writes it, the doorbell rung as the export
//! kernel rings it — several layers back to back, each answered with exactly
//! what each expert alone computes for its rows; and cold experts (§5.4),
//! waited on in a live table and given up through the fault word.

use std::ptr;
use std::sync::atomic::{fence, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use candle::quantized::ko_quant::quantize_ko;
use candle::quantized::GgmlDType;
use candle::Device;

use super::cold::{ColdSource, PagedImages};
use super::kernel::act::{ActTiles, TILE};
use super::kernel::expert::{forward, Expert};
use super::kernel::ko::Ko;
use super::kernel::simd::Isa;
use super::rings::{q8a1024_bytes, CpuRings, Mapped};
use super::service::{CpuService, PoolCores, RowShape};
use super::topology::{physical_cores, pool_cores};
use crate::models::expert_lre::dispatch::AbortWord;
use crate::models::expert_lre::fault::{ExpertFault, FaultWord};
use crate::models::expert_lre::live_table::LiveTable;

const HIDDEN: usize = 2560;
const FF: usize = 640;
const Q8A1024_BLOCK: usize = 1152;
const Q8A1024_META: usize = 1024;
/// The MoE row every layer here is, of the two the service is built for.
const ROW: usize = 1;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// `act`'s rows written into `dst` in the q8a1024 layout the export kernel
/// copies: tile `row · tiles + t`, codes then the `{scale, sum}` slot.
fn write_q8a1024(act: &ActTiles, dst: *mut u8) {
    let tiles = act.tiles;
    for row in 0..act.tokens {
        for t in 0..tiles {
            let flat = row * tiles + t;
            let (q, scale, sum) = act.tile(row, t);
            let qs = (flat >> 3) * Q8A1024_BLOCK + (flat & 7) * TILE;
            let ds = (flat >> 3) * Q8A1024_BLOCK + Q8A1024_META + (flat & 7) * 16;
            // SAFETY: `dst` holds `q8a1024_bytes(act.tokens, tiles)` bytes.
            unsafe {
                ptr::copy_nonoverlapping(q.as_ptr() as *const u8, dst.add(qs), TILE);
                let s = half::f16::from_f32(scale).to_le_bytes();
                let m = half::f16::from_f32(sum).to_le_bytes();
                ptr::copy_nonoverlapping(s.as_ptr(), dst.add(ds), 2);
                ptr::copy_nonoverlapping(m.as_ptr(), dst.add(ds + 2), 2);
            }
        }
    }
}

/// One layer's work slot: the header, then per expert `{expert, ring_start,
/// count, _, gate, up, down}`.
fn write_work(work: &Mapped, row: i32, seq: u32, experts: &[(u32, u32, u32, [u64; 3])]) {
    let n_rows: u32 = experts.iter().map(|e| e.2).sum();
    let p = work.host();
    // SAFETY: inside the slot; the service reads it only after the doorbell.
    unsafe {
        let h = p as *mut u32;
        ptr::write_volatile(h, experts.len() as u32);
        ptr::write_volatile(h.add(1), n_rows);
        ptr::write_volatile(h.add(2), row as u32);
        ptr::write_volatile(h.add(3), seq);
        for (j, &(e, start, count, addrs)) in experts.iter().enumerate() {
            let at = p.add(16 + 40 * j);
            ptr::write_volatile(at as *mut u32, e);
            ptr::write_volatile(at.add(4) as *mut u32, start);
            ptr::write_volatile(at.add(8) as *mut u32, count);
            ptr::write_volatile(at.add(12) as *mut u32, 0);
            for (i, &a) in addrs.iter().enumerate() {
                ptr::write_volatile(at.add(16 + 8 * i) as *mut u64, a);
            }
        }
    }
}

/// Four experts in pinned slot images, and the service over a production-
/// shaped pool.
struct Fixture {
    images: Vec<(Vec<u8>, (usize, usize, usize))>,
    slots: Mapped,
    image_bytes: usize,
    rings: Arc<CpuRings>,
    service: CpuService,
    seed: u64,
    /// The last rung layer's input map: ring row → input row.
    inputs: Vec<usize>,
}

impl Fixture {
    fn new() -> candle::Result<Self> {
        let dt = GgmlDType::Q3_KO;
        let mut seed = 0x5e7_71ce;
        let images: Vec<(Vec<u8>, (usize, usize, usize))> = (0..4)
            .map(|_| {
                let mut w =
                    |n: usize| -> Vec<f32> { (0..n).map(|_| lcg(&mut seed) * 0.05).collect() };
                let (g, u, d) = (
                    quantize_ko(&w(FF * HIDDEN), FF, HIDDEN, dt),
                    quantize_ko(&w(FF * HIDDEN), FF, HIDDEN, dt),
                    quantize_ko(&w(HIDDEN * FF), HIDDEN, FF, dt),
                );
                let off = (0, g.len(), g.len() + u.len());
                ([g, u, d].concat(), off)
            })
            .collect();
        let image_bytes = images[0].0.len();
        let slots = Mapped::new(4 * image_bytes)?;
        for (i, (img, _)) in images.iter().enumerate() {
            // SAFETY: inside the allocation.
            unsafe {
                ptr::copy_nonoverlapping(img.as_ptr(), slots.host().add(i * image_bytes), img.len())
            };
        }
        let shape = RowShape {
            hidden: HIDDEN,
            ff: FF,
            ko: Ko::Q3,
            bytes: [image_bytes / 3; 3],
        };
        let rings = Arc::new(CpuRings::new(128, HIDDEN)?);
        // The pool the expert cache starts (`attach`): every core it may take,
        // so the layer schedule cuts its tasks as production does.
        let cores = pool_cores(&physical_cores().unwrap(), 1);
        let sets: Vec<u32> = cores.iter().map(|c| c.primary()).collect();
        let fast = cores
            .iter()
            .take_while(|c| c.efficiency_class == cores[0].efficiency_class)
            .count();
        let service = CpuService::start(
            rings.clone(),
            vec![shape, shape],
            PoolCores {
                caller: sets[0],
                workers: sets[1..].to_vec(),
                fast_workers: fast - 1,
            },
            Duration::from_millis(5),
            Duration::from_micros(200),
            12e9,
        )?;
        service.wake();
        Ok(Self {
            images,
            slots,
            image_bytes,
            rings,
            service,
            seed,
            inputs: Vec::new(),
        })
    }

    /// Image `e`'s slot image base.
    fn base(&self, e: usize) -> u64 {
        self.slots.host() as u64 + (e * self.image_bytes) as u64
    }

    /// Image `e`'s three projections' addresses.
    fn addrs(&self, e: usize) -> [u64; 3] {
        let (g, u, d) = self.images[e].1;
        let b = self.base(e);
        [b + g as u64, b + u as u64, b + d as u64]
    }

    /// A live table for the two rows, its offsets image 0's — every image's.
    fn table(&self) -> LiveTable {
        let (g, u, d) = self.images[0].1;
        LiveTable::host_only(2, 4, [g as u64, u as u64, d as u64])
    }

    /// Ring layer `seq`: a work slot naming `layer`'s `(image, rows, cold)` —
    /// a cold expert's addresses 0, as bucketize writes them — with expert ids
    /// their images', an input row of its own per ring row, then the doorbell.
    fn ring(&mut self, seq: u64, layer: &[(usize, usize, bool)]) {
        let n_rows: usize = layer.iter().map(|e| e.1).sum();
        self.ring_with_inputs(seq, layer, &(0..n_rows).collect::<Vec<_>>());
    }

    /// The same with each ring row reading input row `inputs[j]`: random input
    /// rows, as many as `inputs` names, and the input map.
    fn ring_with_inputs(&mut self, seq: u64, layer: &[(usize, usize, bool)], inputs: &[usize]) {
        let n_rows: usize = layer.iter().map(|e| e.1).sum();
        assert_eq!(inputs.len(), n_rows);
        let n_in = inputs.iter().max().map_or(0, |&m| m + 1);
        let x: Vec<f32> = (0..n_in * HIDDEN).map(|_| lcg(&mut self.seed)).collect();
        let act = ActTiles::quantize(&x, n_in, HIDDEN);
        assert!(q8a1024_bytes(n_in, HIDDEN / TILE) <= self.rings.rows.bytes());
        write_q8a1024(&act, self.rings.rows.host());
        let map = self.rings.inputs.host() as *mut u32;
        // SAFETY: `1 + cap_rows` mapped words; the service reads them only
        // after the doorbell.
        unsafe {
            ptr::write_volatile(map, n_in as u32);
            for (j, &r) in inputs.iter().enumerate() {
                ptr::write_volatile(map.add(1 + j), r as u32);
            }
        }
        self.inputs = inputs.to_vec();
        let mut entries = Vec::new();
        let mut start = 0u32;
        for &(e, n, cold) in layer {
            let addrs = if cold { [0; 3] } else { self.addrs(e) };
            entries.push((e as u32, start, n as u32, addrs));
            start += n as u32;
        }
        write_work(&self.rings.work, ROW as i32, seq as u32, &entries);
        fence(Ordering::SeqCst);
        // SAFETY: the mapped doorbell.
        unsafe { ptr::write_volatile(self.rings.doorbell.host() as *mut u64, seq) };
    }

    fn done(&self) -> u64 {
        // SAFETY: the mapped done word.
        unsafe { ptr::read_volatile(self.rings.done.host() as *const u64) }
    }

    /// Waits for `done == seq`, then checks every result row against its
    /// expert's `forward` over that row, byte for byte.
    fn expect_answer(&self, seq: u64, layer: &[(usize, usize, bool)], what: &str) {
        let t0 = Instant::now();
        while self.done() != seq {
            assert!(t0.elapsed() < Duration::from_secs(5), "{what}: no answer");
            std::hint::spin_loop();
        }
        fence(Ordering::SeqCst);
        let n_rows: usize = layer.iter().map(|e| e.1).sum();
        let n_in = self.inputs.iter().max().map_or(0, |&m| m + 1);
        // SAFETY: `n_rows · HIDDEN` floats, written before `done`.
        let got = unsafe {
            std::slice::from_raw_parts(self.rings.results.host() as *const f32, n_rows * HIDDEN)
        };
        // The ring's codes re-read the way the service reads them, so the f16
        // sum is the same.
        let ring = unsafe {
            std::slice::from_raw_parts(
                self.rings.rows.host() as *const u8,
                q8a1024_bytes(n_in, HIDDEN / TILE),
            )
        };
        let mut row = 0usize;
        for &(e, n, _) in layer {
            let rows: Vec<usize> = self.inputs[row..row + n].to_vec();
            let sub = ActTiles::from_q8a1024(ring, &rows, HIDDEN / TILE, false);
            let ex = Expert::from_slot(&self.images[e].0, self.images[e].1, HIDDEN, FF, Ko::Q3);
            let mut want = vec![0f32; n * HIDDEN];
            forward(Isa::detect(), &ex, &sub, &mut want);
            for r in 0..n {
                let g = &got[(row + r) * HIDDEN..(row + r + 1) * HIDDEN];
                let w = &want[r * HIDDEN..(r + 1) * HIDDEN];
                assert!(
                    g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{what}: expert {e}, ring row {}",
                    row + r
                );
            }
            row += n;
        }
    }
}

/// Three layers, each giving the service a different subset at different
/// counts — the decode shape (one row each) and a prompt's (several). Every
/// result row must be, byte for byte, its expert's `forward` over that row.
#[test]
fn the_service_answers_each_layer_with_its_experts_rows() -> candle::Result<()> {
    let _context = Device::new_cuda(0)?;
    let mut fx = Fixture::new()?;
    // (image, rows) per layer: decode's single rows, a few at once, then a
    // prompt's — twelve experts (the four images, each standing for three),
    // counts 1 to 30, 104 rows.
    let prompt: Vec<(usize, usize, bool)> = [1, 2, 3, 5, 8, 13, 21, 1, 4, 7, 9, 30]
        .iter()
        .enumerate()
        .map(|(i, &n)| (i % 4, n, false))
        .collect();
    // Then nine to sixteen single-row experts: the prompt layers a cap of 16
    // gives, each expert one row.
    let wide: Vec<Vec<(usize, usize, bool)>> = (9..=16)
        .map(|k| (0..k).map(|i| (i % 4, 1, false)).collect())
        .collect();
    let mut layers: Vec<&[(usize, usize, bool)]> = vec![
        &[(0, 1, false), (2, 1, false)],
        &[(1, 3, false), (3, 2, false), (0, 1, false)],
        &[(2, 1, false)],
        &prompt,
    ];
    layers.extend(wide.iter().map(|w| w.as_slice()));
    for (l, layer) in layers.iter().enumerate() {
        let seq = 0x1000 + l as u64;
        fx.ring(seq, layer);
        fx.expect_answer(seq, layer, &format!("layer {l}"));
    }
    assert_eq!(fx.service.served(), layers.len() as u64);
    Ok(())
}

/// One input row per token, however many experts it routes: a decode token
/// to three experts is one row over the link, and each expert's result row is
/// still its own; then two tokens, one routed to two experts and one to three,
/// interleaved across the ring rows.
#[test]
fn experts_sharing_a_token_read_its_one_input_row() -> candle::Result<()> {
    let _context = Device::new_cuda(0)?;
    let mut fx = Fixture::new()?;
    let decode: &[(usize, usize, bool)] = &[(0, 1, false), (2, 1, false), (3, 1, false)];
    fx.ring_with_inputs(0x4000, decode, &[0, 0, 0]);
    fx.expect_answer(0x4000, decode, "one token, three experts");
    let two: &[(usize, usize, bool)] = &[(1, 2, false), (3, 1, false), (0, 2, false)];
    fx.ring_with_inputs(0x4001, two, &[0, 1, 1, 0, 1]);
    fx.expect_answer(0x4001, two, "two tokens, five assignments");
    Ok(())
}

/// Cold experts: the service computes the experts already in memory, waits
/// for the cold ones' live entries — published here 20 ms after the doorbell,
/// as the stager publishes a pad slot — and answers the whole layer, byte
/// for byte.
#[test]
fn the_service_waits_for_a_cold_expert_and_computes_it() -> candle::Result<()> {
    let _context = Device::new_cuda(0)?;
    let mut fx = Fixture::new()?;
    let table = Arc::new(fx.table());
    let fault = Arc::new(FaultWord::new()?);
    let pinned = [(fx.base(0), fx.base(0) + 4 * fx.image_bytes as u64), (0, 0)];
    fx.service.attach_cold(ColdSource {
        table: table.clone(),
        abort: Arc::new(AbortWord::new()?),
        fault: fault.clone(),
        pinned,
        paged: Arc::new(PagedImages::new(
            table.n_rows(),
            table.n_experts(),
            Arc::new(()),
        )),
        spin_limit: Duration::from_secs(1),
    });
    // A warm decode row, two cold ones; then a cold expert alone.
    let layers: [&[(usize, usize, bool)]; 2] = [
        &[(0, 1, false), (2, 1, true), (3, 2, true)],
        &[(1, 1, true)],
    ];
    for (l, layer) in layers.iter().enumerate() {
        let seq = 0x2000 + l as u64;
        fx.ring(seq, layer);
        let t0 = Instant::now();
        thread::sleep(Duration::from_millis(20));
        assert_ne!(
            fx.done(),
            seq,
            "layer {l}: answered before its cold experts landed"
        );
        for &(e, _, cold) in layer.iter() {
            if cold {
                table.publish(ROW, e, fx.base(e));
            }
        }
        fx.expect_answer(seq, layer, &format!("cold layer {l}"));
        assert!(t0.elapsed() >= Duration::from_millis(20));
        for &(e, _, cold) in layer.iter() {
            if cold {
                table.clear(ROW, e);
            }
        }
    }
    assert!(fault.peek().is_none(), "no wait gave up");
    Ok(())
}

/// Cold experts with a pageable copy: the service computes them from it at
/// once — nothing is ever published, and no pinned range covers them — and
/// answers the layer byte for byte.
#[test]
fn the_service_computes_a_pageable_expert_without_waiting() -> candle::Result<()> {
    let _context = Device::new_cuda(0)?;
    let mut fx = Fixture::new()?;
    let table = Arc::new(fx.table());
    let fault = Arc::new(FaultWord::new()?);
    // The fixture's slots outlive the service, so the images need no owner.
    let mut paged = PagedImages::new(table.n_rows(), table.n_experts(), Arc::new(()));
    for e in [1, 3] {
        paged.set(ROW, e, fx.base(e));
    }
    fx.service.attach_cold(ColdSource {
        table: table.clone(),
        abort: Arc::new(AbortWord::new()?),
        fault: fault.clone(),
        pinned: [(0, 0), (0, 0)],
        paged: Arc::new(paged),
        spin_limit: Duration::from_secs(10),
    });
    let layer: &[(usize, usize, bool)] = &[(1, 1, true), (3, 2, true)];
    let t0 = Instant::now();
    fx.ring(0x5000, layer);
    fx.expect_answer(0x5000, layer, "pageable layer");
    // Nothing is ever published, so a service that waited on the live table
    // would answer only past the 10 s spin limit, if at all.
    assert!(
        t0.elapsed() < Duration::from_secs(1),
        "answered in {:?}: the service waited",
        t0.elapsed()
    );
    assert!(fault.peek().is_none(), "no wait gave up");
    Ok(())
}

/// A cold expert that never lands: once the abort word is raised the service
/// claims the fault word naming the expert's row and id, with the abort flag,
/// and publishes no `done`, so the device's wait on the pool gives up at the
/// word. A cold expert published into device memory, which the pool cannot
/// read, is given up the same way. Each time the next layer is still served.
#[test]
fn the_service_gives_a_lost_cold_expert_up_through_the_fault_word() -> candle::Result<()> {
    let _context = Device::new_cuda(0)?;
    let mut fx = Fixture::new()?;
    let table = Arc::new(fx.table());
    let abort = Arc::new(AbortWord::new()?);
    let fault = Arc::new(FaultWord::new()?);
    let pinned = [(fx.base(0), fx.base(0) + 4 * fx.image_bytes as u64), (0, 0)];
    fx.service.attach_cold(ColdSource {
        table: table.clone(),
        abort: abort.clone(),
        fault: fault.clone(),
        pinned,
        paged: Arc::new(PagedImages::new(
            table.n_rows(),
            table.n_experts(),
            Arc::new(()),
        )),
        spin_limit: Duration::from_secs(10),
    });
    let wait_for_fault = || {
        let t0 = Instant::now();
        while fault.peek().is_none() {
            assert!(t0.elapsed() < Duration::from_secs(5), "no fault claimed");
            thread::sleep(Duration::from_millis(1));
        }
        fault.peek().unwrap()
    };

    // Never published; the abort word raised after 20 ms.
    fx.ring(0x3000, &[(0, 1, false), (3, 1, true)]);
    thread::sleep(Duration::from_millis(20));
    abort.raise();
    let f = wait_for_fault();
    assert_eq!(
        (f.aborted, f.row, f.expert),
        (true, ROW, 3),
        "the fault names the lost expert"
    );
    thread::sleep(Duration::from_millis(20));
    assert_ne!(fx.done(), 0x3000, "a layer given up gets no done");
    fault.clear();
    // The abort word is the dispatch's for good once raised; a fresh one for
    // the rest is not needed — the next layer has no cold expert.
    let next: &[(usize, usize, bool)] = &[(2, 1, false)];
    fx.ring(0x3001, next);
    fx.expect_answer(0x3001, next, "the layer after a give-up");

    // Published into device memory: given up at once.
    let device_addr = 0x7f00_0000_0000u64;
    table.publish(ROW, 1, device_addr);
    fx.ring(0x3002, &[(1, 1, true)]);
    let f = wait_for_fault();
    assert_eq!((f.row, f.expert), (ROW, 1));
    assert_eq!(
        f,
        ExpertFault {
            aborted: true,
            row: ROW,
            expert: 1,
            waited_us: f.waited_us,
        },
        "the abort word is still raised, so the claim carries its flag"
    );
    thread::sleep(Duration::from_millis(20));
    assert_ne!(fx.done(), 0x3002);
    fault.clear();
    fx.ring(0x3003, next);
    fx.expect_answer(0x3003, next, "the layer after an unreadable publish");
    Ok(())
}
