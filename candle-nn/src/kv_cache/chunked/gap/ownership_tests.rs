//! The ownership rules (`docs/kv_quantize_in_the_gap.md` §3.8, §7), as a
//! randomised model test with no device.
//!
//! The real lease, demand, table and block-take code is driven through random
//! interleavings of forwards, gaps, truncates, front evictions and catch-ups;
//! the device's claim is replaced by its sequential reference (a per-class
//! prefix count in table order). Slots are handles that put themselves back in
//! a model pool when dropped, and the pool refuses a slot it already holds, so
//! a slot freed twice fails at the free. After every step every slot is in
//! exactly one place: the pool, the lease, or an installed block; a block is
//! never both installed and handed back for retry; and the cursor never runs
//! past the chunks.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use super::super::size_class::SizeClass;
use super::super::types::CHUNK_SIZE;
use super::assemble::take_block;
use super::demand::{count_bands, gap_demand, PerClass};
use super::lease::Lease;
use super::summary::{HeadRow, BANDS_PER_HEAD};
use super::table::{cursor_after_front_eviction, cursor_after_truncate, take_range, LayerShape};

/// The classes the model's level reaches.
const CLASSES: [usize; 3] = [2, 5, 9];
/// Slots per class in the model pool.
const PER_CLASS: u64 = 48;
const N_KV_HEAD: usize = 2;
const SEQS: usize = 3;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The model pool: each class's free slots, lowest first.
#[derive(Default)]
struct Pool {
    free: PerClass<BTreeSet<u64>>,
}

/// A slot handle that returns itself to the pool when dropped — once.
struct Slot {
    class: usize,
    id: u64,
    pool: Rc<RefCell<Pool>>,
}

impl Drop for Slot {
    fn drop(&mut self) {
        let fresh = self.pool.borrow_mut().free[self.class].insert(self.id);
        assert!(
            fresh,
            "slot {} of class {} freed twice",
            self.id, self.class
        );
    }
}

fn alloc(pool: &Rc<RefCell<Pool>>, class: usize, n: usize) -> Vec<(Slot, u64)> {
    let mut out = Vec::new();
    for _ in 0..n {
        let Some(id) = pool.borrow_mut().free[class].pop_first() else {
            break;
        };
        out.push((
            Slot {
                class,
                id,
                pool: pool.clone(),
            },
            id,
        ));
    }
    out
}

/// A native chunk's identity — what the match check compares.
type Native = Rc<u64>;

struct Block {
    native: Option<Native>,
    installed: Option<Vec<Slot>>,
    /// The class each of its bands selects: fixed by its bytes, as the
    /// selection is deterministic on unchanged native data.
    classes: Vec<usize>,
}

#[derive(Default)]
struct Seq {
    blocks: Vec<Block>,
    /// Tokens in the writer chunk past the last full block.
    writer: u32,
    cursor: usize,
}

impl Seq {
    fn shape(&self) -> LayerShape {
        let mut usage: Vec<u32> = self.blocks.iter().map(|_| CHUNK_SIZE as u32).collect();
        let mut native: Vec<bool> = self.blocks.iter().map(|b| b.installed.is_none()).collect();
        usage.push(self.writer);
        native.push(true);
        LayerShape {
            writer_start: 0,
            first: 0,
            usage,
            native,
        }
    }
}

struct Job {
    seq: usize,
    block: usize,
    pin: Native,
    classes: Vec<usize>,
}

/// The claim's sequential reference: each band's index in its class, in table
/// order, against shared counts; a job deferred when any band falls past its
/// class's lease.
fn reference_claim(jobs: &[Job], lease_len: &PerClass<usize>) -> (Vec<Vec<HeadRow>>, Vec<bool>) {
    let mut taken = [0usize; SizeClass::COUNT];
    let mut rows = Vec::new();
    let mut deferred = Vec::new();
    for job in jobs {
        let mut d = false;
        let mut job_rows = Vec::new();
        for h in 0..N_KV_HEAD {
            let mut row = HeadRow {
                k_pal: vec![0; 32],
                v_pal: vec![0; 32],
                k_fmt: [0; 4],
                v_fmt: [0; 4],
                k_scale: [1.0; 4],
                v_scale: [1.0; 4],
                k_ptr: [0; 4],
                v_ptr: [0; 4],
                lease_idx: [0; BANDS_PER_HEAD],
            };
            for b in 0..BANDS_PER_HEAD {
                let c = job.classes[h * BANDS_PER_HEAD + b];
                let (p, is_v) = (b / 2, b % 2 == 1);
                let fmt = fmt_of(c);
                if is_v {
                    row.v_fmt[p] = fmt;
                } else {
                    row.k_fmt[p] = fmt;
                }
                row.lease_idx[b] = taken[c] as u32;
                d |= taken[c] >= lease_len[c];
                taken[c] += 1;
            }
            job_rows.push(row);
        }
        rows.push(job_rows);
        deferred.push(d);
    }
    (rows, deferred)
}

fn fmt_of(class: usize) -> u8 {
    class as u8 + 1
}

fn class_of(fmt: u8) -> Option<usize> {
    let c = (fmt as usize).checked_sub(1)?;
    CLASSES.contains(&c).then_some(c)
}

struct World {
    pool: Rc<RefCell<Pool>>,
    seqs: Vec<Seq>,
    next_native: u64,
    rng: Rng,
}

impl World {
    fn new(seed: u64) -> Self {
        let pool = Rc::new(RefCell::new(Pool::default()));
        for &c in &CLASSES {
            pool.borrow_mut().free[c] = (0..PER_CLASS).collect();
        }
        Self {
            pool,
            seqs: (0..SEQS).map(|_| Seq::default()).collect(),
            next_native: 0,
            rng: Rng(seed | 1),
        }
    }

    /// A forward appends `tokens` to sequence `s`, completing blocks.
    fn forward(&mut self, s: usize, tokens: u32) {
        let mut writer = self.seqs[s].writer + tokens;
        while writer >= CHUNK_SIZE as u32 {
            writer -= CHUNK_SIZE as u32;
            self.next_native += 1;
            let classes = (0..N_KV_HEAD * BANDS_PER_HEAD)
                .map(|_| CLASSES[self.rng.below(CLASSES.len() as u64) as usize])
                .collect();
            self.seqs[s].blocks.push(Block {
                native: Some(Rc::new(self.next_native)),
                installed: None,
                classes,
            });
        }
        self.seqs[s].writer = writer;
    }

    fn truncate(&mut self, s: usize, keep: usize) {
        let seq = &mut self.seqs[s];
        seq.blocks.truncate(keep);
        seq.writer = 0;
        seq.cursor = cursor_after_truncate(seq.cursor, seq.blocks.len());
    }

    fn evict_front(&mut self, s: usize, n: usize) {
        let seq = &mut self.seqs[s];
        let n = n.min(seq.blocks.len());
        seq.blocks.drain(..n);
        seq.cursor = cursor_after_front_eviction(seq.cursor, n);
    }

    /// The table a gap takes: the retried jobs first, then every sequence's
    /// complete native blocks past its cursor.
    fn table(&mut self, retry: Vec<Job>) -> (Vec<Job>, usize) {
        let mut jobs = retry;
        let n_retry = jobs.len();
        for (s, seq) in self.seqs.iter_mut().enumerate() {
            let range = take_range(seq.cursor, std::slice::from_ref(&seq.shape()));
            for block in range.clone() {
                let b = &seq.blocks[block];
                jobs.push(Job {
                    seq: s,
                    block,
                    pin: b.native.clone().expect("a block past the cursor is native"),
                    classes: b.classes.clone(),
                });
            }
            if !range.is_empty() {
                seq.cursor = range.end;
            }
        }
        (jobs, n_retry)
    }

    /// Every slot is in exactly one place.
    fn check(&self, lease: &Lease<Slot>) {
        let mut seen: PerClass<BTreeSet<u64>> = std::array::from_fn(|_| BTreeSet::new());
        let mut count = |c: usize, id: u64, place: &str| {
            assert!(seen[c].insert(id), "class {c} slot {id} also in {place}");
        };
        for &c in &CLASSES {
            for &id in &self.pool.borrow().free[c] {
                count(c, id, "the pool");
            }
        }
        for seq in &self.seqs {
            assert!(
                seq.cursor <= seq.blocks.len(),
                "the cursor ran past the chunks"
            );
            for b in &seq.blocks {
                assert!(
                    b.native.is_some() != b.installed.is_some(),
                    "a block is native or installed, never both or neither"
                );
                for s in b.installed.iter().flatten() {
                    count(s.class, s.id, "a block");
                }
            }
        }
        for &c in &CLASSES {
            assert!(lease.held()[c] == 0, "nothing is leased between gaps");
            assert_eq!(seen[c].len() as u64, PER_CLASS, "class {c}: a slot is lost");
        }
    }
}

/// One gap and its catch-up, with `between` landing while the gap runs.
fn gap(
    w: &mut World,
    retry: Vec<Job>,
    retried: PerClass<usize>,
    between: &mut dyn FnMut(&mut World),
) -> Vec<Job> {
    let (jobs, n_retry) = w.table(retry);
    // Demand: the retried jobs exactly, the new ones split evenly.
    let new_bands = (jobs.len() - n_retry) * N_KV_HEAD * BANDS_PER_HEAD;
    let mut reach = [false; SizeClass::COUNT];
    for &c in &CLASSES {
        reach[c] = true;
    }
    let demand = gap_demand(&retried, new_bands, None, &reach);
    let mut lease: Lease<Slot> = Lease::default();
    for &c in &CLASSES {
        // The ground gives what it has, sometimes less than asked.
        let short = if w.rng.below(4) == 0 {
            w.rng.below(demand[c] as u64 + 1) as usize
        } else {
            0
        };
        lease.add(c, alloc(&w.pool, c, demand[c] - short));
    }
    let lease_len = lease.held();
    let (rows, deferred) = reference_claim(&jobs, &lease_len);

    between(w);

    // The catch-up.
    let mut next_retry = Vec::new();
    for (j, mut job) in jobs.into_iter().enumerate() {
        let seq = &mut w.seqs[job.seq];
        // At its index, or wherever an eviction moved it, by identity.
        let holds = |b: &Block| b.native.as_ref().is_some_and(|n| Rc::ptr_eq(n, &job.pin));
        let at = if seq.blocks.get(job.block).is_some_and(holds) {
            Some(job.block)
        } else {
            seq.blocks.iter().position(holds)
        };
        let Some(at) = at else {
            continue;
        };
        if deferred[j] {
            job.block = at;
            next_retry.push(job);
            continue;
        }
        let slots =
            take_block(&mut lease, &rows[j], class_of).expect("the summary matches its lease");
        let b = &mut seq.blocks[at];
        b.native = None;
        b.installed = Some(slots);
    }
    drop(lease.release());
    w.check(&lease);
    for job in &next_retry {
        let b = &w.seqs[job.seq].blocks[job.block];
        assert!(b.installed.is_none(), "a retried block is not installed");
    }
    // No complete native block is stranded: each one behind its sequence's
    // cursor is handed back for the next gap.
    for (s, seq) in w.seqs.iter().enumerate() {
        for (i, b) in seq.blocks.iter().enumerate().take(seq.cursor) {
            if b.native.is_some() {
                assert!(
                    next_retry.iter().any(|r| r.seq == s && r.block == i),
                    "seq {s} block {i}: native behind the cursor and not retried"
                );
            }
        }
    }
    next_retry
}

#[test]
fn every_slot_has_one_owner_through_random_interleavings() {
    for seed in 0..400u64 {
        let mut w = World::new(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) + 1);
        let mut retry: Vec<Job> = Vec::new();
        for _ in 0..24 {
            let n_fwd = w.rng.below(4);
            for _ in 0..n_fwd {
                let s = w.rng.below(SEQS as u64) as usize;
                let t = 1 + w.rng.below(70) as u32;
                w.forward(s, t);
            }
            // What lands while the gap is outstanding.
            let op = w.rng.below(6);
            let s = w.rng.below(SEQS as u64) as usize;
            let k = w.rng.below(4) as usize;
            let mut retried = [0usize; SizeClass::COUNT];
            for job in &retry {
                for &c in &job.classes {
                    retried[c] += 1;
                }
            }
            retry = gap(&mut w, retry, retried, &mut |w| match op {
                0 => {
                    let keep = w.seqs[s].blocks.len().saturating_sub(k);
                    w.truncate(s, keep);
                }
                1 => w.evict_front(s, k),
                2 => w.forward(s, 40),
                _ => {}
            });
        }
        // Drop everything: every slot goes home exactly once.
        drop(retry);
        w.seqs.clear();
        let pool = w.pool.borrow();
        for &c in &CLASSES {
            assert_eq!(
                pool.free[c].len() as u64,
                PER_CLASS,
                "seed {seed} class {c}"
            );
        }
    }
}

/// The reference claim and the demand agree with the counting the catch-up
/// uses: a retried job's demand, counted from its rows, is its bands' classes.
#[test]
fn a_retried_jobs_demand_counted_from_its_rows_is_its_bands() {
    let mut w = World::new(7);
    w.forward(0, 200);
    let (jobs, _) = w.table(Vec::new());
    let (rows, _) = reference_claim(&jobs, &[0; SizeClass::COUNT]);
    for (job, rows) in jobs.iter().zip(&rows) {
        let mut want = [0usize; SizeClass::COUNT];
        for &c in &job.classes {
            want[c] += 1;
        }
        assert_eq!(count_bands(rows, class_of).unwrap(), want);
    }
}
