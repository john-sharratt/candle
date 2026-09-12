//! Same-pass checkpoints that capture and panic where the residual first goes
//! non-finite.
//!
//! ## The first panic to fire is the first corruption
//!
//! That is the whole method, and it needs no bookkeeping to work. Checkpoints
//! are placed along the forward in the order the data flows through it, each
//! one checks synchronously, and each one panics on the spot. Execution order
//! **is** causal order, so whichever checkpoint fires is by construction the
//! earliest point at which the value was bad — everything downstream of it is a
//! consequence and everything upstream was clean.
//!
//! Adding a checkpoint therefore only ever narrows the answer: a new one placed
//! before the current winner either fires (and the fault moves up to it) or
//! does not (and the ground between them is cleared). The chain converges by
//! accretion, and no state has to be carried between waves to make it do so.
//!
//! Each checkpoint costs one fence. That is affordable because they fire once —
//! the process stops — and because it has been measured: the gate-GEMM
//! checkpoint reproduced the fault inside two minutes with a fence on every MoE
//! layer of every wave, so this fault is not one a fence hides.
//!
//! ## Why a capture rather than more logging
//!
//! The assert instrumentation says *where* the fault appears and how large the
//! values are, and it has taken that as far as it goes: `moe.gate_out.L38`
//! reaches 5.5e7 from expert weights that are provably bounded by 0.78 (all
//! 33e9 of them, examined at load) and from a gathered int8 operand. More
//! statistics cannot separate the two remaining explanations, because they
//! produce identical statistics:
//!
//! * The bytes that reached the kernel really do multiply out to 5.5e7 — the
//!   operand or a table is wrong *before* the launch.
//! * The bytes that reached the kernel were not the bytes we think — a copy
//!   that had not landed, a buffer reused under an in-flight read, an arena
//!   lease outliving its generation.
//!
//! A replay separates them and nothing else does. Write the exact operands to
//! disk, re-run the same kernel over them offline, and either the bad output
//! reproduces (the inputs are the problem, and they can be bisected) or it does
//! not (production was reading something else — a lifetime or ordering fault,
//! and a completely different search).
//!
//! ## Same pass, not the next one
//!
//! The check is **synchronous and inline**, sitting between the GEMM and its
//! consumer. That placement is the whole design: an asynchronous drain learns
//! about the fault a wave later, by which point the operands are gone and the
//! next pass through this site is carrying *different* ones — so a capture
//! armed by a drain would faithfully record a call that did not fail. The
//! inputs and the output have to be taken together, between the two, which
//! means paying for a synchronisation right here.
//!
//! That cost is why [`capture_gate_gemm`] is the only site that does this. One
//! fence per MoE layer per wave is the serialisation that hides
//! ordering-dependent faults; scattering these would be the probe that suppresses
//! what it hunts.

use candle::quantized::cuda::{ko_repacked_bytes, Q8a128Operand};
use candle::quantized::GgmlDType;
use candle::tensor_assert::{
    assert_device_quant, check_now, check_now_quant, Dump, Finding, QTYPE_Q8A128V,
};
use candle::{cuda_backend::CudaDevice, LiveTensor, Result, Shape};

use super::decode_kv_walk::{nonfinite_rows, read_device};
use super::expert_lre::slot_integrity::{check_watch, watching};
use cudarc::driver::{CudaSlice, DevicePtr};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Once, OnceLock, RwLock};

/// Where the dump goes.
///
/// Absolute, resolved at compile time. The writer is a daemon started from the
/// workspace root and the reader is a test whose working directory is this
/// crate's root, so a relative path names two different places and the test
/// silently reports "no capture" for a dump that exists.
pub const DUMP_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../target/moe_gemm_dump");

type IntegrityProbe = Box<dyn Fn() -> Vec<String> + Send + Sync + 'static>;

fn integrity_probe() -> &'static std::sync::RwLock<Option<IntegrityProbe>> {
    static P: std::sync::OnceLock<std::sync::RwLock<Option<IntegrityProbe>>> =
        std::sync::OnceLock::new();
    P.get_or_init(|| std::sync::RwLock::new(None))
}

/// Register the resident-weight integrity check, so **every** capture reports
/// whether the weights changed since load.
///
/// A global rather than a parameter because the answer is global: the expert
/// grid is one object shared by every layer, and threading a handle to it
/// through each checkpoint would mean the checkpoints furthest from the MoE — a
/// residual, a mixer output — could never ask. Those are exactly the ones where
/// the answer matters most, because a corrupt resident weight explains a bad
/// residual and very little else does.
pub fn set_integrity_probe(f: impl Fn() -> Vec<String> + Send + Sync + 'static) {
    if let Ok(mut p) = integrity_probe().write() {
        *p = Some(Box::new(f));
    }
}

/// Drop both resident-weight probes.
///
/// **The counterpart the registrations never had.** Both hold a fingerprint of
/// the expert grid — raw device addresses plus the bytes expected at them — in a
/// process-lifetime global. When the grid goes away those addresses become
/// ordinary pool memory, and a probe still holding them reads whatever now lives
/// there and reports drift against a grid that no longer exists.
///
/// It also keeps the grid's read-only declaration alive: `SlotIntegrity` gives
/// that up on drop, and an `Arc` parked in a global means the drop never runs.
/// That is what made 23 of the transformers suite's tests fail under
/// `tensor-assert` — each passing alone, then dying in the suite because an
/// earlier test's grid was still declared.
pub fn clear_integrity_probes() {
    if let Ok(mut p) = integrity_probe().write() {
        *p = None;
    }
    if let Ok(mut s) = shard_scan().write() {
        *s = None;
    }
}

type ShardScan = Box<dyn Fn(usize) -> Vec<String> + Send + Sync + 'static>;

fn shard_scan() -> &'static std::sync::RwLock<Option<ShardScan>> {
    static S: std::sync::OnceLock<std::sync::RwLock<Option<ShardScan>>> =
        std::sync::OnceLock::new();
    S.get_or_init(|| std::sync::RwLock::new(None))
}

/// Register the rotating shard scan, driven from [`watch_layer`].
///
/// Global for the same reason [`set_integrity_probe`] is: the layer loop that
/// drives it holds no handle to the expert cache, and threading one through the
/// whole transformer to reach a diagnostic would be the tail wagging the dog.
pub fn set_shard_scan(f: impl Fn(usize) -> Vec<String> + Send + Sync + 'static) {
    if let Ok(mut s) = shard_scan().write() {
        *s = Some(Box::new(f));
    }
}

/// Run the registered probe and record its answer in the dump.
///
/// "Not registered" is recorded distinctly from "registered and found nothing":
/// both would otherwise be an absent list in the manifest, and they mean
/// opposite things — one is a cleared suspect, the other is an unasked question.
fn note_integrity(d: &mut Dump) {
    let Ok(p) = integrity_probe().read() else {
        return;
    };
    let Some(f) = p.as_ref() else {
        d.note("drifted", "probe-not-registered");
        return;
    };
    let drifted = f();
    d.note("drifted.count", drifted.len());
    for (i, s) in drifted.iter().take(64).enumerate() {
        d.note(&format!("drifted.{i:03}"), s);
    }
    if drifted.is_empty() {
        tracing::error!(
            target: "candle_transformers::nan_capture",
            "integrity: NO resident weight has changed since load — the bytes the kernels read \
             are the bytes the fill wrote"
        );
    } else {
        tracing::error!(
            target: "candle_transformers::nan_capture",
            drifted = drifted.len(), first = %drifted[0],
            "integrity: RESIDENT WEIGHTS CHANGED SINCE LOAD — listed in the dump"
        );
    }
}

/// How many sweeps pass between shard checks.
///
/// **Measured, not guessed.** A fingerprint has to fence, and a fence drains the
/// pipeline the race needs full: checking one shard every sweep cost 2350 → 1569
/// t/s, a third of the throughput, which is squarely into the range where the
/// fault stops reproducing at all (a heavily-fenced build once ran 71 minutes
/// clean against a 5-minute production failure). At one shard every eight sweeps
/// the cost is ~4% and the grid is still covered end to end every
/// `SHARD_COUNT * 8` = 512 sweeps — a few minutes of running, far inside any
/// session, and the watch narrows from there anyway.
const SWEEPS_PER_SHARD: usize = 8;

/// Counts sweeps; its quotient picks the shard, so the index still rotates.
static SWEEPS: AtomicUsize = AtomicUsize::new(0);

/// The resident-weight check that runs at **every layer boundary**.
///
/// Two things, in the order that narrows fastest:
///
/// 1. **Sweep a shard — at layer 0, every [`SWEEPS_PER_SHARD`] sweeps.** One
///    `1/SHARD_COUNT` slice of the expert grid. This is what lets a corrupt
///    weight be seen while the run is still producing correct tokens — the
///    whole-grid check in the dump can only ever speak after an activation has
///    already gone non-finite, by which point the write is many waves in the
///    past. Layer 0 and every eighth sweep rather than every layer because a
///    fingerprint has to fence, and fences drain the pipeline the race needs
///    full; see [`SWEEPS_PER_SHARD`] for the measured cost of getting this
///    wrong.
/// 2. **Re-check the watched slot — every layer.** Once a scan has named one,
///    this bounds the write to a single layer of a single sweep and panics
///    there. The extra fences are affordable here precisely because they only
///    start once the run has already misbehaved: the question has narrowed from
///    "does this reproduce" to "where exactly", and the answer is worth a
///    slower wave.
///
/// Costs one relaxed atomic per layer until a scan arms the watch, and nothing
/// at all when the feature is off — the call site is `#[cfg]`-ed out entirely.
pub fn watch_layer(dev: &CudaDevice, layer: usize) {
    if layer == 0 {
        let n = SWEEPS.fetch_add(1, Ordering::Relaxed);
        if n.is_multiple_of(SWEEPS_PER_SHARD) {
            if let Ok(s) = shard_scan().read() {
                if let Some(f) = s.as_ref() {
                    for line in f(n / SWEEPS_PER_SHARD) {
                        tracing::error!(
                            target: "candle_transformers::nan_capture",
                            %line,
                            "integrity: a resident expert weight changed — caught by the \
                             rotating scan, BEFORE any activation went bad"
                        );
                    }
                }
            }
        }
    }
    if !watching() {
        return;
    }
    let Some(what) = check_watch(dev, &format!("layer {layer}")) else {
        return;
    };
    tracing::error!(
        target: "candle_transformers::nan_capture",
        %what,
        "integrity: the watched weight changed DURING this sweep — the writer ran in this window"
    );
    panic!("watched expert weight changed during the sweep: {what}");
}

/// The site a drain has already named, held as its `&'static str` data pointer.
///
/// Zero means nothing is armed, which is every checkpoint until a wave has
/// actually failed. Comparing pointers rather than string contents is what
/// keeps the common path a single relaxed load and a compare — see
/// [`checkpoint`] for why that matters more than it looks.
static ARMED_SITE: AtomicUsize = AtomicUsize::new(0);

/// The drain order stamp of the currently armed site, so the arm can move to an
/// earlier one — see [`arm_from_drain`]. `u32::MAX` means nothing is armed.
///
/// Not atomic with [`ARMED_SITE`], deliberately: a torn pair costs one capture
/// at the wrong site on a diagnostic path, where taking a lock on every finding
/// would cost the budget the asynchronous locate exists to protect.
static ARMED_SEQ: AtomicU32 = AtomicU32::new(u32::MAX);

/// The sites that can actually honour an arm — those that pay a fence and dump
/// when named, as opposed to the far more numerous bare `assert` sites.
///
/// The drain ranks *every* asserted site, and most of the forward is bare
/// asserts. Arming one of those is worse than arming nothing: the arm is taken,
/// no call site ever tests it, and the capture is silently disabled for the rest
/// of the run while the log claims a site was armed. So an arm is only ever
/// placed on a site that has announced itself here by being reached.
fn capture_sites() -> &'static RwLock<HashSet<usize>> {
    static C: OnceLock<RwLock<HashSet<usize>>> = OnceLock::new();
    C.get_or_init(|| RwLock::new(HashSet::new()))
}

/// Announce `name` as a site that captures when armed.
///
/// Called on every pass through a checkpoint, so the read-lock fast path is the
/// one that matters; the write happens once per site per process. Keyed by the
/// `&'static str`'s address, the same identity [`armed_for`] compares.
fn register_capture_site(name: &'static str) {
    let p = name.as_ptr() as usize;
    if let Ok(s) = capture_sites().read() {
        if s.contains(&p) {
            return;
        }
    }
    if let Ok(mut s) = capture_sites().write() {
        s.insert(p);
    }
}

/// Whether an arm placed on `name` would ever be honoured.
fn can_capture(name: &'static str) -> bool {
    capture_sites()
        .read()
        .map(|s| s.contains(&(name.as_ptr() as usize)))
        .unwrap_or(false)
}

/// Register the arming callback. Idempotent; safe to call from any checkpoint.
///
/// The drain reports bad sites first-bad-first by the kernel's own ticket, so
/// the first one it hands us is the earliest site that went non-finite in that
/// wave. That is the site worth paying a fence for on the next wave.
fn arm_from_drain() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        candle::tensor_assert::on_bad(|f: &Finding| {
            // **The arm moves earlier, and only earlier.**
            //
            // It used to latch once, on the first bad finding of the first
            // failing wave, and never move again. That is wrong whenever the
            // waves do not all fail the same way: a wave whose earliest bad site
            // is downstream arms *that*, and a later wave whose fault begins at
            // the true origin can no longer claim the arm. The capture then
            // fences at a site that merely inherited a non-finite value and
            // reports it as the first corruption.
            //
            // Measured twice on the 35B: the drain ranked `attn.ctx_raw` as BAD
            // #1 with `origin=attn.ctx_raw`, while the arm sat on
            // `moe.shared_gated.L{11,27}` from an earlier wave and dumped
            // `Context dumped: []` — naming a consequence and losing the
            // operands that would have explained the cause.
            //
            // `seq` is the drain's order stamp for the first bad observation, so
            // a smaller one is strictly closer to the origin. Re-arming on it
            // converges on the earliest site the run has ever faulted at, rather
            // than on whichever wave happened to fail first.
            let Some(seq) = f.seq else {
                return;
            };
            if seq >= ARMED_SEQ.load(Ordering::Relaxed) {
                return;
            }
            // Recover the interned `&'static str` for this name so the armed
            // site is comparable by pointer at the call sites.
            let Some(s) = candle::tensor_assert::interned(&f.name) else {
                return;
            };
            // **Only a site that captures may hold the arm.** A bare `assert`
            // can be ranked but never tests `armed_for`, so arming one takes the
            // arm out of circulation and disables the capture for the rest of
            // the run. Say so instead: an earlier-seq site with no checkpoint is
            // precisely the place a checkpoint should be added, and naming it is
            // how the chain converges by accretion.
            if !can_capture(s) {
                tracing::error!(
                    target: "candle_transformers::nan_capture",
                    site = %f.name, seq, nan = f.nan, inf = f.inf,
                    "EARLIER than the armed site, but no checkpoint watches it — it can be \
                     ranked and not captured. Put a `checkpoint` here to capture its operands."
                );
                return;
            }
            ARMED_SEQ.store(seq, Ordering::SeqCst);
            ARMED_SITE.store(s.as_ptr() as usize, Ordering::SeqCst);
            tracing::error!(
                target: "candle_transformers::nan_capture",
                site = %f.name, seq, nan = f.nan, inf = f.inf,
                "capture ARMED at the earliest bad site seen so far — the next wave reaching \
                 it pays one fence and dumps its operands"
            );
        });
    });
}

/// Whether a drain has named `name`, for a caller that must decide whether a
/// probe of its own is worth running — see `QuantizedMlp::forward_dynamic`,
/// which uses it to report an operand it could *not* examine rather than skip
/// one in silence.
pub fn is_armed(name: &'static str) -> bool {
    armed_for(name)
}

/// Whether `name` is the site a drain has named. One relaxed load and a compare.
#[inline]
fn armed_for(name: &'static str) -> bool {
    ARMED_SITE.load(Ordering::Relaxed) == name.as_ptr() as usize
}

/// A checkpoint: examine `out` now, and if it is non-finite dump it — with
/// whatever context the caller names — and stop the process.
///
/// The general form, for a value that has no bespoke operand bundle. `extras`
/// are the tensors that would answer "and what went into it": name them and
/// they land in the dump beside the output, which is the difference between
/// knowing *where* and knowing *from what*.
///
/// Returns having done nothing but a launch and a fence when `out` is finite,
/// which is every call until the one that isn't.
pub fn checkpoint(
    name: &'static str,
    out: &LiveTensor<'_>,
    extras: &[(&str, &LiveTensor<'_>)],
    dev: &CudaDevice,
) -> Result<()> {
    arm_from_drain();
    register_capture_site(name);
    // **Free until a drain has already named this site.**
    //
    // The synchronous form costs a device fence, and this fault is one a fence
    // hides: the production build reproduces it in ~5 minutes at 10k t/s while
    // a build fencing at every checkpoint ran 71 minutes clean. Sixteen fenced
    // sites over forty layers is ~640 fences a wave — enough to serialise the
    // sweep and dissolve the race being hunted.
    //
    // So the default is the ASYNCHRONOUS assert: one launch, no readback, no
    // ordering change. It cannot dump operands, but it does not have to — the
    // wave-end drain ranks every bad site by the kernel's own ticket, which
    // names the earliest one. That name arms this site, and only then does the
    // next wave through it pay a single fence to capture its inputs.
    //
    // Locating and capturing are different questions, and only the second one
    // is worth perturbing the program for.
    out.assert(name);
    if !armed_for(name) {
        return Ok(());
    }
    let mut found: Option<Finding> = None;
    check_now(out, name, |f| found = Some(f.clone()));
    let Some(f) = found else {
        return Ok(());
    };

    let mut d = Dump::create(DUMP_DIR)?;
    d.note("checkpoint", name);
    d.note("out_dtype", format!("{:?}", out.dtype()));
    d.note("out_shape", format!("{:?}", out.dims()));
    note_stats(&mut d, &f);
    note_integrity(&mut d);
    dump_tensor(&mut d, "out", out, dev)?;
    for (n, t) in extras {
        d.note(&format!("{n}.dtype"), format!("{:?}", t.dtype()));
        d.note(&format!("{n}.shape"), format!("{:?}", t.dims()));
        dump_tensor(&mut d, n, t, dev)?;
    }
    let dir = d.finish()?;
    panic!(
        "FIRST CORRUPTION at checkpoint {name} → {} (nan={} inf={} of {} finite=[{:?}, {:?}]). \
         Every checkpoint upstream of this one passed on this same pass, so this is where the \
         value first went non-finite — not merely where it was noticed. Context dumped: {:?}.",
        dir.display(),
        f.nan,
        f.inf,
        f.elems,
        f.min,
        f.max,
        extras.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
    );
}

/// [`checkpoint`] for a raw q8a128 activation operand.
///
/// The int8 operands between the norm and the expert GEMMs are not tensors, and
/// on this path they are the only thing left unwatched: the residual entering
/// the layer is checked, the weights are checked, and the corruption is
/// somewhere between them.
///
/// # Safety
///
/// `ptr` must name a complete q8a128 buffer of `rows × cols` logical elements.
pub unsafe fn checkpoint_q8a128(
    name: &'static str,
    ptr: u64,
    rows: usize,
    cols: usize,
    byte_len: usize,
    dev: &CudaDevice,
) -> Result<()> {
    // SAFETY: the caller's contract, passed through unchanged.
    unsafe { checkpoint_q8a128_with(name, ptr, rows, cols, byte_len, dev, &mut |_, _| Ok(())) }
}

/// [`checkpoint_q8a128`] with a context hook.
///
/// `context` runs once, on the armed path only, after the operand has been
/// found bad and its bytes are in the dump. It is handed the rows whose scales
/// are non-finite, so the site that produced the operand can add what only it
/// can reach — the decode attention walks the KV its kernel read — without this
/// module knowing how. Unarmed, the hook costs nothing: it is never called.
///
/// # Safety
///
/// As [`checkpoint_q8a128`].
pub unsafe fn checkpoint_q8a128_with(
    name: &'static str,
    ptr: u64,
    rows: usize,
    cols: usize,
    byte_len: usize,
    dev: &CudaDevice,
    context: &mut dyn FnMut(&mut Dump, &[usize]) -> Result<()>,
) -> Result<()> {
    arm_from_drain();
    register_capture_site(name);
    // **The asynchronous locate, exactly as [`checkpoint`] does it.**
    //
    // `assert_device_quant` folds this buffer's dequantized statistics into
    // `name`'s slot with one kernel — no fence, no readback, no ordering change
    // — so the wave-end drain ranks this site alongside every tensor site, gives
    // it a `seq`, and can arm it on its own merits.
    //
    // This used to gate on `armed_for(name)` with nothing folding into that
    // slot, under a note claiming a raw quantized buffer had no asynchronous
    // form and could only be reached when a neighbouring tensor site named it.
    // That was wrong — the form is [`assert_device_quant`], right here — and the
    // cost was total: an arm that could never be set meant this checkpoint had
    // never once fired, at any of its call sites, in its entire existence.
    //
    // Only after the locate does the fence apply, and only at the one site a
    // drain has already named. The dequant-into-staging pass this performs is
    // far too expensive to run unarmed.
    unsafe { assert_device_quant(name, ptr, QTYPE_Q8A128V, rows * cols, dev) };
    if !armed_for(name) {
        return Ok(());
    }
    let mut found: Option<Finding> = None;
    let bad = unsafe {
        check_now_quant(name, ptr, QTYPE_Q8A128V, rows * cols, dev, |f| {
            found = Some(f.clone())
        })
    };
    if !bad {
        // **A pass is a finding, so it is said out loud.**
        //
        // Silence here is indistinguishable from never having run, and this
        // checkpoint spent its whole existence never running (it gated on an arm
        // that could not be set). Reading a quiet return as "the operand is
        // clean" would repeat that mistake with more confidence.
        tracing::error!(
            target: "candle_transformers::nan_capture",
            site = name, rows, cols, byte_len,
            "operand CLEAN at the armed site — the non-finite value is NOT in this \
             operand, so it enters at or after the consumer that armed the capture"
        );
        return Ok(());
    }
    let f = found.expect("check_now_quant reports bad only through the callback");

    let mut d = Dump::create(DUMP_DIR)?;
    d.note("checkpoint", name);
    d.note("rows", rows);
    d.note("cols", cols);
    // **Who else claims these bytes, asked while the operand is still live.**
    //
    // This operand is written correctly and read back non-finite, so something
    // overwrites it between the two. The between-waves audit cannot see that:
    // it runs when the bump cursor has just reset, so no activation carve
    // exists to collide with anything. Asked here, at the fault, with the wave
    // mid-flight, every tenant's extents are real.
    //
    // A single owner — this buffer's own arena — is the expected answer and
    // says the collision is not a partition error. Two owners names the
    // trespasser outright.
    // **Was the span recycled under this operand?**
    //
    // The address cannot answer it: a bump arena rewinds when the last guard
    // drops, so the buffer that was quantized and the buffer that replaced it
    // occupy the same bytes. The epoch can — it bumps on every rewind, so
    // `made_in != now` means this operand's ground was handed to another carve
    // between the quantize and this read, and what is being examined is not what
    // was written.
    let made_in = candle::wave_provenance::q8a128_epoch_of(ptr);
    let now = candle::wave_provenance::epoch_at(ptr);
    d.note("epoch.made_in", format!("{made_in:?}"));
    d.note("epoch.at_use", format!("{now:?}"));
    match (made_in, now) {
        // Made outside any arena, read inside one: an arena was placed over
        // ground that was already holding a live buffer. Worse than a rewind,
        // because no amount of lifetime discipline on the operand would prevent
        // it — the arena moved onto memory it never owned.
        (Some(candle::wave_provenance::EPOCH_NOT_IN_ARENA), Some(b)) => {
            d.note("epoch.verdict", "ARENA_PLACED_OVER_LIVE_POOL_MEMORY");
            tracing::error!(
                target: "candle_transformers::nan_capture",
                at_use = b,
                "ARENA PLACED OVER LIVE MEMORY: this operand was allocated when its address                  belonged to NO wave arena, and by the time it was read an arena covers it.                  The tier was placed on top of a buffer that was already in use"
            );
        }
        // Made outside any arena and still outside one: the operand is pool
        // memory throughout, so neither the rewind nor the placement story
        // applies and the writer is elsewhere.
        (Some(candle::wave_provenance::EPOCH_NOT_IN_ARENA), None) => {
            d.note("epoch.verdict", "pool-memory-throughout");
            tracing::error!(
                target: "candle_transformers::nan_capture",
                "the operand is pool memory and was never in a wave arena — the arena                  lifetime stories do not apply"
            );
        }
        (Some(a), Some(b)) if a != b => {
            d.note("epoch.verdict", "RECYCLED");
            tracing::error!(
                target: "candle_transformers::nan_capture",
                made_in = a, at_use = b, rewinds = b.saturating_sub(a),
                "USE AFTER REWIND: the arena rewound between this operand being quantized and                  being read, so these bytes belong to a later carve — the operand was never                  corrupted, it was REPLACED"
            );
        }
        (Some(a), Some(b)) => {
            d.note("epoch.verdict", "same-generation");
            tracing::error!(
                target: "candle_transformers::nan_capture",
                made_in = a, at_use = b,
                "the arena has NOT rewound since this operand was quantized — the bytes are                  still its own, so something wrote them without owning them"
            );
        }
        _ => {
            d.note("epoch.verdict", "unknown");
            tracing::error!(
                target: "candle_transformers::nan_capture",
                ?made_in, ?now,
                "epoch unavailable — the operand is not in a wave arena, or was made on                  another thread"
            );
        }
    }
    let owners = candle::span_audit::who_owns(ptr, byte_len);
    d.note("owners.count", owners.len());
    for (i, o) in owners.iter().enumerate() {
        d.note(&format!("owners.{i:02}"), o);
    }
    tracing::error!(
        target: "candle_transformers::nan_capture",
        site = name, ptr = format!("{ptr:#x}"), byte_len, owners = owners.len(),
        "operand owners at the fault: {owners:?}"
    );
    d.note("out_dtype", "F32");
    d.note("out_shape", format!("[{rows}, {cols}]"));
    note_stats(&mut d, &f);
    // The operand's own bytes, in the packed form the GEMM reads. The packed
    // size comes from the operand itself (`Q8a128Operand::byte_len`) rather
    // than being recomputed here — a figure that disagreed with what the
    // quantizer allocated would dump the wrong extent.
    // SAFETY: the caller's contract — `ptr` names a full q8a128 buffer of
    // `byte_len` bytes.
    let stacked = unsafe { read_device(dev, ptr, byte_len)? };
    d.bytes("stacked", &stacked)?;
    // Which rows, read from the scales themselves. On a decode operand a row is
    // a sequence, and that is what the context hook needs to know to follow.
    let bad_rows = match nonfinite_rows(&stacked, rows, cols) {
        Ok(r) => r,
        Err(e) => {
            d.note("bad_rows.error", e);
            Vec::new()
        }
    };
    d.note("bad_rows", format!("{bad_rows:?}"));
    // A hook that fails must not cost the capture: its error is recorded beside
    // everything else and the dump still completes.
    if let Err(e) = context(&mut d, &bad_rows) {
        d.note("context.error", e);
    }
    let dir = d.finish()?;
    panic!(
        "FIRST CORRUPTION at checkpoint {name} → {} (nan={} inf={} of {} finite=[{:?}, {:?}]). \
         A quantized operand's values are integers and cannot be non-finite, so this is a \
         per-group SCALE. Every checkpoint upstream of this one passed on this same pass.",
        dir.display(),
        f.nan,
        f.inf,
        f.elems,
        f.min,
        f.max,
    );
}

/// Record what the assert saw, beside the bytes it saw it in.
fn note_stats(d: &mut Dump, f: &Finding) {
    d.note("saw.nan", f.nan);
    d.note("saw.inf", f.inf);
    d.note("saw.elems", f.elems);
    d.note(
        "saw.min",
        f.min
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".into()),
    );
    d.note(
        "saw.max",
        f.max
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".into()),
    );
}

/// Copy a tensor's own bytes — at its offset, in its dtype — into the dump.
fn dump_tensor(d: &mut Dump, name: &str, t: &LiveTensor<'_>, dev: &CudaDevice) -> Result<()> {
    let (storage, layout) = t.storage_and_layout();
    let candle::Storage::Cuda(cu) = &*storage else {
        // A CPU-side operand is not what any of these checkpoints watch, and
        // silently writing nothing under its name would make the dump lie.
        candle::bail!("capture: {name} is not a CUDA tensor");
    };
    let stream = dev.cuda_stream();
    let esz = t.dtype().size_in_bytes();
    let base = cu.slice.device_ptr(&stream) + (layout.start_offset() * esz) as u64;
    // SAFETY: `elem_count` elements of `esz` bytes from the tensor's own offset.
    unsafe { d.device_ptr(name, dev, base, t.elem_count() * esz) }
}

/// The full argument list of one `grouped_qmatmul_dev_q8a128` call.
///
/// Everything the kernel dereferences, so the replay can rebuild the call
/// exactly rather than approximately — a replay over *similar* inputs answers
/// nothing.
pub struct GemmCall<'a> {
    pub layer: usize,
    pub stacked: &'a Q8a128Operand<'a>,
    /// Device table of per-expert weight addresses; the kernel indexes it at
    /// `expert_base + tile_expert[t]`.
    pub weight_ptrs: &'a CudaSlice<u64>,
    pub expert_base: usize,
    pub num_experts: usize,
    pub weight_dtype: GgmlDType,
    pub weight_nrows: usize,
    pub tile_expert: &'a CudaSlice<i32>,
    pub tile_b_start: &'a CudaSlice<i32>,
    pub tile_b_cnt: &'a CudaSlice<i32>,
    pub launch_tiles: usize,
    pub out: &'a LiveTensor<'a>,
}

/// Check this call's output and, if it is non-finite, dump the call and panic.
///
/// Returns normally — having done nothing beyond one launch and one fence —
/// whenever the output is finite, which is every call in a healthy run.
pub fn capture_gate_gemm(call: &GemmCall<'_>, dev: &CudaDevice) -> Result<()> {
    arm_from_drain();
    let site = candle::tensor_assert::site("moe.capture.gate_out.L", call.layer);
    register_capture_site(site);
    // Asynchronous by default so the GEMM keeps its throughput; the fence and
    // the 400 MB dump are paid only once a drain has named this site. See
    // [`checkpoint`].
    call.out.assert(site);
    if !armed_for(site) {
        return Ok(());
    }
    let mut found: Option<Finding> = None;
    // The callback fires the instant the kernel's verdict is read, and the
    // operands are still live in this scope — which is what makes the dump
    // below describe *this* call rather than some later one.
    check_now(call.out, site, |f| found = Some(f.clone()));
    let Some(f) = found else {
        return Ok(());
    };
    dump(call, &f, dev)
}

fn dump(call: &GemmCall<'_>, f: &Finding, dev: &CudaDevice) -> Result<()> {
    let mut d = Dump::create(DUMP_DIR)?;
    d.note("layer", call.layer);
    d.note("rows", call.stacked.rows);
    d.note("cols", call.stacked.cols);
    d.note("expert_base", call.expert_base);
    d.note("num_experts", call.num_experts);
    d.note("weight_dtype", format!("{:?}", call.weight_dtype));
    d.note("weight_nrows", call.weight_nrows);
    d.note("launch_tiles", call.launch_tiles);
    d.note("out_dtype", format!("{:?}", call.out.dtype()));
    d.note("out_shape", format!("{:?}", call.out.dims()));

    // What the assert actually SAW, recorded beside the bytes. The replay
    // compares against these rather than re-deriving them: if a replay of the
    // same bytes yields a different NaN count or a different range, that is the
    // finding, and it is only visible because production's numbers travelled
    // with the dump instead of living in a log the test cannot read.
    note_stats(&mut d, f);

    // The activation operand, as the bytes the kernel reads.
    let n_stacked = call.stacked.byte_len();
    call.stacked.with_device_ptr(dev, |p| {
        // SAFETY: `byte_len` is the operand's own accounting of its extent, and
        // `with_device_ptr` holds the backing's guards for this closure.
        unsafe { d.device_ptr("stacked", dev, p, n_stacked) }
    })?;

    // The tile tables, over exactly the range the kernel launches.
    let stream = dev.cuda_stream();
    for (name, buf) in [
        ("tile_expert", call.tile_expert),
        ("tile_b_start", call.tile_b_start),
        ("tile_b_cnt", call.tile_b_cnt),
    ] {
        let (p, _g) = buf.device_ptr(&stream);
        // SAFETY: the kernel reads `launch_tiles` i32 from each of these, so
        // that range is readable by construction.
        unsafe { d.device_ptr(name, dev, p, call.launch_tiles * 4)? };
    }

    // Every expert weight this layer can route to, resolved through the same
    // pointer table the kernel indexes. Dumping the whole layer rather than the
    // routed subset keeps the replay's table indexing identical to production's
    // — a remapped table would be a different call.
    let ptrs = dev
        .memcpy_dtov(call.weight_ptrs)
        .map_err(|e| candle::Error::Msg(format!("capture: reading weight table: {e}")))?;
    let shape = Shape::from((call.weight_nrows, call.stacked.cols));
    let w_bytes = ko_repacked_bytes(&shape, call.weight_dtype)?;
    d.note("weight_bytes_each", w_bytes);
    for e in 0..call.num_experts {
        let idx = call.expert_base + e;
        let Some(&ptr) = ptrs.get(idx) else {
            candle::bail!("capture: weight table has no entry {idx}");
        };
        // SAFETY: the table's entries are the addresses the kernel itself
        // dereferences, each naming a full KO weight of `w_bytes`.
        unsafe { d.device_ptr(&format!("weight_{e:05}"), dev, ptr, w_bytes)? };
    }

    // The output as produced in production — what the replay must match.
    dump_tensor(&mut d, "out", call.out, dev)?;

    note_integrity(&mut d);
    let dir = d.finish()?;
    panic!(
        "MoE GEMM captured at layer {} → {} (nan={} inf={} of {} finite=[{:?}, {:?}]). \
         The process stops here on purpose: the dump names arena addresses that the next \
         wave would overwrite, so continuing would leave a bundle that no longer describes \
         what produced it. Replay it with \
         `cargo test -p candle-transformers --features cuda,tensor-assert --test moe_gemm_replay`.",
        call.layer,
        dir.display(),
        f.nan,
        f.inf,
        f.elems,
        f.min,
        f.max,
    );
}
