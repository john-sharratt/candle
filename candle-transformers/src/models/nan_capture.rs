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
use candle::tensor_assert::{check_now, check_now_quant, Dump, Finding, QTYPE_Q8A128V};
use candle::{cuda_backend::CudaDevice, LiveTensor, Result, Shape};

use super::expert_lre::slot_integrity::{check_watch, watching};
use cudarc::driver::{CudaSlice, DevicePtr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

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

type ShardScan = Box<dyn Fn(usize) -> Vec<String> + Send + Sync + 'static>;

fn shard_scan() -> &'static std::sync::RwLock<Option<ShardScan>> {
    static S: std::sync::OnceLock<std::sync::RwLock<Option<ShardScan>>> = std::sync::OnceLock::new();
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

/// Register the arming callback. Idempotent; safe to call from any checkpoint.
///
/// The drain reports bad sites first-bad-first by the kernel's own ticket, so
/// the first one it hands us is the earliest site that went non-finite in that
/// wave. That is the site worth paying a fence for on the next wave.
fn arm_from_drain() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        candle::tensor_assert::on_bad(|f: &Finding| {
            if ARMED_SITE.load(Ordering::Relaxed) != 0 {
                return;
            }
            // Recover the interned `&'static str` for this name so the armed
            // site is comparable by pointer at the call sites.
            let Some(s) = candle::tensor_assert::interned(&f.name) else {
                return;
            };
            if ARMED_SITE
                .compare_exchange(0, s.as_ptr() as usize, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                tracing::error!(
                    target: "candle_transformers::nan_capture",
                    site = %f.name, nan = f.nan, inf = f.inf,
                    "capture ARMED at the first bad site — the next wave reaching it pays one \
                     fence and dumps its operands"
                );
            }
        });
    });
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
    arm_from_drain();
    // Armed like [`checkpoint`], and more strongly so: this one dequantizes the
    // operand into staging before it can look at it, which is a full pass over
    // the buffer on top of the fence. Unarmed it does nothing at all — there is
    // no asynchronous form for a raw quantized buffer, so the drain cannot name
    // this site and it is reached only when an adjacent tensor site names it.
    if !armed_for(name) {
        return Ok(());
    }
    let mut found: Option<Finding> = None;
    let bad = unsafe {
        check_now_quant(
            name,
            ptr,
            QTYPE_Q8A128V,
            rows * cols,
            dev,
            |f| found = Some(f.clone()),
        )
    };
    if !bad {
        return Ok(());
    }
    let f = found.expect("check_now_quant reports bad only through the callback");

    let mut d = Dump::create(DUMP_DIR)?;
    d.note("checkpoint", name);
    d.note("rows", rows);
    d.note("cols", cols);
    d.note("out_dtype", "F32");
    d.note("out_shape", format!("[{rows}, {cols}]"));
    note_stats(&mut d, &f);
    // The operand's own bytes, in the packed form the GEMM reads. The packed
    // size comes from the operand itself (`Q8a128Operand::byte_len`) rather
    // than being recomputed here — a figure that disagreed with what the
    // quantizer allocated would dump the wrong extent.
    // SAFETY: the caller's contract — `ptr` names a full q8a128 buffer of
    // `byte_len` bytes.
    unsafe { d.device_ptr("stacked", dev, ptr, byte_len)? };
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
        f.min.map(|v| v.to_string()).unwrap_or_else(|| "none".into()),
    );
    d.note(
        "saw.max",
        f.max.map(|v| v.to_string()).unwrap_or_else(|| "none".into()),
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
