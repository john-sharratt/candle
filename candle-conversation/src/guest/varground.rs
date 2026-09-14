//! A [`VarBuilder`] whose tensors land in a guest's span ground.
//!
//! # Why this exists
//!
//! Every dense candle model — a UNet, a VAE, a text encoder — is built by
//! asking a `VarBuilder` for named tensors, and every stock backend answers
//! from the CUDA pool. For a guest that is the one thing it may not do: the
//! reservation is the budget, and a model's worth of pool allocations is the
//! largest competitor for the card the engine has ever had.
//!
//! So this is a [`SimpleBackend`] that reads the same safetensors a
//! `VarBuilder` would and copies each tensor into ground instead. Any model
//! that loads through a `VarBuilder` loads into the span unchanged, with no
//! per-architecture work at all — which is what makes the image guest a
//! composition of candle's existing modules rather than a reimplementation of
//! them.
//!
//! # What it does not do
//!
//! It does not allocate *activations*. A forward's intermediate tensors come
//! from the pool as they always have, and that limit is deliberate: during a
//! drain the engine is quiesced and the relief ladder has just run, so the pool
//! is as empty as it ever gets, and the gigabytes — the weights — are what this
//! keeps out of it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::Init;

use super::ground::GuestGround;

/// Reads named tensors from safetensors files and places them in ground.
pub struct GroundVars {
    files: Vec<PathBuf>,
    /// Mapped and indexed once per process — see [`super::checkpoint`]. Held by
    /// `Arc` because the copies below read straight out of it and must not race
    /// its unmapping.
    maps: Arc<candle::safetensors::MmapedSafetensors>,
    /// Every placement made, for the verification pass that runs after the load
    /// has synchronised.
    ///
    /// Verification used to happen per tensor, which meant a device→host read
    /// and a synchronise for each of ~1,550 weights. It is the same check either
    /// way — the placed bytes against the checkpoint's — but done once at the
    /// end it costs one barrier instead of fifteen hundred.
    placed: Mutex<Vec<Placed>>,
    /// Nanoseconds spent on the dtype-conversion path, so the log can separate
    /// "the link is slow" from "the checkpoint is the wrong type for the model".
    host_ns: std::sync::atomic::AtomicU64,
    /// The ground being placed into.
    ///
    /// Shared rather than borrowed because [`SimpleBackend::get`] takes `&self`
    /// and is `'static`, so a `&mut GuestGround` cannot live in the backend. The
    /// lock is not ceremony either: the cursor is a bump pointer, and two
    /// threads placing at once would hand out the same address twice.
    ///
    /// It is also what makes the lifetime rule *checkable*. Every tensor this
    /// hands out views ground, so the model must be dropped before the ground
    /// is — and `drain` reads `Arc::strong_count` after unloading to say so out
    /// loud, where a lifetime comment could only ask.
    ground: Arc<Mutex<GuestGround>>,
    device: Device,
}

impl GroundVars {
    /// Build a `VarBuilder` over `files`, placing every tensor it hands out in
    /// `ground`, and run `f` with it.
    ///
    /// Scoped rather than free so the backend — and the `Arc` clone it holds —
    /// is dropped before this returns, whatever `f` does. A backend that
    /// escaped would keep the ground's refcount above one and the drain would
    /// report it.
    pub fn with<T>(
        files: &[PathBuf],
        dtype: DType,
        device: &Device,
        ground: &Arc<Mutex<GuestGround>>,
        f: impl FnOnce(candle_nn::VarBuilder) -> Result<T, String>,
    ) -> Result<T, String> {
        let maps = super::checkpoint::safetensors(files)?;
        let vars = GroundVars {
            files: files.to_vec(),
            maps: Arc::clone(&maps),
            placed: Mutex::new(Vec::new()),
            host_ns: std::sync::atomic::AtomicU64::new(0),
            ground: Arc::clone(ground),
            device: device.clone(),
        };
        // Kept so the verification below can read what was placed. The backend
        // the `VarBuilder` owns is a second handle on the same maps and ground;
        // both are dropped before this returns, which is what `drain`'s refcount
        // check depends on.
        let record = Arc::new(vars);
        let vb = candle_nn::VarBuilder::from_backend(
            Box::new(Handle(Arc::clone(&record))),
            dtype,
            device.clone(),
        );
        let t_place = Instant::now();
        let built = f(vb)?;
        let place_ms = t_place.elapsed().as_secs_f64() * 1e3;

        // **One barrier for the whole load.**
        //
        // Every copy above reads the mapping, which outlives this function, so
        // none of them needed waiting on individually. They do all have to have
        // landed before anything reads the weights — and before the samples
        // below are read back, which would otherwise race the copies they are
        // checking.
        let t_sync = Instant::now();
        if let Device::Cuda(cuda) = device {
            cuda.cuda_stream()
                .synchronize()
                .map_err(|e| format!("guest weights: {e}"))?;
        }
        let sync_ms = t_sync.elapsed().as_secs_f64() * 1e3;

        let t_verify = Instant::now();
        record.verify_placements()?;
        let (tensors, bytes) = record.placed_totals();
        tracing::info!(
            target: "candle_conversation::guest",
            place_ms,
            sync_ms,
            verify_ms = t_verify.elapsed().as_secs_f64() * 1e3,
            copy_ms = place_ms - record.host_ms(),
            host_convert_ms = record.host_ms(),
            tensors,
            mib = bytes >> 20,
            "guest weights placed"
        );
        Ok(built)
    }

    /// How many tensors were placed, and how many bytes they came to.
    fn placed_totals(&self) -> (usize, u64) {
        self.placed.lock().map_or((0, 0), |p| {
            (
                p.len(),
                p.iter()
                    .map(|q| (q.elems * q.dtype.size_in_bytes()) as u64)
                    .sum(),
            )
        })
    }

    /// Milliseconds spent converting checkpoints stored in a different type than
    /// the model runs in — the slow path, which needs a host buffer and its own
    /// wait. Zero when the checkpoint already holds what the model asked for.
    fn host_ms(&self) -> f64 {
        self.host_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6
    }

    /// Check a sample of what was placed against the checkpoint it came from.
    ///
    /// **A weight that loads wrong does not fail; it draws.** So the load checks
    /// itself rather than trusting that a copy which reported success delivered
    /// the bytes.
    ///
    /// Sampled across the tensors rather than exhaustive: reading every element
    /// of every weight back is as expensive as the load, and every way this has
    /// actually failed — a truncated copy, a stride mismatch, an address handed
    /// out twice — is systematic rather than a single wrong element, so it shows
    /// in any sample of the affected tensor.
    fn verify_placements(&self) -> Result<(), String> {
        let placed = self.placed.lock().map_err(|_| "placement log poisoned")?;
        for p in placed.iter() {
            let view = self
                .maps
                .get(&p.name)
                .map_err(|e| format!("guest weights: re-reading {}: {e}", p.name))?;
            verify_sampled(&self.device, p, view.data())?;
        }
        Ok(())
    }
}

/// One placed weight, kept so the load can check itself once at the end.
struct Placed {
    name: String,
    ptr: u64,
    dtype: DType,
    /// The type the **checkpoint's own bytes** are in, which is not always
    /// `dtype`: a checkpoint stored narrower than the model runs is converted on
    /// the host on the way down. The verification reads those raw bytes, so it
    /// needs the width they are actually laid out at — indexing a bf16 source
    /// with f32 strides walks off the end of the mapping, which is what a
    /// 16-bit VAE did the first time one was loaded here.
    src_dtype: DType,
    elems: usize,
}

/// The `SimpleBackend` the `VarBuilder` owns.
///
/// A thin handle so [`GroundVars::with`] can keep its own reference and read the
/// placement log after the builder has been consumed. `SimpleBackend` takes
/// `&self` and is `'static`, so the state has to be shared rather than borrowed.
struct Handle(Arc<GroundVars>);

impl SimpleBackend for Handle {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        self.0.place(s, name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.maps.get(name).is_ok()
    }
}

impl GroundVars {
    fn place(&self, s: Shape, name: &str, dtype: DType, dev: &Device) -> candle::Result<Tensor> {
        let Device::Cuda(_) = dev else {
            candle::bail!("guest weights: the reservation is a CUDA allocation");
        };
        let view = self.maps.get(name).map_err(|_| {
            candle::Error::Msg(format!(
                "guest weights: {name} is in none of {} file(s) — {:?}",
                self.files.len(),
                self.files
            ))
        })?;
        let shape: Shape = view.shape().to_vec().into();
        if shape != s {
            candle::bail!(
                "guest weights: {name} is {shape:?} in the checkpoint and the model asked for {s:?}"
            );
        }

        let mut ground = self.ground.lock().map_err(|_| {
            candle::Error::Msg("guest weights: the placement lock was poisoned".into())
        })?;

        // **The fast path: the checkpoint already holds the type the model
        // wants, so the bytes go from the mapping to ground untouched.**
        //
        // This used to run `maps.load(name, &Device::Cpu)`, materialising every
        // weight as a host `Tensor` first — a full copy of the checkpoint
        // through the heap on the way to a copy across the link. It also forced
        // a synchronise per tensor, because that host tensor died at the end of
        // the call and an in-flight copy would have read freed memory. The
        // mapping does not die, so neither cost is necessary.
        let elems = s.elem_count();
        // What the mapping holds, which the verification reads back raw.
        let src_dtype = DType::try_from(view.dtype()).unwrap_or(dtype);
        let (placed, ptr) = if DType::try_from(view.dtype()).ok() == Some(dtype) {
            place_bytes(&self.device, &mut ground, view.data(), dtype, s)?
        } else {
            // A checkpoint stored in a different type than the model runs in.
            // Converted on the host, which needs its own buffer — and therefore
            // its own wait, because that buffer dies when this returns.
            let t = Instant::now();
            let host = self.maps.load(name, &Device::Cpu)?.to_dtype(dtype)?;
            let host = host.contiguous()?;
            let out = place_bytes(&self.device, &mut ground, &host_bytes(&host)?, dtype, s)?;
            if let Device::Cuda(cuda) = &self.device {
                cuda.cuda_stream()
                    .synchronize()
                    .map_err(candle::Error::wrap)?;
            }
            self.host_ns.fetch_add(
                t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            out
        };

        if let Ok(mut log) = self.placed.lock() {
            log.push(Placed {
                name: name.to_string(),
                ptr,
                dtype,
                src_dtype,
                elems,
            });
        }
        Ok(placed)
    }
}

/// Copy raw bytes into ground and view them as a device tensor.
///
/// **The one placement path.** Every guest's weights come through here, whether
/// they arrive from a safetensors mapping (via [`GroundVars`]) or from
/// somewhere with no `VarBuilder` behind it at all — an ONNX graph's
/// initializers, say. Two ways of putting bytes in the span would be two sets
/// of the same three mistakes: a wrong length, a pool allocation freed as
/// though it were ground, and a copy read before it landed.
///
/// **Asynchronous, deliberately.** The caller guarantees `raw` outlives the
/// copy — it is either the process-lifetime checkpoint mapping, or a host buffer
/// The lease a guest stamps on the ground it places weights into.
///
/// # It used to be `Foreign`, and that is why a guest never used an arena
///
/// `Foreign` means "memory with no allocator to inherit", and an op reading one
/// allocates its output from the pool. That is right for a KV slot or a pinned
/// staging buffer. It is wrong for a guest's weights, because arena routing is
/// **inheritance-only** — `wave_alloc_attributed` takes its ticket from an input
/// tensor and there is no ambient generation a fresh allocation can pick up. So
/// weights stamped `Foreign` meant every activation derived from them fell to
/// the pool, at every resolution, for the whole life of the guest. Measured on
/// one image drain: `no_ticket_mib=221283`, `arena_full_mib=0` — not one
/// allocation was ever declined for a *full* arena, because not one ever reached
/// an arena to be declined by.
///
/// The seed does not put the weights in the arena; they stay where they were
/// placed. It says where the things *read from* them should be carved, and
/// resolves to the pool whenever no guest generation is open — which is every
/// guest that has not opened one, unchanged.
pub fn guest_origin(device: &Device) -> candle::cuda_backend::wave_provenance::LeaseOrigin {
    use candle::cuda_backend::wave_provenance::LeaseOrigin;

    match device.as_cuda_device() {
        Ok(cuda) => guest_origin_on(cuda.cuda_stream().context().ordinal()),
        // No CUDA device is no arena either, and `Foreign` is what every
        // non-CUDA path already means by it.
        Err(_) => LeaseOrigin::Foreign,
    }
}

/// Copy `src` into the open guest generation, so what reads it carves there too.
///
/// # Seeding the chain, not the weights
///
/// Stamping a routing seed on the *weights* does nothing, and finding out why
/// took a measurement: `no_ticket_mib` did not move at all. Inheritance runs
/// along the activation chain — a matmul takes its ticket from the **lhs**, and
/// for `x.matmul(w)` the lhs is `x`. The weight is the right-hand operand and is
/// never asked. So a forward inherits from whatever started `x`, which is a
/// `randn` or an embedding lookup: owned, ticketless, pool.
///
/// Seeding the root fixes the whole chain, because every tensor after it is
/// derived. One copy at the head of a stage buys the arena for everything that
/// stage allocates.
///
/// Returns `src` unchanged when no generation is open or the arena is full —
/// both mean "the pool", which is where all of this came from before.
pub fn into_arena(src: &Tensor, device: &Device) -> candle::Result<Tensor> {
    use candle::cuda_backend::wave_provenance::{wave_alloc, LeaseOrigin, WaveTicket};

    let Ok(cuda) = device.as_cuda_device() else {
        return Ok(src.clone());
    };
    let ticket = WaveTicket::guest(cuda.cuda_stream().context().ordinal() as u32);
    let bytes = src.elem_count() * src.dtype().size_in_bytes();
    let Some(ptr) = wave_alloc(ticket, bytes, 256) else {
        return Ok(src.clone());
    };
    // SAFETY: the range was just carved from the open generation, which outlives
    // this tensor — the caller drops it before the generation rewinds — and
    // nothing else holds that range while the generation is live.
    let dst = unsafe {
        Tensor::from_leased_cuda_ptr(
            ptr,
            src.dtype(),
            src.dims().to_vec(),
            device,
            LeaseOrigin::Wave(ticket),
        )
    }?;
    dst.slice_set(src, 0, 0)?;
    Ok(dst)
}

/// [`guest_origin`] for a caller that already holds the stream's ordinal.
pub fn guest_origin_on(ordinal: usize) -> candle::cuda_backend::wave_provenance::LeaseOrigin {
    use candle::cuda_backend::wave_provenance::{LeaseOrigin, WaveTicket};

    LeaseOrigin::Wave(WaveTicket::guest(ordinal as u32))
}

/// the caller waits on itself. `GroundVars::with` issues one barrier for the
/// whole load, so the transfers pipeline instead of stopping at every tensor;
/// a caller placing tensors by hand owes the same barrier before it reads them.
pub fn place_bytes(
    device: &Device,
    ground: &mut GuestGround,
    raw: &[u8],
    dtype: DType,
    shape: Shape,
) -> candle::Result<(Tensor, u64)> {
    let bytes = shape.elem_count() * dtype.size_in_bytes();
    if raw.len() != bytes {
        candle::bail!(
            "guest weights: {shape:?} of {dtype:?} needs {bytes} bytes and the checkpoint holds \
             {} — the header does not describe this tensor",
            raw.len()
        );
    }
    let at = ground
        .place(bytes, 256)
        .map_err(|e| candle::Error::Msg(format!("guest weights: placing {shape:?}: {e}")))?;
    let Device::Cuda(cuda) = device else {
        candle::bail!("guest weights: the reservation is a CUDA allocation");
    };
    let stream = cuda.cuda_stream();
    // SAFETY: `at.ptr` names `bytes` of ground the caller holds, and the slice
    // is `forget`ed below rather than dropped, so it never tries to free an
    // address the pool did not allocate.
    let mut dst = unsafe { stream.upgrade_device_ptr::<u8>(at.ptr, bytes) };
    stream
        .memcpy_htod(raw, &mut dst.slice_mut(..bytes))
        .map_err(candle::Error::wrap)?;
    // The slice views ground, so dropping it must not free: the address is
    // inside the VMM reservation and was never a pool allocation.
    std::mem::forget(dst);
    // SAFETY: the bytes are in flight to a range this guest owns, the ground
    // outlives every tensor handed out (the drain drops the model before the
    // ground), and nothing else writes the range. The caller's barrier orders
    // the copy before any read.
    let t = unsafe {
        Tensor::from_leased_cuda_ptr(at.ptr, dtype, shape, device, guest_origin(device))
    }?;
    Ok((t, at.ptr))
}

/// Elements sampled from each placed tensor to prove it arrived.
///
/// Spread across the whole tensor rather than taken from the front, because
/// every way this has actually failed — a truncated async copy, a stride
/// mismatch, an address handed out twice — leaves the beginning correct and
/// diverges later.
const VERIFY_WINDOW: usize = 64;

/// How many of those windows per tensor.
///
/// **One**, on the tensor's tail. Each window is a separate device→host read and
/// each read is a round trip — measured at ~0.9 ms regardless of how few bytes
/// it moves, so the count, not the width, is what verification costs. Four
/// windows across 1,138 tensors came to 4.1 s against copies of 0.7 s: the check
/// cost six times the transfer it was checking.
///
/// The tail is where the evidence is. Every way this has actually failed — a
/// truncated copy, a short placement, an address handed out twice — leaves the
/// beginning of a tensor correct and diverges later, so a window at the end
/// catches what a window at the start cannot, and the intermediate windows were
/// paying full price to re-confirm it.
const VERIFY_WINDOWS: usize = 1;

/// Check that what is on the device is what was sent.
///
/// **A weight that loads wrong does not fail; it draws.** An unsynchronised
/// copy placed UNet weights that were part checkpoint and part freed host
/// memory, and the model still ran, still produced finite numbers at every
/// stage, and rendered banded noise for every prompt. Nothing in the pipeline
/// could report it, because from the pipeline's point of view nothing was
/// wrong.
///
/// So the load checks itself, once, after the barrier: a sample of elements read
/// back off the device and compared against the checkpoint's own bytes. It turns
/// a silently wrong picture into a named tensor.
fn verify_sampled(device: &Device, p: &Placed, src: &[u8]) -> Result<(), String> {
    use candle::cuda_backend::wave_provenance::LeaseOrigin;

    if p.elems == 0 || p.ptr == 0 {
        return Ok(());
    }
    let fail = |e: candle::Error| format!("guest weights: verifying {}: {e}", p.name);
    // SAFETY: the address was placed by this load, its ground is still held, and
    // the caller has synchronised so the copy has landed.
    let placed = unsafe {
        Tensor::from_leased_cuda_ptr(p.ptr, p.dtype, p.elems, device, LeaseOrigin::Foreign)
    }
    .map_err(fail)?;

    // **Windows, not the whole tensor.**
    //
    // Reading every element back to check sixty-four of them moves the entire
    // model across the link a second time — measured at seconds of a load whose
    // transfer is under one. A few small contiguous windows cost a kilobyte each
    // and catch the same failures, because every way this has actually gone
    // wrong is systematic across a tensor rather than a single bad element.
    //
    // The last window is pinned to the tail: a truncated copy, a short
    // placement, and an address handed out twice all leave the beginning of a
    // tensor correct and diverge later, so the end is where the evidence is.
    let width = VERIFY_WINDOW.min(p.elems);
    // The SOURCE's width: `src` is the checkpoint's mapping, and a converted
    // placement leaves it narrower than what landed on the device.
    let esz = p.src_dtype.size_in_bytes();
    let mut offsets = [0usize; VERIFY_WINDOWS];
    for (i, off) in offsets.iter_mut().enumerate() {
        *off = if i + 1 == VERIFY_WINDOWS {
            p.elems - width
        } else {
            (p.elems - width) * i / VERIFY_WINDOWS
        };
    }

    for &off in &offsets {
        let got = placed
            .narrow(0, off, width)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(fail)?;
        // Read at the source's width and widen, which is the same arithmetic the
        // placement did — so a converted weight is still compared against what
        // it was converted *from*, rather than not compared at all.
        let want = Tensor::from_raw_buffer(
            &src[off * esz..(off + width) * esz],
            p.src_dtype,
            &[width],
            &Device::Cpu,
        )
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(fail)?;

        for (j, (a, b)) in want.iter().zip(&got).enumerate() {
            // Bit-identical, not approximate: this is a copy, not a
            // computation. The only tolerance is for a NaN in the checkpoint
            // itself, which compares unequal to itself and is not a transfer
            // error.
            if a != b && !(a.is_nan() && b.is_nan()) {
                return Err(format!(
                    "guest weights: element {} of {} in {} was sent as {a} and arrived as {b} — \
                     the placement did not deliver the tensor it was given",
                    off + j,
                    p.elems,
                    p.name
                ));
            }
        }
    }
    Ok(())
}

/// A contiguous host tensor's raw bytes.
///
/// Goes through the dtype rather than a blanket transmute so a tensor whose
/// storage is not what its dtype says fails here, at the copy, rather than as
/// a wrong number in the guest's output.
fn host_bytes(host: &Tensor) -> candle::Result<Vec<u8>> {
    let flat = host.flatten_all()?;
    Ok(match host.dtype() {
        DType::F32 => bytes_of(&flat.to_vec1::<f32>()?),
        DType::F16 => bytes_of(&flat.to_vec1::<half::f16>()?),
        DType::BF16 => bytes_of(&flat.to_vec1::<half::bf16>()?),
        DType::U8 => flat.to_vec1::<u8>()?,
        DType::U32 => bytes_of(&flat.to_vec1::<u32>()?),
        DType::I64 => bytes_of(&flat.to_vec1::<i64>()?),
        other => candle::bail!("guest weights: {other:?} has no host byte form here"),
    })
}

fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    // SAFETY: `T` is a plain numeric type with no padding and no niches, and
    // the length is derived from its own size.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
        .to_vec()
}

/// Every safetensors file in `dir`, sorted, or an error naming the directory.
///
/// Sorted because a sharded checkpoint's tensors are split across files by
/// name and the index is built by walking them in order — an unsorted walk
/// makes the load non-deterministic across runs on a directory listing that is
/// itself unordered.
pub fn safetensors_in(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("guest weights: reading {dir:?}: {e}"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    if out.is_empty() {
        return Err(format!("guest weights: no .safetensors in {dir:?}"));
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_of_a_float_slice_is_its_little_endian_image() {
        assert_eq!(bytes_of(&[1.0f32, 2.0]), {
            let mut v = 1.0f32.to_le_bytes().to_vec();
            v.extend_from_slice(&2.0f32.to_le_bytes());
            v
        });
    }

    #[test]
    fn an_empty_directory_says_so_rather_than_loading_nothing() {
        let dir = std::env::temp_dir().join("candle-guest-varground-empty");
        let _ = std::fs::create_dir_all(&dir);
        assert!(safetensors_in(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A sharded checkpoint's index is built by walking the files, so the walk
    /// has to be ordered — a directory listing is not.
    #[test]
    fn shards_are_walked_in_a_stable_order() {
        let dir = std::env::temp_dir().join("candle-guest-varground-order");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for n in ["c.safetensors", "a.safetensors", "b.safetensors"] {
            std::fs::write(dir.join(n), b"").unwrap();
        }
        let found = safetensors_in(&dir).unwrap();
        let names: Vec<String> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec!["a.safetensors", "b.safetensors", "c.safetensors"]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
