//! Device allocation and copy, with the guards that refuse a write into memory
//! declared immutable and the reusable staging scratch.
//!
//! `alloc` returns uninitialised memory that the caller must fully write, which
//! is why a buffer a kernel completely overwrites comes from it, and `alloc_zeros`
//! is the safe default. A host-to-device or device-to-device copy writes to an
//! address the allocator did not hand out, so the allocation guards cannot see
//! it, and the copy paths check their destination themselves. Staging scratch is
//! grown by doubling and reused across calls, so a steady wave allocates nothing,
//! and a table uploaded into it is ordered on the stream ahead of the launch that
//! reads it.

use super::*;

impl CudaDevice {
    #[allow(clippy::missing_safety_doc)]
    /// Refuse an allocation that overlaps memory declared immutable after load.
    ///
    /// The pool is supposed to hand out disjoint blocks; a fresh allocation
    /// landing on a resident weight means it handed one out twice, and the
    /// weight is corrupted the moment anything writes through the new handle.
    /// Catching it here names the allocation site instead of leaving a wrong
    /// number to be found several layers downstream.
    ///
    /// Costs two atomic loads and two compares for an address outside every
    /// declared region, which is every allocation in a healthy run — see
    /// [`crate::readonly_regions`].
    #[cfg(feature = "tensor-assert")]
    pub(super) fn guard_fresh_allocation<T>(what: &str, slice: &CudaSlice<T>, bytes: usize) {
        use cudarc::driver::DevicePtr;
        let stream = slice.stream().clone();
        let (base, _g) = slice.device_ptr(&stream);
        crate::readonly_regions::forbid_write(what, base, bytes);
    }

    /// No-op twin. Gated on the whole body rather than relying on
    /// `forbid_write`'s stub, because reading the address is not free: it
    /// clones a stream `Arc` and takes a sync guard, on a path that runs for
    /// every device allocation in the process.
    #[cfg(not(feature = "tensor-assert"))]
    #[inline(always)]
    pub(super) fn guard_fresh_allocation<T>(_what: &str, _slice: &CudaSlice<T>, _bytes: usize) {}

    /// `len` elements of device memory, **uninitialised**.
    ///
    /// # Safety
    ///
    /// The returned slice holds whatever the allocator last left there. The
    /// caller must write every element it goes on to read — a kernel that fully
    /// overwrites its output, or an explicit fill — because reading an
    /// unwritten element is undefined behaviour and, on this backend, silently
    /// returns another tenant's bytes rather than faulting.
    ///
    /// [`Self::alloc_zeros`] is the safe counterpart and is the right choice
    /// unless the buffer is provably fully written: the hot-path rule is that a
    /// buffer a kernel completely overwrites must come from here, so the zeroing
    /// memset is not paid on bytes that are about to be stamped anyway.
    #[track_caller]
    pub unsafe fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>> {
        forbidden_alloc::record("CudaDevice::alloc", len * std::mem::size_of::<T>());
        let _eager = self.pause_capture()?;
        let s = at_site(Location::caller(), || self.cuda_stream().alloc::<T>(len)).w()?;
        Self::guard_fresh_allocation("CudaDevice::alloc", &s, len * std::mem::size_of::<T>());
        Ok(s)
    }

    #[track_caller]
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<CudaSlice<T>> {
        forbidden_alloc::record("CudaDevice::alloc_zeros", len * std::mem::size_of::<T>());
        let _eager = self.pause_capture()?;
        let s = at_site(Location::caller(), || {
            self.cuda_stream().alloc_zeros::<T>(len)
        })
        .w()?;
        Self::guard_fresh_allocation(
            "CudaDevice::alloc_zeros",
            &s,
            len * std::mem::size_of::<T>(),
        );
        Ok(s)
    }

    #[track_caller]
    pub fn memcpy_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: DevicePtrMut<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        // A host→device copy is a bulk write, and its destination is computed
        // by the caller rather than handed out by an allocator — which is why
        // the guards on the allocation paths cannot see it. An upload aimed at
        // the wrong slot writes a contiguous, slot-sized block of plausible
        // bytes over whatever was there.
        //
        // Inside the eager section: reading the destination's address goes
        // through the compute stream, which a thread recording a wave may not
        // touch — the driver refuses it and the refusal invalidates the capture.
        let _eager = self.pause_capture()?;
        self.guard_copy_dst::<T, _>("CudaDevice::memcpy_htod", dst, src.len());
        self.cuda_stream().memcpy_htod(src, dst).w()
    }

    /// Refuse a copy whose destination overlaps memory declared immutable.
    ///
    /// Reads the destination's address through the same accessor the copy will,
    /// so the range checked is the range written. Costs two atomic loads and two
    /// compares when nothing is declared nearby — see
    /// [`crate::readonly_regions`] — and compiles away entirely without the
    /// `tensor-assert` feature.
    #[cfg(feature = "tensor-assert")]
    pub(super) fn guard_copy_dst<T, D: DevicePtrMut<T>>(
        &self,
        what: &str,
        dst: &mut D,
        elems: usize,
    ) {
        let stream = self.stream.clone();
        let (base, _g) = dst.device_ptr_mut(&stream);
        crate::readonly_regions::forbid_write(what, base, elems * std::mem::size_of::<T>());
    }

    /// No-op twin — same reasoning as [`Self::guard_fresh_allocation`], on a
    /// path that runs for every host↔device copy.
    #[cfg(not(feature = "tensor-assert"))]
    #[inline(always)]
    pub(super) fn guard_copy_dst<T, D: DevicePtrMut<T>>(
        &self,
        _what: &str,
        _dst: &mut D,
        _elems: usize,
    ) {
    }

    #[track_caller]
    pub fn memcpy_dtov<T: DeviceRepr, Src: DevicePtr<T>>(&self, src: &Src) -> Result<Vec<T>> {
        let _eager = self.pause_capture()?;
        self.cuda_stream().memcpy_dtov(src).w()
    }

    /// Read `out.len()` elements at device address `addr` into `out`, behind
    /// everything issued on the stream, and wait for them — the readback of
    /// device memory a caller holds by address rather than by slice.
    ///
    /// # Safety
    ///
    /// `addr` names at least `out.len()` live, initialised elements of `T`.
    #[track_caller]
    pub unsafe fn read_raw<T: DeviceRepr>(&self, addr: u64, out: &mut [T]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        let _eager = self.pause_capture()?;
        let stream = self.cuda_stream();
        // SAFETY: the caller's contract, and the wait below keeps `out`
        // borrowed until the copy has landed.
        unsafe { cudarc::driver::result::memcpy_dtoh_async(out, addr, stream.cu_stream()) }.w()?;
        sync_watch::synchronize(&stream).w()
    }

    #[track_caller]
    pub fn memcpy_dtod<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        // Same reasoning as `memcpy_htod`: a bulk write to a caller-computed
        // destination, invisible to every allocation-path guard. Guarded inside
        // the eager section for the reason `memcpy_htod` gives.
        let _eager = self.pause_capture()?;
        self.guard_copy_dst::<T, _>("CudaDevice::memcpy_dtod", dst, src.len());
        self.cuda_stream().memcpy_dtod(src, dst).w()
    }

    #[track_caller]
    pub fn memcpy_stod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
    ) -> Result<CudaSlice<T>> {
        // Allocates as well as copying: the destination slice is fresh device
        // memory, so this is a driver allocation like the two above.
        //
        // The payload is `len` elements, not `size_of_val(src)`: `Src` is a
        // container as often as a slice, and the size of a `Vec<T>` is its
        // three words — every upload from a `Vec` was reported as 24 B, a
        // 3 MiB recurrent-state restore among them.
        let bytes = src.len() * std::mem::size_of::<T>();
        forbidden_alloc::record("CudaDevice::memcpy_stod", bytes);
        let _eager = self.pause_capture()?;
        let s = at_site(Location::caller(), || self.cuda_stream().memcpy_stod(src)).w()?;
        // Allocates as well as copying, so it is an allocation path like the
        // two above and needs the same guard — and unlike them it writes to the
        // range immediately, so a collision here corrupts before anything else
        // gets a chance to notice.
        Self::guard_fresh_allocation("CudaDevice::memcpy_stod", &s, bytes);
        Ok(s)
    }

    /// [`Self::memcpy_stod`] with the destination taken from `origin`'s arena.
    ///
    /// Host uploads are the one wave-path allocation provenance cannot reach on
    /// its own: a table built on the CPU has no device operand to inherit from,
    /// so there is nothing for the rule to follow. What the call sites *do* have
    /// is the arena the table is about to be read alongside — the tile and token
    /// tables are consumed by the very kernels whose operands are already on the
    /// span — so they pass that arena in explicitly.
    ///
    /// The copy itself is unchanged; only the destination moves.
    /// Host upload placed on `origin`'s span, as a plain slice plus its backing.
    ///
    /// The same work as [`Self::memcpy_stod_from`], returning the pieces
    /// `CudaStorage` is built from rather than an [`Uploaded`] guard — which is
    /// what a *tensor* constructor needs, since `CudaStorageSlice` holds the
    /// slice directly.
    ///
    /// This is the only way a host-built table can land on the reservation:
    /// there is no device operand to inherit a ticket from, so the caller names
    /// the span it belongs to. Without it every per-wave descriptor table — the
    /// DeltaNet pointer tables, the rotary layouts, the batched row maps — is a
    /// driver allocation inside the wave.
    pub fn memcpy_stod_leased<T: DeviceRepr + ValidAsZeroBits, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
        origin: Backing,
    ) -> Result<(CudaSlice<T>, Backing)> {
        // An allocation that reaches the driver ends the recording segment on
        // its own; a carve from the wave's arena does not need to.
        let (mut dst, backing) = unsafe { alloc_inheriting::<T>(self, src.len(), origin)? };
        self.upload_into_slice(&mut dst, src)?;
        Ok((dst, backing))
    }

    /// Copy `src` into the front of `dst`. Inside a recording wave capture the
    /// copy is recorded (see [`Self::upload_raw`]); otherwise it is queued on
    /// the compute stream like any upload.
    pub fn upload_into_slice<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        &self,
        dst: &mut CudaSlice<T>,
        src: &Src,
    ) -> Result<()> {
        let (at, _g) = dst.device_ptr_mut(&self.stream);
        // SAFETY: the slice is read on the host right here, before the guard
        // drops; a host buffer has nothing to wait for.
        let (host, _s) = unsafe { src.stream_synced_slice(&self.stream) };
        // SAFETY: `host` is `len` plain device-representable values.
        let bytes = unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, std::mem::size_of_val(host))
        };
        self.upload_raw(at, bytes)
    }

    /// Record the copy of `src` to the device address `dst` into this thread's
    /// recording segment, staged through the wave's ring, and return `true`;
    /// return `false` and do nothing when this thread is not recording or the
    /// ring is full.
    ///
    /// For an uploader with an eager path of its own — pinned staging that
    /// outruns the pageable copy [`Self::upload_raw`] falls back to — which it
    /// then runs inside [`Self::pause_capture`], behind the launches recorded
    /// so far.
    pub fn record_upload(&self, dst: u64, src: &[u8]) -> bool {
        self.capture.record_upload(dst, src)
    }

    /// Copy `src` to the device address `dst`, which holds at least
    /// `src.len()` bytes.
    ///
    /// While this thread records a wave capture, the bytes are staged in the
    /// wave's ring and their copy is recorded into the segment, so the upload
    /// does not end it. In a segment that runs eagerly the copy is queued on the
    /// stream at once, behind the launches already issued and ahead of the ones
    /// that read it; ending the segment there would cut the wave where a
    /// recorded one is not cut, and every segment after it would hold different
    /// layers. Otherwise — or when the ring is full — the recording is paused
    /// and the copy queued behind everything issued before it. The driver has
    /// staged pageable bytes before it returns, so `src` may go.
    #[track_caller]
    pub fn upload_raw(&self, dst: u64, src: &[u8]) -> Result<()> {
        if src.is_empty() || self.record_upload(dst, src) {
            return Ok(());
        }
        let _eager = if self.capture.eager_segment_here() {
            None
        } else {
            self.pause_capture()?
        };
        let stream = self.cuda_stream();
        // SAFETY: the caller's `dst` holds `src.len()` bytes; queued on the
        // stream the launches that read it are on.
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
                stream.cu_stream(),
            )
            .result()
            .w()
        }
    }

    pub fn memcpy_stod_from<T: DeviceRepr + ValidAsZeroBits, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
        origin: Backing,
    ) -> Result<Uploaded<T>> {
        let (mut dst, backing) = unsafe { alloc_inheriting::<T>(self, src.len(), origin)? };
        self.upload_into_slice(&mut dst, src)?;
        Ok(Uploaded {
            slice: std::mem::ManuallyDrop::new(dst),
            backing,
            _anchor: None,
        })
    }

    /// Run `f` with a staging buffer of at least `bytes`, the scratch a
    /// quantized matmul quantizes its activation into before the kernel reads
    /// it.
    ///
    /// **Reused, not allocated per call.** The scratch is written and read by
    /// launches on this handle's stream, so the next call's quantize is
    /// ordered after the previous matmul's read and may take the same bytes.
    /// It grows to the widest request seen — by doubling, so a ramp of widths
    /// settles in a few steps — and is never handed back, which is what makes a
    /// steady-state wave allocation-free. The lock is held while `f` runs, so
    /// two callers on this handle cannot interleave their quantize and read.
    ///
    /// **Inside a recording wave capture a scratch already wide enough is used
    /// in place**, so `f`'s write and read are recorded: `f` must launch only on
    /// [`Self::cuda_stream`]. A segment replays whole on the compute stream, so
    /// this thread's own eager use after a pause runs entirely before that
    /// segment or entirely after it, never between its write and its read.
    /// Pausing here instead ended the segment at every legacy quantized matmul:
    /// measured on Qwen3.5-0.8B, one cut per DeltaNet input projection,
    /// 25–176 µs of idle GPU each — ~1.5 ms of a ~5 ms two-session decode step.
    /// Growth allocates, so it runs eagerly.
    ///
    /// **Every other thread stages through a scratch of its own.** A recording
    /// thread launches its segment whenever it next ends one, and that launch
    /// does not take this lock — so it can land between another thread's copy
    /// into a shared scratch and the kernel that reads it, and the segment's
    /// recorded uploads overwrite what that kernel is about to read. Measured
    /// in zend: the persistence thread's KV record fill read a quantized
    /// activation as its descriptor table and wrote a record to an unmapped
    /// address. Recorded segments never touch the other scratch, so nothing
    /// can be launched between its copy and its read but that thread's own
    /// work, which the lock already orders.
    #[track_caller]
    pub fn with_staging<R>(
        &self,
        bytes: usize,
        f: impl FnOnce(&mut CudaSlice<u8>) -> Result<R>,
    ) -> Result<R> {
        let scratch = if self.capture.capturing_here() {
            &self.staging
        } else {
            &self.foreign_staging
        };
        let mut slot = scratch.lock().unwrap();
        // A scratch its ground wants vacated is taken afresh at its width, where
        // the ground now puts it; one too narrow doubles.
        let moving = slot.as_mut().is_some_and(|s| s.asked_to_move());
        let fits = !moving && slot.as_ref().is_some_and(|s| s.len() >= bytes);
        let _eager = if fits { None } else { self.pause_capture()? };
        if !fits {
            let grown = match slot.as_ref() {
                Some(s) if moving => s.len(),
                Some(s) => s.len() * 2,
                None => 0,
            }
            .max(bytes)
            .max(1);
            // Retired rather than dropped: freed behind every launch that may
            // still read it — at once when nothing records, since the pause
            // above has launched this thread's segment.
            if let Some(old) = slot.take() {
                self.retire(old);
            }
            // SAFETY: staging is written by the quantize the caller launches
            // before its kernel reads it.
            *slot = Some(unsafe { Held::alloc(self, grown)? });
        }
        f(slot.as_mut().expect("sized above"))
    }

    /// Upload `data` into the staging scratch ([`Self::with_staging`]) and run
    /// `f` with its device address — for a launch's small host-built table
    /// (a job list, a carry list) that lives from the upload to the kernel's
    /// read and no further.
    ///
    /// Allocates nothing in the steady state, inside a forward or between two:
    /// the copy and the launch `f` queues are ordered on this stream, and the
    /// scratch is not handed to another caller until `f` returns.
    ///
    /// **Inside a recording wave capture it records rather than pauses**, once
    /// the scratch is already wide enough: the copy is staged through the
    /// wave's ring ([`Self::record_upload`]) and `f`'s launches are recorded
    /// behind it. A recorded chain runs in stream order, so the next call's
    /// copy into the scratch lands after this call's launches have read it —
    /// the same guarantee the eager path buys by running both now — and a
    /// pause before any eager user of the scratch submits everything recorded
    /// ahead of it. Pausing instead ended the wave's segment at every call: a
    /// QSA index append per attention layer, each a graph launch and a host
    /// round trip behind it. Growth still runs eagerly; it allocates.
    #[track_caller]
    pub fn with_staged_upload<T: DeviceRepr, R>(
        &self,
        data: &[T],
        f: impl FnOnce(u64) -> Result<R>,
    ) -> Result<R> {
        let bytes = std::mem::size_of_val(data);
        // SAFETY: `data` is `bytes` of plain device-representable values; read
        // as bytes, it is exactly what the kernel reads back.
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
        if bytes > 0 {
            let mut slot = self.staging.lock().unwrap();
            // A scratch asked to move is moved by the eager path below.
            let moving = slot.as_mut().is_some_and(|s| s.asked_to_move());
            if let Some(buf) = slot.as_ref().filter(|s| !moving && s.len() >= bytes) {
                let (ptr, _guard) = buf.device_ptr(&self.stream);
                if self.record_upload(ptr, src) {
                    // The lock is held while `f` records, as on the eager path:
                    // no other caller's copy can be recorded between this
                    // copy and the launches that read it.
                    return f(ptr);
                }
            }
        }
        // The copy below is issued eagerly, not recorded, so a segment still
        // holding a read of the scratch must be launched ahead of it. A segment
        // that runs eagerly holds none: the copy goes on the stream behind its
        // launches, ahead of `f`'s, and ending it here would cut the wave where
        // a recorded one is not cut (see [`Self::upload_raw`]).
        let _eager = if self.capture.eager_segment_here() {
            None
        } else {
            self.pause_capture()?
        };
        self.with_staging(bytes, |buf| {
            if bytes > 0 {
                self.cuda_stream()
                    .memcpy_htod(src, &mut buf.slice_mut(..bytes))
                    .w()?;
            }
            let (ptr, _guard) = buf.device_ptr(&self.stream);
            f(ptr)
        })
    }

    /// Upload `segments` as one copy into the staging scratch
    /// ([`Self::with_staged_upload`]) and run `f` with each segment's device
    /// address — for a launch fed by several host arrays (a descriptor table
    /// and the arrays it indexes), which would otherwise take one allocation
    /// apiece. Build a segment from a typed slice with [`super::staged::segment`].
    pub fn with_staged_segments<R>(
        &self,
        segments: &[&[u8]],
        f: impl FnOnce(&[u64]) -> Result<R>,
    ) -> Result<R> {
        let (bytes, offsets) = pack_segments(segments);
        self.with_staged_upload(&bytes, |base| {
            let addrs: Vec<u64> = offsets.iter().map(|&o| base + o as u64).collect();
            f(&addrs)
        })
    }

    /// Upload `segments` on `stream` into a held scratch, run `f` with each
    /// segment's device address and the address of `room` further bytes left
    /// for the launch to write, then synchronise `stream` — for work that
    /// reads its results back before it returns, and for a launch table on a
    /// stream other than this handle's (the persistence thread's copy stream),
    /// where the stream-ordered reuse [`Self::with_staged_segments`] relies on
    /// does not hold.
    ///
    /// The room is uninitialised and is not uploaded: a launch writing its
    /// results there costs no transfer for space it is about to fill.
    ///
    /// **Reuse rests on the synchronise, not on stream order.** The scratch is
    /// locked from the upload until `stream` has drained, so the next caller —
    /// on any stream — finds every read of it retired. It grows by doubling
    /// and is never handed back, so a steady stream of launches allocates
    /// nothing. Its own lock: a caller waiting out a long copy here never
    /// holds the forward's [`Self::with_staging`].
    #[track_caller]
    pub fn with_synced_upload<R>(
        &self,
        stream: &Arc<CudaStream>,
        segments: &[&[u8]],
        room: usize,
        f: impl FnOnce(&[u64], u64) -> Result<R>,
    ) -> Result<R> {
        let (bytes, offsets) = pack_segments(segments);
        let room_at = bytes.len().next_multiple_of(SEGMENT_ALIGN);
        let need = room_at + room;
        let _eager = self.pause_capture()?;
        let mut slot = self.synced_staging.lock().unwrap();
        // Every read of the scratch has retired by the time the lock is free, so
        // one its ground wants vacated is simply taken afresh at its width.
        let moving = slot.as_mut().is_some_and(|s| s.asked_to_move());
        if moving || !slot.as_ref().is_some_and(|s| s.len() >= need) {
            let grown = match slot.as_ref() {
                Some(s) if moving => s.len(),
                Some(s) => s.len() * 2,
                None => 0,
            }
            .max(need)
            .max(1);
            *slot = None;
            // SAFETY: every byte a launch reads is written by the upload below.
            *slot = Some(unsafe { Held::alloc(self, grown)? });
            // The pool hands the bytes out in this handle's stream order, and
            // `stream` may be another one: wait for the allocation to retire
            // before a copy on `stream` writes it. Growth only.
            sync_watch::synchronize(&self.cuda_stream()).w()?;
        }
        let buf = slot.as_mut().expect("sized above");
        if !bytes.is_empty() {
            stream
                .memcpy_htod(&bytes, &mut buf.slice_mut(..bytes.len()))
                .w()?;
        }
        let base = buf.device_ptr(stream).0;
        let addrs: Vec<u64> = offsets.iter().map(|&o| base + o as u64).collect();
        let out = f(&addrs, base + room_at as u64);
        sync_watch::synchronize(stream).w()?;
        out
    }

    /// Device copy of a small layout table (the dims/strides blob a strided
    /// kernel reads), memoized by contents and carved from this handle's info
    /// ring — see [`super::info_ring`]. A steady-state wave hits the memo and
    /// uploads nothing; a new shape costs one small copy and no allocation.
    ///
    /// **Leased before the ring is locked.** A thread that is not recording
    /// waits for its lease while a segment records, and the recording thread
    /// takes this lock too: holding it across that wait would leave each
    /// waiting on the other.
    #[track_caller]
    pub fn info_table(&self, info: &[usize]) -> Result<Arc<Uploaded<usize>>> {
        let _stream = self.cuda_stream();
        self.info_tables.lock().unwrap().table(self, info)
    }

    /// The token-major → group-major gather permutation, memoized per `(rows, groups)`:
    ///
    /// ```text
    /// ids[g * rows + t] = t * groups + g        for g in 0..groups, t in 0..rows
    /// ```
    ///
    /// This is the row order the grouped output projection needs. Its source `[rows, groups, w]`
    /// activation is contiguous, so group `g`'s rows are interleaved with stride `groups`; the
    /// grouped matmul wants each group's rows adjacent, and gathering with this table produces
    /// that without ever materialising the permutation in f32.
    ///
    /// **Keyed by shape, not contents.** The table is a pure function of `(rows, groups)`, so a
    /// hit is a two-word hash — no host build, no upload. Building it per call instead was the
    /// same WDDM submission storm [`Self::info_table`] exists to prevent, except worse: it also
    /// spent `O(rows·groups)` of host time per layer per wave.
    ///
    /// Entries are held ([`Held`]) and outlive any single wave; kernels only read them. Cleared
    /// wholesale past a bound, like the info-table cache — in-flight users hold their own `Arc`,
    /// and the cleared entries are retired to the capture hub rather than dropped, so a launch
    /// another thread recorded against one is issued before its bytes go back.
    #[track_caller]
    pub fn group_major_ids(&self, rows: usize, groups: usize) -> Result<Arc<Uploaded<u32>>> {
        let key = (rows, groups);
        let mut cache = self.perm_tables.lock().unwrap();
        if let Some(t) = cache.get(&key) {
            return Ok(t.clone());
        }
        // Clearing frees, and the build below uploads.
        let _eager = self.pause_capture()?;
        if cache.len() >= 1024 {
            // Retired, not dropped: the pause above flushed only this thread's
            // capture, and another thread's recording segment may read these.
            self.retire(std::mem::take(&mut *cache));
        }
        let mut ids: Vec<u32> = Vec::with_capacity(rows * groups);
        for g in 0..groups {
            for t in 0..rows {
                ids.push((t * groups + g) as u32);
            }
        }
        let held = Arc::new(Held::from_host(self, &ids)?);
        let base = held.device_ptr(&self.stream).0;
        // SAFETY: a view of `held`'s `ids.len()` elements, which the anchor
        // keeps alive for as long as the view.
        let view = unsafe { self.stream.upgrade_device_ptr::<u32>(base, ids.len()) };
        let t = Arc::new(Uploaded::leased(view, Some(held as Arc<_>)));
        cache.insert(key, t.clone());
        Ok(t)
    }

    /// Generate an integer arange (`buf[i] = start + i*step`, exact integer arithmetic)
    /// directly on the device — no host-side build, no tiny H2D upload. `Tensor::arange`
    /// index tensors are hot-path gather indices, and per-call host uploads of them were
    /// a measured WDDM submission storm. Integer dtypes only (U8/U32/I64); float aranges
    /// keep the host build (its repeated-addition rounding is the documented semantics,
    /// which the kernel's closed form would not reproduce bit-for-bit). Start/step are
    /// passed as bits per `run_arange_op`.
    pub fn arange_int(
        &self,
        dtype: DType,
        start_bits: u64,
        step_bits: u64,
        len: usize,
    ) -> Result<CudaStorage> {
        let launch = |ptr: u64, fill_dtype: FillDType| unsafe {
            run_arange_op(
                fill_dtype as i32,
                ptr as *mut std::ffi::c_void,
                start_bits,
                step_bits,
                len,
                self.cuda_stream().cu_stream() as *mut std::ffi::c_void,
            );
        };
        let slice = match dtype {
            DType::U8 => {
                let s = unsafe { self.alloc::<u8>(len)? };
                {
                    let (ptr, _g) = s.device_ptr(&self.stream);
                    launch(ptr, FillDType::U8);
                }
                CudaStorageSlice::U8(s)
            }
            DType::U32 => {
                let s = unsafe { self.alloc::<u32>(len)? };
                {
                    let (ptr, _g) = s.device_ptr(&self.stream);
                    launch(ptr, FillDType::U32);
                }
                CudaStorageSlice::U32(s)
            }
            DType::I64 => {
                let s = unsafe { self.alloc::<i64>(len)? };
                {
                    let (ptr, _g) = s.device_ptr(&self.stream);
                    launch(ptr, FillDType::I64);
                }
                CudaStorageSlice::I64(s)
            }
            _ => crate::bail!("arange_int: integer dtypes only, got {dtype:?}"),
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
            backing: Backing::Owned,
            anchor: None,
        })
    }
}
