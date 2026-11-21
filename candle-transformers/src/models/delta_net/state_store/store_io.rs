//! Exports and imports a store's state, and reads the decode addresses off its
//! slots.
//!
//! Export and readback refuse a store mid-wave or with rows pending, because a
//! snapshot has to capture a sealed boundary. On CUDA the readback waits on the
//! stream once for the whole store, and the upload is queued from pinned memory
//! without a host wait. An import validates everything before it touches a tensor,
//! then writes into the existing buffers, which the fused kernels rely on.

use super::*;

impl RecurrentStateStore {
    /// Every recurrent layer's decode-table addresses, in slot order:
    /// `[tail, state, timeline]` of the live half — the three rows
    /// [`build_wave_table`] lays out per layer — without marking any layer
    /// (a sweep may run only part of the stack, and a layer handed a timeline
    /// is one that ran).
    ///
    /// **Read off the slots, not the tensors.** Each half is a view built by
    /// `Half::in_slot` at its slot's base — `s` at the base, the conv tail and
    /// the timeline at their [`SLOT_ALIGN`] boundaries — and every place that
    /// rebuilds a half (`new`, `fork_from`, `relocate`) or exchanges two
    /// (`commit_wave`) moves `held` with it, so the slot always names the buffers
    /// the half stands on. Resolving a tensor's address takes its storage lock
    /// and a stream-guarded device pointer; the table needs three per layer per
    /// decode sequence per forward, which at 256 sequences on an 18-layer
    /// recurrent stack was milliseconds of host time with the device idle.
    ///
    /// Debug builds check every address against the tensor it describes, which
    /// is the one way the slots and the halves could part: a caller rebinding a
    /// half through `layer_state_mut` rather than writing into it.
    ///
    /// [`build_wave_table`]: super::cuda::build_wave_table
    #[cfg(feature = "cuda")]
    pub fn decode_table_addresses(&self) -> Result<impl Iterator<Item = [u64; 3]> + '_> {
        let block = state_block(&self.dims);
        let at = move |slot: &ArenaSlot| {
            [
                slot.ptr() + block.conv_off as u64,
                slot.ptr(),
                slot.ptr() + block.timeline_off as u64,
            ]
        };
        for slot in &self.slots {
            if slot.live.held.is_none() {
                candle::bail!(
                    "recurrent store: layer {} has no state-arena slot, so its decode \
                     addresses are not the slot's — a store built off a CUDA device",
                    slot.layer_index
                );
            }
            #[cfg(debug_assertions)]
            {
                let held = slot.live.held.as_ref().expect("checked above");
                let Device::Cuda(cuda) = &self.device else {
                    candle::bail!("recurrent store: arena slots on a non-CUDA device");
                };
                let want = [
                    tensor_device_ptr(cuda, &slot.live.state.conv_tail)?,
                    tensor_device_ptr(cuda, &slot.live.state.s)?,
                    tensor_device_ptr(cuda, &slot.live.timeline)?,
                ];
                let got = at(held);
                if got != want {
                    candle::bail!(
                        "recurrent store: layer {}'s slot names {got:x?} but its half sits \
                         at {want:x?} — a half was rebound instead of written into",
                        slot.layer_index
                    );
                }
            }
        }
        Ok(self
            .slots
            .iter()
            .map(move |slot| at(slot.live.held.as_ref().expect("checked above"))))
    }

    /// Read every layer back as LE F32 bytes — the turn-seal snapshot body.
    /// Refused mid-wave and with timeline rows pending: a snapshot must capture a
    /// sealed boundary, never a wave in flight or a span not yet applied.
    pub fn export(&self) -> Result<Vec<ExportedLayerState>> {
        if self.open {
            candle::bail!("recurrent store: export mid-wave — seal, then snapshot");
        }
        if self.has_pending() {
            candle::bail!(
                "recurrent store: export with timeline rows pending — fold or discard them, \
                 then snapshot"
            );
        }
        let d = &self.dims;
        let bytes = self.read_back()?;
        Ok(self
            .slots
            .iter()
            .zip(bytes)
            .map(|(slot, (state, conv_tail))| ExportedLayerState {
                layer_index: slot.layer_index as u32,
                n_v_heads: d.n_v_heads as u32,
                d_v: d.head_dim as u32,
                d_k: d.head_dim as u32,
                state,
                conv_channels: d.conv_dim() as u32,
                conv_tail_cols: (d.conv_kernel - 1) as u32,
                conv_tail,
            })
            .collect())
    }

    /// Every layer's live state and conv tail as LE F32 bytes, in slot order.
    ///
    /// **On CUDA, one synchronisation for the whole store.** Every tensor's bytes
    /// are enqueued into one pinned host buffer and the stream is waited on once:
    /// a readback per tensor was two synchronising copies per recurrent layer into
    /// freshly allocated pageable memory — 60 of them on the 35B-A3B, ~27 ms a seal
    /// at ~2.2 GB/s, on the scheduler's loop thread. The buffer is kept between
    /// seals ([`READBACK`]), so neither its pinning nor its first-touch faults are
    /// paid per turn.
    pub(super) fn read_back(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        #[cfg(feature = "cuda")]
        if let Some(first) = self.slots.first() {
            if let Device::Cuda(dev) = first.live.state.s.device() {
                return self.read_back_cuda(dev);
            }
        }
        self.slots
            .iter()
            .map(|slot| {
                let state: Vec<f32> = slot.live.state.s.flatten_all()?.to_vec1()?;
                let tail: Vec<f32> = slot.live.state.conv_tail.flatten_all()?.to_vec1()?;
                Ok((f32_le_bytes(&state), f32_le_bytes(&tail)))
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    pub(super) fn read_back_cuda(&self, dev: &CudaDevice) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let state_bytes = self.dims.state_elems() * 4;
        let tail_bytes = self.dims.conv_state_elems() * 4;
        let total = self.slots.len() * (state_bytes + tail_bytes);
        let mut held = READBACK.lock().unwrap_or_else(|e| e.into_inner());
        if held.as_ref().is_none_or(|b| b.len() < total) {
            // Plain pinned memory, not write-combined: the CPU reads it straight
            // back. Portable, because the buffer is shared by every store in the
            // process, whichever device holds it. Every copy into the buffer it
            // replaces has completed — each call below synchronises before it
            // returns, on success and on error alike.
            *held = Some(PinnedBuf::alloc_owned_portable(total)?);
        }
        let buf = held
            .as_mut()
            .expect("the readback buffer was just sized")
            .as_mut_slice();
        let stream = dev.cuda_stream();
        // Every copy enqueued, then one wait. The wait runs even when an enqueue
        // fails part-way: copies already queued are still writing into `buf`, and
        // returning before they land would hand the next seal a buffer under DMA.
        let enqueued = (|| -> Result<()> {
            let mut at = 0usize;
            for slot in &self.slots {
                for (tensor, bytes) in [
                    (&slot.live.state.s, state_bytes),
                    (&slot.live.state.conv_tail, tail_bytes),
                ] {
                    let (storage, layout) = tensor.storage_and_layout();
                    let Storage::Cuda(cuda) = &*storage else {
                        candle::bail!(
                            "recurrent store: a CUDA store holds a non-CUDA state tensor"
                        );
                    };
                    let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
                        Error::Msg("recurrent store: a state tensor is not contiguous".into())
                    })?;
                    if (end - start) * 4 != bytes {
                        candle::bail!(
                            "recurrent store: layer {} holds {} bytes of state where its \
                             geometry says {bytes}",
                            slot.layer_index,
                            (end - start) * 4,
                        );
                    }
                    cuda.copy_f32_bytes_to_host_on_stream(
                        &mut buf[at..at + bytes],
                        &stream,
                        start,
                        end - start,
                    )?;
                    at += bytes;
                }
            }
            Ok(())
        })();
        let synced = stream
            .synchronize()
            .map_err(|e| Error::Msg(format!("recurrent store: readback sync: {e:?}")));
        enqueued?;
        synced?;
        let mut at = 0usize;
        Ok(self
            .slots
            .iter()
            .map(|_| {
                // Into kept rows: every seal exports the same row sizes, and the
                // rows of the snapshots dropped since are in the pool.
                let state = host_rows::row_from(&buf[at..at + state_bytes]);
                at += state_bytes;
                let tail = host_rows::row_from(&buf[at..at + tail_bytes]);
                at += tail_bytes;
                (state, tail)
            })
            .collect())
    }

    /// [`Self::import`]'s copy on CUDA: every layer's bytes into one pinned host
    /// buffer, then every upload enqueued from it, and no host wait.
    ///
    /// From pageable memory an upload cannot return before the driver's own
    /// staging buffer has drained through the DMA, and that DMA is ordered on
    /// the stream behind whatever forward is still running: a parked sequence's
    /// claim waited out the forward in flight — ~130 ms of a 190 ms admission
    /// begin on BlackWorld (2026-11-15). From pinned memory the copies are
    /// queued and the call returns. The wire form is the device's own,
    /// little-endian F32, so the bytes cross as they are, with no conversion.
    /// The buffer comes from [`UPLOAD`]'s pool, one whose last uploads have
    /// completed, and goes back with the event recorded after these.
    #[cfg(feature = "cuda")]
    pub(super) fn import_cuda(
        &self,
        dev: &CudaDevice,
        layers: &[ExportedLayerState],
    ) -> Result<()> {
        let state_bytes = self.dims.state_elems() * 4;
        let tail_bytes = self.dims.conv_state_elems() * 4;
        let total = self.slots.len() * (state_bytes + tail_bytes);
        let taken = {
            let mut pool = UPLOAD.lock().unwrap_or_else(|e| e.into_inner());
            pool.iter()
                .position(|u| u.buf.len() >= total && u.is_free())
                .map(|i| pool.swap_remove(i))
        };
        let mut up = match taken {
            Some(up) => up,
            // Write-combined: the CPU only writes it, and the device reads it.
            None => PinnedUpload {
                buf: PinnedBuf::alloc_owned(total)?,
                done: None,
            },
        };
        let buf = up.buf.as_mut_slice();
        let mut at = 0usize;
        for l in layers {
            buf[at..at + state_bytes].copy_from_slice(&l.state);
            at += state_bytes;
            buf[at..at + tail_bytes].copy_from_slice(&l.conv_tail);
            at += tail_bytes;
        }
        let stream = dev.cuda_stream();
        let enqueued = (|| -> Result<()> {
            let mut at = 0usize;
            for slot in &self.slots {
                for (tensor, bytes) in [
                    (&slot.live.state.s, state_bytes),
                    (&slot.live.state.conv_tail, tail_bytes),
                ] {
                    if tensor.elem_count() * 4 != bytes {
                        candle::bail!(
                            "recurrent store: layer {} holds {} bytes of state where its \
                             geometry says {bytes}",
                            slot.layer_index,
                            tensor.elem_count() * 4,
                        );
                    }
                    let dst = tensor_device_ptr(dev, tensor)?;
                    // SAFETY: `dst` addresses `bytes` bytes of the slot's own
                    // contiguous state tensor, and the source is `bytes` bytes of
                    // pinned memory nothing writes until the event below has
                    // passed. On the device's stream, behind the slot's clean.
                    unsafe { memcpy_htod_async(dst, &buf[at..at + bytes], stream.cu_stream()) }
                        .map_err(|e| {
                            Error::Msg(format!("recurrent store: restore upload: {e:?}"))
                        })?;
                    at += bytes;
                }
            }
            Ok(())
        })();
        // Recorded even when an enqueue failed part-way: the copies already
        // queued still read the buffer.
        let recorded = match stream.record_event(None) {
            Ok(done) => {
                up.done = Some(done);
                Ok(())
            }
            Err(e) => stream
                .synchronize()
                .map_err(|e| Error::Msg(format!("recurrent store: restore upload sync: {e:?}")))
                .and(Err(Error::Msg(format!(
                    "recurrent store: restore upload event: {e:?}"
                )))),
        };
        UPLOAD.lock().unwrap_or_else(|e| e.into_inner()).push(up);
        recorded?;
        enqueued
    }

    /// Whether a snapshot fits a store of this geometry — its schedule hash, and
    /// every layer's index and dims — without building one. What [`Self::import`]
    /// checks before it touches a tensor, available to a caller that holds the
    /// snapshot back until a store is wanted, so a foreign layout is refused when
    /// it arrives rather than when it is first used.
    pub fn check_snapshot(
        layer_kinds: &[LayerKind],
        dims: &DeltaNetDims,
        snapshot_hash: u64,
        layers: &[ExportedLayerState],
    ) -> Result<()> {
        let hash = schedule_hash(layer_kinds, dims);
        if snapshot_hash != hash {
            candle::bail!(
                "recurrent store: snapshot schedule hash {snapshot_hash:#x} does not match \
                 this model's {hash:#x} — recompute the state instead of scattering a foreign \
                 layout"
            );
        }
        let recurrent: Vec<usize> = layer_kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == LayerKind::DeltaNet)
            .map(|(i, _)| i)
            .collect();
        if layers.len() != recurrent.len() {
            candle::bail!(
                "recurrent store: snapshot has {} layers, store has {}",
                layers.len(),
                recurrent.len()
            );
        }
        for (&layer_index, l) in recurrent.iter().zip(layers) {
            if l.layer_index as usize != layer_index
                || l.n_v_heads as usize != dims.n_v_heads
                || l.d_v as usize != dims.head_dim
                || l.d_k as usize != dims.head_dim
                || l.conv_channels as usize != dims.conv_dim()
                || l.conv_tail_cols as usize != dims.conv_kernel - 1
                || l.state.len() != dims.state_elems() * 4
                || l.conv_tail.len() != dims.conv_state_elems() * 4
            {
                candle::bail!(
                    "recurrent store: snapshot layer {} does not match the store's \
                     geometry",
                    l.layer_index
                );
            }
        }
        Ok(())
    }

    /// Scatter a snapshot back into the store — the resume path. Validates
    /// the schedule hash and every layer's dims before touching any tensor
    /// ([`Self::check_snapshot`]); on any mismatch the store is left untouched
    /// and the caller recomputes. Refused mid-wave and with timeline rows
    /// pending, as `export` is: a step still deciding which rows it keeps is not
    /// a point a restore may land on.
    pub fn import(&mut self, snapshot_hash: u64, layers: &[ExportedLayerState]) -> Result<()> {
        if self.open {
            candle::bail!("recurrent store: import mid-wave");
        }
        self.break_step();
        if self.has_pending() {
            candle::bail!(
                "recurrent store: import with timeline rows pending — fold or discard them, \
                 then restore"
            );
        }
        if snapshot_hash != self.hash {
            candle::bail!(
                "recurrent store: snapshot schedule hash {snapshot_hash:#x} does not match \
                 this model's {:#x} — recompute the state instead of scattering a foreign \
                 layout",
                self.hash
            );
        }
        let d = &self.dims;
        if layers.len() != self.slots.len() {
            candle::bail!(
                "recurrent store: snapshot has {} layers, store has {}",
                layers.len(),
                self.slots.len()
            );
        }
        // Validate everything first — import is all-or-nothing.
        for (slot, l) in self.slots.iter().zip(layers) {
            if l.layer_index as usize != slot.layer_index
                || l.n_v_heads as usize != d.n_v_heads
                || l.d_v as usize != d.head_dim
                || l.d_k as usize != d.head_dim
                || l.conv_channels as usize != d.conv_dim()
                || l.conv_tail_cols as usize != d.conv_kernel - 1
                || l.state.len() != d.state_elems() * 4
                || l.conv_tail.len() != d.conv_state_elems() * 4
            {
                candle::bail!(
                    "recurrent store: snapshot layer {} does not match the store's \
                     geometry",
                    l.layer_index
                );
            }
        }
        self.break_step();
        #[cfg(feature = "cuda")]
        if let Some(Device::Cuda(dev)) = self.slots.first().map(|s| s.live.state.s.device()) {
            let dev = dev.clone();
            self.import_cuda(&dev, layers)?;
            self.seeded = true;
            return Ok(());
        }
        for (slot, l) in self.slots.iter_mut().zip(layers) {
            let state_f: Vec<f32> = l
                .state
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let tail_f: Vec<f32> = l
                .conv_tail
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            // Written into the slot's buffers rather than replacing them: the
            // slot's tensors keep their identity for the store's whole life, and
            // the fused decode kernels rely on that.
            slot.live.state.write_host(&state_f, &tail_f)?;
        }
        // Restored state is state someone put here on purpose. Without this the
        // first wave on a resumed slot standing at offset 0 would reset it, and
        // the conversation would come back fluent and amnesiac — the exact
        // failure resume exists to remove.
        self.seeded = true;
        Ok(())
    }
}
