//! Reuses a cached index for a repeated segment set, and runs a launch with the
//! index's turns held resident.
//!
//! An index is reused only while its turns still hold the runs it recorded. A
//! turn that has moved carries a new run id, and the reuse is refused. Taking the
//! lease holds the turns as a set, without a lock. Every scan path binds the
//! arena's CUDA context first, because the normalisation warm-ups run on threads
//! that never bound it. The hold is released whether or not the launch succeeded.

use super::*;

impl GalleryArena {
    /// Run `launch` over the index for `segments` — reused from the cache when
    /// the same segment set is rescanned, built (and cached) otherwise — with
    /// the index's turns pinned resident for the launch's duration. `launch`
    /// is handed the groups each segment weighs, which the index may exceed.
    pub(super) fn with_index<T>(
        &self,
        segments: &[PagedSegment],
        launch: impl FnOnce(&PagedIndex, &[GroupSet]) -> Result<T>,
    ) -> Result<T> {
        let t_fingerprint = Instant::now();
        let plan = ScanPlan::of(segments);
        let fingerprint_us = t_fingerprint.elapsed().as_micros() as u64;
        self.with_index_keyed(
            &plan,
            fingerprint_us,
            |cover| self.build_index(segments, cover),
            launch,
        )
    }

    /// [`Self::scan_segment_weighted`] over the segments `build` makes, under
    /// the index key `plan` holds for them — `build` is called only when no
    /// index is cached under it, so a caller whose segments do not change
    /// neither builds nor fingerprints them again ([`ScanPlan`]).
    pub fn scan_keyed<'s>(
        &self,
        plan: &ScanPlan,
        build: &dyn Fn() -> Vec<PagedSegment<'s>>,
        probes: &[&[WideQSig]],
        segment_weights: &[&[f32]],
    ) -> Result<Vec<Vec<f32>>> {
        if segment_weights.len() != plan.weighed.len() {
            return Err(candle::Error::Msg(format!(
                "paged scan: {} segment weightings for {} segments",
                segment_weights.len(),
                plan.weighed.len()
            )));
        }
        self.with_index_keyed(
            plan,
            0,
            |cover| self.build_index(&build(), cover),
            |idx, weighed| self.launch_paged(idx, weighed, probes, segment_weights, None),
        )
    }

    /// [`Self::with_index`] under `plan`'s key, the index built by `build`
    /// for the groups it is handed when none is cached; `fingerprint_us` is
    /// what working the key out cost the caller.
    pub(super) fn with_index_keyed<T>(
        &self,
        plan: &ScanPlan,
        fingerprint_us: u64,
        build: impl FnOnce(&[GroupSet]) -> Result<PagedIndex>,
        launch: impl FnOnce(&PagedIndex, &[GroupSet]) -> Result<T>,
    ) -> Result<T> {
        // **Every scan path binds this arena's context first.**
        //
        // Below here are raw driver calls and FFI that read the *calling
        // thread's* current device — `tensor_caps` says so explicitly. That
        // used to be free, because scans only ever ran on threads that already
        // held the context. The normalization warm-ups moved onto their own
        // rayon pool, whose workers never bound it, and the whole path failed
        // with `CUDA_ERROR_INVALID_CONTEXT` — an error that reads like a
        // hardware fault and is really a thread that was never introduced to
        // the device. Binding here makes the assumption true for any caller
        // instead of documenting it and hoping.
        if let Device::Cuda(dev) = &self.device {
            dev.bind_to_thread()?;
        }
        // Reuse the cached index if the same segment set is rescanned and its
        // turns still hold the runs it recorded (the common case, whatever other
        // conversations upload meanwhile) — this leases it, which holds its
        // turns as a set and takes no lock. Otherwise rebuild (which pins each
        // turn) and cache it. Each turn's run id is taken with its addresses
        // under the residency lock, so a turn moved after that point carries a
        // new id and the next reuse rejects the entry.
        let t_index = Instant::now();
        let gen_before = self.residency_gen();
        let fp = plan.key;
        let weighed = &plan.weighed;
        let (idx, lease) = match self.reuse_index(fp, weighed) {
            Ok(lease) => (lease.index(), Some(lease)),
            Err(miss) => {
                tracing::debug!(
                    target: "candle_conversation::provenance::gallery_arena",
                    fingerprint = fp,
                    "scan index rebuilt: {}",
                    miss.reason()
                );
                let t_build = Instant::now();
                scan_split::add(|s| match miss {
                    IndexMiss::Absent => s.index_absent += 1,
                    IndexMiss::Stale => s.index_stale += 1,
                    IndexMiss::Narrower { .. } => s.index_narrower += 1,
                });
                let build_for = match miss {
                    IndexMiss::Narrower { covering } => covering,
                    IndexMiss::Absent | IndexMiss::Stale => weighed.clone(),
                };
                let built = Arc::new(build(&build_for)?);
                let build_us = t_build.elapsed().as_micros() as u64;
                scan_split::add(|s| {
                    s.index_built += 1;
                    s.index_build_us += build_us;
                });
                // What a scan waits for when its working set is not all
                // resident: the uploads of the turns it finds evicted and the
                // index over them, per byte of the set — the stall a turn's
                // residency spares (`eviction`).
                self.note_upload(
                    super::super::pages_for(built.n_tokens) as u64
                        * self.page_bytes_for(built.groups),
                    t_index.elapsed(),
                );
                self.store_index(fp, built.clone());
                (built, None)
            }
        };
        let reused = lease.is_some();
        let index_us = fingerprint_us + t_index.elapsed().as_micros() as u64;
        profile::record(
            if reused {
                "arena:index_reused"
            } else {
                "arena:index_built"
            },
            t_index.elapsed(),
        );
        // Residency mutations (uploads, evictions, moves) on the whole device
        // while this index was found or built — this scan's and any concurrent
        // conversation's. With `reused` it separates "my working set churned"
        // from "someone else's did".
        let mutations = self.residency_gen().saturating_sub(gen_before);
        scan_split::add(|s| {
            s.scans += 1;
            s.segments += plan.weighed.len() as u64;
            s.index_us += index_us;
            s.fingerprint_us += fingerprint_us;
            s.windows += plan.windows as u64;
            s.pinned += idx.pinned_sids.len() as u64;
        });
        let t_launch = Instant::now();
        let mut fresh_before = 0;
        scan_split::add(|s| fresh_before = s.fresh_tokens);
        let result = launch(&idx, weighed);
        if !reused {
            scan_split::add(|s| {
                s.built_fresh_tokens += s.fresh_tokens.saturating_sub(fresh_before);
            });
        }
        let launch_us = t_launch.elapsed().as_micros() as u64;
        profile::record("arena:launch", t_launch.elapsed());
        // Release the scan's hold whether or not the launch succeeded — the pages
        // are no longer being read once the launch has synchronized (or failed).
        // A lease lets go as it drops; a build's pins go in one pass.
        if !reused {
            self.unpin_all(&idx.pinned_sids);
        }
        drop(lease);
        // Index (reuse or rebuild, which pins pages resident) versus launch
        // (which synchronizes and tallies on the host). They have unrelated
        // costs, so a slow scan is attributed to one or the other.
        tracing::trace!(
            target: "candle_conversation::provenance::gallery_arena",
            segments = plan.weighed.len(),
            turns = idx.pinned_sids.len(),
            tokens = idx.n_tokens,
            reused,
            mutations,
            held_mib = self.page_bytes() >> 20,
            index_us,
            launch_us,
            "arena scan"
        );
        result
    }
}
