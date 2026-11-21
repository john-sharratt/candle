//! The device-side forward of a routed MoE layer, which enqueues its work and
//! never waits on the host for the result.
//!
//! The forward enqueues each layer, hands its routing to the pipeline thread and
//! the stager, and returns. The GPU waits only inside an expert GEMM on a cold
//! expert the stager is still reading. A worker that is given up a cold expert
//! raises a fault word, and the forward fails with the row, the expert and the
//! wait named once it has synchronised. The word is cleared last, so a message
//! queued by the faulted forward cannot install a slot the given-up item never
//! wrote. A CPU pool that fails repeatedly is taken out of the split until a
//! restart, and a forward enqueued after the pipeline or the stager has died
//! fails rather than waiting on an expert that will never arrive.

use super::*;

impl ExpertCache {
    /// The routed experts of one MoE layer, on the device — see `dispatch`.
    ///
    /// `acts` is the layer's q8a128 activation `[n_tokens, hidden]`; `weights`
    /// and `indices` are the router's `[n_tokens, k]` output (f32, u32), still
    /// on the device. Returns the routed sum `[n_tokens, hidden]` at
    /// `out_dtype`. The tokens in `decode` are decode rows, which the residency
    /// scoring weights apart from prompt rows.
    ///
    /// Never waits on the host: the layer is enqueued, its routing is handed to
    /// the pipeline thread and the stager, and the call returns. The GPU waits
    /// only in an expert GEMM's worker blocks, on a cold expert the stager is
    /// still reading.
    ///
    /// `look_ahead`, when given, is the next rows' routers applied to this
    /// layer's input: their votes predict those rows' experts for read-ahead
    /// and staging (`votes`). Pass it only while [`Self::reads_ahead`].
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_routed<'w>(
        &self,
        acts: Q8a128Operand<'w>,
        weights: &LiveTensor<'_>,
        indices: &LiveTensor<'_>,
        row: usize,
        decode: &DecodeRows,
        out_dtype: DType,
        wave: Option<&'w WaveGeneration>,
        look_ahead: &[LookAhead<'_>],
    ) -> Result<LiveTensor<'w>> {
        self.dispatch.forward(
            &self.tx,
            &self.stager,
            acts,
            weights,
            indices,
            row,
            decode,
            out_dtype,
            wave,
            look_ahead,
        )
    }

    /// Whether the cache reads ahead at the next decode invocation: while it
    /// copies experts often enough to repay the look-ahead routers, their votes
    /// and bucketize's read-ahead walk (`read_ahead_gate`). A caller builds its
    /// look-ahead routers only when this is true.
    #[cfg(feature = "cuda")]
    pub fn reads_ahead(&self) -> bool {
        self.dispatch.read_ahead.pays()
    }

    /// Fail the segment just swept if its expert launches gave a cold expert up
    /// ([`Self::take_fault`]), first waiting for the device to finish them —
    /// the fault word is complete only then. A cache whose every expert is in
    /// VRAM or pinned host memory never leaves a worker waiting on a cold one,
    /// so it can give nothing up, and it neither waits nor reads: a wait per
    /// segment drains the device before the host can queue what follows, which
    /// on a single decode stream is the whole of the overlap.
    #[cfg(feature = "cuda")]
    pub fn take_segment_fault(&self, device: &Device) -> Result<()> {
        if !self.can_fault {
            return Ok(());
        }
        device.synchronize()?;
        self.take_fault()
    }

    /// Fail the forward that just synchronised if any of its expert launches
    /// gave a cold expert up (`fault`): the error names the row, the expert
    /// and the wait. Call it only once every launch of the forward is complete.
    ///
    /// The word is cleared last. The pipeline thread serves its channel in
    /// order and usually lags the forward thread, so messages of the faulted
    /// forward may still be queued; with the word still set, none of them can
    /// land a demand promotion, and the drop sent behind them removes every one
    /// still pending. Cleared first, a queued landing would read the word clear
    /// and install a slot the given-up item never wrote.
    ///
    /// A pool that fails [`CPU_FAULT_LIMIT`] forwards in a row is taken out of
    /// the split until a restart (`docs/cpu_expert_compute.md` §5.13): its
    /// measured costs are withdrawn, so every later split is all-GPU.
    #[cfg(feature = "cuda")]
    pub fn take_fault(&self) -> Result<()> {
        let Some(fault) = self.dispatch.fault.peek() else {
            self.cpu_faults.store(0, Ordering::Relaxed);
            return Ok(());
        };
        if fault.is_cpu() {
            let n = self.cpu_faults.fetch_add(1, Ordering::Relaxed) + 1;
            if n == CPU_FAULT_LIMIT {
                if let Ok(mut p) = cpu_params().lock() {
                    p.measured = None;
                }
                tracing::error!(
                    forwards = n,
                    "cpu experts: the pool failed {n} forwards in a row — the split stops using \
                     it until a restart"
                );
            }
        } else {
            self.cpu_faults.store(0, Ordering::Relaxed);
        }
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        if self
            .tx
            .send(PipelineMessage::DropDemandPromotions { response_tx })
            .is_ok()
        {
            let _ = response_rx.recv();
        }
        self.dispatch.fault.clear();
        if fault.is_cpu() {
            // What the pool was doing when the device gave up on it.
            let words = self
                .dispatch
                .cpu_words()
                .map_or(String::new(), |(bell, done)| {
                    format!("; doorbell {bell}, done {done}")
                });
            let words = match self.dispatch.cpu_export_words() {
                Some((ticket, blocks, slot)) => format!(
                    "{words}; export counted {blocks} blocks of ticket {ticket}, work slot names {slot}"
                ),
                None => words,
            };
            return Err(candle::Error::Msg(format!(
                "{}; pool: {}{words}",
                fault.error(),
                super::super::cpu_pool::service::state_line()
            )));
        }
        Err(fault.error())
    }

    /// Whether the pipeline thread or the stager has exited. Its guard raised
    /// the abort word as it went, so any expert layer enqueued since gives its
    /// cold experts up and the forward fails ([`Self::take_fault`]).
    #[cfg(feature = "cuda")]
    pub fn pipeline_dead(&self) -> bool {
        self.pipeline_dead.load(Ordering::Acquire) || self.stager_dead.load(Ordering::Acquire)
    }

    /// Send to the pipeline thread; `false` if it is gone (or, in a build
    /// without `cuda`, never existed).
    pub(super) fn send(&self, msg: PipelineMessage) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.tx.send(msg).is_ok()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = msg;
            false
        }
    }

    /// Wait until the pipeline thread has processed every message sent before
    /// this one — every routed layer included — so whatever it writes reads
    /// complete. Returns at once if the thread is gone.
    pub(super) fn settle(&self) {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        if self.send(PipelineMessage::Settle { response_tx }) {
            let _ = response_rx.recv();
        }
    }
}
