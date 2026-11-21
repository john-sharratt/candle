//! Arms and reads the buffers that carry each sequence's last hidden states out
//! of a speculative wave, for the MTP draft head.
//!
//! The buffers are sized before the forward opens, because a wave's storage is
//! claimed up front and the arena refuses allocations made inside the sweep.
//! They are allocated in the wave's activation dtype, so they match the final
//! norm that writes them and the head that reads them, with no conversion at
//! either end. Disarming clears the active set, so an ordinary forward writes no
//! hiddens, but it keeps the row counts, which the accept reads after the forward
//! to find the last accepted row.

use super::*;

impl HybridBatched {
    /// Sequences the current forward must hand back post-final-norm hiddens
    /// for. Empty on every wave but a speculative one.
    pub fn hidden_capture_seqs(&self) -> Result<Vec<usize>> {
        Ok(self
            .hidden_seqs
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: hidden_seqs lock poisoned".into()))?
            .clone())
    }

    /// Name them for the next forward, and size the buffers it will write.
    ///
    /// **Sized here, before the forward opens**: a wave's storage is claimed up
    /// front and the arena refuses a device allocation from inside the sweep.
    ///
    /// `act_dtype` is the wave's activation dtype, and the buffers are
    /// allocated in it rather than in F32. Everything on both sides of these
    /// rows already speaks that type — the trunk's `final_norm` output that
    /// fills them, and the head's input assembly that reads them back as a
    /// seed — so an F32 buffer bought nothing but a conversion at each end,
    /// one launch per sequence per wave in each direction.
    pub fn arm_hidden_capture(&self, seqs: &[(usize, usize)], act_dtype: DType) -> Result<()> {
        let hidden = self.model.cfg.hidden_size;
        {
            let mut buf = self
                .verify_hidden
                .lock()
                .map_err(|_| candle::Error::Msg("qwen35: verify_hidden lock poisoned".into()))?;
            for &(seq, rows) in seqs {
                // Width AND type: a buffer carried over from a session at a
                // different activation dtype is the wrong shape for this wave
                // however many rows it has.
                let fits = buf
                    .get(&seq)
                    .is_some_and(|t| t.dtype() == act_dtype && t.dim(0).is_ok_and(|n| n >= rows));
                if !fits {
                    buf.insert(
                        seq,
                        Tensor::zeros((rows, hidden), act_dtype, &self.model.device)?,
                    );
                }
            }
        }
        {
            let mut rows = self
                .capture_rows
                .lock()
                .map_err(|_| candle::Error::Msg("qwen35: capture_rows lock poisoned".into()))?;
            for &(seq, n) in seqs {
                rows.insert(seq, n);
            }
        }
        *self
            .hidden_seqs
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: hidden_seqs lock poisoned".into()))? =
            seqs.iter().map(|&(s, _)| s).collect();
        Ok(())
    }

    /// Clear the ACTIVE capture set — the buffers stay, and so does the record
    /// of how many rows each of them holds.
    ///
    /// The two are cleared at different times on purpose. The active set gates
    /// the sweep's capture and must be empty the moment the forward ends, or an
    /// ordinary prefill would write hiddens into buffers sized for someone
    /// else's block. The row counts are read by the accept, which runs *after*
    /// the forward, so clearing them here would take away the one number that
    /// says how much of the buffer the wave actually filled.
    pub fn disarm_hidden_capture(&self) {
        if let Ok(mut g) = self.hidden_seqs.lock() {
            g.clear();
        }
    }

    /// How many rows the last armed wave captured for `seq`.
    ///
    /// `None` when no wave has captured for it. Survives
    /// [`Self::disarm_hidden_capture`] — see there for why.
    pub fn captured_rows(&self, seq: usize) -> Result<Option<usize>> {
        Ok(self
            .capture_rows
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: capture_rows lock poisoned".into()))?
            .get(&seq)
            .copied())
    }

    /// The buffer the sweep writes `seq`'s hiddens into.
    pub fn hidden_buffer(&self, seq: usize) -> Result<Option<Tensor>> {
        Ok(self
            .verify_hidden
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: verify_hidden lock poisoned".into()))?
            .get(&seq)
            .cloned())
    }

    /// The draft walk's two carried-hidden buffers, `[n, hidden]` each in
    /// `dtype`.
    ///
    /// A step reads the hidden the step before it wrote and writes its own
    /// beside it, so the walk alternates between two buffers: the seeds are
    /// laid into the first, and each step writes whichever one the step before
    /// did not. The step's other transients are on its forward phase, which
    /// the next step carves over — the carry is what has to outlive it. Held
    /// here and grown by doubling, so a walk allocates nothing for its carry.
    pub fn draft_carry(&self, n: usize, dtype: DType) -> Result<(Tensor, Tensor)> {
        let hidden = self.model.cfg.hidden_size;
        let mut pair = self
            .draft_carry
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: draft carry lock poisoned".into()))?;
        let held = match pair.as_ref() {
            Some((a, _)) if a.dtype() == dtype => a.dim(0)?,
            _ => 0,
        };
        if held < n {
            // Fully written before it is read: the seeds go into the first, and
            // each step writes the buffer it then hands on (invariant 6).
            let dims = ((held * 2).max(n), hidden);
            *pair = Some((
                Tensor::empty(dims, dtype, &self.model.device)?,
                Tensor::empty(dims, dtype, &self.model.device)?,
            ));
        }
        let (a, b) = pair.as_ref().expect("sized above");
        Ok((a.narrow(0, 0, n)?, b.narrow(0, 0, n)?))
    }

    /// A sequence's seed hidden — the trunk's post-`final_norm` output at its
    /// last accepted position. `None` before its first wave.
    pub fn draft_seed(&self, seq: usize) -> Result<Option<Tensor>> {
        Ok(self
            .seed
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: seed lock poisoned".into()))?
            .get(&seq)
            .cloned())
    }
}
