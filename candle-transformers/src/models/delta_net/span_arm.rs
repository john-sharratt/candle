//! Spans of a sequence armed for capture or replay
//! (`docs/recurrent_reply_calibration.md` §6.3, §7.2).
//!
//! An arm names one span of a sequence by its absolute position and length.
//! The wave that runs exactly that span asks for a [`SpanHook`] per DeltaNet
//! layer: a capture copies the span's scan inputs out, a replay runs a
//! record's rows through the span's entering state and writes the result as
//! the span's advanced state. A layer counts as served once its wave commits,
//! so a wave that rolls back serves it again, and a span swept a window of
//! layers at a time keeps its arm until every layer has run it.

use std::collections::HashMap;
use std::sync::Arc;

use candle::{Device, Result};

use super::span_rows::{ReplyRows, SpanRows};
use super::timeline::TIMELINE_ROWS;
use super::types::DeltaNetDims;

/// What one DeltaNet layer does with its span beyond the ordinary scan.
#[derive(Debug, Clone, Default)]
pub enum SpanHook {
    #[default]
    None,
    /// Copy the span's scan inputs into these rows.
    Capture(SpanRows),
    /// Take the span's advanced state from these rows instead of its own.
    Replay(SpanRows),
}

impl SpanHook {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// What a span is armed for.
#[derive(Debug, Clone)]
pub enum ArmKind {
    Capture,
    Replay(Arc<ReplyRows>),
}

struct Arm {
    at: usize,
    len: usize,
    capture: bool,
    /// Per DeltaNet layer, by trunk index: the rows the hook carries.
    rows: HashMap<u32, SpanRows>,
    /// Layers a committed wave ran the span through.
    served: Vec<u32>,
    /// Layers the open wave ran it through.
    ran: Vec<u32>,
}

impl Arm {
    fn complete(&self) -> bool {
        self.served.len() == self.rows.len()
    }
}

/// Every armed span, by sequence.
#[derive(Default)]
pub struct SpanArms {
    arms: HashMap<usize, Arm>,
}

impl SpanArms {
    /// Arm `seq`'s span of `len` rows at absolute position `at`, over the
    /// DeltaNet layers `layers` (trunk indices). Replaces any arm the sequence
    /// held.
    ///
    /// Only a long span can be armed: a short one writes a timeline, which the
    /// fold applies later and a replay has no state to stand in for.
    pub fn arm(
        &mut self,
        seq: usize,
        (at, len): (usize, usize),
        kind: ArmKind,
        layers: &[u32],
        dims: &DeltaNetDims,
        dev: &Device,
    ) -> Result<()> {
        let (conv_dim, n_v_heads) = (dims.conv_dim(), dims.n_v_heads);
        if len <= TIMELINE_ROWS {
            candle::bail!(
                "span arm: a span of {len} rows writes a timeline; only one longer than \
                 {TIMELINE_ROWS} can be captured or replayed"
            );
        }
        let (capture, rows) = match kind {
            ArmKind::Capture => (
                true,
                layers
                    .iter()
                    .map(|&l| Ok((l, SpanRows::alloc(len, conv_dim, n_v_heads, dev)?)))
                    .collect::<Result<HashMap<_, _>>>()?,
            ),
            ArmKind::Replay(rows) => {
                if rows.layers != layers {
                    candle::bail!(
                        "span arm: the rows are of layers {:?}, the model's DeltaNet layers are {:?}",
                        rows.layers,
                        layers
                    );
                }
                if rows.tokens <= TIMELINE_ROWS {
                    candle::bail!(
                        "span arm: rows of {} tokens are too short to replay",
                        rows.tokens
                    );
                }
                if rows.conv_dim != conv_dim || rows.n_v_heads != n_v_heads {
                    candle::bail!(
                        "span arm: rows of conv_dim {} and {} V heads against a model of {conv_dim} \
                         and {n_v_heads}",
                        rows.conv_dim,
                        rows.n_v_heads
                    );
                }
                (false, rows.to_device(dev)?.into_iter().collect())
            }
        };
        self.arms.insert(
            seq,
            Arm {
                at,
                len,
                capture,
                rows,
                served: Vec::new(),
                ran: Vec::new(),
            },
        );
        Ok(())
    }

    /// The hook for `seq`'s span of `len` rows at `at` through `layer`: the arm's,
    /// when the span is the armed one and the layer has not served it yet.
    pub fn hook(&mut self, seq: usize, layer: u32, at: usize, len: usize) -> SpanHook {
        let Some(arm) = self.arms.get_mut(&seq) else {
            return SpanHook::None;
        };
        if arm.at != at || arm.len != len {
            // A long span of an armed sequence that is not the armed one, while
            // the arm still waits for layers: the cut and the arm disagree, and
            // the arm would never be served. A capture every layer has served
            // stays armed only until it is taken, and says nothing.
            if len > TIMELINE_ROWS && !arm.complete() {
                tracing::warn!(
                    target: "candle_transformers::delta_net::span_arm",
                    seq,
                    layer,
                    at,
                    len,
                    armed_at = arm.at,
                    armed_len = arm.len,
                    "a long span of an armed sequence is not the armed span"
                );
            }
            return SpanHook::None;
        }
        if arm.served.contains(&layer) {
            return SpanHook::None;
        }
        let Some(rows) = arm.rows.get(&layer) else {
            return SpanHook::None;
        };
        if arm.ran.is_empty() && arm.served.is_empty() {
            tracing::info!(
                target: "candle_transformers::delta_net::span_arm",
                seq,
                at,
                len,
                capture = arm.capture,
                "armed span reached the mixer"
            );
        }
        if !arm.ran.contains(&layer) {
            arm.ran.push(layer);
        }
        if arm.capture {
            SpanHook::Capture(rows.clone())
        } else {
            SpanHook::Replay(rows.clone())
        }
    }

    /// `seq`'s wave committed: the layers it ran have served the span. A
    /// replay every layer has served is done with.
    pub fn commit(&mut self, seq: usize) {
        let Some(arm) = self.arms.get_mut(&seq) else {
            return;
        };
        let ran = std::mem::take(&mut arm.ran);
        arm.served.extend(ran);
        if !arm.capture && arm.complete() {
            self.arms.remove(&seq);
        }
    }

    /// `seq`'s wave never happened: what it ran serves nothing.
    pub fn rollback(&mut self, seq: usize) {
        if let Some(arm) = self.arms.get_mut(&seq) {
            arm.ran.clear();
        }
    }

    /// The rows a capture took, once every layer has served it, in layer order.
    pub fn take_capture(&mut self, seq: usize) -> Result<Option<ReplyRows>> {
        match self.arms.get(&seq) {
            Some(arm) if arm.capture && arm.complete() => {}
            _ => return Ok(None),
        }
        let arm = self.arms.remove(&seq).expect("checked above");
        let mut layers: Vec<(u32, SpanRows)> = arm.rows.into_iter().collect();
        layers.sort_by_key(|(l, _)| *l);
        ReplyRows::from_device(&layers).map(Some)
    }

    /// Drop whatever `seq` had armed — the sequence is gone.
    pub fn forget(&mut self, seq: usize) {
        self.arms.remove(&seq);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    const LEN: usize = TIMELINE_ROWS + 1;

    /// The smallest geometry: d = 1, one K head, one V head — conv_dim 3.
    const DIMS: DeltaNetDims = DeltaNetDims {
        head_dim: 1,
        n_k_heads: 1,
        n_v_heads: 1,
        conv_kernel: 4,
    };

    fn replay_rows(layers: &[u32], conv_dim: usize) -> ReplyRows {
        ReplyRows {
            layers: layers.to_vec(),
            tokens: LEN,
            conv_dim,
            n_v_heads: 1,
            data: layers
                .iter()
                .map(|_| vec![bf16::from_f32(0.5); LEN * (conv_dim + 2)])
                .collect(),
        }
    }

    fn armed(kind: ArmKind) -> SpanArms {
        let mut arms = SpanArms::default();
        arms.arm(7, (100, LEN), kind, &[0, 2], &DIMS, &Device::Cpu)
            .unwrap();
        arms
    }

    fn fill(t: &candle::Tensor, v: f32) {
        let src = (candle::Tensor::ones(t.dims(), candle::DType::F32, &Device::Cpu).unwrap()
            * v as f64)
            .unwrap();
        t.slice_set(&src, 0, 0).unwrap();
    }

    /// Only the armed span, at its position and length, gets the hook — and a
    /// layer that served it gets it no more.
    #[test]
    fn the_hook_is_the_armed_span_s_alone() {
        let mut arms = armed(ArmKind::Replay(Arc::new(replay_rows(&[0, 2], 3))));
        assert!(arms.hook(7, 0, 99, LEN).is_none());
        assert!(arms.hook(7, 0, 100, LEN + 1).is_none());
        assert!(arms.hook(8, 0, 100, LEN).is_none());
        assert!(
            arms.hook(7, 1, 100, LEN).is_none(),
            "layer 1 is not a DeltaNet layer"
        );
        assert!(matches!(arms.hook(7, 0, 100, LEN), SpanHook::Replay(_)));
        arms.commit(7);
        assert!(arms.hook(7, 0, 100, LEN).is_none());
        assert!(matches!(arms.hook(7, 2, 100, LEN), SpanHook::Replay(_)));
        arms.commit(7);
        assert!(
            arms.hook(7, 2, 100, LEN).is_none(),
            "a served replay is gone"
        );
    }

    /// A wave that rolls back serves nothing: its layers run the span again.
    #[test]
    fn a_rolled_back_wave_serves_nothing() {
        let mut arms = armed(ArmKind::Replay(Arc::new(replay_rows(&[0, 2], 3))));
        assert!(!arms.hook(7, 0, 100, LEN).is_none());
        arms.rollback(7);
        arms.commit(7);
        assert!(!arms.hook(7, 0, 100, LEN).is_none());
    }

    /// A capture is handed back once every layer has run it, and only then.
    #[test]
    fn a_capture_is_taken_once_every_layer_served_it() {
        let mut arms = armed(ArmKind::Capture);
        let SpanHook::Capture(rows) = arms.hook(7, 2, 100, LEN) else {
            panic!("a capture hook");
        };
        for t in [&rows.qkv, &rows.alpha, &rows.beta] {
            fill(t, 1.0);
        }
        arms.commit(7);
        assert_eq!(
            arms.take_capture(7).unwrap(),
            None,
            "layer 0 has not run it"
        );
        let SpanHook::Capture(rows) = arms.hook(7, 0, 100, LEN) else {
            panic!("a capture hook");
        };
        for t in [&rows.qkv, &rows.alpha, &rows.beta] {
            fill(t, 0.0);
        }
        arms.commit(7);
        let taken = arms.take_capture(7).unwrap().expect("complete");
        assert_eq!(taken.layers, vec![0, 2]);
        assert_eq!(
            taken.data[1][0],
            bf16::from_f32(1.0),
            "layer 2's qkv as written"
        );
        assert_eq!(arms.take_capture(7).unwrap(), None, "taken once");
    }

    #[test]
    fn a_short_span_or_foreign_rows_are_refused() {
        let mut arms = SpanArms::default();
        let dev = Device::Cpu;
        assert!(arms
            .arm(1, (0, TIMELINE_ROWS), ArmKind::Capture, &[0], &DIMS, &dev)
            .is_err());
        let replay = |layers: &[u32], conv| ArmKind::Replay(Arc::new(replay_rows(layers, conv)));
        assert!(arms
            .arm(1, (0, LEN), replay(&[0, 1], 3), &[0, 2], &DIMS, &dev)
            .is_err());
        assert!(arms
            .arm(1, (0, LEN), replay(&[0, 2], 4), &[0, 2], &DIMS, &dev)
            .is_err());
        assert!(arms
            .arm(1, (0, LEN), replay(&[0, 2], 3), &[0, 2], &DIMS, &dev)
            .is_ok());
    }
}
