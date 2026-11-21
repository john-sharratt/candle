//! Checks that a wave's recurrent state is committed or rolled back as a whole,
//! and that attention layers have no recurrent slot.
//!
//! A long span writes its prefill scratch rather than the state it reads, so the
//! entry state survives a failed wave untouched, and a successful commit installs
//! the output. Asking an attention layer for recurrent state is an error rather
//! than a silent zero, so a wrong layer map cannot pass for an empty state.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::config::{DeltaNetDims, Qwen35Config};

    use super::*;

    /// The 9B's geometry, without weights — enough to exercise the
    /// bookkeeping that does not touch tensors.
    pub(in super::super) fn cfg() -> Qwen35Config {
        Qwen35Config {
            vocab_size: 248_320,
            hidden_size: 4096,
            intermediate_size: 12_288,
            num_layers: 32,
            layer_kinds: Qwen35Config::schedule_from_interval(32, 4),
            num_attention_heads: 16,
            num_kv_heads: 4,
            attn_head_dim: 256,
            rope_dim: 64,
            rope_sections: [11, 11, 10, 0],
            rope_theta: 1e7,
            rms_norm_eps: 1e-6,
            delta_net: DeltaNetDims {
                head_dim: 128,
                n_k_heads: 16,
                n_v_heads: 32,
                conv_kernel: 4,
            },
            moe: None,
            num_mtp_layers: 0,
            max_position_embeddings: 262_144,
        }
    }

    /// A long span writes the prefill scratch, NOT the state it reads, so the
    /// entry state is intact whether the wave commits or fails — and a commit
    /// installs the wave's output.
    #[test]
    fn recurrent_state_rolls_back_to_the_entry_value() -> Result<()> {
        use crate::models::delta_net::SpanWrite;
        let c = cfg();
        let dev = Device::Cpu;
        let mut store = RecurrentStateStore::new(&c.layer_kinds, &c.delta_net, &dev)?;
        store.ensure_scratch()?;
        // Layer 0 is DeltaNet under the 3:1 schedule.
        let entry = store.layer_state(0)?.s.copy()?;

        // The form the mixer uses for a prefill span: read `live`, write the
        // scratch.
        let advance = |store: &mut RecurrentStateStore| -> Result<()> {
            let (live, write) = store.layer_span_mut(0, TIMELINE_ROWS + 1)?;
            let SpanWrite::State(out) = write else {
                candle::bail!("a long span writes the scratch");
            };
            let bump = Tensor::full(3f32, live.s.shape(), &dev)?;
            out.s.slice_set(&live.s.add(&bump)?, 0, 0)?;
            Ok(())
        };

        store.begin_wave()?;
        advance(&mut store)?;
        let during = store.layer_state(0)?.s.copy()?;
        store.rollback_wave()?;
        let after = store.layer_state(0)?.s.copy()?;

        let max = |t: &candle::Tensor| -> Result<f32> {
            t.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
        };
        assert_eq!(
            max(&during.sub(&entry)?)?,
            0.0,
            "a wave must not write the buffer it is reading"
        );
        assert_eq!(
            max(&after.sub(&entry)?)?,
            0.0,
            "rollback must leave the entry state exactly as it was"
        );

        store.begin_wave()?;
        advance(&mut store)?;
        store.commit_wave();
        let committed = store.layer_state(0)?.s.copy()?;
        assert!(
            max(&committed.sub(&entry)?)? > 1.0,
            "commit must install the wave's output"
        );
        Ok(())
    }

    /// An attention layer has no recurrent slot, and asking for one is an
    /// error rather than a silently-zero state.
    #[test]
    fn attention_layers_have_no_recurrent_slot() -> Result<()> {
        let c = cfg();
        let store = RecurrentStateStore::new(&c.layer_kinds, &c.delta_net, &Device::Cpu)?;
        assert_eq!(store.n_recurrent_layers(), 24, "32 layers, 8 attend");
        assert!(store.layer_state(0).is_ok(), "layer 0 is DeltaNet");
        assert!(
            store.layer_state(3).is_err(),
            "layer 3 attends and must not answer with a state"
        );
        Ok(())
    }
}
