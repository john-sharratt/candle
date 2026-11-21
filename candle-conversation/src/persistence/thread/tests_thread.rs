//! Tests of the gauge wake rule and of which hot turns the gap still owes a
//! quantize.
//!
//! A falling backlog wakes its waiter and a rising or unchanged one does not.
//! A hot copy that still holds native tokens is one the gap has yet to quantize,
//! and empty native chunks do not count. A refusal caused by a forward is a
//! deferral, while a genuine exhaustion of the device is still reported as one.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use candle::DType;
    use candle_nn::kv_cache::{
        ArenaLocation, ChunkGid, HeadGids, KvFormat, QuantFormat, SealedChunk, KV_ARENA_MID_WAVE,
        KV_DEVICE_OOM_MARKER, N_PALETTE, TIER_STANDS_OVER_FREE,
    };

    /// A one-chunk layer of `tokens` tokens, every band in format `tag`.
    fn layer(tag: u8, tokens: u16) -> SealedSequence {
        SealedSequence {
            chunks: vec![SealedChunk {
                gids: HeadGids::uniform(ChunkGid::detached(0), 1),
                offset: 0,
                token_count: tokens,
                k_pal: Arc::new(Vec::new()),
                v_pal: Arc::new(Vec::new()),
                k_scale: Arc::new(Vec::new()),
                v_scale: Arc::new(Vec::new()),
                k_fmt: Arc::new(vec![tag; N_PALETTE]),
                v_fmt: Arc::new(vec![tag; N_PALETTE]),
                byte_size: 0,
                meta: None,
                q_sign: None,
            }],
            token_count: tokens as usize,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }
    }

    /// **A turn the gap has yet to quantize is one whose hot copy holds a
    /// native chunk with tokens** — on any layer. Quantized chunks and empty
    /// native ones are not the gap's.
    #[test]
    fn a_hot_copy_holds_native_only_where_a_native_chunk_has_tokens() {
        let f16 = KvFormat::Float(DType::F16).to_tag();
        let q8 = KvFormat::Quantized(QuantFormat::Q8_0).to_tag();
        let hot = |layers| HotKv::new(layers);
        assert!(!holds_native(&hot(vec![layer(q8, 32), layer(q8, 16)])));
        assert!(!holds_native(&hot(vec![layer(q8, 32), layer(f16, 0)])));
        assert!(holds_native(&hot(vec![layer(q8, 32), layer(f16, 16)])));
    }

    /// **A falling backlog wakes the scheduler; a rising or unchanged one does
    /// not.** The scheduler blocks while the backlog holds admission shut, and
    /// this is its only cue that the drain moved — but a wake on every stamp
    /// would spin it round for nothing. Every stamping path wakes the same way.
    #[test]
    fn a_falling_backlog_wakes_its_waiter_and_nothing_else_does() {
        let gauge = BacklogGauge::default();
        let wakes = Arc::new(AtomicU64::new(0));
        let counted = wakes.clone();
        let _ = gauge.on_fall.set(Box::new(move || {
            counted.fetch_add(1, Ordering::Relaxed);
        }));
        gauge.stamp_warm(500);
        assert_eq!(wakes.load(Ordering::Relaxed), 0, "a rise is no cue");
        gauge.stamp_warm(500);
        assert_eq!(wakes.load(Ordering::Relaxed), 0, "nor is no change");
        gauge.stamp_warm(200);
        assert_eq!(wakes.load(Ordering::Relaxed), 1, "a fall wakes");
        gauge.stamp(0, 7);
        assert_eq!(wakes.load(Ordering::Relaxed), 2, "the full stamp too");
        assert_eq!(gauge.pinned.load(Ordering::Relaxed), 7);
    }

    /// **Both refusals a forward causes are waits, and nothing else is.** The
    /// region pool's "free ground stands above the tier" is the same wait the
    /// arena window's refusal is; a genuine exhaustion is not, and must still
    /// read as one.
    #[test]
    fn a_forwards_refusals_are_deferrals_and_exhaustion_is_not() {
        let window = candle::Error::Msg(format!("{KV_ARENA_MID_WAVE}: class 4096 B"));
        let tier = candle::Error::Msg(format!(
            "{KV_DEVICE_OOM_MARKER}: no region is claimable for class 1152 B — 166 regions \
             above the tier are unowned but {TIER_STANDS_OVER_FREE}."
        ));
        let full = candle::Error::Msg(format!(
            "{KV_DEVICE_OOM_MARKER}: no region is claimable for class 1152 B — every one of \
             1264 regions is live"
        ));
        assert!(is_wave_deferral(&window));
        assert!(is_wave_deferral(&tier));
        assert!(!is_wave_deferral(&full));
    }
}
