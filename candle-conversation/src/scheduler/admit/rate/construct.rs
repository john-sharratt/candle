//! Builds the wave throughput planner, seeding its copy-rate estimate from the
//! link the device measured.
//!
//! The first waves are composed for a slower bus than the card has, at a fraction
//! of the measured link, so the model learns upward from evidence rather than
//! down from optimism. The link rate is the probe's pinned-host-to-device figure,
//! measured before the model loads, which keeps it this machine's own idle bus and
//! not a datasheet's. Dampening is set so one forward moves an estimate only a
//! sixth of the way, because this machine's throughput varies about two times
//! between runs. The constructors also take the link rate and the dials directly
//! for tests.

use super::*;

impl WaveRate {
    /// The estimator's seed as a fraction of the measured link: conservative,
    /// so the first waves are composed for a slower bus than the card has and
    /// the model learns upward from evidence rather than down from optimism.
    pub const SEED_FRACTION: f64 = 0.8;

    /// Default dampening. This machine's throughput has a ~2× run-to-run band,
    /// so one forward moves an estimate by a sixth of the way at most.
    pub const DEFAULT_ALPHA: f64 = 0.15;

    /// Default minimum relative gain an admission must buy.
    pub const DEFAULT_MIN_GAIN: f64 = 0.01;

    /// Build the planner against a device, seeding the estimate at
    /// [`Self::SEED_FRACTION`] of the device's host→device link rate.
    ///
    /// The rate is `link_probe`'s: pinned host memory onto the device's stream,
    /// the path the expert cache streams through, measured before the model
    /// loaded (the builder asks for it then) and cached — so the figure is this
    /// machine's own idle bus, not a datasheet's and not one contended by the
    /// expert cache's startup staging.
    pub fn measure(
        device: &Device,
        geometry: ExpertGeometry,
        model: RateModel,
        decode: DecodeModel,
    ) -> candle::Result<Self> {
        let Device::Cuda(d) = device else {
            candle::bail!("wave rate: the link rate needs a CUDA device");
        };
        let link = link_probe::host_to_device_bytes_per_s(d)?;
        Ok(Self::with_link_rate(link, geometry, model, decode))
    }

    /// Build the planner from a known link rate, in bytes a second. The
    /// constructor every test uses, and the one a caller without a device uses.
    pub fn with_link_rate(
        link_bytes_per_s: f64,
        geometry: ExpertGeometry,
        model: RateModel,
        decode: DecodeModel,
    ) -> Self {
        assert!(
            link_bytes_per_s.is_finite() && link_bytes_per_s > 0.0,
            "a link rate must be a positive finite number of bytes a second"
        );
        assert!(
            decode.initial_tok_per_s > 0.0 && geometry.moe_layers > 0,
            "a decode step rate and a layer count are needed to seed the layer time"
        );
        assert!(
            decode.hit_exponent.is_finite() && decode.hit_exponent > 0.0,
            "a hit exponent is a positive finite power of the resident fraction"
        );
        Self {
            geometry,
            model,
            decode,
            link_bytes_per_s,
            bw_bytes_per_s: link_bytes_per_s * Self::SEED_FRACTION,
            layer_secs: 1.0 / (decode.initial_tok_per_s * geometry.moe_layers as f64),
            decode_fit: DecodeFit::default(),
            decode_fit_at: None,
            decode_context: 0.0,
            alpha: Self::DEFAULT_ALPHA,
            min_gain: Self::DEFAULT_MIN_GAIN,
            trades: TradeWindow::default(),
            samples: 0,
            decode_samples: 0,
            hit_samples: 0,
            fit: CostFit::default(),
            context_per_row: 0.0,
            resident_now: 0,
            prefill_floor: 0,
            decode_floor: 0,
            max_tokens: 0,
            max_decodes: 0,
            tokens: 0,
            decodes: 0,
            routed_per_layer: 0,
            admitted_any: false,
            full: false,
        }
    }

    /// Change the dampening. `alpha` in `(0, 1]`; 1 believes every observation
    /// outright.
    ///
    /// A test dial: production runs [`Self::DEFAULT_ALPHA`], sized to this
    /// machine's ~2x run-to-run band.
    #[cfg(test)]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
        self.alpha = alpha;
        self
    }

    /// Change the minimum relative gain an admission must buy. Zero admits
    /// anything that does not make the rate worse.
    ///
    /// A test dial, like [`Self::with_alpha`]: production runs
    /// [`Self::DEFAULT_MIN_GAIN`].
    #[cfg(test)]
    pub fn with_min_gain(mut self, min_gain: f64) -> Self {
        assert!(min_gain >= 0.0, "a minimum gain cannot be negative");
        self.min_gain = min_gain;
        self
    }

    /// The link rate measured at load, bytes a second.
    pub fn link_bytes_per_s(&self) -> f64 {
        self.link_bytes_per_s
    }

    /// The effective copy rate the model currently believes, bytes a second.
    pub fn effective_bytes_per_s(&self) -> f64 {
        self.bw_bytes_per_s
    }
}
