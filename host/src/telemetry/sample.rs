//! One moment, measured.
//!
//! A [`Sample`] is what the daemon could see at a single instant: the card, the
//! host, and — when there is one — the inference engine. Samples accumulate in
//! [`super::ring::Ring`], which is what gives the performance page a history
//! that survives a reload.
//!
//! Every field is `Option`, and that is load-bearing rather than defensive.
//! `None` means *nobody measured this*; `Some(0.0)` means *measured, and it was
//! zero*. A page that cannot tell those apart shows an idle engine and an absent
//! one identically, which is the single most misleading thing a performance
//! dashboard can do.

use std::time::Instant;

use super::device::{Host, Vram};

/// What an inference engine reports about itself, per sample.
///
/// Nothing fills this in yet. The fields exist because the console already
/// renders them and because this is the shape the engine gets written against —
/// discovering the contract later, by reading a page, is how the two end up
/// disagreeing about what a number means.
#[derive(Debug, Clone, Default)]
pub struct Engine {
    pub decode_tps: Option<f64>,
    pub prefill_tps: Option<f64>,
    /// Characters sharing one decode. The direct check on the claim that a
    /// popular character is the cheapest to run: near 1 means batching is not
    /// happening.
    pub mean_npcs_per_decode: Option<f64>,
    pub max_batch: Option<f64>,
    pub npcs_active: Option<f64>,
    pub ticks_per_sec: Option<f64>,
    /// Events waiting on a character. p99 against p50 separates one character
    /// falling behind from the whole population doing so.
    pub inbox_depth_p50: Option<f64>,
    pub inbox_depth_p99: Option<f64>,
    pub image_queue_depth: Option<f64>,
    /// Not a series — a label, and only the current one is meaningful.
    pub image_queue_state: Option<String>,
    /// The engine's share of card memory, by purpose. The driver can say how
    /// much of the card is in use; only the engine knows what for.
    pub weights_mib: Option<f64>,
    pub kv_mib: Option<f64>,
    pub image_mib: Option<f64>,
}

/// The card, the host and the engine, at one instant.
#[derive(Debug, Clone)]
pub struct Sample {
    /// When this was taken. `Instant`, not a wall clock: the series is served
    /// as ages relative to the moment of the request, so a clock adjustment
    /// mid-window cannot reorder or stretch the history.
    pub at: Instant,
    pub vram: Vram,
    pub host: Host,
    /// Absent when no engine has reported by this point in the window. A window
    /// that spans an engine starting up therefore has `None` at the front and
    /// `Some` at the back, which is exactly what the page needs to draw the
    /// series from the point measurement began rather than from zero.
    pub engine: Option<Engine>,
}

impl Sample {
    /// Host memory in use — what `total - free` means, computed here so no
    /// caller has to remember which of the two `sysinfo` reports.
    pub fn host_used_mib(&self) -> Option<f64> {
        match (self.host.total_mib, self.host.free_mib) {
            (Some(t), Some(f)) => Some(t.saturating_sub(f) as f64),
            _ => None,
        }
    }

    fn engine(&self) -> Option<&Engine> {
        self.engine.as_ref()
    }

    /// Read one engine field, absent if there was no engine at all.
    pub fn eng(&self, f: impl Fn(&Engine) -> Option<f64>) -> Option<f64> {
        self.engine().and_then(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(total: Option<u64>, free: Option<u64>) -> Host {
        Host {
            total_mib: total,
            free_mib: free,
            rss_mib: None,
        }
    }

    fn sample(h: Host, e: Option<Engine>) -> Sample {
        Sample {
            at: Instant::now(),
            vram: Vram::default(),
            host: h,
            engine: e,
        }
    }

    #[test]
    fn host_used_is_total_minus_free_and_absent_if_either_is() {
        assert_eq!(
            sample(host(Some(64_000), Some(20_000)), None).host_used_mib(),
            Some(44_000.0)
        );
        assert_eq!(sample(host(Some(64_000), None), None).host_used_mib(), None);
        assert_eq!(sample(host(None, Some(20_000)), None).host_used_mib(), None);
    }

    /// A free reading above total would underflow a `u64` subtraction into an
    /// enormous positive number — a chart spike out of nowhere. Saturating is
    /// what keeps a driver reporting something odd from becoming a fake event.
    #[test]
    fn an_impossible_reading_clamps_rather_than_wrapping() {
        assert_eq!(
            sample(host(Some(10), Some(99)), None).host_used_mib(),
            Some(0.0)
        );
    }

    /// The distinction the whole module exists for.
    #[test]
    fn no_engine_and_an_engine_reporting_zero_are_different_answers() {
        let absent = sample(host(None, None), None);
        assert_eq!(absent.eng(|e| e.decode_tps), None);

        let idle = sample(
            host(None, None),
            Some(Engine {
                decode_tps: Some(0.0),
                ..Default::default()
            }),
        );
        assert_eq!(idle.eng(|e| e.decode_tps), Some(0.0));

        // Present engine, field it does not report: still absent, not zero.
        let partial = sample(host(None, None), Some(Engine::default()));
        assert_eq!(partial.eng(|e| e.decode_tps), None);
    }
}
