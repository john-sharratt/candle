//! The one path between members (1 GbE through the gateway), as a FIFO: a
//! transfer starts once every transfer before it has finished, and takes its
//! bytes over the link's rate.

/// Bytes a token of KV takes on the wire: twelve full-attention layers.
pub const KV_BYTES_PER_TOKEN: f64 = 12_288.0;

/// Bytes of one recurrent snapshot as exported: 36 DeltaNet layers × 48 value
/// heads × 128 × 128 at f32 (`ExportedLayerState::state`). The card's store
/// reserves more — slot stride and the conv tail — but this is what a record
/// carries.
pub const SNAPSHOT_BYTES: f64 = 36.0 * 48.0 * 128.0 * 128.0 * 4.0;

#[derive(Debug, Clone)]
pub struct Link {
    /// Bytes a second.
    pub rate: f64,
    free_at: f64,
    /// Bytes carried, for the report.
    pub carried: f64,
}

impl Link {
    /// 1 GbE, as the members have.
    pub fn gigabit() -> Link {
        Link::new(110e6)
    }

    pub fn new(rate: f64) -> Link {
        Link {
            rate,
            free_at: 0.0,
            carried: 0.0,
        }
    }

    /// Send `bytes` at `now`: when they have arrived.
    pub fn send(&mut self, now: f64, bytes: f64) -> f64 {
        let start = self.free_at.max(now);
        self.free_at = start + bytes / self.rate;
        self.carried += bytes;
        self.free_at
    }

    /// A turn's state as a transfer moves it: its snapshot and the KV of
    /// `tokens` new tokens.
    pub fn state_bytes(tokens: usize) -> f64 {
        SNAPSHOT_BYTES + KV_BYTES_PER_TOKEN * tokens as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Transfers queue behind each other; an idle link starts one at once.
    #[test]
    fn transfers_queue_in_order() {
        let mut l = Link::new(100.0);
        assert_eq!(l.send(0.0, 100.0), 1.0);
        assert_eq!(l.send(0.5, 50.0), 1.5);
        assert_eq!(l.send(10.0, 100.0), 11.0);
        assert_eq!(l.carried, 250.0);
    }

    /// A turn's state on gigabit takes about 1.25 s for a 2K-token prompt,
    /// nearly all of it the 113 MB snapshot.
    #[test]
    fn a_turns_state_takes_over_a_second() {
        assert_eq!(SNAPSHOT_BYTES, 113_246_208.0);
        let mut l = Link::gigabit();
        let t = l.send(0.0, Link::state_bytes(2_000));
        assert!((t - 1.2529).abs() < 1e-3, "{t}");
    }
}
