//! What a model load says while it runs, for a loading screen.
//!
//! A routed model's load is several jobs in turn, each long enough to be seen:
//! the dense weights into the device reservation, then the expert cache — its
//! host tier pinned, filled from the pack, and VRAM filled from that. A
//! loader tells its caller which it is in and how far through, in the phase's
//! own [`unit`](LoadPhase::unit).

/// The job a load is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadPhase {
    /// The dense weights: one count per transformer block mounted.
    Weights,
    /// Page-locking the expert cache's host tier. Counts nothing: it is one
    /// driver allocation after another, as large as the machine allows.
    PinWarm,
    /// Reading experts from the pack into the host tier.
    ReadWarm,
    /// Uploading experts into VRAM, from the host tier or the pack.
    FillVram,
}

impl LoadPhase {
    /// The line a loading screen prints.
    pub fn label(self) -> &'static str {
        match self {
            LoadPhase::Weights => "Loading the dense weights",
            LoadPhase::PinWarm => "Pinning host memory for the experts",
            LoadPhase::ReadWarm => "Reading experts into host memory",
            LoadPhase::FillVram => "Filling VRAM with experts",
        }
    }

    /// What the phase's `(done, total)` count; empty where it counts nothing.
    pub fn unit(self) -> &'static str {
        match self {
            LoadPhase::Weights => "blocks",
            LoadPhase::PinWarm => "",
            LoadPhase::ReadWarm | LoadPhase::FillVram => "experts",
        }
    }
}

/// Told the phase, and within it `(done, total)`; `total` 0 where the phase
/// counts nothing.
pub type LoadProgress<'a> = &'a dyn Fn(LoadPhase, usize, usize);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every phase names itself, and only the uncounted one has no unit.
    #[test]
    fn every_phase_names_itself_and_its_unit() {
        for p in [
            LoadPhase::Weights,
            LoadPhase::PinWarm,
            LoadPhase::ReadWarm,
            LoadPhase::FillVram,
        ] {
            assert!(!p.label().is_empty(), "{p:?}");
            assert_eq!(p.unit().is_empty(), p == LoadPhase::PinWarm, "{p:?}");
        }
    }
}
