//! The bars a sweep is held to, in bytes: when a volume is low, and what is
//! worth reclaiming.

use crate::config::SweepConfig;

const GB: u64 = 1 << 30;

/// [`SweepConfig`] in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bars {
    /// A volume with less free than this, at least, is low.
    pub low_free: u64,
    /// …or less free than this percent of the volume.
    pub low_free_percent: u8,
    /// What a prune or a clean must bring back to be done at all.
    pub worth: u64,
}

impl Bars {
    pub fn of(cfg: SweepConfig) -> Bars {
        Bars {
            low_free: cfg.low_free_gb.saturating_mul(GB),
            low_free_percent: cfg.low_free_percent.min(100),
            worth: cfg.worth_gb.saturating_mul(GB),
        }
    }

    /// The free space below which a volume of `total` bytes is low.
    pub fn floor(&self, total: u64) -> u64 {
        self.low_free
            .max(total / 100 * u64::from(self.low_free_percent))
    }

    /// Whether a volume with `free` of `total` bytes is low.
    pub fn low(&self, free: u64, total: u64) -> bool {
        free < self.floor(total)
    }

    /// Whether bringing back `bytes` is worth doing.
    pub fn worth(&self, bytes: u64) -> bool {
        bytes >= self.worth
    }
}

/// `bytes` as gigabytes, for a line a person reads.
pub fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / GB as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A volume is low under the larger of its two bars**, and only what
    /// brings back the worth bar is worth doing.
    #[test]
    fn a_volume_is_low_under_the_larger_bar() {
        let bars = Bars::of(SweepConfig::default());
        let tb = 1000 * GB;
        assert_eq!(bars.floor(tb), 150 * GB, "150 GB beats 10% of a terabyte");
        assert_eq!(bars.floor(4 * tb), 400 * GB, "10% of 4 TB beats 150 GB");
        assert!(bars.low(149 * GB, tb));
        assert!(!bars.low(150 * GB, tb));
        assert!(bars.worth(10 * GB) && !bars.worth(10 * GB - 1));
        assert_eq!(gb(3 * GB / 2), "1.5 GB");
    }
}
