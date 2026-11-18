//! Each kind's share of a dialogue slot's room: the slot fill's caps
//! (`docs/hit_driven_projection.md` §7.2 step 4).
//!
//! The shares sum to the room, so no kind crowds another out and the slot never
//! passes the trained window. A share is a cap, never a reservation: a kind
//! with little to place leaves its room to the next round by score.

/// How far the shares' sum may stray from one: the slack of five decimal
/// fractions written in YAML.
const SUM_TOLERANCE: f64 = 1e-6;

/// The share of the room each kind of round may take, each in `[0, 1]`,
/// together exactly the room.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillShares {
    /// File readings' rounds.
    pub readings: f64,
    /// Folder summaries' rounds.
    pub folders: f64,
    /// The workers' guides' rounds.
    pub guides: f64,
    /// Journal notes' rounds.
    pub journals: f64,
    /// The dialogue's own rounds and its inherited lineage, its floors
    /// included.
    pub dialogue: f64,
}

/// Each kind's cap in tokens for one room ([`FillShares::caps`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShareCaps {
    pub readings: usize,
    pub folders: usize,
    pub guides: usize,
    pub journals: usize,
    pub dialogue: usize,
}

impl FillShares {
    /// The shares, or why they cannot be: each must be a finite fraction in
    /// `[0, 1]` and together they must make the whole room.
    pub fn new(
        readings: f64,
        folders: f64,
        guides: f64,
        journals: f64,
        dialogue: f64,
    ) -> Result<Self, String> {
        let shares = Self {
            readings,
            folders,
            guides,
            journals,
            dialogue,
        };
        for (name, share) in shares.named() {
            if !share.is_finite() || !(0.0..=1.0).contains(&share) {
                return Err(format!("shares.{name} must be within [0, 1], got {share}"));
            }
        }
        let sum: f64 = shares.named().iter().map(|(_, s)| s).sum();
        if (sum - 1.0).abs() > SUM_TOLERANCE {
            return Err(format!(
                "shares must sum to 1, the whole room, got {sum} \
                 (readings {readings}, folders {folders}, guides {guides}, \
                 journals {journals}, dialogue {dialogue})"
            ));
        }
        Ok(shares)
    }

    fn named(&self) -> [(&'static str, f64); 5] {
        [
            ("readings", self.readings),
            ("folders", self.folders),
            ("guides", self.guides),
            ("journals", self.journals),
            ("dialogue", self.dialogue),
        ]
    }

    /// Each kind's cap in a room of `room` tokens, to the nearest token: a
    /// share written as a decimal is not exact in binary, and rounding down
    /// would take a token from a cap whose product lands just below a whole.
    pub fn caps(&self, room: usize) -> ShareCaps {
        let of = |share: f64| (room as f64 * share).round() as usize;
        ShareCaps {
            readings: of(self.readings),
            folders: of(self.folders),
            guides: of(self.guides),
            journals: of(self.journals),
            dialogue: of(self.dialogue),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped() -> FillShares {
        FillShares::new(0.225, 0.02, 0.05, 0.105, 0.60).unwrap()
    }

    /// The shipped shares at a 233,000-token room, the design's own column.
    #[test]
    fn the_caps_are_the_shares_of_the_room() {
        assert_eq!(
            shipped().caps(233_000),
            ShareCaps {
                readings: 52_425,
                folders: 4_660,
                guides: 11_650,
                journals: 24_465,
                dialogue: 139_800,
            }
        );
    }

    #[test]
    fn an_empty_room_caps_everything_at_nothing() {
        let c = shipped().caps(0);
        assert_eq!(
            (c.readings, c.folders, c.guides, c.journals, c.dialogue),
            (0, 0, 0, 0, 0)
        );
    }

    #[test]
    fn shares_that_do_not_make_the_room_are_refused() {
        let short = FillShares::new(0.2, 0.02, 0.05, 0.105, 0.60).unwrap_err();
        assert!(short.contains("sum to 1"), "{short}");
        let over = FillShares::new(0.3, 0.02, 0.05, 0.105, 0.60).unwrap_err();
        assert!(over.contains("sum to 1"), "{over}");
    }

    #[test]
    fn a_share_outside_zero_to_one_is_refused() {
        let e = FillShares::new(-0.1, 0.12, 0.05, 0.33, 0.60).unwrap_err();
        assert!(e.contains("shares.readings"), "{e}");
        let e = FillShares::new(f64::NAN, 0.02, 0.05, 0.105, 0.60).unwrap_err();
        assert!(e.contains("shares.readings"), "{e}");
    }
}
