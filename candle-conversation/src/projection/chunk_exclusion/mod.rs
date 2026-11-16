//! Learning a belief group's anomalous-chunk exclusion incrementally
//! (`docs/incremental_chunk_exclusion.md`).
//!
//! A sample of the group's calibration questions is scanned raw against the
//! group's content alone and against its calibrations alone, nothing cut, so
//! the exclusion never learns from a gallery it already shaped. Each sample
//! token's scan statistics are kept as an exact mergeable summary, and every
//! gallery stream is folded in once — when it appears — against the sample:
//!
//! | Module | Concern |
//! |---|---|
//! | `gallery` | A group's galleries and questions as one substrate version holds them |
//! | `scan_jobs` | Scanning in jobs small in both directions, and merging what they hand back |
//! | `fold` | The learner's state, and one fold of what changed into it |
//! | `cut` | The cut a state yields: votes, gate, attribution, outliers |
//! | `learner` | The learner a group keeps, loaded from and written to its records |

mod cut;
mod fold;
mod gallery;
#[cfg(test)]
mod host_arena;
mod learner;
mod scan_jobs;
#[cfg(test)]
mod tests;

pub use cut::CompetitionCut;
pub use fold::FoldCounts;
pub use learner::{ExclusionLearner, FoldReport};
