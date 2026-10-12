//! What a substrate open says while it runs, for a loading screen.
//!
//! An open walks the store into RAM, and a store written before identities is
//! then rewritten whole, once, and walked again (`adoption`). Each counts in
//! its own unit ([`OpenStage::unit`]): a walk counts MiB of log read against
//! the store's size on disk, the rewrite counts records written.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

/// The stage an open is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenStage {
    /// Walking the redo log into RAM.
    Reading,
    /// Rewriting a store written before identities, every record stamped as
    /// this node's own. Counts records written of the live set.
    Stamping,
    /// Walking the rewritten store into RAM.
    Rereading,
}

impl OpenStage {
    /// The line a loading screen prints.
    pub fn label(self) -> &'static str {
        match self {
            OpenStage::Reading => "Reading the substrate",
            OpenStage::Stamping => "Stamping records with this node's identity (once)",
            OpenStage::Rereading => "Reading the rewritten substrate",
        }
    }

    /// What the stage's `(done, total)` count.
    pub fn unit(self) -> &'static str {
        match self {
            OpenStage::Reading | OpenStage::Rereading => "MiB",
            OpenStage::Stamping => "records",
        }
    }
}

/// Told the stage, and within it `(done, total)` in the stage's
/// [`unit`](OpenStage::unit); `total` 0 where there is nothing to count.
pub type OpenProgress<'a> = &'a dyn Fn(OpenStage, usize, usize);

/// Bytes walked, told as whole MiB: [`Self::add`] answers only when a record
/// carries the count past a MiB boundary, so a walk of millions of records
/// reports once per MiB rather than once per record.
#[derive(Debug, Default)]
pub(crate) struct MibMeter {
    bytes: u64,
}

impl MibMeter {
    /// Count a record of `size` bytes; the MiB walked when that crossed into
    /// a new one.
    pub(crate) fn add(&mut self, size: u64) -> Option<usize> {
        let before = self.bytes >> 20;
        self.bytes += size;
        let after = self.bytes >> 20;
        (after > before).then_some(after as usize)
    }
}

/// An [`OpenProgress`] a builder can carry until it opens the substrate.
#[derive(Clone)]
pub struct OpenProgressHook(pub Arc<dyn Fn(OpenStage, usize, usize) + Send + Sync>);

impl OpenProgressHook {
    /// Tell the hook how the open goes.
    pub fn report(&self, stage: OpenStage, done: usize, total: usize) {
        (self.0)(stage, done, total)
    }
}

impl Debug for OpenProgressHook {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("OpenProgressHook")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_stage_names_itself() {
        for s in [
            OpenStage::Reading,
            OpenStage::Stamping,
            OpenStage::Rereading,
        ] {
            assert!(!s.label().is_empty(), "{s:?}");
            assert!(!s.unit().is_empty(), "{s:?}");
        }
    }

    /// The meter speaks only on crossing a MiB boundary, and then says how
    /// many whole MiB are behind it.
    #[test]
    fn the_meter_answers_once_per_mib_crossed() {
        let mut m = MibMeter::default();
        assert_eq!(m.add(600 << 10), None);
        assert_eq!(m.add(600 << 10), Some(1));
        assert_eq!(m.add(100), None);
        assert_eq!(m.add(3 << 20), Some(4));
    }
}
