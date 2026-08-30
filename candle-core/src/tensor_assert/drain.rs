//! Reading the slot table back, and ordering what it says.
//!
//! This is the only place the instrument touches the host, and it is a device→
//! host copy of the whole table — so it belongs at a synchronisation the caller
//! already performs, not at one it introduces. Called there, its marginal cost
//! is one small `memcpy` on a stream that was about to be waited on anyway.
//!
//! The ordering is the diagnostic. Every assert site folds into its slot
//! asynchronously, so host call order says nothing about which tensor went bad
//! first; the kernel's `seq` ticket does. A drained report sorted by `seq` reads
//! as: this is where it started, and everything below is downstream of it.

use crate::cuda_backend::DeviceId;
use crate::{Device, Result};

use super::names;
use super::slots::{self, AssertSlot};

/// One assert site's accumulated statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    /// The name the site was asserted under.
    pub name: String,
    /// Order stamp of the first bad observation; `None` if it never went bad.
    pub seq: Option<u32>,
    pub nan: u32,
    pub inf: u32,
    /// Smallest finite value seen, or `None` if nothing finite was seen.
    pub min: Option<f32>,
    /// Largest finite value seen, or `None` as for `min`.
    pub max: Option<f32>,
    /// Elements examined across every assert that folded into this slot.
    pub elems: u32,
}

impl Finding {
    pub fn is_bad(&self) -> bool {
        self.nan > 0 || self.inf > 0
    }
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rng = match (self.min, self.max) {
            (Some(lo), Some(hi)) => format!("[{lo:e}, {hi:e}]"),
            _ => "[no finite values]".to_string(),
        };
        write!(
            f,
            "#{seq} {name}: nan={nan} inf={inf} of {elems} finite={rng}",
            seq = self.seq.map(|s| s.to_string()).unwrap_or_else(|| "-".into()),
            name = self.name,
            nan = self.nan,
            inf = self.inf,
            elems = self.elems,
        )
    }
}

fn to_finding(idx: usize, slot: &AssertSlot) -> Option<Finding> {
    let name = names::name_of(idx)?;
    Some(Finding {
        name,
        seq: (slot.seq != 0 && slot.seq != u32::MAX).then_some(slot.seq),
        nan: slot.nan,
        inf: slot.inf,
        min: slot.min(),
        max: slot.max(),
        elems: slot.elems,
    })
}

/// Read every claimed slot back, newest fault first.
///
/// Bad slots come first, ordered by when they actually went bad; clean slots
/// follow in registration order. The device is synchronised before the copy,
/// because a slot mid-claim reads as the `u32::MAX` sentinel — which is exactly
/// the ambiguity a synchronisation removes.
pub fn drain(device: &Device) -> Result<Vec<Finding>> {
    let Device::Cuda(dev) = device else {
        return Ok(Vec::new());
    };
    let stream = dev.cuda_stream();
    stream
        .synchronize()
        .map_err(|e| crate::Error::Msg(format!("tensor_assert: drain fence: {e}")))?;
    let raw = slots::with_slots(dev, |sl| sl.read(dev))?;
    let claimed = names::claimed();

    let mut out: Vec<Finding> = raw
        .iter()
        .take(claimed)
        .enumerate()
        .filter_map(|(i, s)| to_finding(i, s))
        .collect();
    out.sort_by_key(|f| match (f.is_bad(), f.seq) {
        (true, Some(s)) => (0u8, s),
        (true, None) => (1, 0),
        (false, _) => (2, 0),
    });
    Ok(out)
}

/// Drain and log every site that went bad, first-bad first. Returns the bad
/// ones, so a caller can act on them as well as read them.
pub fn report(device: &Device) -> Result<Vec<Finding>> {
    let all = drain(device)?;
    if !all.iter().any(Finding::is_bad) {
        return Ok(Vec::new());
    }
    let bad: Vec<Finding> = all.iter().filter(|f| f.is_bad()).cloned().collect();
    // Notify before logging, so a callback that arms a capture is armed by the
    // time the next wave reaches the site — see `super::callback`.
    if super::callback::any_registered() {
        for f in &bad {
            super::callback::fire(f);
        }
    }
    tracing::error!(
        target: "candle_core::tensor_assert",
        bad_sites = bad.len(),
        total_sites = all.len(),
        origin = %bad[0].name,
        "tensor_assert: non-finite values — the full site list follows, first bad first"
    );
    // Every site, not only the bad ones. What a clean site's RANGE was is half
    // the diagnosis: a NaN downstream of an operand whose magnitudes were
    // already implausible is an overflow story, and one downstream of operands
    // that all look ordinary is a corruption story. Printing only the failures
    // throws away the half that tells them apart.
    for f in &all {
        let mark = if f.is_bad() { "BAD " } else { "    " };
        tracing::error!(target: "candle_core::tensor_assert", "  {mark}{f}");
    }
    Ok(bad)
}

/// The device a drain would read, for callers that only hold a [`DeviceId`].
pub fn is_registered(dev: DeviceId, device: &Device) -> bool {
    matches!(device, Device::Cuda(d) if d.id() == dev)
}

/// One site's accumulated statistics by name, for a caller that knows exactly
/// which assert it wants to read rather than the whole report.
pub fn find(device: &Device, name: &str) -> Result<Option<Finding>> {
    Ok(drain(device)?.into_iter().find(|f| f.name == name))
}

#[cfg(test)]
mod tests {
    use super::{to_finding, Finding};
    use crate::tensor_assert::names::slot_for;
    use crate::tensor_assert::slots::AssertSlot;

    fn slot(nan: u32, inf: u32, seq: u32) -> AssertSlot {
        AssertSlot {
            nan,
            inf,
            min_key: u32::MAX,
            max_key: 0,
            seq,
            elems: 10,
            pad0: 0,
            pad1: 0,
        }
    }

    #[test]
    fn a_mid_claim_sentinel_is_not_reported_as_an_order() {
        let idx = slot_for("tensor_assert::drain::sentinel").expect("slot");
        let f = to_finding(idx, &slot(1, 0, u32::MAX)).expect("finding");
        assert!(f.is_bad());
        assert_eq!(f.seq, None, "the claim sentinel must not read as a ticket");
    }

    #[test]
    fn bad_sites_sort_before_clean_ones_and_by_ticket() {
        let mut v = vec![
            Finding { name: "clean".into(), seq: None, nan: 0, inf: 0, min: None, max: None, elems: 1 },
            Finding { name: "second".into(), seq: Some(7), nan: 1, inf: 0, min: None, max: None, elems: 1 },
            Finding { name: "first".into(), seq: Some(2), nan: 0, inf: 3, min: None, max: None, elems: 1 },
        ];
        v.sort_by_key(|f| match (f.is_bad(), f.seq) {
            (true, Some(s)) => (0u8, s),
            (true, None) => (1, 0),
            (false, _) => (2, 0),
        });
        let names: Vec<&str> = v.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["first", "second", "clean"]);
    }
}
