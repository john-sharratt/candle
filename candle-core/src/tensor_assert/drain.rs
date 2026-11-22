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

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

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
            seq = self
                .seq
                .map(|s| s.to_string())
                .unwrap_or_else(|| "-".into()),
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
    super::fence(dev, "drain")?;
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
        static CLEAN: AtomicU64 = AtomicU64::new(0);
        static LAST_BEAT_MS: AtomicU64 = AtomicU64::new(0);
        let n = CLEAN.fetch_add(1, Ordering::Relaxed) + 1;
        if beat_due(now_ms(), &LAST_BEAT_MS) {
            if let Some(line) = heartbeat(n, &all) {
                tracing::info!(target: "candle_core::tensor_assert", "{line}");
            }
        }
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

/// Drop a failed forward's findings: read them, say in one line what they
/// were, fire no callback, and start a fresh epoch.
///
/// **A forward that failed is rolled back whole, so nothing its asserts saw
/// survives it.** A cold expert given up mid-layer leaves that layer's rows
/// unwritten by design and the wave is discarded and run again; drained by the
/// next wave instead, those rows were reported as that wave's corruption and
/// armed captures on it — on WhiteWorld, every non-finite report of a run was a
/// give-up the engine had already handled.
///
/// **Except what the rollback leaves.** A site `survives` names describes state
/// the failed forward hands on — what its rollback restored — so a bad one is
/// not discarded with the forward but reported, each on its own line.
pub fn discard(device: &Device, survives: impl Fn(&str) -> bool) -> Result<()> {
    let (kept, bad): (Vec<Finding>, Vec<Finding>) = drain(device)?
        .into_iter()
        .filter(Finding::is_bad)
        .partition(|f| survives(&f.name));
    if let Some(line) = discarded_line(&bad) {
        tracing::warn!(target: "candle_core::tensor_assert", "{line}");
    }
    for f in &kept {
        tracing::error!(
            target: "candle_core::tensor_assert",
            "tensor_assert: a failed forward left non-finite state behind its rollback: {f}"
        );
    }
    super::epoch(device)
}

/// The line a discarded forward's findings are logged as, `None` when they
/// were all finite.
fn discarded_line(bad: &[Finding]) -> Option<String> {
    let first = bad.first()?;
    Some(format!(
        "tensor_assert: a failed forward's {} non-finite site(s) were discarded with it, \
         first {first}",
        bad.len()
    ))
}

/// The time between two heartbeats.
///
/// **By the clock, not by the count.** Every 1000th clean drain was an hour of
/// WhiteWorld's waves, and the diagnostic run it was meant to vouch for was
/// replaced by a rollout fifteen minutes in, having said nothing.
const HEARTBEAT_MS: u64 = 5 * 60 * 1000;

/// Whether a heartbeat is due at `now_ms`, claiming it if so: the first drain
/// beats at once, and then one in each [`HEARTBEAT_MS`], whichever thread
/// drains.
fn beat_due(now_ms: u64, last: &AtomicU64) -> bool {
    let prev = last.load(Ordering::Relaxed);
    if prev != 0 && now_ms.saturating_sub(prev) < HEARTBEAT_MS {
        return false;
    }
    last.compare_exchange(prev, now_ms.max(1), Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
}

/// Milliseconds since the Unix epoch.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// The heartbeat of a drain that found nothing: how many clean drains so far,
/// how many sites this one read and how many elements they had examined, and
/// the site whose finite values ran widest — `None` when no site was read.
///
/// **A silent instrument has to say it is listening.** A diagnostic motherd on
/// WhiteWorld refused a NaN logit while every probe stayed quiet, and nothing
/// told "every site read finite" apart from "no site was ever read". The range
/// is the other half: a widest site creeping towards f16's ceiling is an
/// overflow story before any value goes non-finite.
fn heartbeat(clean: u64, all: &[Finding]) -> Option<String> {
    let elems: u64 = all.iter().map(|f| u64::from(f.elems)).sum();
    let reach = |f: &Finding| match (f.min, f.max) {
        (Some(lo), Some(hi)) => lo.abs().max(hi.abs()),
        _ => 0.0,
    };
    let widest = all.iter().max_by(|a, b| reach(a).total_cmp(&reach(b)))?;
    Some(format!(
        "tensor_assert: {clean} clean drains — this one read {} sites over {elems} elements; \
         widest {}",
        all.len(),
        widest
    ))
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
    use std::sync::atomic::AtomicU64;

    use super::{beat_due, discarded_line, heartbeat, to_finding, Finding, HEARTBEAT_MS};

    /// **A discarded forward is named by its first bad site**, and a forward
    /// whose sites were all finite says nothing.
    #[test]
    fn a_discarded_forward_names_its_first_bad_site() {
        let bad = [
            Finding {
                name: "moe.routed_out.L3".into(),
                seq: Some(4),
                nan: 2560,
                inf: 0,
                min: Some(-0.5),
                max: Some(0.25),
                elems: 5120,
            },
            Finding {
                name: "q4e.post_moe.res.L3".into(),
                seq: Some(9),
                nan: 10240,
                inf: 0,
                min: None,
                max: None,
                elems: 20480,
            },
        ];
        assert_eq!(
            discarded_line(&bad).as_deref(),
            Some(
                "tensor_assert: a failed forward's 2 non-finite site(s) were discarded with it, \
                 first #4 moe.routed_out.L3: nan=2560 inf=0 of 5120 finite=[-5e-1, 2.5e-1]"
            )
        );
        assert_eq!(discarded_line(&[]), None);
    }

    /// **The first drain beats at once, then one per interval.**
    #[test]
    fn a_heartbeat_is_due_once_per_interval() {
        let last = AtomicU64::new(0);
        assert!(beat_due(1_000, &last), "the first drain beats");
        assert!(
            !beat_due(1_000 + HEARTBEAT_MS - 1, &last),
            "not again inside the interval"
        );
        assert!(
            beat_due(1_000 + HEARTBEAT_MS, &last),
            "once the interval has passed"
        );
        assert!(!beat_due(1_000 + HEARTBEAT_MS + 1, &last));
    }

    fn site(name: &str, min: f32, max: f32, elems: u32) -> Finding {
        Finding {
            name: name.into(),
            seq: None,
            nan: 0,
            inf: 0,
            min: Some(min),
            max: Some(max),
            elems,
        }
    }

    /// **A clean drain's heartbeat names the sites read and the widest one**,
    /// by its largest magnitude either side of zero — and says nothing when no
    /// site was read at all.
    #[test]
    fn a_heartbeat_names_the_sites_and_the_widest() {
        let all = [
            site("q4e.mix.y.L3", -2.0, 3.0, 100),
            site("q4e.post_moe.res.L19", -40.0, 7.0, 200),
        ];
        assert_eq!(
            heartbeat(2000, &all).as_deref(),
            Some(
                "tensor_assert: 2000 clean drains — this one read 2 sites over 300 elements; \
                 widest #- q4e.post_moe.res.L19: nan=0 inf=0 of 200 finite=[-4e1, 7e0]"
            )
        );
        assert_eq!(heartbeat(1000, &[]), None);
    }
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
        let mut v = [
            Finding {
                name: "clean".into(),
                seq: None,
                nan: 0,
                inf: 0,
                min: None,
                max: None,
                elems: 1,
            },
            Finding {
                name: "second".into(),
                seq: Some(7),
                nan: 1,
                inf: 0,
                min: None,
                max: None,
                elems: 1,
            },
            Finding {
                name: "first".into(),
                seq: Some(2),
                nan: 0,
                inf: 3,
                min: None,
                max: None,
                elems: 1,
            },
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
