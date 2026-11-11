//! Each conversation ref's last use as this aggregator has seen it, and when
//! it last told zend's record of it (`docs/private_mode.md` §3.2). Two
//! aggregators serve `/mcp` in turn and share nothing, so what one vouches
//! for is only what it saw: a ref older than that is asked of the record
//! both read.

use std::collections::HashMap;
use std::sync::Mutex;

use mcp_server::refs::{Ref, REF_TOUCH_EVERY_MS};

/// How many refs' marks are kept; past this, the idlest is let go, and a call
/// with it is asked of the record again.
const KEPT: usize = 16_384;

/// One ref's mark: when it was last used here, and when that was last told.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mark {
    used_ms: u64,
    told_ms: u64,
}

/// The refs this process has seen used.
#[derive(Debug, Default)]
pub struct Seen(Mutex<HashMap<String, Mark>>);

impl Seen {
    /// Whether `r` is live at `now_ms` on what this process knows alone:
    /// issued, or used here, within the four hours a ref lives idle.
    pub fn vouches(&self, r: &Ref, now_ms: u64) -> bool {
        let used = self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&r.conversation())
            .map(|m| m.used_ms);
        r.live(used, now_ms)
    }

    /// The record says `r` was last used at `used_ms`.
    pub fn heard(&self, r: &Ref, used_ms: u64) {
        let mut marks = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let mark = marks.entry(r.conversation()).or_insert(Mark {
            used_ms,
            told_ms: used_ms,
        });
        mark.used_ms = mark.used_ms.max(used_ms);
        mark.told_ms = mark.told_ms.max(used_ms);
    }

    /// Note `r` used at `now_ms`; whether its use is now due to be told —
    /// [`REF_TOUCH_EVERY_MS`] since it was last told, or since its issue when
    /// it never was.
    pub fn used(&self, r: &Ref, now_ms: u64) -> bool {
        let mut marks = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let key = r.conversation();
        if marks.len() >= KEPT && !marks.contains_key(&key) {
            let idlest = marks
                .iter()
                .min_by_key(|(_, m)| m.used_ms)
                .map(|(k, _)| k.clone());
            if let Some(idlest) = idlest {
                marks.remove(&idlest);
            }
        }
        let mark = marks.entry(key).or_insert(Mark {
            used_ms: now_ms,
            told_ms: r.issued_ms(),
        });
        mark.used_ms = mark.used_ms.max(now_ms);
        let due = now_ms.saturating_sub(mark.told_ms) >= REF_TOUCH_EVERY_MS;
        if due {
            mark.told_ms = now_ms;
        }
        due
    }

    /// How many refs are marked.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use mcp_server::refs::{Refs, REF_IDLE_MS};

    use super::*;

    const NOW: u64 = 1_794_000_000_000;

    fn r() -> Ref {
        Refs::new([1; 32]).issue("o", false, NOW).1
    }

    /// **A ref is vouched for by its issue, then by its use here**; one this
    /// process never saw used, older than four hours, is not.
    #[test]
    fn a_ref_is_vouched_for_by_its_issue_or_its_use_here() {
        let s = Seen::default();
        let r = r();
        assert!(s.vouches(&r, NOW + REF_IDLE_MS - 1));
        assert!(!s.vouches(&r, NOW + REF_IDLE_MS), "only the record can say now");
        s.heard(&r, NOW + REF_IDLE_MS - 5);
        assert!(s.vouches(&r, NOW + REF_IDLE_MS), "the record said it was used");
        let later = NOW + 3 * REF_IDLE_MS;
        assert!(!s.vouches(&r, later));
        s.used(&r, later);
        assert!(s.vouches(&r, later + REF_IDLE_MS - 1));
    }

    /// **A use is told at most every ten minutes**, the first ten minutes
    /// after the ref's issue.
    #[test]
    fn a_use_is_told_at_most_every_ten_minutes() {
        let s = Seen::default();
        let r = r();
        assert!(!s.used(&r, NOW + 1), "just issued: its issue vouches for it");
        assert!(!s.used(&r, NOW + REF_TOUCH_EVERY_MS - 1));
        assert!(s.used(&r, NOW + REF_TOUCH_EVERY_MS));
        assert!(!s.used(&r, NOW + REF_TOUCH_EVERY_MS + 1));
        assert!(s.used(&r, NOW + 2 * REF_TOUCH_EVERY_MS));
    }

    #[test]
    fn past_the_cap_the_idlest_is_let_go() {
        let s = Seen::default();
        let refs = Refs::new([2; 32]);
        for i in 0..KEPT as u64 + 2 {
            s.used(&refs.issue("o", false, NOW).1, NOW + i);
        }
        assert_eq!(s.len(), KEPT);
    }
}
