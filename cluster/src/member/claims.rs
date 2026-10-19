//! A member's claims (§5.4): the table every gateway relays — this member's
//! own claims among them — and, for its own, when each last served a request.
//!
//! Before serving a sticky key the member checks the table. A key it holds is
//! served. A key another live member holds is not: the request is answered
//! `409` naming the holder. Any other key — unclaimed, or held by a member
//! that has left every cluster — is claimed before serving, at an epoch above
//! every epoch this member has seen for the key, whatever its clock says.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::frame::Epoch;
use crate::gateway::Claims;

/// What serving a key needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Check {
    /// This member holds it.
    Mine,
    /// This member just took it, at this epoch: announce it.
    Took(Epoch),
    /// Another live member holds it.
    Held(u64),
}

#[derive(Debug)]
pub struct MemberClaims {
    node: u64,
    table: Claims,
    /// The highest `ms` seen in any epoch for each key.
    seen: HashMap<Vec<u8>, u64>,
    /// This member's own claims, and when each last served.
    own: HashMap<Vec<u8>, (Epoch, Instant)>,
    /// Of those, the ones a lease stands behind (§9.6): never let go for
    /// idleness, only when the lease is.
    held: HashSet<Vec<u8>>,
    /// The work using each of this member's claims now — a request in flight,
    /// a turn still running ([`Self::begin_use`]).
    uses: HashMap<Vec<u8>, u32>,
    /// Whether a claim lasts only while something uses it
    /// (`docs/cooperative_admission.md` §2.2): let go the moment its last use
    /// ends, so the key's next piece of work is placed afresh, rather than kept
    /// until it has served nothing for the idle timeout.
    follow_use: bool,
    /// Keys this member let go that no other member has claimed since: whatever
    /// the app keeps of one stays current until another member takes it
    /// ([`Self::heard`]), and a request coming back here first finds it intact.
    let_go: HashSet<Vec<u8>>,
}

impl MemberClaims {
    pub fn new(node: u64) -> Self {
        Self {
            node,
            table: Claims::new(),
            seen: HashMap::new(),
            own: HashMap::new(),
            held: HashSet::new(),
            uses: HashMap::new(),
            follow_use: false,
            let_go: HashSet::new(),
        }
    }

    /// `key` is no longer this member's, released by it: its own claim and the
    /// table's entry go, and the key is remembered as let go.
    fn release(&mut self, key: &[u8]) -> Option<Epoch> {
        let (epoch, _) = self.own.remove(key)?;
        self.table.unclaim(key, epoch);
        self.let_go.insert(key.to_vec());
        Some(epoch)
    }

    /// Hold each claim only while something uses it — see the field.
    pub fn follow_use(&mut self) {
        self.follow_use = true;
    }

    /// Whether claims last only while used.
    pub fn follows_use(&self) -> bool {
        self.follow_use
    }

    /// One more piece of work uses this member's claim on `key`.
    pub fn begin_use(&mut self, key: &[u8]) {
        *self.uses.entry(key.to_vec()).or_default() += 1;
    }

    /// A use [`Self::begin_use`] counted has ended. The claim's epoch to
    /// announce as `UNCLAIM` when that was its last and claims follow use;
    /// `None` while it is still used, when claims are kept until idle, or when
    /// this member no longer holds it.
    pub fn end_use(&mut self, key: &[u8]) -> Option<Epoch> {
        let n = self.uses.get_mut(key)?;
        *n = n.saturating_sub(1);
        if *n > 0 {
            return None;
        }
        self.uses.remove(key);
        if !self.follow_use || self.held.contains(key) {
            return None;
        }
        self.release(key)
    }

    /// Claim `key` for a lease this member now holds, whoever the table names:
    /// Paxos has already decided it is this member's. At an epoch above every
    /// one seen for the key. `Some` with the epoch to announce when it is
    /// newly taken; `None` when it was held already.
    pub fn hold(&mut self, key: &[u8], now_ms: u64, now: Instant) -> Option<Epoch> {
        if self.held.contains(key) && self.own.contains_key(key) {
            return None;
        }
        let floor = self.seen.get(key).map_or(0, |ms| ms + 1);
        let epoch = Epoch {
            ms: now_ms.max(floor),
            node: self.node,
        };
        self.saw(key, epoch);
        self.table.claim(key, epoch);
        self.own.insert(key.to_vec(), (epoch, now));
        self.held.insert(key.to_vec());
        self.let_go.remove(key);
        Some(epoch)
    }

    /// Let go of `key`'s claim, its lease gone; the epoch to announce as
    /// `UNCLAIM`, if this member held it.
    pub fn let_go(&mut self, key: &[u8]) -> Option<Epoch> {
        self.held.remove(key);
        self.release(key)
    }

    fn saw(&mut self, key: &[u8], epoch: Epoch) {
        let ms = self.seen.entry(key.to_vec()).or_default();
        *ms = (*ms).max(epoch.ms);
    }

    /// A claim announced by any member — from `ADMIT`'s table or relayed.
    /// Returns whether another member took a key this member held or last let
    /// go: its own claim goes, and whatever it keeps of the key may now fall
    /// behind what the taker writes.
    pub fn heard(&mut self, key: &[u8], epoch: Epoch) -> bool {
        self.saw(key, epoch);
        let took = self.table.claim(key, epoch);
        if !took || epoch.node == self.node {
            return false;
        }
        let held = self.own.remove(key).is_some();
        if held {
            self.held.remove(key);
        }
        let let_go = self.let_go.remove(key);
        held || let_go
    }

    /// A release announced by any member.
    pub fn heard_release(&mut self, key: &[u8], epoch: Epoch) {
        self.saw(key, epoch);
        self.table.unclaim(key, epoch);
    }

    /// Before serving `key` at wall clock `now_ms`; `live` says whether a node
    /// is a member of any cluster this member is joined to.
    pub fn serve(
        &mut self,
        key: &[u8],
        now_ms: u64,
        now: Instant,
        live: impl Fn(u64) -> bool,
    ) -> Check {
        match self.table.holder(key) {
            Some(h) if h == self.node => {
                if let Some((_, served)) = self.own.get_mut(key) {
                    *served = now;
                }
                Check::Mine
            }
            Some(h) if live(h) => Check::Held(h),
            _ => {
                let floor = self.seen.get(key).map_or(0, |ms| ms + 1);
                let epoch = Epoch {
                    ms: now_ms.max(floor),
                    node: self.node,
                };
                self.saw(key, epoch);
                self.table.claim(key, epoch);
                self.own.insert(key.to_vec(), (epoch, now));
                self.let_go.remove(key);
                Check::Took(epoch)
            }
        }
    }

    /// This member's claims that served nothing for `idle`, let go here; each
    /// is announced as `UNCLAIM`. A claim a lease stands behind is not idle.
    pub fn idle(&mut self, now: Instant, idle: Duration) -> Vec<(Vec<u8>, Epoch)> {
        let gone: Vec<(Vec<u8>, Epoch)> = self
            .own
            .iter()
            .filter(|(k, _)| !self.held.contains(*k) && !self.uses.contains_key(*k))
            .filter(|(_, (_, served))| now.duration_since(*served) >= idle)
            .map(|(k, (e, _))| (k.clone(), *e))
            .collect();
        for (key, _) in &gone {
            self.release(key);
        }
        gone
    }

    /// Every claim this member holds, announced again on each admission.
    pub fn own(&self) -> Vec<(Vec<u8>, Epoch)> {
        let mut v: Vec<(Vec<u8>, Epoch)> =
            self.own.iter().map(|(k, (e, _))| (k.clone(), *e)).collect();
        v.sort();
        v
    }

    /// The holder of `key`, as this member knows it.
    pub fn holder(&self, key: &[u8]) -> Option<u64> {
        self.table.holder(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ME: u64 = 1;

    fn e(ms: u64, node: u64) -> Epoch {
        Epoch { ms, node }
    }

    /// A new key is taken at the wall clock; held, it is served; another live
    /// member's is refused naming it; a gone member's is taken over.
    #[test]
    fn a_key_is_taken_served_refused_or_taken_over() {
        let t = Instant::now();
        let mut c = MemberClaims::new(ME);
        assert_eq!(c.serve(b"a", 500, t, |_| true), Check::Took(e(500, ME)));
        assert_eq!(c.serve(b"a", 600, t, |_| true), Check::Mine);

        c.heard(b"b", e(900, 2));
        assert_eq!(c.serve(b"b", 500, t, |n| n == 2), Check::Held(2));
        // Node 2 left every cluster: taken over above its epoch, whatever
        // this member's clock says.
        assert_eq!(c.serve(b"b", 500, t, |_| false), Check::Took(e(901, ME)));
        assert_eq!(
            c.own(),
            vec![(b"a".to_vec(), e(500, ME)), (b"b".to_vec(), e(901, ME))]
        );
    }

    /// A higher epoch relayed for a key this member holds takes it away; a
    /// lower one does not.
    #[test]
    fn a_higher_claim_takes_the_key_away() {
        let t = Instant::now();
        let mut c = MemberClaims::new(ME);
        c.serve(b"k", 100, t, |_| true);
        assert!(!c.heard(b"k", e(50, 2)));
        assert_eq!(c.holder(b"k"), Some(ME));
        assert!(c.heard(b"k", e(100, 2)));
        assert_eq!(c.holder(b"k"), Some(2));
        assert!(c.own().is_empty());
    }

    /// A lease's claim is taken whoever the table names, above every epoch
    /// seen; idleness never lets it go, its lease does.
    #[test]
    fn a_leases_claim_is_held_until_let_go() {
        let t = Instant::now();
        let mut c = MemberClaims::new(ME);
        c.heard(b"w1", e(900, 2));
        assert_eq!(c.hold(b"w1", 500, t), Some(e(901, ME)));
        assert_eq!(c.holder(b"w1"), Some(ME));
        assert_eq!(c.hold(b"w1", 700, t), None, "held already");
        let gone = c.idle(t + Duration::from_secs(3600), Duration::from_secs(600));
        assert!(gone.is_empty(), "a lease's claim is never idle");
        assert_eq!(c.let_go(b"w1"), Some(e(901, ME)));
        assert_eq!(c.holder(b"w1"), None);
        assert_eq!(c.let_go(b"w1"), None);
    }

    /// **A claim that follows use is let go when its last use ends**, and not
    /// before: two uses overlap, the first ending keeps it, the second lets it
    /// go. The key is then taken again above the release.
    #[test]
    fn a_claim_following_use_goes_with_its_last_use() {
        let t = Instant::now();
        let mut c = MemberClaims::new(ME);
        c.follow_use();
        assert_eq!(c.serve(b"conv", 100, t, |_| true), Check::Took(e(100, ME)));
        c.begin_use(b"conv");
        c.begin_use(b"conv");
        assert_eq!(c.end_use(b"conv"), None, "one use still running");
        assert_eq!(c.holder(b"conv"), Some(ME));
        assert_eq!(
            c.end_use(b"conv"),
            Some(e(100, ME)),
            "the last use let it go"
        );
        assert_eq!(c.holder(b"conv"), None);
        assert!(c.own().is_empty());
        assert_eq!(c.end_use(b"conv"), None, "nothing left to end");
        assert_eq!(c.serve(b"conv", 50, t, |_| true), Check::Took(e(101, ME)));
    }

    /// **Another member taking a key this member let go is reported** — what
    /// this member keeps of it may now fall behind. One it never held, or took
    /// back itself in between, is not.
    #[test]
    fn a_key_let_go_and_taken_elsewhere_is_reported() {
        let t = Instant::now();
        let mut c = MemberClaims::new(ME);
        c.follow_use();
        c.serve(b"conv", 100, t, |_| true);
        c.begin_use(b"conv");
        c.end_use(b"conv");
        assert!(c.heard(b"conv", e(200, 2)), "let go here, taken by 2");
        assert!(!c.heard(b"conv", e(300, 2)), "reported once");

        assert!(!c.heard(b"never", e(200, 2)), "never this member's");

        c.serve(b"back", 100, t, |_| true);
        c.begin_use(b"back");
        c.end_use(b"back");
        c.serve(b"back", 150, t, |_| true);
        c.begin_use(b"back");
        assert!(
            c.heard(b"back", e(400, 2)),
            "taken back here, then taken away while held"
        );
        assert_eq!(c.holder(b"back"), Some(2));
    }

    /// **Without following use, a claim outlives its uses** and goes only when
    /// idle — and never while a use is running, however long ago it last
    /// served.
    #[test]
    fn a_claim_kept_until_idle_is_not_idle_while_used() {
        let t = Instant::now();
        let mut c = MemberClaims::new(ME);
        c.serve(b"job", 100, t, |_| true);
        c.begin_use(b"job");
        let later = t + Duration::from_secs(3600);
        assert!(
            c.idle(later, Duration::from_secs(600)).is_empty(),
            "a running use is not idleness"
        );
        assert_eq!(c.end_use(b"job"), None, "kept: claims do not follow use");
        assert_eq!(c.holder(b"job"), Some(ME));
        assert_eq!(
            c.idle(later, Duration::from_secs(600)),
            vec![(b"job".to_vec(), e(100, ME))]
        );
    }

    /// A claim a lease stands behind is the lease's to let go, however its uses
    /// end.
    #[test]
    fn a_leases_claim_outlives_its_uses() {
        let t = Instant::now();
        let mut c = MemberClaims::new(ME);
        c.follow_use();
        c.hold(b"w1", 500, t);
        c.begin_use(b"w1");
        assert_eq!(c.end_use(b"w1"), None);
        assert_eq!(c.holder(b"w1"), Some(ME));
    }

    /// A claim idle past the timeout is let go and reported; a served one
    /// stays. A released key is taken again above the release.
    #[test]
    fn idle_claims_are_released() {
        let t = Instant::now();
        let mut c = MemberClaims::new(ME);
        c.serve(b"old", 100, t, |_| true);
        c.serve(b"new", 100, t, |_| true);
        c.serve(b"new", 200, t + Duration::from_secs(300), |_| true);
        let gone = c.idle(t + Duration::from_secs(600), Duration::from_secs(600));
        assert_eq!(gone, vec![(b"old".to_vec(), e(100, ME))]);
        assert_eq!(c.holder(b"old"), None);
        assert_eq!(c.holder(b"new"), Some(ME));
        assert_eq!(c.serve(b"old", 50, t, |_| true), Check::Took(e(101, ME)));
    }
}
