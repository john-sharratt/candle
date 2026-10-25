//! The requester's side of one catch-up (§7.2), as a state machine with no
//! I/O: it takes the answer's `HEADERS` and `DONE`s and the records that
//! arrive, and says which `FETCH`es to send.
//!
//! 1. **Headers.** Each `HEADERS` lists what its answerer holds of one
//!    complete range. A listed record this member does not hold is wanted;
//!    the `oseq`s listed as gone are handed back for the store to record.
//! 2. **Fetch.** Once the headers end in `DONE`, the wanted records are
//!    fetched from the member that listed them, at most [`FETCH_WINDOW`] bytes
//!    outstanding at once, so the requester sets the pace (§7.4).
//! 3. **Done.** A `FETCH` that ends in `DONE` short of a record asks for it
//!    again; one its answerer cannot deliver after [`FETCH_TRIES`] fetches is
//!    left wanted for a later catch-up rather than holding this one open.

use std::collections::{BTreeMap, HashMap, VecDeque};

use applied_set::AppliedSet;
use serde::{Deserialize, Serialize};

use crate::frame::{Fetch, Headers};

/// The most bytes of fetched records outstanding at once.
///
/// A catch-up moves one window a round trip, and a round trip is the
/// answerer's reads, the gateway hop and the spare-capacity gate — about
/// 170 ms on `.5`, so 8 MiB moved ~46 MB/s, under half the LAN, and a member
/// whose peer's npcd wrote faster than that could never close its gap (34 and
/// 40 GB backlogs each way, 2026-10-17). Half the gateway's catch-up cap, so a
/// window in flight never fills the lane on its own.
pub const FETCH_WINDOW: u64 = 64 << 20;

/// How many times one record is fetched before it is left for a later
/// catch-up.
pub const FETCH_TRIES: u8 = 3;

/// Why a catch-up runs (§7.1, §7.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    /// A socket was admitted: the gateway picks the answerer. The socket is
    /// ready once it finishes.
    Admission,
    /// An origin's `high_water` stayed above what this member holds.
    Gap(u64),
    /// A socket has been down for 30 s.
    Periodic,
    /// A restarted member offered more than this member holds.
    Offer(u64),
}

/// What a catch-up has done so far, for the loading page (§7.6).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Progress {
    /// Records the headers listed that this member lacked, and their bytes.
    pub wanted: u64,
    pub wanted_bytes: u64,
    /// Records the headers listed that this member already held.
    pub skipped: u64,
    pub fetched: u64,
    pub fetched_bytes: u64,
    /// Records left for a later catch-up.
    pub left: u64,
    /// The member answering.
    pub answerer: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct Want {
    answerer: u64,
    origin: u64,
    oseq: u64,
    len: u64,
}

#[derive(Debug)]
struct Fetching {
    wants: Vec<Want>,
    bytes: u64,
}

/// One catch-up in progress. See the module documentation.
#[derive(Debug)]
pub struct Requester {
    pub request: u64,
    /// The socket it runs on: its answers come back on it.
    pub socket: usize,
    pub purpose: Purpose,
    node: u64,
    headers_done: bool,
    wanted: VecDeque<Want>,
    fetching: BTreeMap<u64, Fetching>,
    outstanding: u64,
    tries: HashMap<(u64, u64), u8>,
    progress: Progress,
    /// `oseq`s the answer listed as gone.
    gone_learned: u64,
    /// What this member held when it asked: the request's `since`.
    since: AppliedSet,
}

impl Requester {
    /// Catch-up `request`, run by member `node` on `socket`, asking for what
    /// exists after `since`.
    pub fn new(
        request: u64,
        socket: usize,
        purpose: Purpose,
        node: u64,
        since: AppliedSet,
    ) -> Self {
        Self {
            request,
            socket,
            purpose,
            node,
            headers_done: false,
            wanted: VecDeque::new(),
            fetching: BTreeMap::new(),
            outstanding: 0,
            tries: HashMap::new(),
            progress: Progress::default(),
            gone_learned: 0,
            since,
        }
    }

    /// One `HEADERS` of the answer: what this member lacks of it becomes
    /// wanted, and its gone `oseq`s are returned for the store to record.
    /// `held` says what this member holds, or another of its catch-ups
    /// already wants; the records this one wants are returned with the gone.
    pub fn headers(
        &mut self,
        h: Headers,
        held: impl Fn(u64, u64) -> bool,
    ) -> (Vec<u64>, Vec<(u64, u64)>) {
        self.progress.answerer = Some(h.answerer);
        let mut wanted = Vec::new();
        for r in &h.records {
            if held(h.origin, r.oseq) {
                self.progress.skipped += 1;
                continue;
            }
            self.progress.wanted += 1;
            self.progress.wanted_bytes += r.len as u64;
            wanted.push((h.origin, r.oseq));
            self.wanted.push_back(Want {
                answerer: h.answerer,
                origin: h.origin,
                oseq: r.oseq,
                len: r.len as u64,
            });
        }
        self.gone_learned += h.gone.len() as u64;
        (h.gone, wanted)
    }

    /// Every record this catch-up still wants or is fetching.
    pub fn outstanding_wants(&self) -> Vec<(u64, u64)> {
        self.wanted
            .iter()
            .chain(self.fetching.values().flat_map(|f| f.wants.iter()))
            .map(|w| (w.origin, w.oseq))
            .collect()
    }

    /// The answer's `DONE`: the headers are complete, and fetching starts.
    pub fn headers_done(&mut self, next_request: impl FnMut() -> u64) -> Vec<Fetch> {
        self.headers_done = true;
        self.fill(next_request)
    }

    /// Whether `request` is one of this catch-up's fetches.
    pub fn owns_fetch(&self, request: u64) -> bool {
        self.fetching.contains_key(&request)
    }

    /// A fetched record arrived.
    pub fn record(&mut self, len: u64) {
        self.progress.fetched += 1;
        self.progress.fetched_bytes += len;
    }

    /// Fetch `request` ended in `DONE`: a record of it this member still does
    /// not hold is wanted again, or left once tried [`FETCH_TRIES`] times.
    /// Returns the next fetches, and the records this catch-up no longer
    /// wants — held now, or left.
    pub fn fetch_done(
        &mut self,
        request: u64,
        held: impl Fn(u64, u64) -> bool,
        next_request: impl FnMut() -> u64,
    ) -> (Vec<Fetch>, Vec<(u64, u64)>) {
        let mut settled = Vec::new();
        if let Some(f) = self.fetching.remove(&request) {
            self.outstanding -= f.bytes;
            for w in f.wants {
                if held(w.origin, w.oseq) {
                    settled.push((w.origin, w.oseq));
                    continue;
                }
                let tries = self.tries.entry((w.origin, w.oseq)).or_insert(0);
                *tries += 1;
                if *tries >= FETCH_TRIES {
                    self.progress.left += 1;
                    settled.push((w.origin, w.oseq));
                } else {
                    self.wanted.push_back(w);
                }
            }
        }
        (self.fill(next_request), settled)
    }

    /// The fetches to send now: wanted records grouped by answerer and
    /// origin, while the window has room. A record larger than the window is
    /// fetched alone once nothing else is outstanding.
    fn fill(&mut self, mut next_request: impl FnMut() -> u64) -> Vec<Fetch> {
        let mut out = Vec::new();
        while let Some(first) = self.wanted.front().copied() {
            let room = FETCH_WINDOW.saturating_sub(self.outstanding);
            if first.len > room && self.outstanding > 0 {
                break;
            }
            let mut wants = vec![self.wanted.pop_front().expect("front")];
            let mut bytes = first.len;
            while let Some(next) = self.wanted.front() {
                if next.answerer != first.answerer
                    || next.origin != first.origin
                    || bytes + next.len > room
                {
                    break;
                }
                bytes += next.len;
                wants.push(self.wanted.pop_front().expect("front"));
            }
            let request = next_request();
            out.push(Fetch {
                request,
                reply_to: self.node,
                from_node: first.answerer,
                origin: first.origin,
                oseqs: wants.iter().map(|w| w.oseq).collect(),
            });
            self.outstanding += bytes;
            self.fetching.insert(request, Fetching { wants, bytes });
        }
        out
    }

    /// The answer is in and nothing is outstanding.
    pub fn finished(&self) -> bool {
        self.headers_done && self.wanted.is_empty() && self.fetching.is_empty()
    }

    /// A finished admission catch-up whose answer moved this member on — a
    /// record it lacked is now held, or an `oseq` learned gone — asks again
    /// before the socket is ready: an answer is bounded (`answer.rs`), so one
    /// that brought anything may have stopped short. One that brought nothing
    /// was the whole of what its answerer covers.
    ///
    /// Moved on is judged by what this member holds `now` against what it
    /// held when it asked, not by what the answer listed. An answer can list
    /// the same `oseq`s gone every time — records of a type this store never
    /// takes, which the next request's `since` does not cover either — and a
    /// count of them alone asked again for ever: WhiteWorld's admission on
    /// d7f850f79 (2026-10-20) ran 660,000 catch-ups at ~400 a second and never
    /// served. The same answer to the same question is the whole answer.
    pub fn asks_again(&self, now: &AppliedSet) -> bool {
        self.purpose == Purpose::Admission
            && self.finished()
            && (self.progress.wanted > self.progress.left || self.gone_learned > 0)
            && *now != self.since
    }

    pub fn progress(&self) -> &Progress {
        &self.progress
    }

    /// Bytes of fetched records outstanding.
    pub fn outstanding(&self) -> u64 {
        self.outstanding
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::RecordHeader;

    fn header(oseq: u64, len: u32) -> RecordHeader {
        RecordHeader {
            oseq,
            record_type: 0,
            stream_id: 0,
            chunk_index: 0,
            len,
            digest: [0; 16],
        }
    }

    fn listing(answerer: u64, origin: u64, from: u64, to: u64, lens: &[u32]) -> Headers {
        Headers {
            request: 1,
            reply_to: 7,
            answerer,
            origin,
            from,
            to,
            gone: vec![],
            records: lens
                .iter()
                .enumerate()
                .map(|(i, len)| header(from + i as u64, *len))
                .collect(),
        }
    }

    /// What a member holds once an answer has moved it on from nothing.
    fn moved() -> AppliedSet {
        let mut a = AppliedSet::default();
        a.insert(9, 1);
        a
    }

    fn counter() -> impl FnMut() -> u64 {
        let mut n = 100;
        move || {
            n += 1;
            n
        }
    }

    /// Only records not held are wanted; fetches go to the member that listed
    /// them, grouped by origin.
    #[test]
    fn what_is_held_is_skipped_and_the_rest_fetched_from_its_answerer() {
        let mut r = Requester::new(1, 0, Purpose::Admission, 7, AppliedSet::default());
        let (gone, wanted) = r.headers(listing(3, 9, 1, 3, &[10, 20, 30]), |_, oseq| oseq == 2);
        assert!(gone.is_empty());
        assert_eq!(wanted, vec![(9, 1), (9, 3)]);
        r.headers(listing(3, 8, 5, 5, &[40]), |_, _| false);
        assert_eq!(r.outstanding_wants(), vec![(9, 1), (9, 3), (8, 5)]);
        let fetches = r.headers_done(counter());
        assert_eq!(
            fetches,
            vec![
                Fetch {
                    request: 101,
                    reply_to: 7,
                    from_node: 3,
                    origin: 9,
                    oseqs: vec![1, 3],
                },
                Fetch {
                    request: 102,
                    reply_to: 7,
                    from_node: 3,
                    origin: 8,
                    oseqs: vec![5],
                },
            ]
        );
        assert_eq!(
            (
                r.progress().wanted,
                r.progress().skipped,
                r.progress().wanted_bytes
            ),
            (3, 1, 80)
        );
        assert!(!r.finished());
        let (next, settled) = r.fetch_done(101, |_, _| true, counter());
        assert!(next.is_empty());
        assert_eq!(settled, vec![(9, 1), (9, 3)]);
        assert!(r.fetch_done(102, |_, _| true, counter()).0.is_empty());
        assert!(r.finished());
        assert!(r.outstanding_wants().is_empty());
    }

    /// No more than `FETCH_WINDOW` is outstanding; the next fetch goes once a
    /// `DONE` frees the window.
    #[test]
    fn fetches_stay_within_the_window() {
        let mut r = Requester::new(1, 0, Purpose::Admission, 7, AppliedSet::default());
        let eighth = (FETCH_WINDOW / 8) as u32;
        r.headers(listing(3, 9, 1, 12, &[eighth; 12]), |_, _| false);
        let first = r.headers_done(counter());
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].oseqs, (1..=8).collect::<Vec<_>>());
        assert_eq!(r.outstanding(), FETCH_WINDOW);
        let (next, _) = r.fetch_done(first[0].request, |_, _| true, counter());
        assert_eq!(next[0].oseqs, (9..=12).collect::<Vec<_>>());
    }

    /// A record bigger than the window is fetched alone once nothing else is
    /// outstanding.
    #[test]
    fn an_outsized_record_is_fetched_alone() {
        let mut r = Requester::new(1, 0, Purpose::Admission, 7, AppliedSet::default());
        let outsized = (FETCH_WINDOW + 1) as u32;
        r.headers(listing(3, 9, 1, 2, &[1 << 20, outsized]), |_, _| false);
        let first = r.headers_done(counter());
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].oseqs, vec![1]);
        let (next, _) = r.fetch_done(first[0].request, |_, _| true, counter());
        assert_eq!(next[0].oseqs, vec![2]);
    }

    /// A `FETCH` that ends short is fetched again; after three tries the
    /// record is left for a later catch-up and this one finishes.
    #[test]
    fn a_short_fetch_is_fetched_again_then_left() {
        let mut r = Requester::new(1, 0, Purpose::Admission, 7, AppliedSet::default());
        r.headers(listing(3, 9, 1, 2, &[10, 10]), |_, _| false);
        let f = r.headers_done(counter());
        let (again, _) = r.fetch_done(f[0].request, |_, oseq| oseq == 1, counter());
        assert_eq!(again[0].oseqs, vec![2]);
        let (third, _) = r.fetch_done(again[0].request, |_, oseq| oseq == 1, counter());
        assert_eq!(third[0].oseqs, vec![2]);
        let (none, settled) = r.fetch_done(third[0].request, |_, oseq| oseq == 1, counter());
        assert!(none.is_empty());
        assert_eq!(settled, vec![(9, 2)], "left");
        assert!(r.finished());
        assert_eq!(r.progress().left, 1);
    }

    /// An answer that lists nothing finishes at its `DONE`, and an admission
    /// asks no more of it.
    #[test]
    fn an_empty_answer_finishes_at_once() {
        let mut r = Requester::new(1, 0, Purpose::Admission, 7, AppliedSet::default());
        assert!(r.headers_done(counter()).is_empty());
        assert!(r.finished());
        assert!(!r.asks_again(&moved()));
    }

    /// An admission whose answer brought a record, or taught an `oseq` gone,
    /// asks again once it finishes — and not before.
    #[test]
    fn an_admission_that_moved_on_asks_again() {
        let mut fetched = Requester::new(1, 0, Purpose::Admission, 7, AppliedSet::default());
        fetched.headers(listing(3, 9, 1, 1, &[10]), |_, _| false);
        let f = fetched.headers_done(counter());
        assert!(!fetched.asks_again(&moved()), "still fetching");
        fetched.fetch_done(f[0].request, |_, _| true, counter());
        assert!(fetched.asks_again(&moved()));

        let mut gone = Requester::new(1, 0, Purpose::Admission, 7, AppliedSet::default());
        let mut h = listing(3, 9, 1, 4, &[]);
        h.gone = vec![4];
        gone.headers(h, |_, _| false);
        gone.headers_done(counter());
        assert!(gone.asks_again(&moved()));
    }

    /// **The same answer to the same question is the whole answer.** An
    /// answer listing `oseq`s gone that leave this member holding exactly what
    /// it asked from — records of a type it never takes — does not ask again:
    /// counting them asked again for ever, 660,000 times on WhiteWorld.
    #[test]
    fn an_answer_that_left_the_member_where_it_was_asks_no_more() {
        let since = moved();
        let mut r = Requester::new(1, 0, Purpose::Admission, 7, since.clone());
        let mut h = listing(3, 9, 2, 4, &[]);
        h.gone = vec![3, 4];
        r.headers(h, |_, _| false);
        r.headers_done(counter());
        assert!(r.finished());
        assert!(!r.asks_again(&since), "nothing it holds changed");
        let mut further = since;
        further.insert(9, 3);
        assert!(r.asks_again(&further), "what it holds moved: ask again");
    }

    /// Nothing moved on — every record held already, or left undeliverable —
    /// and the admission is done. Other purposes never ask again: the gap
    /// watch and the periodic catch-up come round on their own.
    #[test]
    fn an_admission_that_moved_nothing_is_done() {
        let mut held = Requester::new(1, 0, Purpose::Admission, 7, AppliedSet::default());
        held.headers(listing(3, 9, 1, 2, &[10, 10]), |_, _| true);
        held.headers_done(counter());
        assert!(held.finished());
        assert!(!held.asks_again(&moved()));

        let mut left = Requester::new(1, 0, Purpose::Admission, 7, AppliedSet::default());
        left.headers(listing(3, 9, 1, 1, &[10]), |_, _| false);
        let mut f = left.headers_done(counter());
        for _ in 0..FETCH_TRIES {
            f = left.fetch_done(f[0].request, |_, _| false, counter()).0;
            if f.is_empty() {
                break;
            }
        }
        assert!(left.finished());
        assert!(!left.asks_again(&moved()));

        let mut gap = Requester::new(1, 0, Purpose::Gap(9), 7, AppliedSet::default());
        gap.headers(listing(3, 9, 1, 1, &[10]), |_, _| false);
        let f = gap.headers_done(counter());
        gap.fetch_done(f[0].request, |_, _| true, counter());
        assert!(gap.finished());
        assert!(!gap.asks_again(&moved()));
    }
}
