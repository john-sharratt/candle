//! The meeting point behind [`super::EngineSlot`]: every tenant arrives with
//! its requests, the highest-ranked builds once all have arrived or withdrawn,
//! and the rest attach to what it built.
//!
//! Generic over what a build hands the others (`H`), so the coordination is
//! tested on its own, with no model behind it.

use std::mem::take;
use std::sync::{Condvar, Mutex, MutexGuard};

use super::CoreRequests;
use crate::error::ConversationError;
use crate::tenant_rank::TenantRank;

pub(super) struct Rendezvous<H> {
    /// Every tenant expected, by name.
    expected: Vec<&'static str>,
    state: Mutex<State<H>>,
    changed: Condvar,
}

struct State<H> {
    /// Each arrived tenant's rank and requests, until the build takes them.
    arrivals: Vec<(TenantRank, CoreRequests)>,
    /// The requests of daemons left out of the process, at the ranks they
    /// would run at: merged into the build, never building or attaching.
    standing: Vec<(TenantRank, CoreRequests)>,
    /// Every tenant that has arrived, by name — kept after the build takes
    /// the requests.
    arrived: Vec<&'static str>,
    /// Tenants that will never arrive, by name.
    withdrawn: Vec<&'static str>,
    /// Set once the build has started, so exactly one tenant builds.
    building: bool,
    /// What the build hands the others, or why it failed.
    built: Option<Result<H, String>>,
}

impl<H: Clone> Rendezvous<H> {
    pub(super) fn new(
        expected: &[&'static str],
        standing: Vec<(TenantRank, &'static str, CoreRequests)>,
    ) -> Self {
        Self {
            expected: expected.to_vec(),
            state: Mutex::new(State {
                arrivals: Vec::new(),
                standing: standing.into_iter().map(|(r, _, a)| (r, a)).collect(),
                arrived: Vec::new(),
                withdrawn: Vec::new(),
                building: false,
                built: None,
            }),
            changed: Condvar::new(),
        }
    }

    pub(super) fn expected(&self) -> usize {
        self.expected.len()
    }

    fn lock(&self) -> MutexGuard<'_, State<H>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The expected tenants that have neither arrived nor withdrawn.
    fn missing(&self, st: &State<H>) -> Vec<&'static str> {
        self.expected
            .iter()
            .copied()
            .filter(|n| !st.arrived.contains(n) && !st.withdrawn.contains(n))
            .collect()
    }

    /// Arrive as tenant `rank` (named `name`) asking `requests`. Blocks until
    /// this tenant can proceed: it builds with every tenant's merged requests
    /// when it is the highest-ranked of those that arrive, and otherwise
    /// attaches to what that one built.
    ///
    /// `waiting` is told the tenants this one is waiting for, each time that
    /// set changes, and told none once it proceeds — what its loading screen
    /// shows while another tenant is still on its way.
    pub(super) fn arrive<T>(
        &self,
        rank: TenantRank,
        name: &'static str,
        requests: CoreRequests,
        waiting: &dyn Fn(&[&'static str]),
        build: impl FnOnce(CoreRequests) -> crate::Result<(T, H)>,
        attach: impl FnOnce(H) -> crate::Result<T>,
    ) -> crate::Result<T> {
        let mut st = self.lock();
        st.arrivals.push((rank, requests));
        st.arrived.push(name);
        self.changed.notify_all();
        let mut told: Vec<&'static str> = Vec::new();
        loop {
            let missing = self.missing(&st);
            if missing != told {
                waiting(&missing);
                told = missing;
            }
            if let Some(built) = st.built.clone() {
                drop(st);
                if !told.is_empty() {
                    waiting(&[]);
                }
                let handed = built.map_err(|e| {
                    ConversationError::Other(format!(
                        "{name}: the shared engine was not built: {e}"
                    ))
                })?;
                return attach(handed);
            }
            let first = st.arrivals.iter().map(|(r, _)| *r).min() == Some(rank);
            if told.is_empty() && first && !st.building {
                st.building = true;
                let mut asks: Vec<(TenantRank, CoreRequests)> = take(&mut st.arrivals);
                asks.append(&mut st.standing);
                drop(st);
                asks.sort_by_key(|(r, _)| *r);
                let merged = CoreRequests::merge(asks.into_iter().map(|(_, a)| a).collect())
                    .expect("this tenant's own requests are among them");
                let result = build(merged);
                let mut st = self.lock();
                st.built = Some(match &result {
                    Ok((_, handed)) => Ok(handed.clone()),
                    Err(e) => Err(e.to_string()),
                });
                self.changed.notify_all();
                return result.map(|(own, _)| own);
            }
            st = self.changed.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Tenant `name`, which will never arrive.
    pub(super) fn withdraw(&self, name: &'static str) {
        let mut st = self.lock();
        st.withdrawn.push(name);
        self.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::guest::GuestRegistry;
    use crate::models::StrataTokens;

    fn ask(max_concurrent: usize) -> CoreRequests {
        CoreRequests {
            prefill_pass_tokens: None,
            min_rope_factor: None,
            qsa_selection_budget: None,
            qsa_strata: StrataTokens::DEFAULT,
            max_response_tokens: 16_384,
            max_concurrent,
            thinking: None,
            guests: GuestRegistry::new(),
        }
    }

    fn rank(i: usize) -> TenantRank {
        TenantRank::nth(i).unwrap()
    }

    const BOTH: &[&str] = &["zend", "npcd"];

    fn no_wait(_: &[&'static str]) {}

    /// One tenant builds at once — what a daemon run alone does — and waits
    /// for nobody.
    #[test]
    fn a_lone_tenant_builds_at_once() {
        let r: Rendezvous<u32> = Rendezvous::new(&["zend"], Vec::new());
        let got = r.arrive(
            rank(0),
            "zend",
            ask(4),
            &|w| panic!("a lone tenant waits for nobody, told {w:?}"),
            |m| Ok((m.max_concurrent, 7)),
            |_| panic!("a lone tenant never attaches"),
        );
        assert_eq!(got.unwrap(), 4);
    }

    /// The second tenant arrives first and waits; the first builds with both
    /// tenants' requests merged and the second attaches to what it built.
    #[test]
    fn the_first_tenant_builds_with_every_request_and_the_second_attaches() {
        let r: Arc<Rendezvous<u32>> = Arc::new(Rendezvous::new(BOTH, Vec::new()));
        let second = {
            let r = Arc::clone(&r);
            std::thread::spawn(move || {
                r.arrive(
                    rank(1),
                    "npcd",
                    ask(3),
                    &no_wait,
                    |_| panic!("npcd never builds"),
                    |h| Ok(h + 100),
                )
            })
        };
        // Let the second park before the first arrives.
        while r.lock().arrivals.is_empty() {
            std::thread::yield_now();
        }
        let first = r.arrive(
            rank(0),
            "zend",
            ask(4),
            &no_wait,
            |m| Ok((m.max_concurrent, 42)),
            |_| panic!("zend builds"),
        );
        assert_eq!(first.unwrap(), 7, "built with both tenants' conversations");
        assert_eq!(
            second.join().unwrap().unwrap(),
            142,
            "attached to the build"
        );
    }

    /// **The first to arrive is told whom it waits for**, and told nobody
    /// once they have come — what its loading screen shows as "waiting for
    /// npcd" while npcd is still opening its substrate.
    #[test]
    fn a_tenant_waiting_is_told_whom_it_waits_for() {
        let r: Arc<Rendezvous<u32>> = Arc::new(Rendezvous::new(BOTH, Vec::new()));
        let told: Arc<Mutex<Vec<Vec<&'static str>>>> = Arc::new(Mutex::new(Vec::new()));
        let first = {
            let (r, told) = (Arc::clone(&r), Arc::clone(&told));
            std::thread::spawn(move || {
                r.arrive(
                    rank(0),
                    "zend",
                    ask(4),
                    &|w| told.lock().unwrap().push(w.to_vec()),
                    |m| Ok((m.max_concurrent, 1)),
                    |_| panic!("zend builds"),
                )
            })
        };
        while told.lock().unwrap().is_empty() {
            std::thread::yield_now();
        }
        let second = r.arrive(
            rank(1),
            "npcd",
            ask(3),
            &no_wait,
            |_| panic!("npcd never builds"),
            |h| Ok(h + 10),
        );
        assert_eq!(first.join().unwrap().unwrap(), 7);
        assert_eq!(second.unwrap(), 11);
        assert_eq!(*told.lock().unwrap(), vec![vec!["npcd"], vec![]]);
    }

    /// A tenant that withdraws is not waited for.
    #[test]
    fn a_withdrawn_tenant_is_not_waited_for() {
        let r: Rendezvous<u32> = Rendezvous::new(BOTH, Vec::new());
        r.withdraw("npcd");
        let got = r.arrive(
            rank(0),
            "zend",
            ask(4),
            &|w| panic!("npcd withdrew, yet zend was told {w:?}"),
            |m| Ok((m.max_concurrent, 1)),
            |_| panic!("zend builds"),
        );
        assert_eq!(got.unwrap(), 4);
    }

    /// **A daemon left out still asks.** npcd run alone, at the rank it holds
    /// beside zend, builds at once — waiting for nobody — on the merge it
    /// would have had: zend's standing asks first, its own after.
    #[test]
    fn a_standing_seat_is_merged_but_never_waited_for() {
        let mut zend = ask(4);
        zend.min_rope_factor = Some(2.0);
        let r: Rendezvous<u32> = Rendezvous::new(&["npcd"], vec![(rank(0), "zend", zend)]);
        let got = r.arrive(
            rank(1),
            "npcd",
            ask(3),
            &|w| panic!("a standing seat is not waited for, told {w:?}"),
            |m| {
                assert_eq!(m.min_rope_factor, Some(2.0), "zend's floor");
                Ok((m.max_concurrent, 1))
            },
            |_| panic!("npcd is the only tenant that runs, so it builds"),
        );
        assert_eq!(got.unwrap(), 7, "both seats' conversations");
    }

    /// A failed build reaches every tenant waiting on it, named.
    #[test]
    fn a_failed_build_reaches_the_tenants_waiting_on_it() {
        let r: Arc<Rendezvous<u32>> = Arc::new(Rendezvous::new(BOTH, Vec::new()));
        let second = {
            let r = Arc::clone(&r);
            std::thread::spawn(move || {
                r.arrive(
                    rank(1),
                    "npcd",
                    ask(3),
                    &no_wait,
                    |_| panic!("npcd never builds"),
                    Ok,
                )
            })
        };
        while r.lock().arrivals.is_empty() {
            std::thread::yield_now();
        }
        let first: crate::Result<u32> = r.arrive(
            rank(0),
            "zend",
            ask(4),
            &no_wait,
            |_| Err(ConversationError::Other("no weights".into())),
            |_| panic!("zend builds"),
        );
        assert!(first.is_err());
        let err = second.join().unwrap().unwrap_err().to_string();
        assert!(
            err.contains("npcd: the shared engine was not built: no weights"),
            "{err}"
        );
    }
}
