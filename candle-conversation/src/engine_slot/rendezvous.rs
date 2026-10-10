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
    expected: usize,
    state: Mutex<State<H>>,
    changed: Condvar,
}

struct State<H> {
    /// Each arrived tenant's rank and requests, until the build takes them.
    arrivals: Vec<(TenantRank, CoreRequests)>,
    /// Tenants that will never arrive.
    withdrawn: usize,
    /// Set once the build has started, so exactly one tenant builds.
    building: bool,
    /// What the build hands the others, or why it failed.
    built: Option<Result<H, String>>,
}

impl<H: Clone> Rendezvous<H> {
    pub(super) fn new(expected: usize) -> Self {
        Self {
            expected,
            state: Mutex::new(State {
                arrivals: Vec::new(),
                withdrawn: 0,
                building: false,
                built: None,
            }),
            changed: Condvar::new(),
        }
    }

    pub(super) fn expected(&self) -> usize {
        self.expected
    }

    fn lock(&self) -> MutexGuard<'_, State<H>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Arrive as tenant `rank` (named `name`) asking `requests`. Blocks until
    /// this tenant can proceed: it builds with every tenant's merged requests
    /// when it is the highest-ranked of those that arrive, and otherwise
    /// attaches to what that one built.
    pub(super) fn arrive<T>(
        &self,
        rank: TenantRank,
        name: &str,
        requests: CoreRequests,
        build: impl FnOnce(CoreRequests) -> crate::Result<(T, H)>,
        attach: impl FnOnce(H) -> crate::Result<T>,
    ) -> crate::Result<T> {
        let mut st = self.lock();
        st.arrivals.push((rank, requests));
        self.changed.notify_all();
        loop {
            if let Some(built) = st.built.clone() {
                drop(st);
                let handed = built.map_err(|e| {
                    ConversationError::Other(format!(
                        "{name}: the shared engine was not built: {e}"
                    ))
                })?;
                return attach(handed);
            }
            let all_here = st.arrivals.len() + st.withdrawn == self.expected;
            let first = st.arrivals.iter().map(|(r, _)| *r).min() == Some(rank);
            if all_here && first && !st.building {
                st.building = true;
                let mut asks: Vec<(TenantRank, CoreRequests)> = take(&mut st.arrivals);
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

    /// A tenant that will never arrive.
    pub(super) fn withdraw(&self) {
        let mut st = self.lock();
        st.withdrawn += 1;
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

    /// One tenant builds at once — what a daemon run alone does.
    #[test]
    fn a_lone_tenant_builds_at_once() {
        let r: Rendezvous<u32> = Rendezvous::new(1);
        let got = r.arrive(
            rank(0),
            "zend",
            ask(4),
            |m| Ok((m.max_concurrent, 7)),
            |_| panic!("a lone tenant never attaches"),
        );
        assert_eq!(got.unwrap(), 4);
    }

    /// The second tenant arrives first and waits; the first builds with both
    /// tenants' requests merged and the second attaches to what it built.
    #[test]
    fn the_first_tenant_builds_with_every_request_and_the_second_attaches() {
        let r: Arc<Rendezvous<u32>> = Arc::new(Rendezvous::new(2));
        let second = {
            let r = Arc::clone(&r);
            std::thread::spawn(move || {
                r.arrive(
                    rank(1),
                    "npcd",
                    ask(3),
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

    /// A tenant that withdraws is not waited for.
    #[test]
    fn a_withdrawn_tenant_is_not_waited_for() {
        let r: Rendezvous<u32> = Rendezvous::new(2);
        r.withdraw();
        let got = r.arrive(
            rank(0),
            "zend",
            ask(4),
            |m| Ok((m.max_concurrent, 1)),
            |_| panic!("zend builds"),
        );
        assert_eq!(got.unwrap(), 4);
    }

    /// A failed build reaches every tenant waiting on it, named.
    #[test]
    fn a_failed_build_reaches_the_tenants_waiting_on_it() {
        let r: Arc<Rendezvous<u32>> = Arc::new(Rendezvous::new(2));
        let second = {
            let r = Arc::clone(&r);
            std::thread::spawn(move || {
                r.arrive(rank(1), "npcd", ask(3), |_| panic!("npcd never builds"), Ok)
            })
        };
        while r.lock().arrivals.is_empty() {
            std::thread::yield_now();
        }
        let first: crate::Result<u32> = r.arrive(
            rank(0),
            "zend",
            ask(4),
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
