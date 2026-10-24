//! The catch-ups and fetches a gateway has handed to an answerer and not yet
//! seen a `DONE` for (§7.2). When an answerer's socket ends, each of its open
//! requests will get no answer, and its requester is told so (`LOST`) rather
//! than left to a silence that a reply still arriving over a slow link also
//! gives.

use std::collections::HashMap;

/// One request in the open: its app, requester and id.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    app: String,
    requester: u64,
    request: u64,
}

/// What a request asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Asked {
    Catchup,
    Fetch,
}

/// Open requests, each with the member answering it.
#[derive(Debug, Default)]
pub struct OpenAnswers {
    open: HashMap<Key, (u64, Asked)>,
}

impl OpenAnswers {
    /// `requester`'s catch-up `request` in `app` was handed to `answerer`. An
    /// answerer runs one answer per requester: a newer catch-up cancels the
    /// one before it, which then sends no `DONE`, so the older is forgotten
    /// here too.
    pub fn asked_catchup(&mut self, app: &str, answerer: u64, requester: u64, request: u64) {
        self.open.retain(|k, (a, asked)| {
            !(k.app == app
                && k.requester == requester
                && *a == answerer
                && *asked == Asked::Catchup)
        });
        self.insert(app, answerer, requester, request, Asked::Catchup);
    }

    /// `requester`'s fetch `request` in `app` was handed to `answerer`.
    pub fn asked_fetch(&mut self, app: &str, answerer: u64, requester: u64, request: u64) {
        self.insert(app, answerer, requester, request, Asked::Fetch);
    }

    fn insert(&mut self, app: &str, answerer: u64, requester: u64, request: u64, asked: Asked) {
        self.open.insert(
            Key {
                app: app.to_string(),
                requester,
                request,
            },
            (answerer, asked),
        );
    }

    /// The `DONE` of `requester`'s `request` passed: it is answered.
    pub fn done(&mut self, app: &str, requester: u64, request: u64) {
        self.open.remove(&Key {
            app: app.to_string(),
            requester,
            request,
        });
    }

    /// `node`'s socket in `app` ended. Returns the requests it was answering
    /// for others, as `(requester, request)`: none of them will be answered.
    /// Its own requests are forgotten too — nobody is left to tell.
    pub fn left(&mut self, app: &str, node: u64) -> Vec<(u64, u64)> {
        let mut lost = Vec::new();
        self.open.retain(|k, (answerer, _)| {
            if k.app != app {
                return true;
            }
            if *answerer == node && k.requester != node {
                lost.push((k.requester, k.request));
                return false;
            }
            k.requester != node
        });
        lost.sort_unstable();
        lost
    }

    /// How many requests are open.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.open.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An answerer leaving loses what it was answering, and only that: an
    /// answered request, another app's, and another answerer's stay as they
    /// are.
    #[test]
    fn an_answerer_leaving_loses_only_its_open_requests() {
        let mut o = OpenAnswers::default();
        o.asked_catchup("zend", 0xB, 0xA, 1);
        o.asked_fetch("zend", 0xB, 0xA, 2);
        o.asked_catchup("zend", 0xB, 0xC, 7);
        o.asked_catchup("zend", 0xD, 0xA, 3);
        o.asked_catchup("npcd", 0xB, 0xA, 4);
        o.asked_fetch("zend", 0xB, 0xA, 5);
        o.done("zend", 0xA, 5);

        assert_eq!(o.left("zend", 0xB), [(0xA, 1), (0xA, 2), (0xC, 7)]);
        assert_eq!(o.len(), 2, "zend 0xD's and npcd's remain");
        assert!(o.left("zend", 0xB).is_empty(), "told once");
    }

    /// A requester leaving takes its own requests with it: there is nobody
    /// to tell.
    #[test]
    fn a_requester_leaving_forgets_its_own_requests() {
        let mut o = OpenAnswers::default();
        o.asked_catchup("zend", 0xB, 0xA, 1);
        o.asked_catchup("zend", 0xA, 0xC, 2);
        assert_eq!(o.left("zend", 0xA), [(0xC, 2)]);
        assert_eq!(o.len(), 0);
    }

    /// A requester asking the same answerer again cancels its answer before,
    /// which sends no `DONE`: only the newer stands, so re-asking every 30 s
    /// for hours leaves one entry, not thousands. Its fetches, and its
    /// catch-ups of other answerers, stand.
    #[test]
    fn a_newer_catch_up_replaces_the_one_its_answerer_cancels() {
        let mut o = OpenAnswers::default();
        for request in 1..=1000 {
            o.asked_catchup("zend", 0xB, 0xA, request);
        }
        o.asked_fetch("zend", 0xB, 0xA, 5000);
        o.asked_catchup("zend", 0xD, 0xA, 6000);
        assert_eq!(o.len(), 3);
        assert_eq!(o.left("zend", 0xB), [(0xA, 1000), (0xA, 5000)]);
    }
}
