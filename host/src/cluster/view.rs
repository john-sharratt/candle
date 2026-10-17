//! The member's cluster as it reports it (`docs/replicated_substrate.md` §7.6):
//! the `connecting` and `catchup` loading steps' detail, and the `cluster`
//! block of its own `/v1/status` — per app, the link, every cluster's socket,
//! and its replication: its own events in flight and unsent, and any catch-up
//! in progress.

use candle_conversation::models::identity::ModelKey;
use cluster::frame::ModelId;
use cluster::member::{ClusterView, Link};
use cluster::replicate::CatchupView;
use serde_json::{json, Map, Value};

use super::Cluster;

/// The model a substrate is bound to, as `HELLO` names it (§3.6).
pub fn model_id(key: &ModelKey) -> ModelId {
    ModelId {
        arch: key.arch.clone(),
        weights: key.weights,
        tokenizer: key.tokenizer,
        rope: key.rope.clone(),
        describe: key.describe(),
    }
}

/// The link as one word, and the reason when stopped.
pub fn link_name(link: &Link) -> (&'static str, Option<String>) {
    match link {
        Link::Starting => ("connecting", None),
        Link::Joined => ("joined", None),
        Link::CutOff => ("cut_off", None),
        Link::Rejoining => ("rejoining", None),
        Link::Reloading => ("reloading", None),
        Link::Stopped(close) => ("stopped", Some(close.to_string())),
    }
}

/// The `catchup` step's progress: records and bytes fetched against what the
/// headers listed.
pub fn catchup_progress(catchups: &[CatchupView]) -> (u64, u64) {
    catchups.iter().fold((0, 0), |(done, total), c| {
        (done + c.progress.fetched, total + c.progress.wanted)
    })
}

/// The `catchup` step's detail: per catch-up, records and bytes fetched
/// against the total the headers listed, records skipped as already held, and
/// which member is answering. One no member has answered says so: "0 of 0
/// records" read as a catch-up with nothing left to do.
pub fn catchup_detail(catchups: &[CatchupView]) -> String {
    if catchups.is_empty() {
        return "waiting to be asked for what this member lacks".into();
    }
    catchups
        .iter()
        .map(|c| {
            let p = &c.progress;
            if p.answerer.is_none() && p.wanted == 0 {
                return "asked — no member has answered yet".to_string();
            }
            let mut line = format!(
                "{} of {} records, {} of {} MiB",
                p.fetched,
                p.wanted,
                p.fetched_bytes >> 20,
                p.wanted_bytes >> 20,
            );
            if p.skipped > 0 {
                line.push_str(&format!(", {} already held", p.skipped));
            }
            if let Some(a) = p.answerer {
                line.push_str(&format!(", from {a:016x}"));
            }
            line
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// How many clusters have admitted this member.
pub fn admitted(views: &[ClusterView]) -> u64 {
    views.iter().filter(|v| v.state == "admitted").count() as u64
}

/// The `connecting` step's detail: each cluster's URL, the attempt, the next
/// retry, and the last close's reason.
pub fn connecting_detail(views: &[ClusterView]) -> String {
    views
        .iter()
        .map(|v| {
            let mut line = v.url.clone();
            match v.state {
                "admitted" => line.push_str(&format!(
                    " — admitted, session {}",
                    v.session.unwrap_or_default()
                )),
                "dialling" => line.push_str(" — dialling"),
                state => {
                    line.push_str(&format!(" — {state}, attempt {}", v.attempts));
                    if let Some(s) = v.retry_in_s {
                        line.push_str(&format!(", next in {s} s"));
                    }
                    if let Some(close) = &v.last_close {
                        line.push_str(&format!(": {close}"));
                    }
                }
            }
            line
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The `cluster` block: per app, its link, node, and every cluster's socket.
pub fn status_block(cluster: &Cluster) -> Value {
    let build = cluster.build();
    let apps: Map<String, Value> = cluster
        .links()
        .map(|link| {
            let (state, reason) = link_name(&link.state());
            (
                link.app().to_string(),
                json!({
                    "link": state,
                    "reason": reason,
                    "node": link.member().map(|m| format!("{:016x}", m.node())),
                    "clusters": link.clusters(),
                    "replication": link.replication(),
                }),
            )
        })
        .collect();
    json!({
        "build": {"count": build.count, "hash": build.hash, "local": build.local},
        "apps": apps,
    })
}

#[cfg(test)]
mod tests {
    use std::slice::from_ref;

    use super::*;
    use cluster::close::{Close, CloseCode};
    use cluster::replicate::{Progress, Purpose};

    fn view(url: &str, state: &'static str) -> ClusterView {
        ClusterView {
            url: url.into(),
            state,
            attempts: 3,
            session: Some(9),
            retry_in_s: Some(4),
            last_close: Some("4410: build 4812 (a1b2c3d) is below the zend floor 4815".into()),
        }
    }

    /// Each cluster in a line: an admitted one by its session, a retrying one by
    /// its attempt, its wait and the close that a 4410 names both builds in.
    #[test]
    fn the_connecting_detail_names_each_cluster() {
        let views = [view("https://a", "admitted"), view("https://b", "retrying")];
        assert_eq!(
            connecting_detail(&views),
            "https://a — admitted, session 9; https://b — retrying, attempt 3, next in 4 s: \
             4410: build 4812 (a1b2c3d) is below the zend floor 4815"
        );
        assert_eq!(admitted(&views), 1);
    }

    /// Each catch-up in a line: what it fetched of what the headers listed,
    /// what it skipped, and who answers.
    #[test]
    fn the_catchup_detail_names_each_catch_up() {
        let c = CatchupView {
            cluster: 0,
            purpose: Purpose::Admission,
            progress: Progress {
                wanted: 120,
                wanted_bytes: 64 << 20,
                skipped: 7,
                fetched: 30,
                fetched_bytes: 16 << 20,
                left: 0,
                answerer: Some(0xab),
            },
        };
        assert_eq!(
            catchup_detail(from_ref(&c)),
            "30 of 120 records, 16 of 64 MiB, 7 already held, from 00000000000000ab"
        );
        assert_eq!(catchup_progress(&[c.clone(), c]), (60, 240));
    }

    /// **A catch-up nobody has answered is not one with nothing to do.** Its
    /// "0 of 0 records" read as finished on a member whose every catch-up was
    /// going unanswered.
    #[test]
    fn an_unanswered_catch_up_says_so() {
        let c = CatchupView {
            cluster: 0,
            purpose: Purpose::Admission,
            progress: Progress::default(),
        };
        assert_eq!(
            catchup_detail(from_ref(&c)),
            "asked — no member has answered yet"
        );
        assert_eq!(
            catchup_detail(&[]),
            "waiting to be asked for what this member lacks"
        );
    }

    #[test]
    fn a_link_is_one_word_and_a_stop_its_reason() {
        assert_eq!(link_name(&Link::Joined), ("joined", None));
        assert_eq!(link_name(&Link::CutOff), ("cut_off", None));
        assert_eq!(link_name(&Link::Rejoining), ("rejoining", None));
        let stop = Link::Stopped(Close::new(CloseCode::Superseded, "rebuild"));
        assert_eq!(link_name(&stop), ("stopped", Some("4410: rebuild".into())));
    }
}
