//! The member's cluster as it reports it (`docs/replicated_substrate.md` §7.6):
//! the `connecting` and `catchup` loading steps' detail, and the `cluster`
//! block of its own `/v1/status` — per app, the link, every cluster's socket,
//! its replication: its own events in flight and unsent, and any catch-up in
//! progress — and its upgrade: the floor it is held below, and the build it is
//! fetching to reach it.

use candle_conversation::models::identity::ModelKey;
use cluster::coord::UpgradeView;
use cluster::frame::{BuildId, ModelId};
use cluster::member::{ClusterView, Link};
use cluster::replicate::{CatchupView, Waiting};
use cluster::upgrade::decisions::build_name;
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

/// The `catchup` step's progress: records fetched against what the headers
/// listed — or, while every catch-up waits for an answer, seconds waited
/// against the seconds after which an admission catch-up stops waiting and
/// the member serves on what it holds, so the bar moves toward a point it
/// will reach.
pub fn catchup_progress(catchups: &[CatchupView]) -> (u64, u64) {
    let records = catchups.iter().fold((0, 0), |(done, total), c| {
        (done + c.progress.fetched, total + c.progress.wanted)
    });
    if records.1 > 0 {
        return records;
    }
    catchups
        .iter()
        .filter_map(|c| {
            let w = c.waiting?;
            let ask = w.ask?;
            let waited = u64::from(ask.n - 1) * w.quiet_s + w.silent_s.min(w.quiet_s);
            Some((waited, u64::from(ask.of) * w.quiet_s))
        })
        .fold((0, 0), |(done, total), (d, t)| (done + d, total + t))
}

/// What a catch-up nobody has answered waits on, and what happens when.
fn waiting_detail(w: &Waiting) -> String {
    let who = if w.lost {
        "no member that holds what this one lacks is up"
    } else {
        "waiting for the member asked to answer"
    };
    let next = w.quiet_s.saturating_sub(w.silent_s);
    let then = match w.ask {
        Some(ask) if ask.n >= ask.of => {
            format!("serving on what this member holds in {next} s")
        }
        Some(ask) => format!(
            "asking again in {next} s (ask {} of {}; after {} unanswered it serves on what it \
             holds)",
            ask.n, ask.of, ask.of
        ),
        None => format!("asking again in {next} s"),
    };
    format!("{who} — {then}")
}

/// The `catchup` step's detail: per catch-up, records and bytes fetched
/// against the total the headers listed, records skipped as already held, and
/// which member is answering. One no member has answered says what it waits
/// on and when that ends ([`Waiting`]): "0 of 0 records" read as a catch-up
/// with nothing left to do, and "no member has answered yet" at 0 % for a
/// minute as a dead daemon.
pub fn catchup_detail(catchups: &[CatchupView]) -> String {
    if catchups.is_empty() {
        return "waiting to be asked for what this member lacks".into();
    }
    catchups
        .iter()
        .map(|c| {
            let p = &c.progress;
            if p.answerer.is_none() && p.wanted == 0 {
                return match &c.waiting {
                    Some(w) => waiting_detail(w),
                    None => "asked — no member has answered yet".to_string(),
                };
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

/// An app's upgrade: the floor a gateway admitted it below, when it is — so it
/// takes no traffic and replicates nothing until it has the floor's build —
/// the step it is at, why no build can be installed, and how far the build it
/// is fetching has come.
pub fn upgrade_block(below: Option<&BuildId>, view: &UpgradeView) -> Value {
    json!({
        "below_floor": below.map(build_name),
        "step": view.step,
        "stranded": view.stranded,
        "fetching": view.fetching,
    })
}

/// The `cluster` block: per app, its link, node, every cluster's socket, its
/// replication and its upgrade.
pub fn status_block(cluster: &Cluster) -> Value {
    let build = cluster.build();
    let apps: Map<String, Value> = cluster
        .links()
        .map(|link| {
            let (state, reason) = link_name(&link.state());
            let below = link.member().and_then(|m| m.upgrading());
            (
                link.app().to_string(),
                json!({
                    "link": state,
                    "reason": reason,
                    "node": link.member().map(|m| format!("{:016x}", m.node())),
                    "clusters": link.clusters(),
                    "replication": link.replication(),
                    "upgrade": upgrade_block(
                        below.as_ref(),
                        &cluster.upgrader().view(link.app()),
                    ),
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
    use cluster::replicate::{Ask, Progress, Purpose};

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
            waiting: None,
        };
        assert_eq!(
            catchup_detail(from_ref(&c)),
            "30 of 120 records, 16 of 64 MiB, 7 already held, from 00000000000000ab"
        );
        assert_eq!(catchup_progress(&[c.clone(), c]), (60, 240));
    }

    fn unanswered(purpose: Purpose, waiting: Waiting) -> CatchupView {
        CatchupView {
            cluster: 0,
            purpose,
            progress: Progress::default(),
            waiting: Some(waiting),
        }
    }

    fn admission_wait(silent_s: u64, lost: bool, n: u32) -> Waiting {
        Waiting {
            silent_s,
            quiet_s: 30,
            lost,
            ask: Some(Ask { n, of: 2 }),
        }
    }

    /// **A catch-up nobody has answered says what it waits on and when that
    /// ends.** "asked — no member has answered yet" at 0 % for a minute read
    /// as a dead daemon while a member started alone.
    #[test]
    fn an_unanswered_catch_up_says_what_it_waits_on_and_until_when() {
        let first = unanswered(Purpose::Admission, admission_wait(7, true, 1));
        assert_eq!(
            catchup_detail(from_ref(&first)),
            "no member that holds what this one lacks is up — asking again in 23 s (ask 1 of 2; \
             after 2 unanswered it serves on what it holds)"
        );
        let last = unanswered(Purpose::Admission, admission_wait(25, true, 2));
        assert_eq!(
            catchup_detail(from_ref(&last)),
            "no member that holds what this one lacks is up — serving on what this member holds \
             in 5 s"
        );
        let gap = unanswered(
            Purpose::Gap(3),
            Waiting {
                silent_s: 10,
                quiet_s: 30,
                lost: false,
                ask: None,
            },
        );
        assert_eq!(
            catchup_detail(from_ref(&gap)),
            "waiting for the member asked to answer — asking again in 20 s"
        );
        assert_eq!(
            catchup_detail(&[]),
            "waiting to be asked for what this member lacks"
        );
    }

    /// **While nobody answers, the bar moves toward serving anyway**: the
    /// seconds waited across every ask, against every ask's quiet.
    #[test]
    fn an_unanswered_admission_moves_the_bar_toward_serving() {
        let first = unanswered(Purpose::Admission, admission_wait(15, true, 1));
        assert_eq!(catchup_progress(from_ref(&first)), (15, 60));
        let last = unanswered(Purpose::Admission, admission_wait(45, true, 2));
        assert_eq!(
            catchup_progress(from_ref(&last)),
            (60, 60),
            "silence capped at the quiet"
        );
    }

    /// **A member held below the floor says so, and how its fetch is going.**
    /// WhiteWorld sat `cut_off` on 3585 under a 3593 floor for hours, its
    /// replication idle and its fetch of the floor's build at 0 bytes, and
    /// nothing in its status said why.
    #[test]
    fn a_member_below_the_floor_shows_the_floor_and_its_fetch() {
        use cluster::coord::FetchLine;

        let floor = BuildId {
            count: 3593,
            hash: "ef604210".into(),
            local: String::new(),
        };
        let view = UpgradeView {
            step: "upgrading".into(),
            fetching: Some(FetchLine {
                build: "3593-ef604210".into(),
                kind: "motherd".into(),
                bytes: 0,
                size: 406_964_736,
                resumes: 4,
            }),
            ..UpgradeView::default()
        };
        assert_eq!(
            upgrade_block(Some(&floor), &view),
            json!({
                "below_floor": "3593-ef604210",
                "step": "upgrading",
                "stranded": null,
                "fetching": {
                    "build": "3593-ef604210",
                    "kind": "motherd",
                    "bytes": 0,
                    "size": 406_964_736,
                    "resumes": 4,
                },
            })
        );
        assert_eq!(
            upgrade_block(None, &UpgradeView::default()),
            json!({"below_floor": null, "step": "", "stranded": null, "fetching": null})
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
