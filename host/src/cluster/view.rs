//! The member's cluster as it reports it (`docs/replicated_substrate.md` §7.6):
//! the `connecting` loading step's detail, and the `cluster` block of its own
//! `/v1/status` — per app, the link and every cluster's socket.

use candle_conversation::models::identity::ModelKey;
use cluster::frame::ModelId;
use cluster::member::{ClusterView, Link};
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
        Link::Stopped(close) => ("stopped", Some(close.to_string())),
    }
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
    use super::*;
    use cluster::close::{Close, CloseCode};

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

    #[test]
    fn a_link_is_one_word_and_a_stop_its_reason() {
        assert_eq!(link_name(&Link::Joined), ("joined", None));
        assert_eq!(link_name(&Link::CutOff), ("cut_off", None));
        let stop = Link::Stopped(Close::new(CloseCode::Superseded, "rebuild"));
        assert_eq!(link_name(&stop), ("stopped", Some("4410: rebuild".into())));
    }
}
