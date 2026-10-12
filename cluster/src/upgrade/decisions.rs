//! The upgrade's state as the group records it (§3.8, §3.11): Paxos decisions
//! in every member's substrate, so every member and — through `COORD` — every
//! gateway reads the same verdicts, rollouts and pauses, and a restarted
//! gateway learns them from the first member back.
//!
//! | Key | Value |
//! |---|---|
//! | `certify/<app>/<count>-<hash>/<i>` | a [`Verdict`]; the highest `i` holds |
//! | `rollout/<app>/<n>` | a [`Rollout`], a forced rollout's plan; the highest `n` holds |
//! | `pause/<app>/<n>` | a [`Pause`]; the highest `n` holds |
//!
//! A decision is made once, so a later verdict on the same build —
//! `certify-anyway`, `reject-build`, a downgrade's rejection — is decided under
//! the next `i`, and the latest says where the build stands.

use serde::{Deserialize, Serialize};

use crate::frame::BuildId;

/// A build as decisions name it: `count-hash`.
pub fn build_name(build: &BuildId) -> String {
    format!("{}-{}", build.count, build.hash)
}

/// The `(count, hash)` a `count-hash` names.
pub fn parse_build(text: &str) -> Option<(u32, String)> {
    let (count, hash) = text.split_once('-')?;
    let count = count.parse().ok()?;
    (!hash.is_empty() && !hash.contains('-')).then(|| (count, hash.to_string()))
}

pub fn certify_prefix(app: &str, build: &BuildId) -> String {
    format!("certify/{app}/{}/", build_name(build))
}

pub fn certify_key(app: &str, build: &BuildId, i: u64) -> String {
    format!("{}{i}", certify_prefix(app, build))
}

pub fn rollout_prefix(app: &str) -> String {
    format!("rollout/{app}/")
}

pub fn pause_prefix(app: &str) -> String {
    format!("pause/{app}/")
}

/// How a verdict settled a build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Settled {
    /// Certified, or accepted by hand: the automatic rollout may install it.
    Pass,
    /// Rejected: never installed automatically, never sets a floor, and a
    /// member running it is admitted drained.
    Fail,
}

/// `certify/<app>/<build>/<i>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    pub result: Settled,
    /// `certification`, or the admin who settled it by hand.
    pub by: String,
    pub at_ms: u64,
    /// Why, in a line: the failing routes, or the command that settled it.
    pub reason: String,
    /// The figures it rests on, one gateway's per entry (§3.8), as JSON.
    pub figures: Vec<String>,
}

/// One artifact a rollout installs, by what identifies it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactLine {
    pub kind: String,
    pub platform: String,
    pub size: u64,
    pub sha256: String,
}

/// `rollout/<app>/<n>`: a forced rollout (§3.11), as its plan was confirmed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rollout {
    /// The build installed: `count-hash`.
    pub build: String,
    /// The member it comes from, its node in hex.
    pub from: String,
    /// The members it installs on, nodes in hex.
    pub targets: Vec<String>,
    /// How many install at once.
    pub parallel: u32,
    pub artifacts: Vec<ArtifactLine>,
    /// For a downgrade, the builds it rejects: `count-hash`.
    pub rejects: Vec<String>,
    pub by: String,
    pub at_ms: u64,
}

/// `pause/<app>/<n>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pause {
    pub paused: bool,
    pub by: String,
    pub at_ms: u64,
}

pub fn encode<T: Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec(value).expect("a decision always serializes")
}

/// The decision under `prefix` with the highest number, and that number;
/// `decisions` is every `(key, value)` the store holds under it.
pub fn latest<T: for<'de> Deserialize<'de>>(
    decisions: &[(String, Vec<u8>)],
    prefix: &str,
) -> Option<(u64, T)> {
    decisions
        .iter()
        .filter_map(|(k, v)| {
            let n: u64 = k.strip_prefix(prefix)?.parse().ok()?;
            Some((n, v))
        })
        .max_by_key(|(n, _)| *n)
        .and_then(|(n, v)| Some((n, serde_json::from_slice(v).ok()?)))
}

/// The number the next decision under `prefix` takes.
pub fn next(decisions: &[(String, Vec<u8>)], prefix: &str) -> u64 {
    decisions
        .iter()
        .filter_map(|(k, _)| k.strip_prefix(prefix)?.parse::<u64>().ok())
        .max()
        .unwrap_or(0)
        + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build() -> BuildId {
        BuildId {
            count: 4815,
            hash: "9f8e7d6".into(),
            local: String::new(),
        }
    }

    #[test]
    fn keys_name_the_app_and_the_build() {
        assert_eq!(
            certify_key("npcd", &build(), 2),
            "certify/npcd/4815-9f8e7d6/2"
        );
        assert_eq!(rollout_prefix("zend"), "rollout/zend/");
        assert_eq!(pause_prefix("zend"), "pause/zend/");
        assert_eq!(parse_build("4815-9f8e7d6"), Some((4815, "9f8e7d6".into())));
        assert_eq!(parse_build("4815"), None);
        assert_eq!(parse_build("x-9f8e7d6"), None);
        assert_eq!(parse_build("4815-"), None);
    }

    /// A verdict is exact JSON, and the latest number holds.
    #[test]
    fn the_latest_verdict_holds() {
        let pass = Verdict {
            result: Settled::Pass,
            by: "certification".into(),
            at_ms: 1,
            reason: String::new(),
            figures: vec!["{}".into()],
        };
        assert_eq!(
            String::from_utf8(encode(&pass)).unwrap(),
            r#"{"result":"pass","by":"certification","at_ms":1,"reason":"","figures":["{}"]}"#
        );
        let fail = Verdict {
            result: Settled::Fail,
            by: "ops@tokera.com".into(),
            reason: "reject-build".into(),
            figures: vec![],
            at_ms: 2,
        };
        let prefix = certify_prefix("zend", &build());
        let held = vec![
            (format!("{prefix}10"), encode(&fail)),
            (format!("{prefix}9"), encode(&pass)),
            ("certify/zend/1-other/11".to_string(), encode(&pass)),
        ];
        assert_eq!(latest::<Verdict>(&held, &prefix), Some((10, fail)));
        assert_eq!(next(&held, &prefix), 11);
        assert_eq!(latest::<Verdict>(&[], &prefix), None);
        assert_eq!(next(&[], &prefix), 1);
    }
}
