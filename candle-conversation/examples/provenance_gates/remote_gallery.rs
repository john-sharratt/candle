//! The gallery a running daemon holds resident, scanned there: each batch of
//! probes goes to `POST /v1/substrate/zend/scan`, the daemon scores it raw on
//! its GPU gallery arena, and every probe token's vote per fold group comes
//! back for the host to weigh, gate and fold under every setting a gate tries.
//!
//! The daemon's scan is the one step that needs the card; everything this
//! harness iterates on runs here. A second process uploading its own gallery
//! beside a daemon that fills the card overcommits it, and the first kernel to
//! touch an unbacked granule faults.

use std::array::from_fn;
use std::collections::HashMap;

use anyhow::Context;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::provenance::window_score::Vote;
use candle_conversation::provenance::{encode_wide_sigs, WideQSig, FOLD_GROUPS};
use serde::Deserialize;
use serde_json::json;

use crate::corpus::File;

/// The `code_reading` group whose files the gates rank.
const GROUP: &str = "scopes";

/// The admin identity a member's routes accept from the gateway.
const ADMIN_HEADERS: [(&str, &str); 3] = [
    ("x-tokera-user", "forged-admin"),
    ("x-tokera-provider", "google"),
    ("x-tokera-email", "johnathan.sharratt@gmail.com"),
];

/// Where the daemon is: its HTTP listener, and the member that answers.
pub struct Daemon {
    pub url: String,
    /// The member's 16-hex node: a joined member forwards to its gateway,
    /// which routes by it.
    pub node: String,
}

/// What a scan's gallery holds beside the files' own turns.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Questions {
    /// The files' own turns alone.
    Without,
    /// Each file's calibration questions too, each voting for its file — but
    /// the one a mask names.
    With,
}

/// The daemon's answer to one scan.
#[derive(Deserialize)]
struct ScanView {
    files: Vec<String>,
    groups: usize,
    probes: Vec<ProbeVotes>,
}

#[derive(Deserialize)]
struct ProbeVotes {
    tokens: usize,
    cases: Vec<i32>,
    weights: Vec<f32>,
}

/// The daemon's gallery, addressed in this harness's file order.
pub struct RemoteGallery<'a> {
    daemon: &'a Daemon,
    files: &'a [File],
    /// Each file's index here, by its timeline.
    index_of: HashMap<u64, usize>,
    agent: ureq::Agent,
}

impl<'a> RemoteGallery<'a> {
    pub fn new(files: &'a [File], daemon: &'a Daemon) -> RemoteGallery<'a> {
        RemoteGallery {
            daemon,
            files,
            index_of: files
                .iter()
                .enumerate()
                .map(|(i, f)| (f.timeline, i))
                .collect(),
            agent: ureq::AgentBuilder::new().build(),
        }
    }

    /// Each probe's per-token, per-fold-group vote — the file it went to and
    /// its `z × margin` weight — against every turn `masked` does not name, as
    /// `(file, turn)`, and with [`Questions::With`] every calibration question
    /// `masked_question` does not name, as `(file, question)`. A vote does not
    /// carry the gallery token it was won at; a vote for a file this harness
    /// did not load is no vote.
    pub fn votes(
        &self,
        probes: &[&[WideQSig]],
        masked: &dyn Fn(usize, usize) -> bool,
        questions: Questions,
        masked_question: &dyn Fn(usize, usize) -> bool,
    ) -> anyhow::Result<Vec<Vec<[Option<Vote>; FOLD_GROUPS]>>> {
        self.votes_without(probes, masked, questions, masked_question, &[])
    }

    /// [`Self::votes`] with the token ranges `excluded` names — `(turn stream
    /// id, start, end)` — cut from the gallery, so the scan never sees them
    /// and their votes go to whatever agrees next best.
    pub fn votes_without(
        &self,
        probes: &[&[WideQSig]],
        masked: &dyn Fn(usize, usize) -> bool,
        questions: Questions,
        masked_question: &dyn Fn(usize, usize) -> bool,
        excluded: &[(u64, usize, usize)],
    ) -> anyhow::Result<Vec<Vec<[Option<Vote>; FOLD_GROUPS]>>> {
        let mut masked_sids: Vec<String> = Vec::new();
        for (fi, f) in self.files.iter().enumerate() {
            for t in (0..f.turns.len()).filter(|&t| masked(fi, t)) {
                masked_sids.push(turn_stream_id(f.timeline, t as u32).0.to_string());
            }
            if questions == Questions::With {
                for (qi, q) in f.questions.iter().enumerate() {
                    if masked_question(fi, qi) {
                        masked_sids.push(turn_stream_id(q.timeline, q.turn).0.to_string());
                    }
                }
            }
        }
        let body = json!({
            "group": GROUP,
            "calibrations": questions == Questions::With,
            "masked": masked_sids,
            "excluded": excluded
                .iter()
                .map(|&(turn, start, end)| json!({ "turn": turn.to_string(), "start": start, "end": end }))
                .collect::<Vec<_>>(),
            "probes": probes
                .iter()
                .map(|p| STANDARD.encode(encode_wide_sigs(p)))
                .collect::<Vec<_>>(),
        });
        let mut req = self
            .agent
            .post(&format!("{}/v1/substrate/zend/scan", self.daemon.url))
            .set("x-tokera-node", &self.daemon.node);
        for (k, v) in ADMIN_HEADERS {
            req = req.set(k, v);
        }
        let view: ScanView = match req.send_json(body) {
            Ok(resp) => resp.into_json().context("reading the daemon's scan")?,
            Err(ureq::Error::Status(code, resp)) => anyhow::bail!(
                "the daemon refused the scan: {code} {}",
                resp.into_string().unwrap_or_default()
            ),
            Err(e) => return Err(e).context("reaching the daemon"),
        };
        local_votes(&view, &self.index_of)
    }
}

/// `view`'s votes with each case named by this harness's file index.
fn local_votes(
    view: &ScanView,
    index_of: &HashMap<u64, usize>,
) -> anyhow::Result<Vec<Vec<[Option<Vote>; FOLD_GROUPS]>>> {
    anyhow::ensure!(
        view.groups == FOLD_GROUPS,
        "the daemon scans {} fold groups, this harness {FOLD_GROUPS}",
        view.groups
    );
    let local: Vec<Option<usize>> = view
        .files
        .iter()
        .map(|f| {
            f.parse::<u64>()
                .ok()
                .and_then(|tl| index_of.get(&tl).copied())
        })
        .collect();
    Ok(view
        .probes
        .iter()
        .map(|p| {
            (0..p.tokens)
                .map(|t| {
                    from_fn(|g| {
                        let i = t * FOLD_GROUPS + g;
                        let case = usize::try_from(p.cases[i]).ok()?;
                        local[case].map(|file| Vote {
                            token: usize::MAX,
                            file,
                            weight: p.weights[i],
                        })
                    })
                })
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The daemon's cases come back as this harness's files**: its file
    /// order differs, one of its files is not loaded here, and a token with
    /// no vote stays none.
    #[test]
    fn a_daemons_votes_are_named_by_local_file() {
        let view = ScanView {
            files: vec!["900".into(), "41".into(), "42".into()],
            groups: FOLD_GROUPS,
            probes: vec![ProbeVotes {
                tokens: 1,
                cases: vec![2, 0, -1],
                weights: vec![1.5, 2.0, 0.0],
            }],
        };
        let index_of: HashMap<u64, usize> = [(41, 0), (42, 1)].into();
        let votes = local_votes(&view, &index_of).unwrap();
        assert_eq!(votes.len(), 1);
        assert_eq!(
            votes[0][0],
            [
                Some(Vote {
                    token: usize::MAX,
                    file: 1,
                    weight: 1.5
                }),
                None,
                None
            ]
        );
    }
}
