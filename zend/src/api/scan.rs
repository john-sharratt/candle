//! `POST /v1/substrate/zend/scan` — score a batch of probes raw against a
//! belief group on the daemon's GPU gallery arena, and hand back every probe
//! token's vote per fold group, before the needle gate and the tally.
//!
//! The primitive an offline harness builds on: the daemon holds the gallery
//! resident and runs the scan, the one step that needs the card, and the
//! harness does everything else — weighing, gating, normalizing, folding,
//! grading — on the host, where it iterates without the daemon being rebuilt
//! (`candle-conversation/examples/provenance_gates`). A second process
//! uploading its own gallery beside a daemon that fills the card overcommits
//! it, and the first kernel to touch an unbacked granule faults.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};

use candle_conversation::persistence::streams::StreamId;
use candle_conversation::projection::GroupVotes;
use candle_conversation::provenance::{decode_wide_sigs, WideQSig};
use candle_conversation::ScanRefused;

use crate::session::ZendSession;

/// The largest request the route reads: a batch of a few thousand probe
/// tokens at 192 bytes each is megabytes of base64 on its own, and a gallery
/// cut can name tens of thousands of ranges.
pub const BODY_LIMIT: usize = 64 << 20;

/// What to score, and against what.
#[derive(Deserialize)]
pub struct ScanReq {
    /// The belief group, by its schema name (`scopes`).
    pub group: String,
    /// Whether the group's calibration questions stand in the gallery.
    #[serde(default)]
    pub calibrations: bool,
    /// Turns left out of the gallery, by stream id — decimal strings, so a
    /// JavaScript client never rounds them.
    #[serde(default)]
    pub masked: Vec<String>,
    /// Token ranges cut from the gallery, so the scan never sees them and
    /// their votes go to whatever agrees next best.
    #[serde(default)]
    pub excluded: Vec<Excluded>,
    /// Each probe's signatures, base64 of `encode_wide_sigs`.
    pub probes: Vec<String>,
}

/// One turn's `[start, end)` tokens cut from the gallery.
#[derive(Deserialize)]
pub struct Excluded {
    /// The turn's stream id, a decimal string.
    pub turn: String,
    pub start: usize,
    pub end: usize,
}

/// The votes: each case's file, and per probe its tokens' votes.
#[derive(Serialize, Debug, PartialEq)]
pub struct ScanView {
    /// Each case's timeline — a decimal string, as every timeline id on the
    /// wire is.
    pub files: Vec<String>,
    /// Fold groups per token.
    pub groups: usize,
    pub probes: Vec<ProbeVotes>,
}

/// One probe's votes, token-major then fold group: entry `t × groups + g`.
#[derive(Serialize, Debug, PartialEq)]
pub struct ProbeVotes {
    /// Tokens scanned (a token narrower than the gallery's fold is not).
    pub tokens: usize,
    /// The case each vote went to, `-1` where it went to none.
    pub cases: Vec<i32>,
    /// Each vote's `z × margin` weight, `0` where there was none.
    pub weights: Vec<f32>,
}

pub async fn scan(
    State(session): State<Arc<ZendSession>>,
    Json(req): Json<ScanReq>,
) -> Result<Json<ScanView>, (StatusCode, String)> {
    let bad = |e: String| (StatusCode::BAD_REQUEST, e);
    let probes = decode_probes(&req.probes).map_err(bad)?;
    let masked = decode_masked(&req.masked).map_err(bad)?;
    let excluded = decode_excluded(&req.excluded).map_err(bad)?;
    let group = session
        .projection_builder()
        .id_for_group(&req.group)
        .ok_or_else(|| bad(format!("no group named `{}`", req.group)))?;
    let scanner = session.group_scanner().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "the model is not loaded".to_string(),
    ))?;
    // The scan runs on the scheduler thread at ingest priority; this thread
    // only waits for it, off the async runtime's workers.
    let calibrations = req.calibrations;
    let out = tokio::task::spawn_blocking(move || {
        scanner.scan(group, calibrations, masked, excluded, probes)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    match out {
        Ok(Ok(votes)) => Ok(Json(view(&votes))),
        Ok(Err(e)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("scan failed: {e}"),
        )),
        Err(ScanRefused::NoArena) => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "this daemon has no GPU gallery arena".to_string(),
        )),
        Err(ScanRefused::Gone) => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "the scheduler has shut down".to_string(),
        )),
    }
}

/// Each probe's signatures, from base64 of `encode_wide_sigs`.
fn decode_probes(probes: &[String]) -> Result<Vec<Vec<WideQSig>>, String> {
    probes
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let bytes = STANDARD
                .decode(p)
                .map_err(|e| format!("probe {i} is not base64: {e}"))?;
            decode_wide_sigs(&bytes).ok_or_else(|| format!("probe {i} holds no signatures"))
        })
        .collect()
}

/// The masked turns' stream ids, from decimal strings.
fn decode_masked(masked: &[String]) -> Result<HashSet<StreamId>, String> {
    masked
        .iter()
        .map(|m| {
            m.parse::<u64>()
                .map(StreamId)
                .map_err(|e| format!("masked turn `{m}` is not a stream id: {e}"))
        })
        .collect()
}

/// The excluded ranges, by turn.
fn decode_excluded(excluded: &[Excluded]) -> Result<HashMap<StreamId, Vec<Range<usize>>>, String> {
    let mut by_turn: HashMap<StreamId, Vec<Range<usize>>> = HashMap::new();
    for x in excluded {
        let turn = x
            .turn
            .parse::<u64>()
            .map_err(|e| format!("excluded turn `{}` is not a stream id: {e}", x.turn))?;
        by_turn
            .entry(StreamId(turn))
            .or_default()
            .push(x.start..x.end);
    }
    Ok(by_turn)
}

/// The wire form of `votes`.
fn view(votes: &GroupVotes) -> ScanView {
    let raw = &votes.votes;
    let groups = raw.n_groups;
    let probes = (0..raw.per_probe.len())
        .map(|p| {
            let first = raw.first_token(p);
            let tokens = raw.per_probe[p];
            let mut cases = Vec::with_capacity(tokens * groups);
            let mut weights = Vec::with_capacity(tokens * groups);
            for t in first..first + tokens {
                for g in 0..groups {
                    match raw.at(t, g, 0) {
                        Some((case, weight)) => {
                            cases.push(case as i32);
                            weights.push(weight);
                        }
                        None => {
                            cases.push(-1);
                            weights.push(0.0);
                        }
                    }
                }
            }
            ProbeVotes {
                tokens,
                cases,
                weights,
            }
        })
        .collect();
    ScanView {
        files: votes.files.iter().map(|f| f.raw().to_string()).collect(),
        groups,
        probes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::projection::TimelineId;
    use candle_conversation::provenance::encode_wide_sigs;
    use candle_conversation::provenance::gallery_arena::TokenVotes;

    /// **A probe goes in as base64 of its encoded signatures and comes out
    /// whole**; text that is not base64, or bytes that hold no signatures,
    /// are refused by index.
    #[test]
    fn probes_decode_from_base64_and_bad_ones_are_named() {
        let sigs = vec![
            WideQSig {
                n_heads: 12,
                words: vec![0x0F0F_0F0F_0F0F_0F0F; 24],
            },
            WideQSig {
                n_heads: 12,
                words: vec![0x3333_3333_3333_3333; 24],
            },
        ];
        let wire = STANDARD.encode(encode_wide_sigs(&sigs));
        assert_eq!(decode_probes(&[wire]).unwrap(), vec![sigs]);
        assert!(decode_probes(&["@@".into()])
            .unwrap_err()
            .starts_with("probe 0 is not base64"));
        assert_eq!(
            decode_probes(&[STANDARD.encode([1u8, 2, 3])]).unwrap_err(),
            "probe 0 holds no signatures"
        );
    }

    #[test]
    fn excluded_ranges_gather_by_turn() {
        let x = |turn: &str, start, end| Excluded {
            turn: turn.into(),
            start,
            end,
        };
        let got = decode_excluded(&[x("7", 0, 32), x("9", 64, 96), x("7", 96, 128)]).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[&StreamId(7)], vec![0..32, 96..128]);
        assert_eq!(got[&StreamId(9)], vec![64..96]);
        assert!(decode_excluded(&[x("seven", 0, 1)]).is_err());
    }

    #[test]
    fn masked_turns_parse_as_decimal_stream_ids() {
        assert_eq!(
            decode_masked(&["7".into(), "18446744073709551615".into()]).unwrap(),
            [StreamId(7), StreamId(u64::MAX)].into()
        );
        assert!(decode_masked(&["x".into()]).is_err());
    }

    /// **The view lays each probe's votes token-major, then group**: two
    /// probes of one and two tokens over two groups, one vote missing.
    #[test]
    fn votes_are_laid_token_major_then_group() {
        let votes = GroupVotes {
            files: vec![
                TimelineId::from_raw(41).unwrap(),
                TimelineId::from_raw(42).unwrap(),
            ],
            votes: TokenVotes {
                case: vec![0, 1, 1, -1, 0, 0],
                vote: vec![1.5, 2.5, 3.0, 0.0, 4.0, 5.0],
                per_probe: vec![1, 2],
                n_groups: 2,
                n_segments: 1,
            },
        };
        assert_eq!(
            view(&votes),
            ScanView {
                files: vec!["41".into(), "42".into()],
                groups: 2,
                probes: vec![
                    ProbeVotes {
                        tokens: 1,
                        cases: vec![0, 1],
                        weights: vec![1.5, 2.5],
                    },
                    ProbeVotes {
                        tokens: 2,
                        cases: vec![1, -1, 0, 0],
                        weights: vec![3.0, 0.0, 4.0, 5.0],
                    },
                ],
            }
        );
    }
}
