//! A question's probe and production's pick for it, from a running daemon's
//! `POST /v1/substrate/zend/project`: the question prefilled under the
//! dialogue's system prompt as a thinking turn, `decode` tokens of the
//! model's answer decoded after it, the whole turn's wide-Q captured, and the
//! projection that probe selects — nothing written to the substrate.
//!
//! **Every answer is kept** under [`CACHE_DIR`], keyed by the question's text
//! and `decode`, so the lab asks the daemon once per question and every
//! formula after that scores the same probes. Empty the folder to probe the
//! daemon as it is now.

use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use candle_conversation::persistence::content_hash::ContentHasher;
use candle_conversation::provenance::{decode_wide_sigs, WideQSig};
use serde::Deserialize;
use serde_json::json;

use crate::remote_gallery::{Daemon, ADMIN_HEADERS};

/// Where the daemon's answers are kept, from the repository root.
const CACHE_DIR: &str = "target/provenance-probe-cache";

/// The `code_reading` layer as a projection names it.
const CODE_READING: &str = "code_reading";

/// One question as the daemon probed it.
pub struct Probed {
    /// The turn's wide-Q, one signature a token: the prefilled question's,
    /// then the decoded answer's.
    pub probe: Vec<WideQSig>,
    /// Signatures of `probe` that are the prefilled question's.
    pub question: usize,
    /// The `code_reading` conversations production placed for it, by
    /// timeline, best first.
    pub placed: Vec<u64>,
}

#[derive(Deserialize)]
struct ProjectView {
    tiles: Vec<Tile>,
    probe: String,
    question: usize,
}

#[derive(Deserialize)]
struct Tile {
    selected: bool,
    layer: String,
    #[serde(default)]
    timeline: Option<String>,
}

/// `question` probed on `daemon` with `decode` tokens of its answer, from the
/// cache when it holds it.
pub fn probed(daemon: &Daemon, question: &str, decode: usize) -> anyhow::Result<Probed> {
    let kept = cache_path(question, decode);
    let answer = match fs::read_to_string(&kept) {
        Ok(answer) => answer,
        Err(_) => {
            let answer = ask(daemon, question, decode)?;
            fs::create_dir_all(CACHE_DIR).context("making the probe cache")?;
            fs::write(&kept, &answer).context("keeping the daemon's probe")?;
            answer
        }
    };
    parse(&answer)
}

/// The probe and pick in the daemon's `answer`.
fn parse(answer: &str) -> anyhow::Result<Probed> {
    let view: ProjectView =
        serde_json::from_str(answer).context("reading the daemon's projection")?;
    anyhow::ensure!(
        !view.probe.is_empty(),
        "the daemon's projection carries no probe — it predates the probe field, or probed nothing"
    );
    let bytes = STANDARD
        .decode(&view.probe)
        .context("the probe is not base64")?;
    let probe = decode_wide_sigs(&bytes).context("the probe does not decode")?;
    anyhow::ensure!(
        view.question <= probe.len(),
        "the probe's question ({}) is longer than the probe ({})",
        view.question,
        probe.len()
    );
    let placed = view
        .tiles
        .iter()
        .filter(|t| t.selected && t.layer == CODE_READING)
        .filter_map(|t| t.timeline.as_deref()?.parse().ok())
        .fold(Vec::new(), |mut out: Vec<u64>, tl| {
            if !out.contains(&tl) {
                out.push(tl);
            }
            out
        });
    Ok(Probed {
        probe,
        question: view.question,
        placed,
    })
}

/// The daemon's projection of `question`, as it sent it.
fn ask(daemon: &Daemon, question: &str, decode: usize) -> anyhow::Result<String> {
    let mut req = ureq::AgentBuilder::new()
        .build()
        .post(&format!("{}/v1/substrate/zend/project", daemon.url))
        .set("x-tokera-node", &daemon.node)
        .set("content-type", "application/json");
    for (k, v) in ADMIN_HEADERS {
        req = req.set(k, v);
    }
    match req.send_string(&json!({ "text": question, "decode": decode }).to_string()) {
        Ok(resp) => resp
            .into_string()
            .context("reading the daemon's projection"),
        Err(ureq::Error::Status(code, resp)) => anyhow::bail!(
            "the daemon refused the projection: {code} {}",
            resp.into_string().unwrap_or_default()
        ),
        Err(e) => Err(e).context("reaching the daemon"),
    }
}

/// Where the answer for `question` decoded `decode` tokens is kept.
fn cache_path(question: &str, decode: usize) -> PathBuf {
    let mut h = ContentHasher::new();
    h.update(question.as_bytes());
    h.update(&(decode as u64).to_le_bytes());
    let key = h.finish();
    PathBuf::from(CACHE_DIR).join(format!("{:016x}{:016x}.json", key.hi, key.lo))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::provenance::encode_wide_sigs;

    /// **A projection gives back its probe and its code_reading pick**, in
    /// order, each file once — sections, other layers and unselected tiles
    /// left out.
    #[test]
    fn a_projection_gives_its_probe_and_its_pick() {
        let sigs = vec![
            WideQSig {
                n_heads: 12,
                words: vec![7; 24],
            };
            3
        ];
        let answer = json!({
            "query_tokens": 3,
            "question": 2,
            "probe": STANDARD.encode(encode_wide_sigs(&sigs)),
            "tiles": [
                { "selected": true, "layer": "code_reading", "timeline": "41" },
                { "selected": true, "layer": "dialogue", "timeline": "9" },
                { "selected": false, "layer": "code_reading", "timeline": "42" },
                { "selected": true, "layer": "code_reading", "timeline": "43" },
                { "selected": true, "layer": "code_reading", "timeline": "41" },
                { "selected": true, "layer": "" }
            ]
        })
        .to_string();
        let p = parse(&answer).unwrap();
        assert_eq!(p.probe, sigs);
        assert_eq!(p.question, 2);
        assert_eq!(p.placed, vec![41, 43]);
    }

    /// **A projection with no probe is refused**: the daemon predates it.
    #[test]
    fn a_projection_without_a_probe_is_refused() {
        let answer =
            json!({ "query_tokens": 0, "question": 0, "probe": "", "tiles": [] }).to_string();
        assert!(parse(&answer)
            .err()
            .unwrap()
            .to_string()
            .contains("no probe"));
    }
}
