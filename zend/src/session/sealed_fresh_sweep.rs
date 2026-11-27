//! The anchor sweep: one [`SealedFreshReq`]'s sealed side decoded once, then
//! its fresh side run with its own K/V throughout — the reference — and once
//! per `(head, tail)` of a grid, each section's first `head` and last `tail`
//! tokens kept from the fresh prefill over a sealed K/V transplant
//! ([`SealedFreshReq::anchor_head`]). Whether sealed sections read as fresh
//! ones do with their context restored at their ends, and how much of each
//! end it takes, apart or together.

use serde::{Deserialize, Serialize};

use super::sealed_fresh::{SealedFreshReq, Source};
use super::ZendSession;

/// The sweep asked for: the run every cell starts from, and the grid.
#[derive(Debug, Clone, Deserialize)]
pub struct AnchorSweepReq {
    /// Every cell's request; its `fresh_kv` and anchors are the sweep's.
    #[serde(flatten)]
    pub base: SealedFreshReq,
    /// The heads swept, in tokens.
    pub heads: Vec<usize>,
    /// The tails swept, in tokens.
    pub tails: Vec<usize>,
}

/// One cell of the grid, against the reference.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnchorCell {
    pub head: usize,
    pub tail: usize,
    /// Chunks kept from the fresh prefill.
    pub anchored_chunks: usize,
    /// What the anchored fresh side decoded.
    pub text: String,
    /// The characters it shares with the reference before they part; the
    /// reference's whole length when it decoded the same.
    pub agrees_for: usize,
    pub same_as_reference: bool,
}

/// The sweep's answer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnchorSweep {
    /// The sealed side's decode, which every cell shares.
    pub sealed_text: String,
    /// The fresh side's decode over its own K/V everywhere.
    pub reference_text: String,
    /// Tokens of the prompt, and the files placed after the needle.
    pub prompt_tokens: usize,
    pub after_tokens: usize,
    pub cells: Vec<AnchorCell>,
}

/// Every `(head, tail)` of the grid, heads outer, in the order asked.
pub fn grid(heads: &[usize], tails: &[usize]) -> Vec<(usize, usize)> {
    heads
        .iter()
        .flat_map(|&h| tails.iter().map(move |&t| (h, t)))
        .collect()
}

/// The characters `a` and `b` share before they part.
pub fn common_prefix_chars(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

impl ZendSession {
    /// Run [`AnchorSweepReq`]: the sealed side once, then the reference and
    /// every cell of the grid as fresh sides over it, each transplanting from
    /// the one capture.
    pub fn sealed_fresh_anchor_sweep(&self, req: &AnchorSweepReq) -> anyhow::Result<AnchorSweep> {
        anyhow::ensure!(
            !req.heads.is_empty() && !req.tails.is_empty(),
            "a sweep needs at least one head and one tail"
        );
        // Dropped on every way out, releasing the capture the cells share.
        let half = self.sealed_half(&req.base)?;
        let reference = half.fresh(&SealedFreshReq {
            fresh_kv: Source::Fresh,
            anchor_head: 0,
            anchor_tail: 0,
            ..req.base.clone()
        })?;
        let mut cells = Vec::new();
        for (head, tail) in grid(&req.heads, &req.tails) {
            let run = half.fresh(&SealedFreshReq {
                fresh_kv: Source::Sealed,
                anchor_head: head,
                anchor_tail: tail,
                ..req.base.clone()
            })?;
            let agrees_for = common_prefix_chars(&run.fresh_text, &reference.fresh_text);
            cells.push(AnchorCell {
                head,
                tail,
                anchored_chunks: run.anchored_chunks,
                same_as_reference: run.fresh_text == reference.fresh_text,
                agrees_for,
                text: run.fresh_text,
            });
        }
        Ok(AnchorSweep {
            sealed_text: half.sealed_text(),
            reference_text: reference.fresh_text,
            prompt_tokens: reference.prompt_tokens,
            after_tokens: half.after_tokens(),
            cells,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The grid is every head against every tail, heads outer.**
    #[test]
    fn the_grid_is_every_head_against_every_tail() {
        assert_eq!(
            grid(&[0, 32], &[0, 64, 128]),
            vec![(0, 0), (0, 64), (0, 128), (32, 0), (32, 64), (32, 128)]
        );
        assert!(grid(&[], &[1]).is_empty());
    }

    /// **The shared prefix counts characters up to the first that differs.**
    #[test]
    fn the_shared_prefix_stops_at_the_first_difference() {
        assert_eq!(common_prefix_chars("serde 1.0.228", "serde 1.0.171"), 10);
        assert_eq!(common_prefix_chars("abc", "abc"), 3);
        assert_eq!(common_prefix_chars("—a", "—b"), 1);
        assert_eq!(common_prefix_chars("", "x"), 0);
    }

    /// **The sweep reads its grid beside the base request's own fields.**
    #[test]
    fn the_sweep_reads_its_grid_beside_the_base_request() {
        let req: AnchorSweepReq = serde_json::from_str(
            r#"{"question":"q","n":8,"file":"a/b","heads":[0,32],"tails":[64]}"#,
        )
        .unwrap();
        assert_eq!(req.heads, vec![0, 32]);
        assert_eq!(req.tails, vec![64]);
        assert_eq!(req.base.file.as_deref(), Some("a/b"));
    }
}
