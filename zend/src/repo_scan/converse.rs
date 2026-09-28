//! The `repo_map` layer's per-folder conversation — a REAL agentic tool loop.
//!
//! [`super::render::render_chain`] seeds the folder's conversation with one
//! prefilled pair (the request, and the `file_list` call it provokes) and the
//! listing's `<tool_response>`. What the model writes in answer to that response
//! is normally the folder's summary, and the conversation is done in one decode.
//!
//! **But a decode is not obliged to answer.** A model handed a tool context can
//! issue another `<tool_call>` instead — most often `file_list` on a
//! subdirectory it just saw named in the listing. There used to be nothing here
//! to run it: the single decode's text was taken verbatim as the folder's
//! summary, so a unit that ended on a call sealed that call *as* its summary,
//! and — because the resume-cache hash was written on any non-error decode — was
//! then skipped on every later pass, permanently. Measured on one workspace: 20
//! of 88 units, and 40% of the units whose listing contained a subdirectory.
//!
//! So the loop is the fix, not a guard against looping. The turn count is bounded
//! by the request instead ([`super::render`]'s `SUMMARY_ASK` asks for ONE
//! sentence and says not to read anything), with [`MAX_FOLDER_ROUNDS`] as the
//! backstop for a model that ignores it — and the backstop *forces an answer*
//! rather than accepting a call as one.
//!
//! The shape mirrors `crate::code_read::run_file_conversation`, which is the same
//! loop over a file's `file_read` calls; the difference is only that a folder
//! starts from a prefilled seed rather than an opening instruction.

use std::sync::Arc;

use candle_conversation::stencil::TriggerRegistry;
use candle_conversation::{Sequence, TurnText};
use zend_tools::ToolContext;

use super::dir_unit::DirUnit;
use super::render::CHAIN_TOOLS;
use super::{dir_tags, FOLDER_SUMMARY_MAX_TOKENS};
use crate::tool_round;
use crate::tools::format_tool_responses;

/// Tool rounds a folder's conversation may run past its seed before being forced
/// to answer on what it has already gathered.
///
/// The request asks for one sentence from the listing alone, so the ordinary unit
/// uses NONE of these — the seed's decode is the summary. The budget is for the
/// model that walks a subdirectory or two first, and is deliberately small: a
/// folder is summarised from what is directly in it, and a conversation still
/// listing on its fifth round is not converging on that.
const MAX_FOLDER_ROUNDS: usize = 4;

/// A folder's completed conversation. The summary itself is not returned — it is
/// a sealed turn on `conv`, which is where the layer reads it from; what the
/// caller needs back is the assurance that one exists (an `Err` otherwise) and
/// the cost of getting it.
pub struct FolderSummary {
    /// Tokens prefilled plus decoded across every round.
    pub tokens: usize,
    /// Tool rounds run past the seed. Zero for a unit that answered its listing
    /// directly, which is what the request asks for.
    pub tool_rounds: usize,
}

/// Drive one folder's conversation to a summary: decode, and for as long as the
/// model answers with a `<tool_call>` instead, run the call and feed the
/// response back as the next round's user turn.
///
/// `prefilled` and `decode_user` are [`super::render::render_chain`]'s output and
/// seed the FIRST round only — from the second round on, the call was decoded
/// rather than prefilled, so there is nothing to write verbatim and the tool
/// response stands alone as the user turn.
///
/// Fails when the conversation never produces a summary (see
/// [`check_summary`]). That is deliberately the same `Err` a decode failure
/// returns, so `process_one_dir`'s existing handling applies unchanged: this
/// attempt's partial is tombstoned, the prior generation stays live, the content
/// key is NOT written, and the unit is retried on the next pass. A unit that
/// ends on a tool call must not be able to commit itself as done.
pub fn run_folder_conversation(
    conv: &mut Sequence,
    unit: &DirUnit,
    prefilled: &[(TurnText, String)],
    decode_user: TurnText,
    triggers: Arc<TriggerRegistry>,
    tool_ctx: &ToolContext,
) -> anyhow::Result<FolderSummary> {
    let tags = dir_tags(unit);
    let force_tools: Vec<String> = CHAIN_TOOLS.iter().map(|t| t.to_string()).collect();
    let mut seed = prefilled;
    let mut user = decode_user;
    let mut tokens = 0usize;

    for round in 0..=MAX_FOLDER_ROUNDS {
        if candle_conversation::ingest_cancelled() {
            anyhow::bail!("shutdown cancelled mid-directory");
        }
        let closing = round == MAX_FOLDER_ROUNDS;
        if closing {
            tracing::warn!(
                target: "zend::repo_scan::converse",
                dir = %unit.dir,
                rounds = round,
                "hit the folder tool-round cap — forcing a summary from what has been listed",
            );
        }
        let got = conv
            .ingest_roundtrip_chain(
                seed,
                user,
                tags.clone(),
                FOLDER_SUMMARY_MAX_TOKENS,
                &force_tools,
                Arc::clone(&triggers),
                closing,
            )
            .map_err(|e| anyhow::anyhow!("ingest_roundtrip_chain: {e}"))?;
        tokens += got.tokens;
        // The seed is spent: every later round's call was DECODED, so it is
        // already a sealed turn and has no prefilled half to write.
        seed = &[];

        let steps = tool_round::plan(&got.text);
        if steps.is_empty() || closing {
            check_summary(&got.text)?;
            return Ok(FolderSummary {
                tokens,
                tool_rounds: round,
            });
        }

        let results = tool_round::run(tool_ctx, steps);
        let response = format_tool_responses(&results);
        // A round whose tools produced no response text must NOT spawn a
        // follow-up turn — there would be nothing in it. The text already
        // decoded stands, and since it was a call, `summary_of` refuses it and
        // the unit retries.
        if response.is_blank() {
            check_summary(&got.text)?;
            return Ok(FolderSummary {
                tokens,
                tool_rounds: round,
            });
        }
        // The round-trip is now certain — the tool returned real output and the
        // follow-up turn is submitted below — so couple the just-sealed call turn
        // to it, by its OWN sealed index.
        conv.couple_turn(got.turn_index)
            .map_err(|e| anyhow::anyhow!("couple_turn: {e}"))?;
        tracing::debug!(
            target: "zend::repo_scan::converse",
            dir = %unit.dir,
            round,
            calls = results.len(),
            "folder conversation ran a follow-up tool round",
        );
        user = response;
    }
    unreachable!("the closing round at MAX_FOLDER_ROUNDS always returns")
}

/// Whether the decoded text is a folder summary — `Err` naming why it is not.
///
/// Two things disqualify it, and both used to be committed as the folder's
/// `repo_map` entry: a `<tool_call>` (the model asking for more evidence, with
/// nowhere left to send the request), and nothing at all (a decode that spent its
/// whole budget elsewhere). Neither is a summary, and a unit holding one is worse
/// than a unit holding nothing — its content key would retire it from every
/// later pass.
fn check_summary(text: &str) -> anyhow::Result<()> {
    if !tool_round::plan(text).is_empty() {
        anyhow::bail!("folder conversation ended on a tool call, not a summary");
    }
    if text.trim().is_empty() {
        anyhow::bail!("folder conversation decoded no summary");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-sentence answer is a summary.
    #[test]
    fn a_plain_sentence_is_a_summary() {
        assert!(check_summary("The `a/` folder holds the widget loader.").is_ok());
    }

    /// The defect this module exists to fix: a decode that ends on a tool call
    /// is NOT a summary, so the caller cannot write the resume hash for it.
    #[test]
    fn a_tool_call_is_refused_as_a_summary() {
        let text = "<tool_call>\n{\"name\": \"file_list\", \"arguments\": {\"prefix\": \"a/b/\"}}\n</tool_call>";
        let err = check_summary(text).unwrap_err().to_string();
        assert!(err.contains("ended on a tool call"), "{err}");
    }

    /// A call with prose in front of it is still a call — the round has work
    /// left to do, so it must not commit.
    #[test]
    fn prose_followed_by_a_call_is_refused() {
        let text = "Let me look inside.\n<tool_call>\n{\"name\": \"file_list\", \
             \"arguments\": {\"prefix\": \"a/b/\"}}\n</tool_call>";
        assert!(check_summary(text).is_err());
    }

    /// An empty decode is refused too: a blank entry would take the folder's
    /// resume hash and retire it from every later pass.
    #[test]
    fn an_empty_decode_is_refused() {
        let err = check_summary("   \n").unwrap_err().to_string();
        assert!(err.contains("decoded no summary"), "{err}");
    }

    /// The backstop is small on purpose: the request asks for no reading at all,
    /// so the budget is for a model that strays, not for a folder that needs it.
    #[test]
    fn the_round_cap_is_a_backstop_not_a_budget() {
        assert!(
            (1..=8).contains(&MAX_FOLDER_ROUNDS),
            "a folder summarised from its own listing needs no rounds; \
             {MAX_FOLDER_ROUNDS} would be a budget, not a backstop"
        );
    }
}
