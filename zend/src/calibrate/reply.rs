//! A unit's reply calibration: for each page, the recurrent rows of the
//! `in_context` reply a served read of that page answers with, as a reader's
//! recurrence takes them (`docs/recurrent_reply_calibration.md` §6).
//!
//! Each page is calibrated on a draft of its own — a fork onto the unit, its
//! reading the draft's opening and its recurrent memory the one the reading
//! left ([`on_a_draft`]). The draft prefills the request a read of the page
//! answers and the call, then the reply with its block armed for capture; the
//! rows are kept as the page's record ([`crate::reply_record`]). A page whose
//! record already carries the current recipe is left as it is.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;

use candle_conversation::projection::TimelineId;
use candle_conversation::recurrent_span::{ArmKind, ReplyRows, TextSpan};
use candle_conversation::{
    ConversationEngine, SamplingConfig, Sequence, TurnEvent, TurnOptions, TurnText,
};
use serde_json::json;
use zend_tools::tools::file::render::file_anchor;

use super::run::{draft_selection, on_a_draft, TAG};
use super::spec::CalibrateSpec;
use crate::fast_path::{pages_of, served_response};
use crate::ingest_lease::{self, Share};
use crate::mcp::records::Records;
use crate::reply_record::{self, recipe, slot_key};
use crate::tools::{format_tool_blocks, ToolCall, ToolResult};

/// What a page's draft prefills after the unit.
pub struct PageTurns {
    /// The request the call answers.
    pub request: String,
    /// The call, as the assistant half of the request's turn.
    pub call: String,
    /// The served reply, as the next turn's user half.
    pub reply: TurnText,
    /// The reply's block, by its bytes.
    pub block: Range<usize>,
}

impl PageTurns {
    /// Page `page` of `pages` of the file `subject` (`<repo>/<path>`), under
    /// `spec` — `None` when the layer records no replies.
    pub fn of(spec: &CalibrateSpec, subject: &str, page: u32, pages: u32) -> Option<Self> {
        let request = spec.reply_for(subject, page)?;
        let (repo, path) = crate::repo_path::split(subject);
        let call = ToolCall {
            name: "file_read".to_string(),
            arguments: json!({"repo": repo, "path": path, "page": page}),
        };
        let text = format!(
            "<tool_call>\n{}\n</tool_call>",
            json!({"name": call.name, "arguments": call.arguments})
        );
        let (reply, blocks) = format_tool_blocks(&[ToolResult {
            response: served_response(file_anchor(repo, path), page, pages),
            call,
        }]);
        Some(Self {
            request,
            call: text,
            reply,
            block: blocks.into_iter().next()?,
        })
    }

    /// The recipe a record of these turns is kept under.
    pub fn recipe(&self, system_prompt: &str) -> String {
        recipe(&self.request, &self.call, &self.reply.text(), system_prompt)
    }
}

/// Calibrate every page of the unit at `unit` — keyed `content_key`, of the
/// file `subject` — whose record is missing or of another recipe. Returns how
/// many it calibrated; a page another member holds the lease for is theirs.
pub fn calibrate_replies(
    engine: &ConversationEngine,
    spec: &CalibrateSpec,
    base: &Sequence,
    unit: (TimelineId, &str, &str),
    system_prompt: &str,
) -> anyhow::Result<usize> {
    let (timeline, subject, content_key) = unit;
    let Some(pages) = pages_of(engine, timeline) else {
        return Ok(0);
    };
    let store = engine.conversation().shared();
    let mut done = 0;
    for page in 1..=pages {
        if crate::ingest_cancel::cancelled() {
            break;
        }
        let Some(turns) = PageTurns::of(spec, subject, page, pages) else {
            return Ok(0);
        };
        let key = slot_key(content_key, page);
        let recipe = turns.recipe(system_prompt);
        let fresh =
            |store: &_| reply_record::record(store, &key).is_some_and(|r| r.recipe == recipe);
        if fresh(&store) {
            continue;
        }
        let _lease = match ingest_lease::share_calibration(&format!("{key}|{recipe}")) {
            Share::Theirs => continue,
            Share::Mine(lease) => {
                if lease.recheck() && fresh(&store) {
                    continue;
                }
                lease
            }
        };
        let rows = on_a_draft(engine, &spec.layer, base, timeline, subject, |conv| {
            capture(conv, &turns)
        })?;
        reply_record::keep(&store, &key, &recipe, &rows)?;
        done += 1;
    }
    Ok(done)
}

/// End every record whose unit is not among `present` (content keys), its rows
/// with it. Returns how many it ended.
pub fn retire_gone(engine: &ConversationEngine, present: &HashSet<&str>) -> usize {
    let store = engine.conversation().shared();
    store
        .under(reply_record::PREFIX)
        .into_iter()
        .filter(|(key, _)| !unit_of(key).is_some_and(|k| present.contains(k)))
        .filter(|(key, _)| reply_record::forget(&store, key).is_ok())
        .count()
}

/// The content key a record's key names: `recurrent/<content key>/<page>`.
fn unit_of(key: &str) -> Option<&str> {
    key.strip_prefix(reply_record::PREFIX)?
        .rsplit_once('/')
        .map(|(unit, _)| unit)
}

/// Prefill `turns` on the draft `conv` and capture the reply block's rows.
fn capture(conv: &mut Sequence, turns: &PageTurns) -> anyhow::Result<ReplyRows> {
    let token = crate::ingest_cancel::token();
    let handle = conv
        .submit_prefilled_turn(
            turns.request.as_str(),
            &turns.call,
            crate::tool_def::PROJECTION_MARKER,
            draft_selection(),
            vec![TAG.to_string()],
        )
        .map_err(|e| anyhow::anyhow!("the request: {e}"))?;
    let resp = handle
        .wait_cancellable(token)
        .map_err(|e| anyhow::anyhow!("the request: {e}"))?;
    conv.finish_turn(handle, &resp)
        .map_err(|e| anyhow::anyhow!("the request: {e}"))?;

    // The reply prefills with its block armed; the one token decoded after it
    // is the draft's, and goes with it.
    let options = TurnOptions {
        max_tokens: Some(1),
        sampling: Some(SamplingConfig::argmax()),
        selection: draft_selection(),
        tags: vec![TAG.to_string()],
        recurrent_spans: vec![TextSpan {
            bytes: turns.block.clone(),
            kind: ArmKind::Capture,
        }],
        ..Default::default()
    };
    let handle = conv
        .submit_turn_with_options(turns.reply.clone(), options)
        .map_err(|e| anyhow::anyhow!("the reply: {e}"))?;
    let mut rows = None;
    let mut done = None;
    for event in handle.stream() {
        match event {
            TurnEvent::RecurrentCapture(r) => rows = Some(r),
            TurnEvent::Done(resp) => done = Some(resp),
            TurnEvent::Error(e) => anyhow::bail!("the reply: {e}"),
            _ => {}
        }
    }
    let resp = done.ok_or_else(|| anyhow::anyhow!("the reply never finished"))?;
    conv.finish_turn(handle, &resp)
        .map_err(|e| anyhow::anyhow!("the reply: {e}"))?;
    let rows = rows.ok_or_else(|| anyhow::anyhow!("the reply's block was never captured"))?;
    Ok(Arc::unwrap_or_clone(rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> CalibrateSpec {
        CalibrateSpec {
            layer: "code_reading".into(),
            questions: 1,
            think: 1,
            prompt: "q".into(),
            reply: Some("Read page <<page>> of `<<path>>` in the `<<repo>>` repository.".into()),
        }
    }

    /// A record's key names its unit even when the unit's key holds slashes.
    #[test]
    fn a_record_key_names_its_unit() {
        assert_eq!(
            unit_of(&slot_key("v3:candle/a/b.rs@ff", 2)),
            Some("v3:candle/a/b.rs@ff")
        );
        assert_eq!(unit_of("mcp/x/1"), None);
    }

    /// A page's turns, exactly as a live serve of that page would put them.
    #[test]
    fn a_page_s_turns_are_the_live_serve_s() {
        let t = PageTurns::of(&spec(), "candle/zend/src/a.rs", 2, 3).unwrap();
        assert_eq!(
            t.request,
            "Read page 2 of `zend/src/a.rs` in the `candle` repository."
        );
        assert_eq!(
            t.call,
            "<tool_call>\n{\"name\":\"file_read\",\"arguments\":{\"repo\":\"candle\",\"path\":\"zend/src/a.rs\",\"page\":2}}\n</tool_call>"
        );
        let text = t.reply.text();
        assert_eq!(
            &text[t.block.clone()],
            "<tool_response>{\"status\":\"in_context\",\"anchor\":\"file=candle/zend/src/a.rs page=2 pages=3\"}</tool_response>"
        );
        assert_eq!(t.recipe("sys"), t.recipe("sys"));
        assert_ne!(t.recipe("sys"), t.recipe("sys2"));
        let mut none = spec();
        none.reply = None;
        assert!(PageTurns::of(&none, "candle/a.rs", 1, 1).is_none());
    }
}
