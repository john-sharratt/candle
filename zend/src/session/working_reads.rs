//! Keeps the knowledge base consistent with what a turn has read.
//!
//! A file a round read stands verbatim in the dialogue, so it is withheld from
//! the knowledge base for the rest of the turn. Placing it again would carry
//! the file twice. Reads of drawn guides and journals are sealed as
//! use-exemplars of their unit on a thread of their own, and the turn does not
//! wait for them.

use super::*;

/// Keep the readings of every file `results` read out of `timeline`'s working
/// set for the rest of its turn's loop (`docs/zend_working_set.md` §4.5): the
/// page each read returned stands verbatim in the dialogue, so placing the
/// reading again would carry the file twice. A read that failed returned no
/// page and withholds nothing. On the blocking pool: finding the version a
/// read returned is git work.
pub(super) async fn withhold_read_files(
    state: &Arc<InferenceState>,
    timeline: TimelineId,
    files: &Arc<RepoFiles>,
    results: &[ToolResult],
) {
    if state.working_set.is_none() {
        return;
    }
    let read: Vec<ToolCall> = results
        .iter()
        .filter(|r| r.response.get("error").is_none())
        .map(|r| r.call.clone())
        .collect();
    if read.is_empty() {
        return;
    }
    let state = Arc::clone(state);
    let files = Arc::clone(files);
    if let Err(e) = tokio::task::spawn_blocking(move || {
        let read_files = corpus::read_files(&files, &state.workspace, &read);
        state.engine.files_withhold(timeline, &read_files);
    })
    .await
    {
        tracing::warn!(
            timeline = timeline.raw(),
            "withholding the round's read files panicked: {e}"
        );
    }
}

/// The reads of drawn units — a guide, a journal — among `results`; a read
/// that failed read nothing.
pub(super) fn drawn_reads(state: &InferenceState, results: &[ToolResult]) -> Vec<guidance::Read> {
    results
        .iter()
        .filter(|r| r.response.get("error").is_none())
        .filter_map(|r| guidance::read_path(&r.call, &state.workspace))
        .filter(|read| state.retrieval_scope.draws(&read.path()))
        .collect()
}

/// The drawn units `timeline`'s working set places now, by path, each with
/// the fresh score the last scan gave it (`WorkingSet::fresh_of`).
pub(super) fn placed_guidance(state: &InferenceState, timeline: TimelineId) -> Vec<(String, f32)> {
    let conversation = state.engine.conversation();
    let substrate = conversation.read();
    let Some(ws) = substrate.working_set(timeline) else {
        return Vec::new();
    };
    ws.members()
        .into_iter()
        .filter(|member| {
            substrate
                .placing_group(timeline, *member)
                .is_some_and(|g| state.retrieval_scope.is_drawing(g))
        })
        .filter_map(|member| {
            let path = substrate.custom_of(member)?.get(PATH_KEY).cloned()?;
            Some((path, ws.fresh_of(member)))
        })
        .collect()
}

/// Log what the turn's guidance came to (`guidance::tally`), when it placed or
/// read any; a placed unit unread is relevant once a scan matched it to the
/// working set's `min_momentum`.
pub(super) fn log_guidance_tally(conv_id: &str, tally: &guidance::Tally, min_momentum: f32) {
    if tally.is_empty() {
        return;
    }
    let counts = tally.counts(min_momentum);
    tracing::info!(
        target: "zend::guidance",
        conv_id = %conv_id,
        missed = counts.missed.len(),
        read_anyway = counts.read_anyway.len(),
        relevant = counts.relevant.len(),
        idle = counts.idle.len(),
        misses = %counts.missed.join(","),
        read_anyways = %counts.read_anyway.join(","),
        relevants = %counts.relevant.join(","),
        idles = %counts.idle.join(","),
        "guidance this turn",
    );
}

/// Seal each of `reads` — a round's reads of drawn units — as a use-exemplar
/// of its unit (`crate::guidance`): the reasoning the turn wrote before the
/// calls, under the request `asked`, is the state the unit was wanted in. On a
/// thread of its own: finding the unit is git work and the seal is a prefill,
/// and the turn waits for neither. A round with no reasoning ahead of its calls
/// teaches nothing.
pub(super) fn calibrate_guidance_reads(
    state: &Arc<InferenceState>,
    files: &Arc<RepoFiles>,
    reads: Vec<guidance::Read>,
    asked: &str,
    answer: &TurnText,
) {
    if reads.is_empty() {
        return;
    }
    let Some(thought) = thinking_of(answer) else {
        return;
    };
    let Some(case) = guidance::use_case(asked, &thought) else {
        return;
    };
    let state = Arc::clone(state);
    let files = Arc::clone(files);
    std::thread::spawn(move || {
        let frame = CalibrateFrame {
            system_prompt: &state.formatted_prompt,
            config: &state.refresh_config,
            thinking: turn_triggers(&state, ThinkMode::Balanced, ToolMode::None),
        };
        for read in reads {
            let Some((unit, key)) = guidance::unit_of(&state.engine, &files, &read) else {
                tracing::debug!(
                    target: "zend::guidance",
                    subject = %read.path(),
                    "no finished unit for a drawn read at this base: it seals no exemplar",
                );
                continue;
            };
            let subject = read.path();
            match guidance::record_use(&state.engine, &frame, unit, &key, &subject, case.clone()) {
                Ok(retired) => tracing::info!(
                    target: "zend::guidance",
                    subject = %subject,
                    retired,
                    "a read sealed as its unit's use-exemplar",
                ),
                Err(e) => tracing::warn!(
                    target: "zend::guidance",
                    subject = %subject,
                    "sealing a read's use-exemplar failed: {e:#}",
                ),
            }
        }
    });
}
