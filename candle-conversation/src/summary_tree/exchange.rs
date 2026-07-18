//! Grouping `Normal` turns into **exchanges** — the unit of summarisation and
//! selection (`docs/immutable_summary_forest.md`, *Exchanges*).
//!
//! A tool round-trip spans more than one turn. The model answers with `<think>`
//! plus a `<tool_call>`; the tool's output arrives as the *next* turn's user half
//! (`<tool_response>`); only then does the assistant give its real answer. So a
//! single turn is often only *half* an exchange, and treating it as a whole one
//! breaks three ways at once:
//!
//! - a summary over the call turn alone has **no answer among its children**, so
//!   the compressor invents one (an observed leaf claimed "15:30 UTC" when the
//!   tool had returned 12:21);
//! - a summary over the response turn alone has **no question**, so its scope
//!   becomes a raw `<tool_response>{…}` JSON blob;
//! - selecting one without the other injects a tool response that nothing
//!   requested, or a request whose result never arrives.
//!
//! An **exchange** is the maximal run of turns joined by [`TurnCoupling`]
//! records: a head turn plus every following turn coupled to its predecessor.
//!
//! # Why a coupling record rather than per-turn flags
//!
//! The obvious alternative — mark each turn "I expect a tool response" / "I am a
//! tool response" at its own seal — cannot work, because of *when* the facts are
//! known. A turn's `TurnDecl` is written by the scheduler at seal, and only
//! afterwards does zend parse the assistant text for tool calls and decide
//! whether to run them. By the time the authoritative answer exists, the record
//! is already durable.
//!
//! Deriving the flag from the decode instead (the stencil's tool-call open) is
//! available at seal but is *not* authoritative: it fires in capture mode, where
//! calls are deliberately never executed, and on a malformed call that yields no
//! runnable tool — both claiming a continuation that never comes.
//!
//! A coupling sidesteps both. Because the tool response is always the next
//! `Normal` turn, the record needs only its `from` index, so zend can write it in
//! the one window where the fact is certain: after the tools have actually
//! returned output, and before the response turn is submitted. So a coupling
//! exists **iff** the round-trip really happened, and it is always durable
//! *before* the turn it points at exists — the summariser can never observe the
//! response turn without also observing the coupling, and therefore can never
//! freeze a leaf over half an exchange.
//!
//! "Next `Normal` turn" is precise, not loose: summary turns share the index
//! space and the summariser is asynchronous, so a leaf can be recorded between a
//! call and its response. Everything here works in positions over the `Normal`
//! subsequence — see [`over_normals`].

use std::ops::Range;

use ahash::{AHashMap, AHashSet};

use crate::projection::TurnIndex;

/// Positions of the turns that couple to the one after them, **in a timeline's
/// `Normal` subsequence** — not raw turn indices. Build with
/// [`over_normals`].
///
/// `pos ∈ set` reads as "the Normal turn after position `pos` is its tool
/// response".
pub type Couplings = AHashSet<u32>;

/// Project persisted coupling records — which name raw turn indices — into
/// positions over `normals`, the timeline's `Normal` turns in chronological
/// order.
///
/// This indirection is load-bearing. Summary turns are recorded into the *same*
/// index space as conversation turns and the summariser runs asynchronously, so
/// a leaf can land between a tool call and its response:
///
/// ```text
///   #2 NORMAL (calls a tool)   #3 SoT←[2]   #4 NORMAL (<tool_response>)
/// ```
///
/// The response is `#4`, not `#3`. "The response is the next turn" therefore
/// holds only over the `Normal` subsequence — against raw indices it would
/// couple a call turn to its own summary. Working in positions also keeps the
/// record writable before the response turn exists: the writer names only the
/// call turn, so it never has to predict an index the summariser might take
/// first.
///
/// A coupling naming a turn that is not a `Normal` (or is absent) is dropped —
/// with a warning, because that is a *lost exchange*, not a no-op (the usual
/// cause is a caller that coupled a non-`Normal` index, e.g. a summary turn
/// mistaken for the call turn).
pub fn over_normals(raw: &AHashSet<u32>, normals: &[TurnIndex]) -> Couplings {
    // Build the turn-index → position map once (O(normals + couplings)) rather
    // than scanning `normals` per coupling (O(normals × couplings)).
    let pos_of: AHashMap<u32, u32> = normals
        .iter()
        .enumerate()
        .map(|(pos, n)| (n.0, pos as u32))
        .collect();
    let mut out = Couplings::default();
    for &from in raw {
        match pos_of.get(&from) {
            Some(&pos) => {
                out.insert(pos);
            }
            None => tracing::warn!(
                target: "candle_conversation::summariser",
                from,
                n_normals = normals.len(),
                "coupling names a non-Normal turn — dropping (its exchange will not form)",
            ),
        }
    }
    out
}

/// Whether turn `i` runs on into turn `i + 1` — the single rule this module is
/// built from.
///
/// A coupling is only ever written once its response turn is certain, so a
/// coupling whose successor is not yet recorded means the output is still in
/// flight, not that the run ended.
fn continues_into(couplings: &Couplings, total: usize, i: usize) -> bool {
    couplings.contains(&(i as u32)) && i + 1 < total
}

/// Group a timeline's `total` `Normal` turns into exchanges. The returned ranges
/// are contiguous, non-overlapping, and cover every turn, so each turn belongs to
/// exactly one exchange.
pub fn exchanges(couplings: &Couplings, total: usize) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < total {
        let mut end = start;
        while continues_into(couplings, total, end) {
            end += 1;
        }
        out.push(start..end + 1);
        start = end + 1;
    }
    out
}

/// The exchange containing turn `i`, expanded in **both** directions.
///
/// This is what provenance uses: a scan hit on any member — the question, a tool
/// response, the final answer — pulls in the whole run, so the model never sees
/// half a round-trip. Walking back and walking forward read the same rule from
/// the two ends, so this agrees with [`exchanges`] by construction.
///
/// Returns an empty range if `i` is out of bounds.
pub fn exchange_of(couplings: &Couplings, total: usize, i: usize) -> Range<usize> {
    if i >= total {
        return 0..0;
    }
    let mut start = i;
    while start > 0 && continues_into(couplings, total, start - 1) {
        start -= 1;
    }
    let mut end = i;
    while continues_into(couplings, total, end) {
        end += 1;
    }
    start..end + 1
}

/// Whether an exchange is **settled** — a later `Normal` turn already exists
/// beyond it, so it can never grow and its leaf is safe to seal.
///
/// This is deliberately a *frontier* test (`range.end < total`), not a
/// coupling-presence test, and that is what makes sealing race-free. A coupling
/// is written by the caller only after the tool returns — i.e. *after* the call
/// turn has already sealed — so "does the last turn carry a coupling yet" cannot
/// distinguish "not a tool call" from "tool still running", and a leaf sealed on
/// that basis freezes over half an exchange (the fabrication bug).
///
/// The timing guarantee behind the frontier test: the caller writes the coupling
/// *before* it submits the response turn, and the response precedes any later
/// turn. So by the time **any** turn exists past an exchange, that exchange's
/// couplings are all already in the log — the run is both complete (a non-member
/// turn has started) and fully grouped. The newest exchange is left unsealed
/// until the next turn arrives; that is correct, since the live frontier is
/// anchored verbatim by recency and needs no summary yet.
///
/// `finalize` overrides the frontier test: when the conversation can take no more
/// turns (it has been archived), the newest exchange will never gain a successor,
/// so waiting would leave it permanently unsummarised — a hole in the peak cover.
/// A finalising pass seals it too. It is safe because an archived conversation is
/// terminal: no in-flight tool call can still extend the frontier.
pub fn is_settled(range: &Range<usize>, total: usize, finalize: bool) -> bool {
    !range.is_empty() && (finalize || range.end < total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn couplings(from: &[u32]) -> Couplings {
        from.iter().copied().collect()
    }

    fn raw(from: &[u32]) -> AHashSet<u32> {
        from.iter().copied().collect()
    }

    /// The regression this whole indirection exists for. A summariser leaf lands
    /// between a tool call and its response, so raw indices are NOT contiguous:
    ///
    /// ```text
    ///   #0 NORMAL  #1 SoT←[0]  #2 NORMAL(calls)  #3 SoT←[2]  #4 NORMAL(response)
    /// ```
    ///
    /// The record couples raw turn 2. Read literally as "2 + 1" that is the SoT
    /// #3 — the call turn's own summary. Over the Normal subsequence it is
    /// position 1, whose successor is #4: the real tool response.
    #[test]
    fn couplings_project_past_interleaved_summary_turns() {
        let normals = [TurnIndex(0), TurnIndex(2), TurnIndex(4)];
        let projected = over_normals(&raw(&[2]), &normals);
        assert_eq!(projected, couplings(&[1]), "raw #2 is Normal position 1");
        // Position 1 (#2) runs into position 2 (#4) — the response, not the SoT.
        assert_eq!(exchanges(&projected, normals.len()), vec![0..1, 1..3]);
    }

    /// A coupling naming a turn that is not a Normal can only be noise — it must
    /// not shift every later position by silently mis-projecting.
    #[test]
    fn couplings_naming_a_non_normal_turn_are_dropped() {
        let normals = [TurnIndex(0), TurnIndex(2), TurnIndex(4)];
        assert_eq!(over_normals(&raw(&[3]), &normals), couplings(&[]));
        assert_eq!(over_normals(&raw(&[99]), &normals), couplings(&[]));
    }

    #[test]
    fn uncoupled_turns_are_their_own_exchanges() {
        let c = couplings(&[]);
        assert_eq!(exchanges(&c, 3), vec![0..1, 1..2, 2..3]);
    }

    /// The observed case: `what time is it?` + its `<tool_response>` are ONE
    /// exchange, not two half-exchanges.
    #[test]
    fn a_tool_round_trip_is_one_exchange() {
        // Turn 1 called a tool; turn 2 carries its output.
        let c = couplings(&[1]);
        assert_eq!(exchanges(&c, 4), vec![0..1, 1..3, 3..4]);
    }

    /// Multiple tool calls in one assistant turn still produce a single response
    /// turn (results are batched into one message), so the run is unchanged — a
    /// coupling needs no count. A chain is just consecutive couplings.
    #[test]
    fn a_chain_of_round_trips_is_one_exchange() {
        let c = couplings(&[0, 1, 2]);
        assert_eq!(exchanges(&c, 4), vec![0..4]);
    }

    /// Capture mode and malformed calls emit no coupling, so the next real user
    /// message is never swallowed — the failure the stencil-derived flag would
    /// have caused.
    #[test]
    fn an_uncoupled_call_does_not_swallow_the_next_turn() {
        let c = couplings(&[]);
        assert_eq!(exchanges(&c, 2), vec![0..1, 1..2]);
    }

    #[test]
    fn no_turns_means_no_exchanges() {
        assert!(exchanges(&couplings(&[]), 0).is_empty());
    }

    /// The provenance requirement: a hit on EITHER end pulls in both.
    #[test]
    fn expands_from_either_member_of_a_pair() {
        let c = couplings(&[1]);
        assert_eq!(exchange_of(&c, 3, 1), 1..3, "hit the call turn");
        assert_eq!(exchange_of(&c, 3, 2), 1..3, "hit the response turn");
        assert_eq!(exchange_of(&c, 3, 0), 0..1, "uncoupled turn stands alone");
    }

    /// …and from anywhere in the middle of a longer chain.
    #[test]
    fn expands_from_the_middle_of_a_chain() {
        let c = couplings(&[1, 2]);
        for i in 1..=3 {
            assert_eq!(exchange_of(&c, 4, i), 1..4, "expanding from {i}");
        }
    }

    #[test]
    fn expand_out_of_bounds_is_empty() {
        assert_eq!(exchange_of(&couplings(&[]), 0, 0), 0..0);
        assert_eq!(exchange_of(&couplings(&[]), 1, 5), 0..0);
    }

    /// `exchange_of` and `exchanges` are two readings of one rule — they must
    /// never disagree, or provenance would pull a different set than the
    /// summariser sealed.
    #[test]
    fn expansion_agrees_with_grouping_everywhere() {
        let c = couplings(&[1, 2, 4]);
        for group in exchanges(&c, 7) {
            for i in group.clone() {
                assert_eq!(exchange_of(&c, 7, i), group, "member {i} of {group:?}");
            }
        }
    }

    /// Every turn lands in exactly one exchange, and the cover is contiguous.
    #[test]
    fn exchanges_partition_every_turn() {
        let c = couplings(&[0, 3, 4]);
        let groups = exchanges(&c, 6);
        let mut next = 0;
        for g in &groups {
            assert_eq!(g.start, next, "gap or overlap before {g:?}");
            assert!(g.end > g.start, "empty exchange {g:?}");
            next = g.end;
        }
        assert_eq!(next, 6, "cover is short");
    }

    /// An exchange is settled once a later turn exists beyond it; the newest
    /// exchange (the frontier) is not, regardless of its couplings.
    #[test]
    fn only_a_non_frontier_exchange_is_settled() {
        // 3 turns total. An exchange ending before turn 2 has a successor…
        assert!(is_settled(&(0..1), 3, false), "a later turn exists");
        assert!(is_settled(&(1..2), 3, false), "a later turn exists");
        // …the one ending at the frontier does not.
        assert!(!is_settled(&(2..3), 3, false), "nothing after it yet");
    }

    /// The fabrication guard, at the predicate level: a completed round-trip that
    /// is still the whole conversation so far is the frontier — NOT settled — so
    /// its leaf waits for the next turn rather than sealing while live. Coupling
    /// presence is irrelevant to this decision.
    #[test]
    fn a_frontier_round_trip_is_not_settled_until_a_later_turn_lands() {
        let c = couplings(&[0]);
        assert_eq!(exchanges(&c, 2), vec![0..2]);
        assert!(
            !is_settled(&(0..2), 2, false),
            "the round-trip is the frontier"
        );
        // A third turn lands → the round-trip now has a successor → settled.
        assert!(is_settled(&(0..2), 3, false));
    }

    /// Finalising (the conversation is archived, so no successor will ever come)
    /// seals the frontier that would otherwise wait forever — but never an empty
    /// range.
    #[test]
    fn finalize_seals_the_frontier_but_not_an_empty_range() {
        assert!(!is_settled(&(2..3), 3, false), "live: frontier waits");
        assert!(is_settled(&(2..3), 3, true), "archived: frontier seals");
        assert!(!is_settled(&(0..0), 5, true), "empty range never seals");
    }

    #[test]
    fn an_empty_range_is_not_settled() {
        assert!(!is_settled(&(0..0), 5, false));
    }
}
