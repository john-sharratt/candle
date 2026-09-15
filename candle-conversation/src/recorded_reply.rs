//! Replaying a recorded turn through the decode loop.
//!
//! A turn submitted with the ids an earlier run of it sealed
//! ([`TurnOptions::recorded_turn`](crate::TurnOptions::recorded_turn)) decodes
//! as a live turn does — one token per step, the page cut at the prefill/decode
//! boundary and at every break token, the reasoning boundary, the mid-decode
//! reprojections — but commits the recorded id at each step in place of the
//! sampled one. What it seals is the recorded turn in everything the substrate
//! keeps: its ids, and the index pages its decode cut them into.
//!
//! Prefilling the same text cannot do that. It reproduces the ids at best — a
//! decode's non-canonical token splits re-tokenize canonically — and it cuts
//! none of the pages a decode does, so the turn records no reasoning span and
//! every later projection injects its reasoning where the recorded turn's was
//! windowed out.
//!
//! The turn's stencils run as they did live, because the static runs they play
//! are part of the recording and are forwarded without the page cut a decoded
//! token earns. Each run is checked against the recording ([`departure`]); a
//! turn whose steering leaves it fails rather than sealing a turn that never
//! happened.

/// The ids a replay of `grid` decodes: what follows this submission's
/// `prefill`, less the `tail` the scheduler writes after the decode.
///
/// `Err` when the recorded grid does not begin with `prefill` or does not end
/// with `tail`: the recorded turn was submitted differently, and forcing its
/// reply onto this prefill would seal a turn that never existed.
pub(crate) fn recorded_reply(
    grid: &[u32],
    prefill: &[u32],
    tail: &[u32],
) -> Result<Vec<u32>, String> {
    if !grid.starts_with(prefill) {
        let at = grid
            .iter()
            .zip(prefill)
            .position(|(a, b)| a != b)
            .unwrap_or(grid.len());
        return Err(format!(
            "the recorded turn parts from this submission's {}-id prefill at id {at}",
            prefill.len()
        ));
    }
    let body = &grid[prefill.len()..];
    if !body.ends_with(tail) {
        return Err(format!(
            "the recorded turn does not end with this submission's {}-id closing tail",
            tail.len()
        ));
    }
    Ok(body[..body.len() - tail.len()].to_vec())
}

/// A recorded reply being decoded again, and where its steering last refused
/// one of its ids.
#[derive(Debug, Clone)]
pub(crate) struct Replay {
    ids: Vec<u32>,
    refused_at: Option<usize>,
}

impl Replay {
    pub(crate) fn new(ids: Vec<u32>) -> Self {
        Self {
            ids,
            refused_at: None,
        }
    }

    /// The recorded reply.
    pub(crate) fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Note that steering refused the recorded id at `at`, and say whether the
    /// replay has left its recording.
    ///
    /// **Once is what the recording holds.** A think-steer tree drops the
    /// model's own `</think>` and plays its own close as a static run, so the
    /// recorded close is refused at its position and then played there — the
    /// run is checked by [`departure`]. Refused a second time at the same
    /// position, the steering will not play what the recording holds, and
    /// forcing it again would only be refused again.
    pub(crate) fn refuse(&mut self, at: usize) -> bool {
        let again = self.refused_at == Some(at);
        self.refused_at = Some(at);
        again
    }
}

/// The id a replayed turn commits at step `at`: the recording's, then `eos`
/// once the recording is spent, which ends the turn where the recording did.
pub(crate) fn replayed_step(reply: &[u32], at: usize, eos: u32) -> u32 {
    reply.get(at).copied().unwrap_or(eos)
}

/// Where a static run a stencil plays from step `at` departs from the
/// recording: the offset of its first id that is not the recorded one, or —
/// past the recording's end — that does not end the turn. `None` when it
/// follows the recording throughout.
pub(crate) fn departure(
    reply: &[u32],
    at: usize,
    run: &[u32],
    is_eos: impl Fn(u32) -> bool,
) -> Option<usize> {
    run.iter()
        .enumerate()
        .position(|(k, &t)| match reply.get(at + k) {
            Some(&r) => r != t,
            None => !is_eos(t),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reply_is_what_lies_between_the_prefill_and_the_tail() {
        assert_eq!(
            recorded_reply(&[1, 2, 3, 4, 5, 9, 8], &[1, 2, 3], &[9, 8]),
            Ok(vec![4, 5])
        );
        assert_eq!(recorded_reply(&[1, 2, 9], &[1, 2], &[9]), Ok(vec![]));
    }

    #[test]
    fn a_grid_submitted_with_another_prefill_is_refused() {
        assert_eq!(
            recorded_reply(&[1, 7, 3, 4], &[1, 2, 3], &[]),
            Err("the recorded turn parts from this submission's 3-id prefill at id 1".to_string())
        );
        assert_eq!(
            recorded_reply(&[1, 2], &[1, 2, 3], &[]),
            Err("the recorded turn parts from this submission's 3-id prefill at id 2".to_string())
        );
    }

    #[test]
    fn a_grid_closed_by_another_tail_is_refused() {
        assert_eq!(
            recorded_reply(&[1, 2, 3, 4], &[1], &[9]),
            Err(
                "the recorded turn does not end with this submission's 1-id closing tail"
                    .to_string()
            )
        );
    }

    #[test]
    fn a_run_that_plays_the_recording_does_not_depart() {
        let eos = |t: u32| t == 99;
        assert_eq!(departure(&[4, 5, 6], 1, &[5, 6], eos), None);
        // A closing run ends the turn exactly where the recording does.
        assert_eq!(departure(&[4, 5], 1, &[5, 99], eos), None);
    }

    #[test]
    fn a_run_departs_at_its_first_unrecorded_id() {
        let eos = |t: u32| t == 99;
        assert_eq!(departure(&[4, 5, 6], 0, &[4, 7, 6], eos), Some(1));
        assert_eq!(departure(&[4], 0, &[4, 5], eos), Some(1));
    }

    #[test]
    fn a_replay_leaves_its_recording_only_when_refused_twice_at_one_id() {
        let mut replay = Replay::new(vec![4, 5, 6]);
        assert!(
            !replay.refuse(2),
            "the steering's own close is refused once"
        );
        assert!(replay.refuse(2), "refused again at the same id");
        assert!(!replay.refuse(3), "a later id starts over");
        assert_eq!(replay.ids(), &[4, 5, 6]);
    }

    #[test]
    fn a_spent_recording_ends_the_turn() {
        let reply = [4, 5];
        assert_eq!(replayed_step(&reply, 0, 99), 4);
        assert_eq!(replayed_step(&reply, 1, 99), 5);
        assert_eq!(replayed_step(&reply, 2, 99), 99);
        assert_eq!(replayed_step(&[], 0, 99), 99);
    }
}
