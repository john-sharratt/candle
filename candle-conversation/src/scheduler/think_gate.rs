//! Whether a decode boundary reprojects, given where the decode stands against
//! the turn's think block and the turn's reprojection (`ReprojectionMode`,
//! over `SequenceConfig::reproject_in_thinking`).
//!
//! Every boundary the rule fires reprojects for a turn that reprojects in
//! full. For one held in thinking, a boundary inside the block is passed over,
//! so the block decodes on the projection it opened on, and the committed
//! `</think>` reprojects, so the answer starts on a selection the whole
//! thought informed. A turn that selects only at its opening reprojects at no
//! decode boundary: its opening boundary (`ReprojectionPolicy::opening_chunk`)
//! is the one selection the question drives.

/// Where a committed token leaves the decode against the think block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Thinking {
    /// Outside any think block.
    Outside,
    /// Inside one, its opening tag included.
    Inside,
    /// The token closed the block.
    Closed,
}

impl Thinking {
    /// Where `token` leaves the decode: `in_segment` is the sampler's
    /// think-block flag after it, `close` the `</think>` id (`< 0` when the
    /// dialect has none).
    pub(super) fn after(token: u32, in_segment: bool, close: i32) -> Thinking {
        if in_segment {
            Thinking::Inside
        } else if u32::try_from(close).is_ok_and(|c| c == token) {
            Thinking::Closed
        } else {
            Thinking::Outside
        }
    }
}

/// Which decode boundaries a turn reprojects at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeReprojection {
    /// Every boundary the rule fires, inside the think block as out.
    Every,
    /// None inside the think block; its close, and every boundary after.
    HoldThinking,
    /// None: the turn decodes on what its opening selected.
    Never,
}

/// Whether a reprojection follows the token: `fired` is the boundary rule's
/// verdict, `when` the turn's decode reprojection.
pub(super) fn reprojects(fired: bool, thinking: Thinking, when: DecodeReprojection) -> bool {
    match (when, thinking) {
        (DecodeReprojection::Never, _) => false,
        (DecodeReprojection::Every, _) => fired,
        (DecodeReprojection::HoldThinking, Thinking::Inside) => false,
        (DecodeReprojection::HoldThinking, Thinking::Closed) => true,
        (DecodeReprojection::HoldThinking, Thinking::Outside) => fired,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLOSE: i32 = 151_668;
    const ALL: [Thinking; 3] = [Thinking::Outside, Thinking::Inside, Thinking::Closed];

    #[test]
    fn the_sampler_flag_and_the_close_id_place_the_token() {
        assert_eq!(Thinking::after(7, true, CLOSE), Thinking::Inside);
        assert_eq!(
            Thinking::after(CLOSE as u32, false, CLOSE),
            Thinking::Closed
        );
        assert_eq!(Thinking::after(7, false, CLOSE), Thinking::Outside);
        assert_eq!(
            Thinking::after(CLOSE as u32, false, -1),
            Thinking::Outside,
            "a dialect with no close id has no block to close"
        );
    }

    /// **In full, the rule alone decides**, inside the block or out.
    #[test]
    fn reprojecting_in_full_every_fired_boundary_reprojects() {
        for thinking in ALL {
            assert!(reprojects(true, thinking, DecodeReprojection::Every));
            assert!(!reprojects(false, thinking, DecodeReprojection::Every));
        }
    }

    /// **Held, the block keeps its projection and its close reprojects once.**
    #[test]
    fn held_the_block_holds_and_its_close_reprojects() {
        let held = DecodeReprojection::HoldThinking;
        assert!(!reprojects(true, Thinking::Inside, held));
        assert!(reprojects(false, Thinking::Closed, held));
        assert!(reprojects(true, Thinking::Closed, held));
        assert!(reprojects(true, Thinking::Outside, held));
        assert!(!reprojects(false, Thinking::Outside, held));
    }

    /// **At the opening only, no decode boundary reprojects** — not the rule's,
    /// not the think block's close.
    #[test]
    fn an_opening_only_turn_reprojects_at_no_decode_boundary() {
        for thinking in ALL {
            for fired in [true, false] {
                assert!(!reprojects(fired, thinking, DecodeReprojection::Never));
            }
        }
    }
}
