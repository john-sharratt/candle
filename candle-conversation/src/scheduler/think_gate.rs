//! Whether a boundary reprojects, given where the decode stands against the
//! turn's think block (`SequenceConfig::reproject_in_thinking`).
//!
//! With reprojection in thinking on, every boundary the rule fires
//! reprojects. With it off, a boundary inside the block is passed over, so the
//! block decodes on the projection it opened on, and the committed `</think>`
//! reprojects, so the answer starts on a selection the whole thought informed.

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

/// Whether a reprojection follows the token: `fired` is the boundary rule's
/// verdict, `in_thinking` whether boundaries inside the block reproject.
pub(super) fn reprojects(fired: bool, thinking: Thinking, in_thinking: bool) -> bool {
    match thinking {
        _ if in_thinking => fired,
        Thinking::Inside => false,
        Thinking::Closed => true,
        Thinking::Outside => fired,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLOSE: i32 = 151_668;

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

    /// **On, the rule alone decides**, inside the block or out.
    #[test]
    fn with_reprojection_in_thinking_every_fired_boundary_reprojects() {
        for thinking in [Thinking::Outside, Thinking::Inside, Thinking::Closed] {
            assert!(reprojects(true, thinking, true));
            assert!(!reprojects(false, thinking, true));
        }
    }

    /// **Off, the block holds its projection and its close reprojects once.**
    #[test]
    fn without_it_the_block_holds_and_its_close_reprojects() {
        assert!(!reprojects(true, Thinking::Inside, false));
        assert!(reprojects(false, Thinking::Closed, false));
        assert!(reprojects(true, Thinking::Closed, false));
        assert!(reprojects(true, Thinking::Outside, false));
        assert!(!reprojects(false, Thinking::Outside, false));
    }
}
