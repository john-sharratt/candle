//! Checks where a console's stream starts: returning, from the future,
//! unparseable, and for a character that has never ticked.
//!
//! A returning console resumes from the tick after the one it holds. A tick from
//! the future falls back to a fresh attachment rather than a silent connection,
//! since a daemon restart resets tick numbering. Junk in the query string, and a
//! character that has never ticked, also start fresh, rather than underflowing
//! into a replay from zero.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::resume_from;

    /// The ordinary case: nobody names a tick, so the attachment starts at the
    /// newest one and does not replay the character's whole afternoon.
    #[test]
    fn a_fresh_attachment_starts_at_the_newest_tick() {
        assert_eq!(resume_from(None, 400), 400);
    }

    /// **The point of the parameter.** A console that has already read up to
    /// tick 380 gets 381 onwards, which is what it missed while its reader was
    /// looking at another page.
    #[test]
    fn a_returning_console_resumes_where_it_left_off() {
        assert_eq!(resume_from(Some("380"), 400), 380);
    }

    /// **A tick from the future must not stall the stream.** Unclamped, `seen`
    /// would start above every tick the character has taken and the loop would
    /// send nothing at all — a connection that opens and stays silent, which is
    /// indistinguishable from a mind that has stopped thinking. A daemon
    /// restart resets tick numbering, so a console outliving one arrives here
    /// with exactly that.
    #[test]
    fn a_tick_from_the_future_falls_back_to_a_fresh_attachment() {
        assert_eq!(resume_from(Some("9000"), 400), 400);
    }

    /// Junk in the query string is not a reason to serve a broken stream.
    #[test]
    fn an_unparseable_since_is_a_fresh_attachment() {
        for junk in ["", "abc", "-5", "3.5", "١٢٣"] {
            assert_eq!(resume_from(Some(junk), 400), 400, "since={junk:?}");
        }
    }

    /// A character that has never ticked has no history to resume into, and
    /// asking for one must not underflow into replaying from zero for ever.
    #[test]
    fn a_character_that_has_never_ticked_resumes_at_nothing() {
        assert_eq!(resume_from(Some("12"), 0), 0);
        assert_eq!(resume_from(None, 0), 0);
    }
}
