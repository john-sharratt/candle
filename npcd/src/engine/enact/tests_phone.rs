//! Tests of the acts made on the phone: promises made across it, and reading a group there.
//!
//! A promise made on the phone stands for the party of the pair's thread, wherever they are, and no room
//! hears it. A reminder can be recalled only for a promise that was really made. Reading a group on the
//! phone marks what was read, so nothing is left unread to remind the character of it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use serde_json::json;

    use super::super::tests_acts::tests::{act, waste};

    #[test]
    fn a_promise_stands_and_can_then_be_recalled_but_only_the_real_one() {
        let h = waste();
        assert!(perform(
            &h,
            "c1",
            &act(
                "promise",
                json!({"to":"Soren","what":"the eastern sweep","by":"dusk"})
            )
        )
        .happened());

        // Soren reminding Wren of it works; reminding of something never
        // promised does not.
        let good = perform(
            &h,
            "c2",
            &act("remind", json!({"who":"Wren","which":"the eastern sweep"})),
        );
        assert!(good.happened(), "{good:?}");
        let bad = perform(
            &h,
            "c2",
            &act("remind", json!({"who":"Wren","which":"the western sweep"})),
        );
        assert!(!bad.happened(), "a promise nobody made was recalled");
    }

    /// **A promise is made on the phone too** (§7.1): to the party of the
    /// pair's thread, wherever they are, kept in the ledger, and heard by no
    /// room.
    #[test]
    fn a_promise_on_the_phone_stands_without_the_party_in_the_room() {
        let h = waste();
        h.with_sim(|s| {
            let id = s
                .threads
                .reach(&Who::new("c1", "Wren"), &Who::new("u_mira", "Mira"));
            assert!(!id.is_empty());
        });
        let out = perform(
            &h,
            "c1",
            &act(
                "promise",
                json!({"to":"Mira","what":"the garden report","by":"Friday"}),
            ),
        );
        assert_eq!(
            out,
            Outcome::Did(
                "You promise Mira: the garden report, by Friday. It stands until you keep it."
                    .into()
            )
        );
        assert_eq!(h.sim(|s| s.owed_to("Mira", "Wren")), ["the garden report"]);
        let refused = perform(
            &h,
            "c1",
            &act("promise", json!({"to":"Nobody","what":"x","by":"y"})),
        );
        assert!(!refused.happened(), "{refused:?}");
    }

    /// **`read` reads a group on the phone**, by the name the character calls
    /// it, or every one with something unread as its messages — marked read
    /// as it is handed over, so nothing is left unread to remind it of — and
    /// shows the reader its own last line there.
    #[test]
    fn read_reads_a_group_on_the_phone() {
        let h = waste();
        h.with_sim(|s| {
            crate::sim::seed::issue_handset(s, "c1", "Wren");
            let crew = s.threads.open_group(
                "the crew",
                &[Who::new("c1", "Wren"), Who::new("c2", "Soren")],
            );
            s.threads
                .send_on(&crew, "c1", Said::words("Back at noon."))
                .unwrap();
            s.threads
                .send_on(&crew, "c2", Said::words("How much ore is left?"))
                .unwrap();
            let ops = s
                .threads
                .open_group("ops", &[Who::new("c1", "Wren"), Who::new("c3", "Kit")]);
            s.threads
                .send_on(&ops, "c3", Said::words("Hello."))
                .unwrap();
        });
        let out = perform(&h, "c1", &act("read", json!({"what": "the crew"})));
        assert_eq!(
            out.line().unwrap(),
            "On the crew: Soren: How much ore is left? (you last said, just now: Back at noon.)\n\
             To answer, `reply` on the conversation it was said on, by that name."
        );
        let again = perform(&h, "c1", &act("read", json!({"what": "The Crew"})));
        assert!(!again.happened(), "{again:?}");
        assert_eq!(
            again.line().unwrap(),
            "You have read everything on the crew. Nothing has been said since."
        );

        let all = perform(&h, "c1", &act("read", json!({"what": "messages"})));
        assert_eq!(
            all.line().unwrap(),
            "On ops: Kit: Hello.\nTo answer, `reply` on the conversation it was said on, by \
             that name."
        );
        assert!(
            h.sim(|s| s.threads.unread_threads("c1").is_empty()),
            "nothing left unread"
        );
        let none = perform(&h, "c1", &act("read", json!({"what": "messages"})));
        assert_eq!(none.line().unwrap(), "Nothing on your phone is unread.");
    }
}
