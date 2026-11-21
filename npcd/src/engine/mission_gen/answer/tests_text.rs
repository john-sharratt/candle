//! Tests the loop, kept-field and link rules that govern how generated text is checked.
//!
//! A field that goes round in a loop is refused, but a field that ends in a loop is kept up to it when what came before is a field. A repeated sentence in ordinary prose is not a loop, so nothing is cut. A long field is kept to its cap ending on a whole sentence, and one with no sentence end inside the cap is refused. A markdown link reads as its words when a quote is checked.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_event::tests::HAPPENS;

    #[test]
    fn a_field_that_goes_round_in_a_loop_is_refused() {
        assert!(loops(
            "She is Zen, and nothing matters. She is Zen, and everything matters. She is Zen, \
             and neither is true."
        ));
        assert!(!loops(HAPPENS));
    }

    /// **A field that ends in a loop is kept up to it**, when what came
    /// before is a field; one that is all loop is still refused.
    #[test]
    fn a_field_is_kept_up_to_its_loop() {
        let tail = "The year 3012 is in the Contested Cities. The draft repeats its first \
                    paragraph. I am done. I will output. I am done. I am ready. I am done. \
                    I will output.";
        assert_eq!(
            before_the_loop(tail),
            "The year 3012 is in the Contested Cities. The draft repeats its first paragraph."
        );
        let args: serde_json::Map<String, serde_json::Value> = serde_json::from_value(
            serde_json::json!({ "checked": tail, "all": "She is Zen, and \
             nothing. She is Zen, and all. She is Zen, and none." }),
        )
        .unwrap();
        let f = Fields(&args);
        assert_eq!(
            f.words("checked", 5, 400, "what you checked").unwrap(),
            "The year 3012 is in the Contested Cities. The draft repeats its first paragraph."
        );
        assert!(f
            .words("all", 5, 400, "what happens")
            .unwrap_err()
            .contains("repeats itself"));
        // Said twice is prose, not a loop: nothing is cut.
        let twice = "The draft repeats a line. The draft repeats a name.";
        assert_eq!(before_the_loop(twice), twice);
    }

    /// **A long field kept to its cap ends on a whole sentence**; one with no
    /// sentence end inside the cap is refused as before.
    #[test]
    fn a_long_field_is_kept_to_its_last_whole_sentence() {
        let long = "It is set in 3012. The Houses fit. The voice is right. And then it goes on";
        assert_eq!(within_words(long, 9), "It is set in 3012. The Houses fit.");
        let args: serde_json::Map<String, serde_json::Value> = serde_json::from_value(
            serde_json::json!({ "checked": long, "run": "one run on sentence with no end at all" }),
        )
        .unwrap();
        let f = Fields(&args);
        assert_eq!(
            f.words_kept("checked", 3, 9, "what you checked").unwrap(),
            "It is set in 3012. The Houses fit."
        );
        assert!(f
            .words_kept("run", 3, 4, "x")
            .unwrap_err()
            .contains("must be under 4"));
        assert_eq!(
            f.words_kept("checked", 3, 400, "x").unwrap(),
            long,
            "under the cap"
        );
    }

    #[test]
    fn a_link_reads_as_its_words_when_a_quote_is_checked() {
        assert_eq!(
            unlink("mobile fortress-[towers](/tower) behind [walls](/w) and [a] note"),
            "mobile fortress-towers behind walls and [a] note"
        );
        assert!(contains(
            "house them in mobile fortress-[towers](/tower) behind dimensional walls",
            "house them in mobile fortress-towers behind dimensional walls"
        ));
    }
}
