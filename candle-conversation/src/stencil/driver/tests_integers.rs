//! Tests of integer arguments: which digits and signs the grammar offers at each
//! step, and how the number ends.
//!
//! An unsigned integer offers no sign. A minimum of one offers no zero as its first
//! digit, and a one-digit maximum offers no digit past it, so the grammar never offers
//! a value the schema forbids. A leading zero ends the number, as JSON requires. The
//! lookahead tests cover a number merged with its delimiter, which heals, and one that
//! arrives as its own token, which does not.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::stencil::vocab::TestVocab;

    use super::super::tests_strings::tests::{driver_at_first_value, follow, tool_tree};

    /// A driver stepped to an integer value's first masked choice.
    pub(in super::super) fn at_integer(catalog: &str) -> (StencilDriver, AllowedSet) {
        let mut driver = StencilDriver::new(tool_tree(catalog));
        assert!(matches!(driver.step(), StepMask::Prefill(_)));
        let StepMask::Branch(set) = driver.step() else {
            panic!("an integer value begins with a masked choice");
        };
        (driver, set)
    }

    /// **A `minimum` of zero — every `u32` in the catalog — offers no sign**, so
    /// a page, a line number or a count cannot be decoded negative.
    #[test]
    fn an_unsigned_integer_is_offered_no_sign() {
        let (_, set) = at_integer(
            r#"[{"name":"read","params":[
                 {"name":"page","type":"integer","minimum":0.0,"required":true}]}]"#,
        );
        assert!(!set.contains(b'-' as TokenId), "{set:?}");
        for d in b'0'..=b'9' {
            assert!(set.contains(d as TokenId), "digit {}", d as char);
        }
    }

    /// **A `minimum` of one offers no `0`**: a one-based page has no page 0, so
    /// the grammar never offers the digit that would write it, while every
    /// other first digit — and so 10, 20, 100 — stays open.
    #[test]
    fn a_one_based_integer_cannot_begin_with_zero() {
        let (mut driver, set) = at_integer(
            r#"[{"name":"read","params":[
                 {"name":"page","type":"integer","minimum":1.0,"required":true}]}]"#,
        );
        assert!(!set.contains(b'0' as TokenId), "{set:?}");
        assert!(!set.contains(b'-' as TokenId), "{set:?}");
        for d in b'1'..=b'9' {
            assert!(set.contains(d as TokenId), "digit {}", d as char);
        }
        // A zero after the first digit is an ordinary digit: page 10.
        assert_eq!(driver.accept(b'1' as TokenId, b"1"), Healed::No);
        let StepMask::Branch(next) = driver.step() else {
            panic!("a second digit or the end is a masked choice");
        };
        assert!(next.contains(b'0' as TokenId), "{next:?}");
    }

    /// **A one-digit maximum offers no digit past it**: `context` 0–5 cannot
    /// be written 6. Live, a worker asked for six lines of context wrote
    /// `"context": 6` and spent a round on the refusal.
    #[test]
    fn a_one_digit_maximum_offers_no_digit_past_it() {
        let (_, set) = at_integer(
            r#"[{"name":"grep","params":[
                 {"name":"context","type":"integer","minimum":0.0,"maximum":5.0,
                  "max_tokens":1,"required":true}]}]"#,
        );
        for d in b'0'..=b'5' {
            assert!(set.contains(d as TokenId), "digit {}", d as char);
        }
        for d in b'6'..=b'9' {
            assert!(!set.contains(d as TokenId), "digit {}", d as char);
        }
    }

    /// **A leading zero ends the number**, as JSON requires — `07` is not a
    /// number — so after it only what follows the value can be written.
    #[test]
    fn a_leading_zero_ends_the_number() {
        let (mut driver, _) = at_integer(
            r#"[{"name":"read","params":[
                 {"name":"page","type":"integer","minimum":0.0,"required":true}]}]"#,
        );
        assert_eq!(driver.accept(b'0' as TokenId, b"0"), Healed::No);
        let after = driver.step();
        let allowed = match &after {
            StepMask::Branch(set) => set.clone(),
            StepMask::Prefill(_) => return, // the close is the only way on
            other => panic!("unexpected {other:?}"),
        };
        assert!(!allowed.contains(b'5' as TokenId), "{allowed:?}");
    }

    /// **Past the first digit, each step is another digit or the value's end**,
    /// and the end carries what follows it: here the call's close.
    #[test]
    fn an_integer_writes_its_digits_then_what_follows() {
        let v = TestVocab::new();
        let tree = tool_tree(
            r#"[{"name":"read","params":[
                 {"name":"page","type":"integer","minimum":0.0,"required":true}]}]"#,
        );
        let target =
            "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"page\": 42}}\n</tool_call>";
        assert_eq!(follow(tree, target, &v), target);
    }

    /// A `number` is still a free, lookahead-terminated value — an integer is
    /// not (it is written under the mask) — so the lookahead mechanics are
    /// exercised through one.
    pub(in super::super) const NUM_ONLY: &str = r#"[{"name":"wait","params":[
        {"name":"secs","type":"number","required":true}]}]"#;

    #[test]
    fn lookahead_value_merged_with_delimiter() {
        // Number value: lookahead terminator.  `30}` is one token — the `30`
        // is the value (consumed=2), the `}` is the lookahead delimiter.
        let v = TestVocab::new().with_special("30}", 300);
        let mut d = driver_at_first_value(NUM_ONLY, &v);
        assert_eq!(
            d.accept(300, b"30}"),
            Healed::Rewrite {
                bytes: b"30".to_vec()
            }
        );
    }

    #[test]
    fn lookahead_clean_delimiter_does_not_heal() {
        // The delimiter arrives as its own token: a clean lookahead (consumed=0),
        // handled by push-back — not a heal.
        let v = TestVocab::new();
        let mut d = driver_at_first_value(NUM_ONLY, &v);
        assert_eq!(d.accept(b'3' as TokenId, b"3"), Healed::No);
        assert_eq!(d.accept(b'0' as TokenId, b"0"), Healed::No);
        assert_eq!(d.accept(b'}' as TokenId, b"}"), Healed::No); // whole token = delimiter
    }
}
