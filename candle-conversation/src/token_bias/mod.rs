//! Steering a sampler by token class: bitsets built once, weighted per turn.
//!
//! # The shape
//!
//! A **class** is any property of a token worth steering on. Each class owns a
//! [`TokenBitset`] over the vocabulary — one bit per token id, 32 to a word, so
//! membership is `(w[t >> 5] >> (t & 31)) & 1`: one coalesced load and two
//! instructions, against the O(|set|) scan a token *list* forces on every
//! vocabulary element.
//!
//! Selecting composes those sets into a [`TokenBias`] — one bitset to push
//! down, one to nudge up — and the sampler applies each with its own weight.
//! The sets never move; only the weights change.
//!
//! # Why classes and not an allow-list
//!
//! The obvious way to keep a model writing English is to allow only English
//! tokens. It is also wrong: an allow-list must be *complete*, and anything
//! forgotten is silently unsayable. Digits, operators, brackets, identifiers,
//! URLs, `<tool_call>` — a coding assistant lives on tokens belonging to no
//! language at all.
//!
//! So a class holds the tokens that positively *are* that class, and a token no
//! class claims is untouchable by construction rather than by having been
//! remembered. Three populations get that protection for free: tokens of no
//! class, exempt tokens (protocol markup and the tokenizer's specials), and
//! whichever class is declared always-kept.
//!
//! # Tenants
//!
//! [`script`] is the first: writing systems, which is the axis a model drifting
//! from English into Chinese moves along. The core in [`classes`] knows nothing
//! about it — a second taxonomy (tool syntax, a per-turn ban list) supplies its
//! own classifier and shares every line.
//!
//! It biases scripts, not languages: English against Chinese, yes; English
//! against French, never — they share every byte.

mod bitset;
mod classes;
mod script;

pub use bitset::TokenBitset;
pub use classes::{TokenBias, TokenClasses};
pub use script::{
    is_protocol_markup, script_classes, scripts_of, Script, ScriptClasses, ScriptSet,
    PROTOCOL_SCRIPT,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// A miniature vocabulary with one token per interesting case.
    fn vocab() -> Vec<(u32, &'static str)> {
        vec![
            (0, "hello"),         // Latin
            (1, "world"),         // Latin
            (2, "中"),            // Han
            (3, "文档"),          // Han
            (4, "ひらがな"),      // Kana
            (5, "한국"),          // Hangul
            (6, "Привет"),        // Cyrillic
            (7, " "),             // unclassified
            (8, "42"),            // unclassified
            (9, "->"),            // unclassified
            (10, "{"),            // unclassified
            (11, "漢字かな"),     // Han + Kana (Japanese, genuinely mixed)
            (12, "café"),         // Latin
            (13, "</tool_call>"), // exempt by markup
            (14, "system"),       // exempt by declaration (a template word)
            // A frame marker that is NOT one token, as `</think>` tokenizes on
            // a checkpoint it was not trained into. The outer pieces carry
            // markup; the middle one is the ordinary English word.
            (15, "</"),
            (16, "think"),
            (17, ">"),
        ]
    }

    fn build() -> ScriptClasses {
        let v = vocab();
        let owned: Vec<(u32, &[u8])> = v.iter().map(|(i, s)| (*i, s.as_bytes())).collect();
        script_classes(20, owned, &[14])
    }

    #[test]
    fn classes_hold_exactly_the_tokens_that_emit_them() {
        let b = build();
        assert_eq!(
            b.set_of(Script::Han).unwrap().iter().collect::<Vec<_>>(),
            vec![2, 3, 11]
        );
        assert_eq!(
            b.set_of(Script::Latin).unwrap().iter().collect::<Vec<_>>(),
            vec![0, 1, 12, 16],
            "markup and the declared template word are exempt, but `think` is an \
             ordinary English word and belongs here"
        );
        assert_eq!(
            b.set_of(Script::Kana).unwrap().iter().collect::<Vec<_>>(),
            vec![4, 11]
        );
    }

    #[test]
    fn exempt_tokens_join_no_class() {
        let b = build();
        assert!(b.exempt().contains(13), "markup");
        assert!(b.exempt().contains(14), "declared special");
        assert!(
            b.exempt().contains(15) && b.exempt().contains(17),
            "marker edges"
        );
        for s in Script::ALL {
            if let Some(set) = b.set_of(s) {
                for id in [13u32, 14, 15, 17] {
                    assert!(!set.contains(id), "{s:?} claimed exempt token {id}");
                }
            }
        }
    }

    /// The regression the exemption exists to prevent: a non-Latin target must
    /// not cost the model its tool-call closer.
    #[test]
    fn exempt_tokens_survive_every_target() {
        let b = build();
        for target in [
            vec![],
            vec![Script::Han],
            vec![Script::Kana],
            Script::ALL.to_vec(),
        ] {
            let bias = b.bias_for_target(&target);
            for id in [13u32, 14, 15, 17] {
                assert!(
                    !bias.suppress().contains(id),
                    "exempt {id} suppressed for target={target:?}"
                );
            }
        }
    }

    #[test]
    fn unclassified_tokens_survive_every_target() {
        let b = build();
        for target in [vec![], vec![Script::Han], Script::ALL.to_vec()] {
            let bias = b.bias_for_target(&target);
            for t in [7u32, 8, 9, 10] {
                assert!(!bias.suppress().contains(t), "token {t} target={target:?}");
            }
        }
    }

    /// Why the raw primitive is private, demonstrated rather than asserted in
    /// prose. It breaks a split frame marker; the public door cannot.
    ///
    /// The breakage is asserted FIRST — if it ever stops happening, the
    /// guarantee below is being tested against a problem that no longer exists.
    #[test]
    fn the_raw_selection_breaks_a_split_marker_and_the_public_one_cannot() {
        let b = build();
        // `bias_for` without the protocol class is the raw selection.
        let raw = b.bias_for(&[Script::Han], &[]);
        assert!(!raw.suppress().contains(15), "`</` is markup");
        assert!(!raw.suppress().contains(17), "`>` is markup");
        assert!(
            raw.suppress().contains(16),
            "the raw selection must break `think`, or the fix below is untested"
        );

        for target in [vec![Script::Han], vec![Script::Kana], vec![]] {
            let safe = b.bias_for_target(&target);
            for piece in [15u32, 16, 17] {
                assert!(
                    !safe.suppress().contains(piece),
                    "target={target:?} suppressed marker piece {piece}"
                );
            }
        }
    }

    #[test]
    fn a_target_always_keeps_the_protocol_script() {
        let b = build();
        let latin: Vec<u32> = b.set_of(Script::Latin).unwrap().iter().collect();
        for target in [vec![Script::Han], vec![], vec![Script::Latin]] {
            let bias = b.bias_for_target(&target);
            for id in &latin {
                assert!(
                    !bias.suppress().contains(*id),
                    "target={target:?} suppressed Latin {id}"
                );
            }
        }
    }

    // ── The boost half ──────────────────────────────────────────────────────

    /// English alone boosts nothing: it is the floor every target stands on.
    #[test]
    fn an_english_only_target_boosts_nothing() {
        let b = build();
        for target in [vec![], vec![Script::Latin]] {
            let bias = b.bias_for_target(&target);
            assert!(
                bias.boost().is_empty(),
                "target={target:?} should boost nothing"
            );
            assert!(!bias.suppress().is_empty(), "but should still suppress");
        }
    }

    /// Selecting another language boosts exactly that language.
    #[test]
    fn a_non_english_target_boosts_exactly_its_own_tokens() {
        let b = build();
        let bias = b.bias_for_target(&[Script::Han]);
        assert_eq!(
            bias.boost().iter().collect::<Vec<_>>(),
            vec![2, 3, 11],
            "the Han set, mixed Han+Kana token included"
        );
        // And English is kept but NOT boosted — it is not competing.
        for id in [0u32, 1, 12, 16] {
            assert!(
                !bias.boost().contains(id),
                "English {id} must not be boosted"
            );
            assert!(!bias.suppress().contains(id), "nor suppressed");
        }
    }

    /// The two halves can never disagree about a token.
    #[test]
    fn suppress_and_boost_are_always_disjoint() {
        let b = build();
        for target in [
            vec![],
            vec![Script::Han],
            vec![Script::Kana],
            vec![Script::Han, Script::Cyrillic],
            Script::ALL.to_vec(),
        ] {
            let bias = b.bias_for_target(&target);
            for t in 0..b.vocab() as u32 {
                assert!(
                    !(bias.suppress().contains(t) && bias.boost().contains(t)),
                    "token {t} both suppressed and boosted for target={target:?}"
                );
            }
        }
    }

    /// Nothing exempt or unclassified is ever boosted either — a boost is as
    /// much an intervention as a suppression.
    #[test]
    fn exempt_and_unclassified_tokens_are_never_boosted() {
        let b = build();
        for target in [vec![Script::Han], vec![Script::Kana], Script::ALL.to_vec()] {
            let bias = b.bias_for_target(&target);
            for id in [7u32, 8, 9, 10, 13, 14, 15, 17] {
                assert!(
                    !bias.boost().contains(id),
                    "token {id} boosted for target={target:?}"
                );
            }
        }
    }

    // ── The frame half ──────────────────────────────────────────────────────

    /// The frame set is what was DECLARED, not what the markup rule inferred.
    /// `</` and `>` are exempt from suppression because that is free; they are
    /// not protocol and must not be encouraged.
    #[test]
    fn the_frame_set_is_the_declared_specials_only() {
        let b = build();
        let bias = b.bias_for_target(&[Script::Han]);
        assert_eq!(
            bias.frame().iter().collect::<Vec<_>>(),
            vec![14],
            "only the declared token, not the markup-inferred ones"
        );
        for inferred in [13u32, 15, 17] {
            assert!(
                !bias.frame().contains(inferred),
                "markup-inferred {inferred} must not be encouraged"
            );
            assert!(
                b.exempt().contains(inferred),
                "but it is still exempt from suppression"
            );
        }
    }

    /// All three sets are mutually disjoint — no token is pushed two ways.
    #[test]
    fn suppress_boost_and_frame_are_mutually_disjoint() {
        let b = build();
        for target in [
            vec![],
            vec![Script::Han],
            vec![Script::Kana],
            vec![Script::Han, Script::Cyrillic],
            Script::ALL.to_vec(),
        ] {
            let bias = b.bias_for_target(&target);
            for t in 0..b.vocab() as u32 {
                let s = bias.suppress().contains(t);
                let o = bias.boost().contains(t);
                let f = bias.frame().contains(t);
                assert!(
                    (s as u8) + (o as u8) + (f as u8) <= 1,
                    "token {t} in more than one set for target={target:?}"
                );
            }
        }
    }

    /// The frame set does not depend on the target: the protocol is the
    /// protocol whatever language is being written.
    #[test]
    fn the_frame_set_is_the_same_for_every_target() {
        let b = build();
        let first = b.bias_for_target(&[]).frame().iter().collect::<Vec<_>>();
        for target in [vec![Script::Han], vec![Script::Kana], Script::ALL.to_vec()] {
            assert_eq!(
                b.bias_for_target(&target)
                    .frame()
                    .iter()
                    .collect::<Vec<_>>(),
                first,
                "target={target:?} changed the frame set"
            );
        }
    }

    #[test]
    fn a_multi_class_token_survives_if_either_class_is_kept() {
        let b = build();
        assert!(!b.bias_for_target(&[Script::Han]).suppress().contains(11));
        assert!(!b.bias_for_target(&[Script::Kana]).suppress().contains(11));
        assert!(b
            .bias_for_target(&[Script::Cyrillic])
            .suppress()
            .contains(11));
    }

    #[test]
    fn keeping_everything_suppresses_nothing() {
        let b = build();
        assert!(b.bias_for_target(&Script::ALL).suppress().is_empty());
    }

    #[test]
    fn suppressed_and_sayable_partition_the_vocabulary() {
        let b = build();
        for target in [vec![], vec![Script::Han], Script::ALL.to_vec()] {
            let bias = b.bias_for_target(&target);
            let w = b.sayable_for(&target, &[PROTOCOL_SCRIPT]);
            assert_eq!(
                bias.suppress().len() + w.len(),
                b.vocab(),
                "target={target:?}"
            );
            for t in 0..b.vocab() as u32 {
                assert_ne!(
                    bias.suppress().contains(t),
                    w.contains(t),
                    "token {t} target={target:?}"
                );
            }
        }
    }

    #[test]
    fn ids_outside_the_vocabulary_are_skipped_not_folded() {
        let toks: Vec<(u32, &[u8])> = vec![(0, "hi".as_bytes()), (99, "中".as_bytes())];
        let b = script_classes(8, toks, &[]);
        assert!(b.set_of(Script::Latin).unwrap().contains(0));
        assert!(b.set_of(Script::Han).map(|l| l.is_empty()).unwrap_or(true));
    }

    #[test]
    fn markup_is_detected_in_byte_fragments_too() {
        assert!(is_protocol_markup(b"</think>"));
        assert!(is_protocol_markup(b"<|im_start|>"));
        assert!(
            is_protocol_markup(&[0xE4, b'<']),
            "invalid utf8 with markup"
        );
        assert!(!is_protocol_markup("中文".as_bytes()));
        assert!(!is_protocol_markup(b"hello"));
    }
}
