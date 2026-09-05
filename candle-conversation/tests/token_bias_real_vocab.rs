//! [`language_bias`] against a real tokenizer, not a synthetic one.
//!
//! The unit tests prove the set algebra on fourteen hand-written tokens. They
//! cannot prove the thing that actually decides whether this ships: that a
//! production vocabulary classifies the way the design assumes. Qwen's is
//! byte-level BPE over 248k tokens whose surface forms are byte-map artifacts
//! (`ä¸­`, not `中`), so the classifier is fed DECODED bytes and this test is
//! what checks that the decode is the right one.
//!
//! Skipped, loudly, when the tokenizer is not on this machine — it is a real
//! model file, not a fixture, and a silent skip would let the interesting half
//! of the coverage rot.

use candle_conversation::token_bias::{script_classes, scripts_of, Script, ScriptClasses};
use tokenizers::Tokenizer;

/// Every Qwen tokenizer in the HF hub cache, as `(model name, path)`.
///
/// All of them, not the first: the cache holds checkpoints with different
/// vocabularies (151k and 248k here), and a classifier that works on one and
/// not the other is exactly the failure this is meant to catch. Sorted so a
/// failure names the same checkpoint on every run.
fn find_tokenizers() -> Vec<(String, std::path::PathBuf)> {
    let Ok(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) else {
        return Vec::new();
    };
    let hub = std::path::Path::new(&home)
        .join(".cache")
        .join("huggingface")
        .join("hub");
    let Ok(models) = std::fs::read_dir(&hub) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for m in models.flatten() {
        let name = m.file_name().to_string_lossy().to_string();
        if !name.contains("Qwen") {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(m.path().join("snapshots")) else {
            continue;
        };
        for s in entries.flatten() {
            let p = s.path().join("tokenizer.json");
            if p.is_file() {
                out.push((name.trim_start_matches("models--").to_string(), p));
                break;
            }
        }
    }
    out.sort();
    out
}

/// Run `check` against every cached Qwen tokenizer, naming each. Skips loudly
/// when the machine has none rather than passing vacuously.
fn for_each_tokenizer(check: impl Fn(&str, &Tokenizer)) {
    let found = find_tokenizers();
    if found.is_empty() {
        eprintln!("SKIP: no Qwen tokenizer in the HF hub cache on this machine");
        return;
    }
    for (name, path) in found {
        let Ok(tok) = Tokenizer::from_file(&path) else {
            panic!("{name}: tokenizer.json failed to parse");
        };
        eprintln!("── {name}");
        check(&name, &tok);
    }
}

/// Build the script sets from a tokenizer by decoding each token to the bytes
/// it would actually emit, and declaring its special tokens structural.
fn build(tok: &Tokenizer) -> (ScriptClasses, usize) {
    let vocab = tok.get_vocab(true);
    let size = vocab
        .values()
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0);
    // `decode` on a single id yields the token's text, which is what the model
    // emits — as opposed to `get_vocab`'s key, which is the byte-map surface
    // form and would classify `ä¸­` as Latin.
    let decoded: Vec<(u32, String)> = vocab
        .values()
        .map(|id| (*id, tok.decode(&[*id], false).unwrap_or_default()))
        .collect();
    let refs: Vec<(u32, &[u8])> = decoded.iter().map(|(i, s)| (*i, s.as_bytes())).collect();
    // The tokenizer's own added tokens: role words and sentinels that carry no
    // markup character and so are invisible to the heuristic.
    let special: Vec<u32> = tok
        .get_added_vocabulary()
        .get_vocab()
        .values()
        .copied()
        .collect();
    (script_classes(size, refs, &special), size)
}

#[test]
fn real_vocabulary_classifies_as_the_design_assumes() {
    for_each_tokenizer(|name, tok| {
        let (b, vocab) = build(tok);
        let neutral = b.sayable_for(&[], &[]).len();
        eprintln!(
            "   vocab={vocab} neutral+structural={neutral} structural={}",
            b.exempt().len()
        );
        for s in b.classes() {
            eprintln!("     {s:?}: {}", b.set_of(s).unwrap().len());
        }

        // Han and Latin must both be substantial: these are bilingual
        // checkpoints, and a classifier finding almost no Han would mean the
        // decode is yielding byte-map surface forms (`ä¸­`) rather than emitted
        // text — the trap this whole test exists to catch.
        let han = b.set_of(Script::Han).expect("a Qwen vocab contains Han");
        let latin = b.set_of(Script::Latin).expect("and Latin");
        assert!(
            han.len() > 1000,
            "{name}: only {} Han tokens — the decode is probably yielding \
             byte-map surface forms rather than emitted text",
            han.len()
        );
        assert!(latin.len() > 1000, "{name}: only {} Latin", latin.len());

        // And a real share of the vocabulary must be neutral. A classifier that
        // returned a script for everything would pass the two checks above and
        // would suppress the model's ability to write a number.
        assert!(
            neutral > vocab / 100,
            "{name}: only {neutral} of {vocab} tokens are neutral"
        );
    });
}

/// How another lab's control tokens tokenize here, and what that does to the
/// exemption rules.
///
/// Qwen3.8-Flash-Next emits Kimi K2's tool-call syntax
/// (`<|tool_call_begin|>`) under pressure, despite having no such token — a
/// training-exposure artifact, visible only because the two tokenizers
/// disagree. It therefore arrives SPELLED OUT across several ordinary tokens.
///
/// Reported rather than asserted: the decomposition is a property of a
/// checkpoint, not of this code. What matters for the bias is the last line —
/// every piece carries `<`, `|` or `>`, so the markup rule already exempts
/// them and no language target can suppress a half-written control token.
#[test]
fn report_foreign_control_token_decomposition() {
    for_each_tokenizer(|_name, tok| {
        let (b, _) = build(tok);
        for s in [
            "<|tool_call_begin|>",
            "<|tool_calls_section_begin|>",
            "<|tool_call_argument_begin|>",
            "<tool_call>",
            "<|im_start|>",
        ] {
            let Ok(enc) = tok.encode(s, false) else {
                continue;
            };
            let pieces: Vec<String> = enc
                .get_ids()
                .iter()
                .map(|id| tok.decode(&[*id], false).unwrap_or_default())
                .collect();
            let exempt = enc.get_ids().iter().all(|id| b.exempt().contains(*id));
            eprintln!(
                "   {s:<32} {} token(s) {:?} all-exempt={exempt}",
                enc.get_ids().len(),
                pieces
            );
        }
    });
}

/// How many tokens straddle two scripts, and what they look like.
///
/// Not an assertion about a threshold — a report. The whole difference between
/// "suppress anything emitting a disallowed script" and "keep anything the
/// target can also use" is this population, and the design should be chosen
/// against its real size and content rather than against an intuition about it.
#[test]
fn report_mixed_and_structural_populations() {
    for_each_tokenizer(|_name, tok| {
        let vocab = tok.get_vocab(true);
        let mut latin_han = Vec::new();
        let mut latin_other = 0usize;
        let mut structural_with_letters = Vec::new();
        let mut multi = 0usize;

        for id in vocab.values().copied() {
            let text = tok.decode(&[id], false).unwrap_or_default();
            let set = scripts_of(text.as_bytes());
            let n = set.iter().count();
            if n > 1 {
                multi += 1;
                if set.contains(Script::Latin) && set.contains(Script::Han) {
                    latin_han.push(text.clone());
                } else if set.contains(Script::Latin) {
                    latin_other += 1;
                }
            }
            // Tokens carrying markup AND letters — the chat-template and
            // tool-call machinery, which must never be suppressed for any
            // target because the protocol depends on them.
            if !set.is_empty() && text.chars().any(|c| matches!(c, '<' | '>' | '|')) {
                structural_with_letters.push(text.clone());
            }
        }

        eprintln!("   multi-script tokens: {multi}");
        eprintln!("     Latin+Han: {}", latin_han.len());
        eprintln!("     Latin+other: {latin_other}");
        latin_han.sort();
        for s in latin_han.iter().take(12) {
            eprintln!("       {s:?}");
        }
        eprintln!(
            "   tokens with markup AND letters: {}",
            structural_with_letters.len()
        );
        structural_with_letters.sort();
        for s in structural_with_letters.iter().take(12) {
            eprintln!("       {s:?}");
        }
    });
}

/// The property the design rests on, on the real vocabulary: biasing toward
/// Latin must not cost the model a digit, a bracket, or an operator.
#[test]
fn biasing_to_latin_keeps_every_token_a_coding_assistant_needs() {
    for_each_tokenizer(|name, tok| {
        let (b, _) = build(tok);
        let suppress = b.bias_for_target(&[Script::Latin]).suppress().clone();

        // Each string is encoded, then every token of the encoding is checked.
        // Encoding rather than looking up literals is deliberate: it asks the
        // question the sampler faces — "can the model still produce this
        // string" — rather than "is this particular token neutral".
        for text in [
            "fn main() { let x = 42; }",
            "0x1F + 1e-9 * 3.14",
            "for (i = 0; i < n; i++) { arr[i] += 1; }",
            "SELECT * FROM t WHERE a >= 10 AND b != 'x';",
            "<tool_call>{\"name\": \"file_read\"}</tool_call>",
            "https://example.com/path?q=1&r=2#frag",
            "impl<T: Clone> Trait for Vec<T> {}",
        ] {
            let enc = tok.encode(text, false).expect("encodes");
            for &id in enc.get_ids() {
                assert!(
                    !suppress.contains(id),
                    "{name}: biasing to Latin would suppress token {id} ({:?}) \
                     from {text:?}",
                    tok.decode(&[id], false).unwrap_or_default()
                );
            }
        }
    });
}

/// The other half: biasing toward Latin must actually suppress Chinese, or the
/// whole exercise is decorative.
#[test]
fn biasing_to_latin_suppresses_chinese_text() {
    for_each_tokenizer(|name, tok| {
        let (b, _) = build(tok);
        let suppress = b.bias_for_target(&[Script::Latin]).suppress().clone();

        for text in ["你好，我很好", "用户用中文提问", "这是一个测试"] {
            let enc = tok.encode(text, false).expect("encodes");
            let ids = enc.get_ids();
            let hit = ids.iter().filter(|id| suppress.contains(**id)).count();
            // Punctuation inside the string is legitimately neutral, so the
            // assertion is that the SUBSTANCE is suppressed, not every token.
            assert!(
                hit * 2 >= ids.len(),
                "{name}: only {hit} of {} tokens suppressed for {text:?}",
                ids.len()
            );
        }
    });
}

/// Switching target at sample time is a selection over the same immutable
/// blacklists — verified on the real vocabulary, where the two targets must
/// genuinely disagree.
/// The regression the structural carve-out exists to prevent, on the real
/// vocabulary and against the targets that would trip it.
///
/// `</think>` and `</tool_call>` carry Latin letters. Without the carve-out
/// they sit in the Latin set, and selecting ANY non-Latin script suppresses the
/// model's ability to close a reasoning block or a tool call. Asserted against
/// Han, Kana and Cyrillic targets, because a Latin target could never catch it.
///
/// **Scoped to atomic frame markers, deliberately.** On a checkpoint where
/// `</think>` is one special token — which is every checkpoint that trains on
/// it, our production one included — the carve-out covers it entirely. On an
/// older vocabulary it decomposes into `</` + `think` + `>`, and the middle
/// piece is the ordinary English word: correctly Latin, and correctly penalised
/// by a Han target. That is not a protocol break because the penalty is SOFT
/// and the continuation after `</` is near-deterministic; it would become one
/// if the penalty were ever raised to a ban, which is the reason this comment
/// exists.
#[test]
fn the_protocol_survives_a_non_latin_target() {
    for_each_tokenizer(|name, tok| {
        let (b, _) = build(tok);
        for keep in [
            vec![Script::Han],
            vec![Script::Kana],
            vec![Script::Cyrillic],
            vec![],
        ] {
            let sup = b.bias_for_target(&keep).suppress().clone();

            // Universal: every token the tokenizer declares special, and every
            // token carrying markup, survives any target whatsoever.
            for id in b.exempt().iter() {
                assert!(
                    !sup.contains(id),
                    "{name}: keep={keep:?} suppresses structural token {id} ({:?})",
                    tok.decode(&[id], false).unwrap_or_default()
                );
            }

            // And the frame markers themselves, wherever they are atomic.
            for frame in FRAMES {
                let Ok(enc) = tok.encode(*frame, false) else {
                    continue;
                };
                if enc.get_ids().len() != 1 {
                    continue;
                }
                let id = enc.get_ids()[0];
                assert!(
                    !sup.contains(id),
                    "{name}: keep={keep:?} suppresses the atomic frame token \
                     {id} ({frame})"
                );
            }
        }
    });
}

/// Protocol strings a chat model must be able to emit.
const FRAMES: &[&str] = &[
    "</think>",
    "<think>",
    "</tool_call>",
    "<tool_call>",
    "<|im_start|>",
    "<|im_end|>",
    "<|im_start|>assistant\n",
    "<tool_response>",
    "</tool_response>",
    "{\"name\": \"file_read\", \"arguments\": {\"path\": \"a.rs\"}}",
];

/// The general answer to the multi-token frame marker, and the reason
/// [`suppression_for_target`] exists.
///
/// The previous test has to skip `</think>` wherever it is not atomic, because
/// its middle piece is the English word `think` and a Han target suppresses it
/// correctly. Keeping Latin unconditionally removes the exception: EVERY token
/// of EVERY frame survives, on every checkpoint, whether or not the tokenizer
/// happens to have the marker as one token — and without anything enumerating
/// which markers exist.
///
/// This is the assertion that would have caught the `</` + `think` + `>` case,
/// so it is written to cover the whole tokenization rather than the atomic ones.
#[test]
fn keeping_the_protocol_script_makes_every_frame_sayable_under_any_target() {
    for_each_tokenizer(|name, tok| {
        let (b, _) = build(tok);
        for target in [
            vec![Script::Han],
            vec![Script::Kana],
            vec![Script::Cyrillic],
            vec![Script::Arabic],
            vec![Script::Thai],
            vec![],
        ] {
            let sup = b.bias_for_target(&target).suppress().clone();
            for frame in FRAMES {
                let Ok(enc) = tok.encode(*frame, false) else {
                    continue;
                };
                for &id in enc.get_ids() {
                    assert!(
                        !sup.contains(id),
                        "{name}: target={target:?} suppresses token {id} ({:?}) \
                         of {frame:?} — the protocol would stop parsing",
                        tok.decode(&[id], false).unwrap_or_default()
                    );
                }
            }
        }
    });
}

/// The boost half, on a real vocabulary: selecting a language nudges that
/// language's own tokens and nothing else.
#[test]
fn selecting_a_language_boosts_exactly_that_language() {
    for_each_tokenizer(|name, tok| {
        let (b, _) = build(tok);

        // English alone has nothing to encourage — it is the floor.
        assert!(
            b.bias_for_target(&[]).boost().is_empty(),
            "{name}: an English-only target must boost nothing"
        );
        assert!(
            b.bias_for_target(&[Script::Latin]).boost().is_empty(),
            "{name}: naming the protocol script explicitly must not boost it"
        );

        let bias = b.bias_for_target(&[Script::Han]);
        let boost = bias.boost();
        assert!(!boost.is_empty(), "{name}: a Han target must boost Han");

        // Chinese text is boosted…
        let chinese = tok.encode("你好世界", false).expect("encodes");
        assert!(
            chinese.get_ids().iter().any(|id| boost.contains(*id)),
            "{name}: Chinese text should be boosted by a Han target"
        );
        // …English and the protocol are not. A boost is an intervention, and
        // encouraging the frame would distort the very tokens that must stay
        // neutral.
        for text in ["The quick brown fox", "fn main() { let x = 42; }"] {
            let enc = tok.encode(text, false).expect("encodes");
            for &id in enc.get_ids() {
                assert!(
                    !boost.contains(id),
                    "{name}: {text:?} token {id} ({:?}) was boosted",
                    tok.decode(&[id], false).unwrap_or_default()
                );
            }
        }
        for id in b.exempt().iter().take(2000) {
            assert!(!boost.contains(id), "{name}: exempt token {id} was boosted");
        }
    });
}

/// The frame set on a real vocabulary: the protocol markers are in it, and the
/// incidental markup the heuristic exempted is not.
///
/// The distinction is the whole point. `</tool_call>` should be encouraged so a
/// bias toward another language does not make it fractionally harder to reach;
/// `<Account` was exempted only because exempting it was free, and encouraging
/// it would be a preference nobody asked for.
#[test]
fn the_frame_set_holds_the_protocol_and_not_incidental_markup() {
    for_each_tokenizer(|name, tok| {
        let (b, _) = build(tok);
        let bias = b.bias_for_target(&[Script::Han]);
        let frame = bias.frame();
        assert!(!frame.is_empty(), "{name}: no declared protocol tokens");

        // Every atomic frame marker is in it.
        let mut found = 0;
        for marker in FRAMES {
            let Ok(enc) = tok.encode(*marker, false) else {
                continue;
            };
            if enc.get_ids().len() == 1 {
                assert!(
                    frame.contains(enc.get_ids()[0]),
                    "{name}: atomic marker {marker:?} is not in the frame set"
                );
                found += 1;
            }
        }
        assert!(found > 0, "{name}: no atomic markers to check");

        // And nothing in it is ordinary language. A declared special that
        // classified as a script would mean the exemption ran too late.
        for id in frame.iter() {
            for s in Script::ALL {
                if let Some(set) = b.set_of(s) {
                    assert!(
                        !set.contains(id),
                        "{name}: frame token {id} also classified {s:?}"
                    );
                }
            }
        }
    });
}

/// The three sets never disagree about a token, on the real vocabulary.
#[test]
fn suppress_and_boost_never_overlap() {
    for_each_tokenizer(|name, tok| {
        let (b, vocab) = build(tok);
        for target in [
            vec![],
            vec![Script::Han],
            vec![Script::Cyrillic],
            vec![Script::Han, Script::Kana],
        ] {
            let bias = b.bias_for_target(&target);
            for t in 0..vocab as u32 {
                let s = bias.suppress().contains(t);
                let o = bias.boost().contains(t);
                let f = bias.frame().contains(t);
                assert!(
                    (s as u8) + (o as u8) + (f as u8) <= 1,
                    "{name}: token {t} in more than one set for {target:?}"
                );
            }
        }
    });
}

/// Keeping the protocol script must not defeat the bias it is protecting: a Han
/// target still has to suppress Chinese, and a Cyrillic one still has to
/// suppress Russian.
#[test]
fn keeping_the_protocol_script_still_suppresses_other_languages() {
    for_each_tokenizer(|name, tok| {
        let (b, _) = build(tok);
        for (target, sample) in [
            (Script::Han, "你好世界，这是一个测试"),
            (Script::Cyrillic, "Привет мир, это тест"),
        ] {
            // Steering AWAY from `sample`'s script: keep everything except it.
            let others: Vec<Script> = Script::ALL.into_iter().filter(|s| *s != target).collect();
            let sup = b.bias_for_target(&others).suppress().clone();
            let enc = tok.encode(sample, false).expect("encodes");
            let ids = enc.get_ids();
            let hit = ids.iter().filter(|id| sup.contains(**id)).count();
            assert!(
                hit * 2 >= ids.len(),
                "{name}: only {hit} of {} tokens of {sample:?} suppressed",
                ids.len()
            );
        }
    });
}

#[test]
fn adding_a_script_to_the_target_stops_suppressing_it() {
    for_each_tokenizer(|name, tok| {
        let (b, _) = build(tok);
        // English-only, then English plus Chinese — the selection a bilingual
        // conversation makes at sample time, over the same immutable sets.
        let english_only = b.bias_for_target(&[]).suppress().clone();
        let bilingual = b.bias_for_target(&[Script::Han]).suppress().clone();

        let english = tok.encode("The quick brown fox", false).expect("encodes");
        let chinese = tok.encode("你好世界", false).expect("encodes");

        // English survives both, because the protocol script is never dropped.
        for &id in english.get_ids() {
            assert!(!english_only.contains(id), "{name}: English suppressed");
            assert!(!bilingual.contains(id), "{name}: English suppressed");
        }
        // Chinese is suppressed by the first and spared by the second. That
        // difference IS the steering knob.
        assert!(
            chinese
                .get_ids()
                .iter()
                .any(|id| english_only.contains(*id)),
            "{name}: an English-only target must suppress some Chinese"
        );
        for &id in chinese.get_ids() {
            assert!(
                !bilingual.contains(id),
                "{name}: adding Han to the target must stop suppressing Chinese"
            );
        }
    });
}
