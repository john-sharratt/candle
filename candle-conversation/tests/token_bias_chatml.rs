//! Script bias against full ChatML conversations, on a fabricated vocabulary.
//!
//! No model and no tokenizer file: the point here is the set algebra, and a
//! hand-built vocabulary is the only way to assert against *known* ids. Every
//! script gets a complete ChatML exchange — system frame, user turn, a
//! `<think>` block, a tool call, an assistant answer — and each conversation is
//! checked in both directions:
//!
//! * **negative** — with its own script kept, NOTHING in the conversation is
//!   suppressed. Not the language, not the frame, not the punctuation.
//! * **positive** — with a different script kept, every language token IS
//!   suppressed while the frame and punctuation are untouched.
//!
//! The negative half is the one that matters. A bias that suppresses the wrong
//! thing is not a weaker bias, it is a broken model: lose `</tool_call>` and the
//! protocol stops parsing; lose the digits and it cannot write a number.

use candle_conversation::token_bias::{script_classes, Script, ScriptClasses, TokenBitset};

// ── A fabricated vocabulary ──────────────────────────────────────────────────
//
// Ids are assigned in blocks so a failure message points at a category rather
// than a number: 0..20 frame, 20..40 neutral, 100+ language words.

/// ChatML frame and tool protocol. Some carry markup and are caught by the
/// heuristic; the role words do not and must be DECLARED special, which is
/// exactly the gap the heuristic cannot close on its own.
const FRAME: &[(u32, &str)] = &[
    (0, "<|im_start|>"),
    (1, "<|im_end|>"),
    (2, "<think>"),
    (3, "</think>"),
    (4, "<tool_call>"),
    (5, "</tool_call>"),
    (6, "<tool_response>"),
    (7, "</tool_response>"),
    (8, "system"),
    (9, "user"),
    (10, "assistant"),
    (11, "<|endoftext|>"),
];

/// The role words carry no markup character, so nothing but a declaration can
/// keep them out of the Latin set.
const SPECIAL_IDS: &[u32] = &[8, 9, 10];

/// A frame marker that is NOT one token — `</` + `think` + `>`, which is how
/// `</think>` tokenizes on a checkpoint the marker was not trained into.
///
/// `think` carries no markup and is not special: it is the ordinary English
/// word, correctly Latin. That is the whole difficulty, reproduced here so it
/// can be asserted on rather than discovered on a real vocabulary.
const SPLIT_FRAME: &[(u32, &str)] = &[(40, "</"), (41, "think"), (42, ">")];

/// Punctuation, digits, whitespace, and the JSON a tool call is made of.
const NEUTRAL: &[(u32, &str)] = &[
    (20, "\n"),
    (21, " "),
    (22, "."),
    (23, ","),
    (24, "?"),
    (25, "!"),
    (26, ":"),
    (27, "\""),
    (28, "{"),
    (29, "}"),
    (30, "42"),
    (31, "3.14"),
    (32, "-"),
    (33, "_"),
    (34, "("),
    (35, ")"),
];

/// Four words per script: a greeting, a noun, a verb, a closing word. Enough
/// for a conversation to read as one, and enough that a partial suppression
/// shows up as a count rather than a coin flip.
fn words(s: Script) -> [&'static str; 4] {
    match s {
        Script::Latin => ["Hello", "codebase", "explain", "thanks"],
        Script::Han => ["你好", "代码", "解释", "谢谢"],
        Script::Kana => ["こんにちは", "コード", "せつめい", "ありがとう"],
        Script::Hangul => ["안녕하세요", "코드", "설명", "감사합니다"],
        Script::Cyrillic => ["Привет", "код", "объясни", "спасибо"],
        Script::Greek => ["Γεια", "κώδικα", "εξήγησε", "ευχαριστώ"],
        Script::Arabic => ["مرحبا", "الشفرة", "اشرح", "شكرا"],
        Script::Hebrew => ["שלום", "קוד", "הסבר", "תודה"],
        Script::Thai => ["สวัสดี", "รหัส", "อธิบาย", "ขอบคุณ"],
        Script::Devanagari => ["नमस्ते", "कोड", "समझाओ", "धन्यवाद"],
    }
}

/// First id of a script's word block.
fn word_base(s: Script) -> u32 {
    100 + 10 * (Script::ALL.iter().position(|x| *x == s).unwrap() as u32)
}

fn vocab_size() -> usize {
    (word_base(Script::Devanagari) + 10) as usize
}

/// Every token, as `(id, text)`.
fn all_tokens() -> Vec<(u32, String)> {
    let mut v: Vec<(u32, String)> = Vec::new();
    for (i, t) in FRAME.iter().chain(NEUTRAL.iter()).chain(SPLIT_FRAME.iter()) {
        v.push((*i, (*t).to_string()));
    }
    for s in Script::ALL {
        for (k, w) in words(s).iter().enumerate() {
            v.push((word_base(s) + k as u32, (*w).to_string()));
        }
    }
    v
}

fn build() -> ScriptClasses {
    let owned = all_tokens();
    let refs: Vec<(u32, &[u8])> = owned.iter().map(|(i, s)| (*i, s.as_bytes())).collect();
    script_classes(vocab_size(), refs, SPECIAL_IDS)
}

// ── A ChatML conversation ────────────────────────────────────────────────────

/// The three token classes of a conversation, kept apart so the assertions can
/// speak about each.
struct Conversation {
    script: Script,
    /// Frame + protocol tokens, in order of use.
    frame: Vec<u32>,
    /// Punctuation, digits, whitespace.
    neutral: Vec<u32>,
    /// The words carrying the language.
    language: Vec<u32>,
    /// The full interleaved stream, as the model would see it.
    stream: Vec<u32>,
}

/// A complete ChatML exchange in `s`: system frame, user question, an assistant
/// turn containing a think block, a tool call with JSON, a tool response, and a
/// closing answer.
fn conversation(s: Script) -> Conversation {
    let w = word_base(s);
    let (greet, noun, verb, thank) = (w, w + 1, w + 2, w + 3);
    // (id, class) — class 0 frame, 1 neutral, 2 language.
    let script: Vec<(u32, u8)> = vec![
        (0, 0),
        (8, 0),
        (20, 1), // <|im_start|>system\n
        (verb, 2),
        (21, 1),
        (noun, 2),
        (22, 1), // "explain the codebase."
        (1, 0),
        (20, 1), // <|im_end|>\n
        (0, 0),
        (9, 0),
        (20, 1), // <|im_start|>user\n
        (greet, 2),
        (23, 1),
        (21, 1),
        (verb, 2),
        (21, 1),
        (noun, 2),
        (24, 1), // greeting, question
        (1, 0),
        (20, 1), // <|im_end|>\n
        (0, 0),
        (10, 0),
        (20, 1), // <|im_start|>assistant\n
        (2, 0),
        (20, 1), // <think>\n
        (verb, 2),
        (21, 1),
        (noun, 2),
        (22, 1),
        (3, 0),
        (20, 1), // </think>\n
        (4, 0),
        (28, 1),
        (27, 1),
        (33, 1),
        (27, 1),
        (26, 1),
        (30, 1),
        (29, 1), // <tool_call>{"_":42}
        (5, 0),
        (20, 1), // </tool_call>\n
        (6, 0),
        (31, 1),
        (7, 0),
        (20, 1), // <tool_response>3.14</tool_response>\n
        (greet, 2),
        (25, 1),
        (21, 1),
        (noun, 2),
        (21, 1),
        (thank, 2),
        (22, 1),
        (1, 0), // <|im_end|>
    ];
    let mut c = Conversation {
        script: s,
        frame: Vec::new(),
        neutral: Vec::new(),
        language: Vec::new(),
        stream: Vec::new(),
    };
    for (id, class) in script {
        c.stream.push(id);
        match class {
            0 => c.frame.push(id),
            1 => c.neutral.push(id),
            _ => c.language.push(id),
        }
    }
    c
}

fn decode(v: &ScriptClasses, id: u32) -> String {
    let _ = v;
    all_tokens()
        .into_iter()
        .find(|(i, _)| *i == id)
        .map(|(_, s)| s)
        .unwrap_or_else(|| format!("<{id}>"))
}

/// Nothing in `ids` may be suppressed.
fn assert_none_suppressed(v: &ScriptClasses, s: &TokenBitset, ids: &[u32], what: &str) {
    for id in ids {
        assert!(
            !s.contains(*id),
            "{what}: token {id} ({:?}) was suppressed",
            decode(v, *id)
        );
    }
}

/// Every id in `ids` must be suppressed.
fn assert_all_suppressed(v: &ScriptClasses, s: &TokenBitset, ids: &[u32], what: &str) {
    for id in ids {
        assert!(
            s.contains(*id),
            "{what}: token {id} ({:?}) was NOT suppressed",
            decode(v, *id)
        );
    }
}

// ── The tests ────────────────────────────────────────────────────────────────

/// Negative half, every script: keeping a conversation's own script must leave
/// the entire conversation sayable — frame, punctuation and language alike.
#[test]
fn a_conversation_is_untouched_when_its_own_script_is_kept() {
    let v = build();
    for s in Script::ALL {
        let c = conversation(s);
        let sup = v.bias_for_target(&[s]).suppress().clone();
        assert_none_suppressed(&v, &sup, &c.stream, &format!("{:?} kept", c.script));
    }
}

/// Positive half: a language that is not the target goes, while the frame and
/// punctuation stay. Every ordered pair of non-Latin scripts, so no script is
/// asserted only against one rival.
///
/// Latin is excluded as a *spoken* language because it can no longer be
/// suppressed at all — the protocol script is kept unconditionally, which is
/// asserted separately below.
#[test]
fn a_non_target_language_is_suppressed_and_the_frame_is_spared() {
    let v = build();
    for spoken in Script::ALL {
        if spoken == Script::Latin {
            continue;
        }
        let c = conversation(spoken);
        for kept in Script::ALL {
            if kept == spoken {
                continue;
            }
            let sup = v.bias_for_target(&[kept]).suppress().clone();
            let what = format!("{spoken:?} spoken, {kept:?} target");
            assert_all_suppressed(&v, &sup, &c.language, &what);
            assert_none_suppressed(&v, &sup, &c.frame, &what);
            assert_none_suppressed(&v, &sup, &c.neutral, &what);
        }
    }
}

/// The invariant that replaces "Latin is suppressible": whatever the target,
/// an English conversation survives whole. This is what makes every protocol
/// string sayable without any of them being enumerated.
#[test]
fn an_english_conversation_survives_every_target() {
    let v = build();
    let c = conversation(Script::Latin);
    for target in Script::ALL {
        let sup = v.bias_for_target(&[target]).suppress().clone();
        assert_none_suppressed(&v, &sup, &c.stream, &format!("target={target:?}"));
    }
    let sup = v.bias_for_target(&[]).suppress().clone();
    assert_none_suppressed(&v, &sup, &c.stream, "target=[]");
}

/// The frame is the protocol. Whatever is kept — including nothing at all — a
/// model must still be able to close a think block, emit a tool call, and end
/// its turn.
#[test]
fn the_chatml_frame_survives_every_target_including_the_empty_one() {
    let v = build();
    let frame: Vec<u32> = FRAME.iter().map(|(i, _)| *i).collect();
    let neutral: Vec<u32> = NEUTRAL.iter().map(|(i, _)| *i).collect();

    let mut targets: Vec<Vec<Script>> = vec![vec![], Script::ALL.to_vec()];
    for s in Script::ALL {
        targets.push(vec![s]);
    }
    targets.push(vec![Script::Latin, Script::Han]);

    for keep in targets {
        let sup = v.bias_for_target(&keep).suppress().clone();
        let what = format!("keep={keep:?}");
        assert_none_suppressed(&v, &sup, &frame, &what);
        assert_none_suppressed(&v, &sup, &neutral, &what);
    }
}

/// A bilingual target keeps both and suppresses the rest — the case a mixed
/// English/Chinese conversation needs.
#[test]
fn a_two_script_target_keeps_both_conversations_whole() {
    let v = build();
    let sup = v
        .bias_for_target(&[Script::Latin, Script::Han])
        .suppress()
        .clone();
    for s in [Script::Latin, Script::Han] {
        let c = conversation(s);
        assert_none_suppressed(&v, &sup, &c.stream, &format!("{s:?} in a bilingual target"));
    }
    for s in [Script::Cyrillic, Script::Thai, Script::Hangul] {
        let c = conversation(s);
        assert_all_suppressed(&v, &sup, &c.language, &format!("{s:?} against Latin+Han"));
    }
}

/// Every script must actually be represented, or a conversation could pass its
/// positive test by being empty. Guards the fixture, not the code.
#[test]
fn every_script_has_words_and_they_classify_as_that_script() {
    let v = build();
    for s in Script::ALL {
        let set = v
            .set_of(s)
            .unwrap_or_else(|| panic!("{s:?} has no tokens in the fixture"));
        let c = conversation(s);
        assert!(!c.language.is_empty(), "{s:?} conversation has no language");
        for id in &c.language {
            assert!(
                set.contains(*id),
                "{s:?} word {id} ({:?}) did not classify as {s:?}",
                decode(&v, *id)
            );
        }
    }
}

/// The multi-token frame marker survives every target.
///
/// `</think>` as `</` + `think` + `>`: the outer pieces are markup, and the
/// middle one is the ordinary English word that a script-only classifier has no
/// reason to spare. It survives because the protocol script is always kept —
/// which covers this marker and every other one without naming any of them.
///
/// That the *raw* primitive breaks it is asserted where the raw primitive
/// lives, in the module's own tests; it is private precisely so this test
/// cannot be written the wrong way round.
#[test]
fn a_split_frame_marker_survives_every_target() {
    let v = build();
    let pieces: Vec<u32> = SPLIT_FRAME.iter().map(|(i, _)| *i).collect();
    for target in [
        vec![Script::Han],
        vec![Script::Kana],
        vec![Script::Cyrillic],
        vec![Script::Thai],
        vec![],
    ] {
        let sup = v.bias_for_target(&target).suppress().clone();
        assert_none_suppressed(
            &v,
            &sup,
            &pieces,
            &format!("split marker under target={target:?}"),
        );
    }
}

/// Keeping the protocol script must not defeat the bias: the target's rivals
/// still go.
#[test]
fn keeping_latin_still_suppresses_the_other_languages() {
    let v = build();
    let sup = v.bias_for_target(&[Script::Han]).suppress().clone();
    for s in Script::ALL {
        let c = conversation(s);
        if s == Script::Han || s == Script::Latin {
            assert_none_suppressed(&v, &sup, &c.stream, &format!("{s:?} kept"));
        } else {
            assert_all_suppressed(&v, &sup, &c.language, &format!("{s:?} against Han+Latin"));
        }
    }
}

/// Suppressing everything still leaves a machine able to speak protocol and
/// arithmetic — the strongest statement of the carve-out.
#[test]
fn suppressing_all_scripts_leaves_exactly_the_frame_and_the_neutrals() {
    let v = build();
    let sup = v.bias_for_target(&[]).suppress().clone();
    for s in Script::ALL {
        // Latin is the protocol script and is kept unconditionally, so an empty
        // target means "English and the machinery, nothing else" rather than
        // "nothing at all".
        if s == Script::Latin {
            continue;
        }
        let c = conversation(s);
        assert_all_suppressed(&v, &sup, &c.language, &format!("{s:?} under target=[]"));
    }
    // Over the ids the fixture actually defines. The blocks leave gaps, and an
    // id no token occupies is in no set and so trivially survives — including
    // it would assert something about the fixture's numbering rather than about
    // the suppression.
    let mut defined: Vec<u32> = all_tokens().into_iter().map(|(i, _)| i).collect();
    defined.sort();
    let survivors: Vec<u32> = defined
        .iter()
        .copied()
        .filter(|t| !sup.contains(*t))
        .collect();
    // The frame, the neutrals, the whole split marker (its middle piece is
    // English and English is always kept), and the English words themselves.
    let mut expected: Vec<u32> = FRAME
        .iter()
        .chain(NEUTRAL.iter())
        .chain(SPLIT_FRAME.iter())
        .map(|(i, _)| *i)
        .collect();
    expected.extend(conversation(Script::Latin).language);
    expected.sort();
    expected.dedup();
    assert_eq!(
        survivors, expected,
        "an empty target should leave the machinery and English, and nothing else"
    );
}
