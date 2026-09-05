//! What writing system a token's bytes commit the model to emitting.
//!
//! # Script, not language
//!
//! A token's bytes identify a **script**, and only sometimes a language. Han
//! characters are written by Chinese and Japanese alike; Cyrillic by Russian,
//! Ukrainian and Bulgarian; Latin by English, French, German and a hundred
//! others. Nothing in a token's bytes separates English from French, so no
//! amount of care here will bias one against the other — that needs a prompt,
//! not a bitset.
//!
//! What it *does* separate is Latin from Han, Hangul, Cyrillic, Greek, Arabic,
//! Hebrew, Thai and Devanagari, which is exactly the axis a model drifting from
//! English into Chinese moves along. The type is named for what it can actually
//! tell apart.
//!
//! # Tokens are byte fragments, not text
//!
//! Byte-level BPE splits wherever the merges fell, so a token can hold a
//! trailing partial character, a leading continuation byte, or a whole
//! character's worth of bytes and nothing else. Classification therefore walks
//! UTF-8 by hand: a complete codepoint is classified, an incomplete tail is
//! classified from the lead byte's range alone (which already fixes the block),
//! and a stray continuation byte says nothing and is skipped.

/// A writing system a token can commit the model to emitting.
///
/// Deliberately coarse. Every variant is a range this classifier can decide
/// from bytes with no dictionary, and each is a block a model can drift into
/// wholesale.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Script {
    /// ASCII letters and the Latin-1/Extended blocks — English, and every
    /// other language written in this alphabet.
    Latin,
    /// CJK ideographs. Chinese and Japanese kanji both.
    Han,
    /// Japanese kana, which (unlike Han) is Japanese and nothing else.
    Kana,
    /// Korean.
    Hangul,
    Cyrillic,
    Greek,
    Arabic,
    Hebrew,
    Thai,
    Devanagari,
}

impl Script {
    /// Every script this classifier decides.
    pub const ALL: [Script; 10] = [
        Script::Latin,
        Script::Han,
        Script::Kana,
        Script::Hangul,
        Script::Cyrillic,
        Script::Greek,
        Script::Arabic,
        Script::Hebrew,
        Script::Thai,
        Script::Devanagari,
    ];

    /// The script a codepoint belongs to, or `None` for one that belongs to no
    /// language in particular.
    ///
    /// **`None` is the important answer.** Digits, punctuation, whitespace,
    /// symbols, and every byte of source code land here, which is what keeps
    /// them out of every blacklist and therefore out of every suppression set.
    /// Suppressing a language must never cost the model its ability to write a
    /// number or close a bracket.
    pub fn of_char(c: char) -> Option<Script> {
        let u = c as u32;
        match u {
            // Latin: ASCII letters, then Latin-1 Supplement letters through
            // Latin Extended-B. ASCII digits and punctuation are deliberately
            // absent — they are neutral.
            0x0041..=0x005A | 0x0061..=0x007A => Some(Script::Latin),
            0x00C0..=0x024F => Some(Script::Latin),
            0x0370..=0x03FF | 0x1F00..=0x1FFF => Some(Script::Greek),
            // Cyrillic and its Supplement, contiguous.
            0x0400..=0x052F => Some(Script::Cyrillic),
            0x0590..=0x05FF => Some(Script::Hebrew),
            0x0600..=0x06FF | 0x0750..=0x077F => Some(Script::Arabic),
            0x0900..=0x097F => Some(Script::Devanagari),
            0x0E00..=0x0E7F => Some(Script::Thai),
            0x3040..=0x309F | 0x30A0..=0x30FF | 0x31F0..=0x31FF => Some(Script::Kana),
            0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => Some(Script::Hangul),
            // CJK ideographs: the main block, Extension A, and compatibility.
            0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF => Some(Script::Han),
            // Supplementary ideographic plane.
            0x20000..=0x2FA1F => Some(Script::Han),
            // CJK punctuation (0x3000..0x303F) is deliberately NOT Han: the
            // ideographic space and corner brackets carry no language on their
            // own, and a model writing English prose never reaches for them.
            _ => None,
        }
    }

    /// The script a UTF-8 lead byte's codepoint must fall in, for a sequence
    /// cut short by the end of a token.
    ///
    /// A 3-byte lead pins the codepoint to a 4096-wide block, which is coarser
    /// than a full decode but enough to separate CJK from everything else. Used
    /// only for a truncated tail, where the alternative is to classify nothing
    /// and let a half-emitted character through.
    fn of_truncated(lead: u8, second: Option<u8>) -> Option<Script> {
        // 3-byte forms: 1110xxxx. The codepoint is
        // ((lead & 0x0F) << 12) | ((second & 0x3F) << 6) | ...
        if (0xE0..=0xEF).contains(&lead) {
            let hi = ((lead & 0x0F) as u32) << 12;
            let mid = second.map(|b| ((b & 0x3F) as u32) << 6).unwrap_or(0);
            // Probe the block's low end; every range this classifier decides is
            // at least 64 wide, so the low corner lands in the same block.
            return char::from_u32(hi | mid).and_then(Script::of_char);
        }
        // 4-byte forms: 11110xxx — the supplementary planes, where the only
        // range decided is ideographic.
        if (0xF0..=0xF4).contains(&lead) {
            let hi = ((lead & 0x07) as u32) << 18;
            let mid = second.map(|b| ((b & 0x3F) as u32) << 12).unwrap_or(0);
            return char::from_u32(hi | mid).and_then(Script::of_char);
        }
        // 2-byte forms cover Latin-1 through Arabic; without the continuation
        // byte the block is ambiguous, so nothing is claimed.
        None
    }
}

/// The scripts a token's bytes would emit.
///
/// A small ordered set — a token holds a handful of characters at most, and
/// callers only ever ask "does this contain script S". Returned rather than a
/// single script because a token CAN straddle two, and a straddling token emits
/// both: it must be suppressed if either is unwanted.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct ScriptSet {
    scripts: Vec<Script>,
}

impl ScriptSet {
    pub fn contains(&self, s: Script) -> bool {
        self.scripts.contains(&s)
    }

    pub fn is_empty(&self) -> bool {
        self.scripts.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = Script> + '_ {
        self.scripts.iter().copied()
    }

    fn add(&mut self, s: Script) {
        if !self.scripts.contains(&s) {
            self.scripts.push(s);
        }
    }
}

/// Classify a token's raw bytes.
///
/// `bytes` is the token's decoded byte string — what the detokenizer would
/// append to the output — not the byte-level-BPE surface form. A token that
/// emits no letters in any decided script returns an empty set and is therefore
/// in no blacklist.
pub fn scripts_of(bytes: &[u8]) -> ScriptSet {
    let mut out = ScriptSet::default();
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        let width = utf8_width(b);
        match width {
            // A continuation byte with no lead: the token began mid-character.
            // Which character is unknowable from here, so it claims nothing.
            0 => {
                i += 1;
            }
            1 => {
                if let Some(s) = Script::of_char(b as char) {
                    out.add(s);
                }
                i += 1;
            }
            w => {
                if i + w <= bytes.len() {
                    match std::str::from_utf8(&bytes[i..i + w]) {
                        Ok(s) => {
                            if let Some(sc) = s.chars().next().and_then(Script::of_char) {
                                out.add(sc);
                            }
                        }
                        // Well-formed width, malformed content. Fall back to the
                        // lead byte rather than skipping: the block is still
                        // pinned.
                        Err(_) => {
                            if let Some(sc) = Script::of_truncated(b, bytes.get(i + 1).copied()) {
                                out.add(sc);
                            }
                        }
                    }
                    i += w;
                } else {
                    // Truncated tail — the token ends mid-character. The lead
                    // byte still fixes the block.
                    if let Some(sc) = Script::of_truncated(b, bytes.get(i + 1).copied()) {
                        out.add(sc);
                    }
                    break;
                }
            }
        }
    }
    out
}

/// UTF-8 sequence width from a lead byte; `0` for a continuation byte.
fn utf8_width(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 0,
    }
}

// ── The script tenant of the generic facility ───────────────────────────────

use super::classes::{TokenBias, TokenClasses};

/// Token classes keyed by writing system.
pub type ScriptClasses = TokenClasses<Script>;

/// The script every protocol is written in.
///
/// Chat templates, tool-call syntax, JSON keys and function names are ASCII by
/// universal convention, so a selection that keeps Latin keeps every piece a
/// protocol string can decompose into — `think` in `</` + `think` + `>`
/// included — without knowing which strings exist.
pub const PROTOCOL_SCRIPT: Script = Script::Latin;

/// Characters marking a token as protocol rather than language.
///
/// Deliberately tiny. These three appear in chat templates and markup and
/// essentially never inside running prose in any script, so treating a token
/// carrying one as exempt costs nothing and protects the template.
const MARKUP: [char; 3] = ['<', '>', '|'];

/// Does this token carry protocol markup?
pub fn is_protocol_markup(bytes: &[u8]) -> bool {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.chars().any(|c| MARKUP.contains(&c)),
        // A token that is not valid UTF-8 on its own is a byte fragment; the
        // markup characters are ASCII, so a byte scan answers it exactly.
        Err(_) => bytes.iter().any(|b| matches!(b, b'<' | b'>' | b'|')),
    }
}

/// Classify a vocabulary by writing system.
///
/// `special` is the tokenizer's special/added token ids plus anything the chat
/// template uses structurally. Markup-bearing tokens are detected without it;
/// a special token that looks like an ordinary word is not, and only the
/// tokenizer knows.
pub fn script_classes<'a, I>(vocab: usize, tokens: I, special: &[u32]) -> ScriptClasses
where
    I: IntoIterator<Item = (u32, &'a [u8])>,
{
    TokenClasses::build(vocab, tokens, special, &is_protocol_markup, &|bytes| {
        scripts_of(bytes).iter().collect()
    })
}

impl ScriptClasses {
    /// The bias for steering toward `target`, with the protocol script kept
    /// unconditionally.
    ///
    /// The reason the protocol script is never dropped is the one problem the
    /// markup exemption cannot reach: a frame marker is only atomic if the
    /// tokenizer was trained with it, and where it was not, `</think>` is
    /// `</` + `think` + `>`. The middle piece is the ordinary English word —
    /// correctly Latin, correctly classified — and a Han target suppresses it,
    /// leaving the marker unsayable through nobody's mistake. Keeping Latin
    /// fixes that for every protocol string at once, without enumerating any.
    ///
    /// **What it gives up:** English can never be suppressed, so a "target
    /// Chinese, suppress English" mode is not expressible — deliberately. For
    /// an assistant whose protocol, code and identifiers are English there is
    /// no such mode.
    pub fn bias_for_target(&self, target: &[Script]) -> TokenBias {
        self.bias_for(target, &[PROTOCOL_SCRIPT])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scripts(s: &str) -> Vec<Script> {
        let mut v: Vec<Script> = scripts_of(s.as_bytes()).iter().collect();
        v.sort();
        v
    }

    #[test]
    fn latin_text_is_latin() {
        assert_eq!(scripts("hello"), vec![Script::Latin]);
        assert_eq!(scripts("Hello"), vec![Script::Latin]);
        assert_eq!(scripts("café"), vec![Script::Latin]);
        assert_eq!(scripts("Grüße"), vec![Script::Latin]);
    }

    #[test]
    fn cjk_and_friends_are_separated() {
        assert_eq!(scripts("中文"), vec![Script::Han]);
        assert_eq!(scripts("你好"), vec![Script::Han]);
        assert_eq!(scripts("ひらがな"), vec![Script::Kana]);
        assert_eq!(scripts("カタカナ"), vec![Script::Kana]);
        assert_eq!(scripts("한국어"), vec![Script::Hangul]);
        assert_eq!(scripts("Привет"), vec![Script::Cyrillic]);
        assert_eq!(scripts("Ελληνικά"), vec![Script::Greek]);
        assert_eq!(scripts("مرحبا"), vec![Script::Arabic]);
        assert_eq!(scripts("שלום"), vec![Script::Hebrew]);
        assert_eq!(scripts("ไทย"), vec![Script::Thai]);
        assert_eq!(scripts("हिन्दी"), vec![Script::Devanagari]);
    }

    /// The property the whole design rests on: anything that is not a letter of
    /// some script belongs to no language, so it can never land in a blacklist
    /// and can never be suppressed. Numbers, punctuation and operators are the
    /// tokens a coding assistant cannot afford to lose.
    #[test]
    fn neutral_tokens_claim_no_script() {
        for s in [
            " ", "\n", "\t", "", "0", "42", "3.14", ".", ",", ";", "!", "?", "()", "{}", "[]",
            "->", "=>", "::", "+=", "/*", "*/", "#", "$", "%", "&", "|", "^", "~", "@", "\"\"",
            "''", "`", "…", "—", "€", "©",
        ] {
            assert!(
                scripts_of(s.as_bytes()).is_empty(),
                "{s:?} must be neutral, got {:?}",
                scripts_of(s.as_bytes())
            );
        }
    }

    /// Source code is not neutral, and should not be: `0x1F` and `1e-9` carry
    /// Latin letters, identifiers are Latin words, and keywords are Latin. That
    /// is the correct classification rather than a leak — code survives because
    /// an English target KEEPS Latin, not because code was special-cased into
    /// neutrality. Asserted so the distinction stays deliberate.
    #[test]
    fn code_is_latin_and_therefore_kept_by_an_english_target() {
        for s in [
            "0x1F", "1e-9", "fn", "let", "self", "u32", "HashMap", "to_vec",
        ] {
            assert_eq!(
                scripts(s),
                vec![Script::Latin],
                "{s:?} should classify as Latin"
            );
        }
    }

    /// CJK punctuation is neutral on purpose — the ideographic space and corner
    /// brackets are not a language, and treating them as Han would suppress
    /// them for a model writing about Japanese typography in English.
    #[test]
    fn cjk_punctuation_is_neutral() {
        assert!(scripts_of("、".as_bytes()).is_empty());
        assert!(scripts_of("。".as_bytes()).is_empty());
        assert!(scripts_of("「".as_bytes()).is_empty());
        assert!(scripts_of("\u{3000}".as_bytes()).is_empty());
    }

    #[test]
    fn a_token_straddling_two_scripts_reports_both() {
        assert_eq!(scripts("API文档"), vec![Script::Latin, Script::Han]);
        assert_eq!(scripts("中1文"), vec![Script::Han]);
    }

    /// Byte-level BPE cuts wherever the merges fell, so a token can be the
    /// first two bytes of a three-byte character. The lead byte still fixes the
    /// block, and a half-emitted Han character must be classified Han or the
    /// suppression leaks.
    #[test]
    fn a_truncated_character_is_classified_from_its_lead_byte() {
        let full = "中".as_bytes();
        assert_eq!(full.len(), 3);
        let head = &full[..2];
        assert!(
            scripts_of(head).contains(Script::Han),
            "a truncated Han character must still read as Han"
        );
        let lead_only = &full[..1];
        assert!(scripts_of(lead_only).contains(Script::Han));
    }

    /// The mirror case: a token that BEGINS with a continuation byte cannot say
    /// what character it is finishing, and must not guess.
    #[test]
    fn a_stray_continuation_byte_claims_nothing() {
        let full = "中".as_bytes();
        assert!(scripts_of(&full[1..]).is_empty());
        assert!(scripts_of(&full[2..]).is_empty());
    }

    #[test]
    fn supplementary_plane_ideographs_are_han() {
        assert_eq!(scripts("\u{20000}"), vec![Script::Han]);
    }

    #[test]
    fn empty_and_invalid_bytes_are_survivable() {
        assert!(scripts_of(&[]).is_empty());
        assert!(scripts_of(&[0xFF, 0xFE]).is_empty());
    }
}
