//! Checks a dream brief and its axis for the faults that make it unusable as a
//! dream, so that a retry or repair is taken only when it is an improvement.
//!
//! The checks are cheap and structural. A brief too short to be a dream fails, as
//! does one not addressed to the dreamer, one that ends on a conclusion, a
//! question, or a cut-off sentence, one that has the dreamer know it is dreaming,
//! and an assumption longer than one short line. Between two faulty attempts only
//! the presence of a brief is weighed, since ranking faults against each other
//! would be a judgement no structural check can make. A repair pass must also keep
//! the dream's nouns, so a pass that replaced the dream is refused.

use super::*;

/// The shortest brief that is a dream rather than a line about one.
///
/// A floor, not a quality bar. Briefs land anywhere from sixty to three hundred
/// words and the long ones are not better; what this catches is the decode that
/// answers the format instead of the question — one sentence under the `BRIEF:`
/// label, which carries no dream for the second conversation to decode.
pub(super) const BRIEF_FLOOR_WORDS: usize = 40;

/// The window a brief has to land in, in words.
pub(super) const BRIEF_MIN_WORDS: usize = 120;

pub(super) const BRIEF_MAX_WORDS: usize = 280;

/// How many words a dream's assumption may run to — "one short line".
///
/// See [`assumption_fault`] for the measured spread this sits inside.
pub(super) const ASSUMPTION_MAX_WORDS: usize = 30;

/// How much of a revision has to still be the thing it was revising.
///
/// A first setting rather than a derived one. Measured on the run that made this
/// necessary, a pass that kept the dream scored well above it and the pass that
/// replaced it scored far below, so the gap is wide — but the threshold itself
/// is a judgement and should move if a good revision is ever refused.
pub(super) const REVISION_OVERLAP: f32 = 0.35;

/// Whether a pass rewrote the dream instead of revising it.
///
/// **The repair passes are calibration, not regeneration.** That was the intent
/// and the wording did not hold it: every pass said *write that dream again*,
/// which is a generation instruction, and the model obliged. One measured run
/// went from a warm stone on a desk, to live wires and liquid fire, to a mouth
/// made of teeth eating the air — three different dreams in three passes, each a
/// competent answer to what it was literally asked.
///
/// Overlap of content words is the cheap test. A revision keeps its nouns: the
/// room, the objects, the one thing that is untrue. A fresh dream keeps almost
/// none of them, and there is no wording that makes that distinction as reliably
/// as counting does.
pub(super) fn rewritten(before: &str, after: &str) -> bool {
    let content = |s: &str| -> std::collections::HashSet<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 3)
            .map(|w| w.to_ascii_lowercase())
            .collect()
    };
    let (was, now) = (content(before), content(after));
    if now.is_empty() {
        return false;
    }
    let kept = now.iter().filter(|w| was.contains(*w)).count();
    (kept as f32 / now.len() as f32) < REVISION_OVERLAP
}

/// The dream call's check beyond the brief's own length: the other field.
pub(super) fn dream_call_fault(call: &str) -> Option<String> {
    assumption_fault(&field(call, "assumption").unwrap_or_default())
}

/// What is wrong with a dream's `assumption`, if anything.
///
/// **The contract is one short line, and nothing enforced it.** The tool says
/// so and so does `mind/layers/dreams/_dream-prompt-system.md`; the checks
/// looked only at `brief`. Measured: a repair pass was shown the dream as it
/// stood and filled `assumption` — the first parameter the grammar asks for —
/// with that whole dream, about a hundred and ninety words. It passed every
/// check and was adopted, and the next pass was told the entire dream was "the
/// assumption it suspends, which does not change", so the error pinned itself in
/// place for every pass after. It is the form-completing move the reflection's
/// opening turn describes, reached through the other field.
///
/// Every well-formed assumption measured was one sentence of nine to fifteen
/// words; the failure was two orders of magnitude longer.
/// [`ASSUMPTION_MAX_WORDS`] sits well clear of both.
pub(super) fn assumption_fault(a: &str) -> Option<String> {
    let t = a.trim();
    if t.is_empty() {
        return Some("the assumption is empty".into());
    }
    let n = sentences(t);
    if n > 1 {
        return Some(format!(
            "the assumption runs to {n} sentences; it is the one thing that has stopped being \
             true, in one short line"
        ));
    }
    let w = words(t);
    if w > ASSUMPTION_MAX_WORDS {
        return Some(format!(
            "the assumption is {w} words; it is the one thing that has stopped being true, in \
             one short line"
        ));
    }
    None
}

/// A brief as it came back, before it is known to be well-formed.
pub(super) struct Brief {
    pub(super) assumption: Option<String>,
    pub(super) brief: Option<String>,
}

impl Brief {
    /// Split a brief out of the call the answer grammar forced.
    ///
    /// Both fields are named parameters, so there is nothing to find and nothing
    /// to fail at — the grammar cannot emit a call missing either one.
    pub(super) fn from_call(answer: &str) -> Self {
        Self {
            assumption: field(answer, "assumption"),
            brief: field(answer, "brief"),
        }
    }

    /// Whether this attempt beats `other` when both are faulty.
    ///
    /// A clean attempt wins on its own and never reaches here. Between two faulty
    /// ones the only distinction worth drawing is whether there is a brief at all:
    /// a brief in the wrong person still carries a dream the second conversation
    /// can decode, and `None` carries nothing. Anything finer would be ranking
    /// faults against each other, which is a judgement no structural check has
    /// the standing to make.
    pub(super) fn beats(&self, other: &Self) -> bool {
        self.brief.is_some() && other.brief.is_none()
    }

    /// What is wrong with this brief, if anything — the gate on the retry.
    ///
    /// Cheap structural checks, no model involved. They catch the failures that
    /// actually recurred in testing and nothing else: a missing axis line, an
    /// axis that is a paragraph rather than a line, an axis the generator was
    /// explicitly steered away from and used anyway, a brief that ends on a
    /// realisation instead of on something happening, a brief written in the
    /// wrong person, a brief too short to be a dream, and one whose decode
    /// stopped mid-sentence.
    pub(super) fn fault(&self, steered_away_from: &[String]) -> Option<String> {
        let Some(a) = &self.assumption else {
            return Some("no assumption".into());
        };
        let Some(b) = &self.brief else {
            return Some("no brief".into());
        };
        // Here as well as refused in the exchange, because the exchange stops
        // arguing after `MAX_REFUSALS` and hands back its last answer anyway — and
        // a repair pass whose assumption is the whole dream must not be adopted
        // just because nobody was still refusing it.
        if let Some(w) = assumption_fault(a) {
            return Some(w);
        }
        let al = a.to_ascii_lowercase();
        if steered_away_from
            .iter()
            .any(|s| s.to_ascii_lowercase() == al)
        {
            return Some(format!("assumption repeats a sampled axis: {a}"));
        }
        // The last sentence is where the pull toward a closing thought lands,
        // and "do not conclude" was measurably too abstract to prevent it — a
        // decode once wrote the instruction itself into the prose. This checks
        // the one place it shows up.
        let tail = b
            .trim_end_matches(['.', '"', '\''])
            .rsplit(['.', '?', '!'])
            .find(|s| !s.trim().is_empty())
            .unwrap_or(b)
            .to_ascii_lowercase();
        const CONCLUDING: &[&str] = &[
            "realise",
            "realize",
            "understand",
            "means",
            "meaning",
            "what it meant",
        ];
        if b.trim_end().ends_with('?') {
            return Some("brief ends on a question".into());
        }
        if let Some(w) = CONCLUDING.iter().find(|w| tail.contains(**w)) {
            return Some(format!("brief ends on a realisation (\"{w}\")"));
        }
        // **The other shape a conclusion takes, and the word list misses it
        // entirely.** A brief closed with *"they do not exist to be met because
        // no one ever addressed you. The silence is not theirs, or even yours;
        // it is the absence of any one who would ever speak"* — no flagged word,
        // no question mark, and the dreamer has still arrived at what it meant.
        //
        // A causal connective in the final sentence is the signal, because it is
        // the grammar of explaining rather than of happening. The design already
        // applies this exact test to the line that crosses back to the character
        // (`docs/reflection_and_dreams.md` §6, "it contains no causal
        // connective") for the same reason: an account of *why* is a layer doing
        // the reflection layer's work.
        const EXPLAINING: &[&str] = &["because", "which is why", "the reason", "so that"];
        if let Some(w) = EXPLAINING.iter().find(|w| tail.contains(**w)) {
            return Some(format!("brief ends on an explanation (\"{w}\")"));
        }
        // **The brief is decoded by a conversation that is not this one.** It
        // arrives there as the instruction for a dream the dreamer is inside, so
        // "you are four levels deep" is the form and "I walk through the casting
        // hall" is a report of somebody else's dream. The prompt asks for `you`
        // and a decode drifts out of it anyway, roughly one run in three, and
        // the drift is whole-brief rather than a slip — which is what makes the
        // absence of the word a sound test rather than a heuristic. A brief
        // genuinely addressed to the dreamer cannot avoid saying it.
        // **The hard constraint, and nothing was watching it.** A non-lucid dream
        // has no metacognition — the dreamer does not know it is dreaming, which
        // `mind/layers/dreams/_dream-prompt-system.md` calls the one part of rule
        // three that is absolute. An escalating repair pass wrote *"everything
        // else is just a dream"* into the prose and every check passed it.
        const LUCID: &[&str] = &[
            "is just a dream",
            "is only a dream",
            "it is a dream",
            "this is a dream",
            "you are dreaming",
            "you wake up",
            "none of this is real",
        ];
        let all = b.to_ascii_lowercase();
        if let Some(w) = LUCID.iter().find(|w| all.contains(**w)) {
            return Some(format!("the dreamer knows it is dreaming (\"{w}\")"));
        }
        if !addresses_the_dreamer(b) {
            return Some("brief is not addressed to the dreamer as \"you\"".into());
        }
        if b.split_whitespace().count() < BRIEF_FLOOR_WORDS {
            return Some("brief is too short to carry a dream".into());
        }
        // **A decode that stopped mid-sentence passes every check above.** It
        // has both labels, it is in the second person, it is long enough, and
        // its last sentence is neither a question nor a realisation — because it
        // has no last sentence. Two briefs came back live ending on "the map
        // table on Level" and "sixteen orders spread across", and nothing
        // flagged either. Terminal punctuation is the one cheap signal that the
        // decode reached an end rather than being cut at one.
        if !b.trim_end().ends_with(ENDINGS) {
            return Some("brief stops mid-sentence".into());
        }
        None
    }
}

/// What the last character of a finished brief may be.
///
/// A closing quote or bracket counts: a brief may legitimately end on quoted
/// speech or a parenthetical, and the sentence inside it is still finished.
pub(super) const ENDINGS: [char; 9] = ['.', '!', '…', '"', '\'', '”', '’', ')', ']'];

/// Whether a brief is written in the second person.
///
/// Word-boundary matching rather than `contains`, because `your` is the hit that
/// matters and substring search would also take it out of the middle of another
/// word — and because the one thing this must never do is pass a first-person
/// brief that happens to mention a `young` Maker.
pub(super) fn addresses_the_dreamer(brief: &str) -> bool {
    const SECOND_PERSON: &[&str] = &["you", "your", "yours", "yourself", "you're"];
    brief
        .split(|c: char| !c.is_ascii_alphabetic() && c != '\'')
        .any(|w| {
            let w = w.to_ascii_lowercase();
            SECOND_PERSON.contains(&w.as_str())
        })
}
