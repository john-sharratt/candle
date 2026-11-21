//! Holds what one reflection produced, and checks the line that crosses back to
//! the character for the register the design requires.
//!
//! That line is one sentence with no causal connective, and it has to be framed as
//! an experience rather than stated as a claim, because an unframed line read back
//! as the character's own conclusion. Its length is counted in words, not tokens,
//! since a model asked to count its tokens cannot. The result keeps the answer, the
//! raw attempts, the retry and each repair pass apart, so that a response can say
//! what each one changed.

use super::*;

/// How many words the reflection may run to before it is refused.
///
/// **Words, not tokens.** The refusal is written back to the model, and a model
/// asked to count its own tokens cannot — told "that was 47 tokens, produce 200"
/// it pads rather than writes. Words are a unit it can actually hold.
pub(super) const REFLECTION_MAX_WORDS: usize = 40;

/// What one reflection produced, in the parts that mean different things.
///
/// Kept apart rather than returned as one blob because the pieces have
/// genuinely different lifetimes and audiences: `reflection` goes back to the
/// character, `assumption` is stored as the dream's diversity axis, `brief` is
/// handed to a separate conversation, and `raw` exists so a malformed answer can
/// be read rather than guessed at.
#[derive(Debug, Serialize)]
pub struct Reflection {
    /// Serialised as a base-36 string ([`crate::npcs::npc_id_wire`]) like
    /// every other character id on the wire — a `u64` past 2^53 does not
    /// survive a JavaScript client as a bare JSON number.
    #[serde(serialize_with = "crate::npcs::npc_id_wire_serde")]
    pub npc_id: u64,
    /// Echoed back so a caller reading a stored reflection knows what it was
    /// about without holding the request.
    pub situation: String,
    pub feeling: String,
    /// The behaviour-space cell this brief was drawn from.
    pub domain: String,
    /// The axes the generator was steered away from — a *sample* of the corpus,
    /// never all of it. See [`Reflect::run`].
    pub sampled_axes: Vec<String>,
    /// **Question one's answer. The only part the character ever sees.**
    pub reflection: String,
    /// The dream's axis, parsed from question two. `None` when the model did not
    /// produce the line, which is what `fault` reports on.
    pub assumption: Option<String>,
    /// The brief itself, parsed from question two.
    pub brief: Option<String>,
    /// **The brief that won** — question two's answer, or whichever retry or
    /// repair pass replaced it. This is what the dream conversation gets.
    pub raw: String,
    /// Question two's own answer, verbatim, before any retry or repair touched
    /// it.
    ///
    /// Identical to `raw` on a run where nothing displaced it, and the only
    /// record of what the review started from on a run where something did.
    /// Without it a response showing a repaired brief cannot say what the repair
    /// changed — the question the whole review exists to answer.
    pub original_raw: String,
    /// Whether the retry turn was sent.
    pub retried: bool,
    /// Why the first attempt was retried, if it was.
    pub fault: Option<String>,
    /// The retry's answer, **only when it was rejected and the first attempt
    /// kept**. `None` both when no retry was sent and when the retry was the one
    /// taken, in which case it is already `raw`.
    pub retry_raw: Option<String>,
    /// What was wrong with that rejected retry.
    pub retry_fault: Option<String>,
    /// What each repair pass did, in order. Empty when the mind authors none.
    pub repairs: Vec<Repair>,
    /// What is wrong with the reflection that crosses back, if anything.
    ///
    /// `docs/reflection_and_dreams.md` §6 names two invariants for this line and
    /// says both are testable and that nothing else would flag a regression: it
    /// is one sentence, and it carries no causal connective. Reported rather than
    /// retried — the line is already the character's, and a second ask for the
    /// same sentence is how a reflection becomes a negotiation.
    pub reflection_fault: Option<String>,
    /// Every assistant turn in order, verbatim — refused attempts included.
    ///
    /// **The refused ones are the point.** A response that showed only the
    /// answers that stood could not say whether the refusal loop is working, how
    /// many rounds it took, or what the model does when it is told its answer is
    /// the wrong length. Without this the conversation is not reconstructable
    /// from its own record, and reconstructing it is the only way to read it —
    /// the timeline is transient and reaches no disk.
    pub transcript: Vec<String>,
    /// Tokens decoded per turn, in order and one-to-one with `transcript`.
    ///
    /// Reported because a brief that stops mid-sentence is indistinguishable
    /// from its text alone from one the model chose to end there, and the two
    /// have different fixes. A count sitting on the EOS failsafe is a budget; a
    /// short one is the decode stopping on its own.
    pub tokens: Vec<usize>,
    /// The transient timeline this ran on, already tombstoned by the time a
    /// caller reads this. Reported for tracing, not for use.
    pub timeline: u64,
    pub ms: u64,
    /// The frame the character actually read.
    ///
    /// Reported because the alternative is guessing. A reflection that comes
    /// back reasoning about which act to call has been handed the acting
    /// instruction from somewhere, and no amount of reading the code settles
    /// where — the prompt does. The `simulate` route reports the daemon's own
    /// prompt for the same reason.
    pub system_prompt: String,
}

impl Reflection {
    /// The brief and assumption to dream from, or why there is no dream.
    ///
    /// **A brief whose assumption is not one short line is not dreamt.** The
    /// exchange refuses such an answer twice and then hands back what it has, and
    /// a faulty brief is otherwise still taken — a brief in the wrong person still
    /// carries a dream. The assumption is different: it is stored as the dream's
    /// axis and sampled back into every later reflection as something to steer
    /// away from. Measured: eleven of fifty kept dreams carried an "assumption" of
    /// one to two hundred words — the reflection's own loop of what the character
    /// had been doing, run on with *and… and…* — and the dreams decoded from those
    /// briefs repeated it nearly verbatim.
    pub fn dream(&self) -> Result<(&str, &str), String> {
        let (Some(brief), Some(assumption)) = (self.brief.as_deref(), self.assumption.as_deref())
        else {
            return Err("the reflection produced no brief".into());
        };
        match assumption_fault(assumption) {
            Some(fault) => Err(fault),
            None => Ok((brief, assumption)),
        }
    }
}

/// What one pass of the repair review did to the brief.
///
/// Recorded whether or not it was taken, because a pass that was rejected is the
/// only evidence of whether the question is working — a response that shows the
/// final brief and nothing else cannot distinguish a review that improved it
/// three times from one that was discarded three times.
#[derive(Debug, Serialize)]
pub struct Repair {
    /// 1-based, matching the order the mind authors them in.
    pub pass: usize,
    /// Whether this pass's brief replaced the one before it.
    pub took: bool,
    /// Why it did not, when it did not.
    pub fault: Option<String>,
    /// The pass's answer verbatim.
    pub raw: String,
}

/// What is wrong with the line that crosses back to the character.
///
/// Three invariants, all about register rather than content: a line that runs
/// to several sentences is a paragraph of reasoning; a line that explains itself
/// has stopped being an inclination and become an argument (§6: *"A response
/// that starts explaining itself is a regression, and nothing else would flag
/// it."*); and a line that states its image as a fact has stopped being a
/// reflection at all — see [`states_it_as_fact`].
pub(super) fn reflection_fault(line: &str) -> Option<String> {
    let t = line.trim();
    if t.is_empty() {
        return Some("the reflection is empty".into());
    }
    let n = sentences(t);
    if n > 1 {
        return Some(format!("the reflection runs to {n} sentences"));
    }
    let lower = t.to_ascii_lowercase();
    const EXPLAINING: &[&str] = &[
        " because ",
        " since ",
        " so that ",
        " in order to ",
        " which is why ",
    ];
    if let Some(w) = EXPLAINING.iter().find(|w| lower.contains(**w)) {
        return Some(format!("the reflection explains itself (\"{}\")", w.trim()));
    }
    if states_it_as_fact(t) {
        // Said back as the refusal, so it is written as something the model
        // can act on: what was wrong, and the shape of what would be right.
        return Some(
            "it says it as a fact rather than as how it is for you — say it as what you feel, \
             what it seems like, or what you find yourself wanting (\"I feel like…\", \"it seems \
             as if…\", \"I find myself…\")"
                .into(),
        );
    }
    None
}

/// Whether a reflection states what came to the character as a fact.
///
/// **A reflection reports an inner state; it does not make claims.** What
/// crosses back lands in the character's own context, where it is read the next
/// turn as something the character holds. Framed — *"I feel like I am the room
/// itself"* — it is an image the character had. Unframed — *"I am the room
/// itself"* — it is a statement about what the character is, and read back as
/// its own conclusion it becomes one: measured live, a character whose
/// reflections came back as *"I am the room itself"*, *"I am the vibration in
/// the pipe"*, *"I am the space between your words"*, one after another.
///
/// So the line has to carry an experiential frame — a feeling, a seeming, a
/// wanting, a comparison held as a comparison. The frames are the ways English
/// marks a thing as experienced rather than asserted; a line with none of them
/// is an assertion.
pub(super) fn states_it_as_fact(line: &str) -> bool {
    const FRAMES: &[&str] = &[
        "i feel",
        "i'm feeling",
        "i am feeling",
        "it feels",
        "feels like",
        "feels as",
        "i find myself",
        "i sense",
        "i wonder",
        "i want",
        "wanting",
        "i'd rather",
        "i would rather",
        "inclined",
        "i'm thinking",
        "i am thinking",
        "i think",
        "i can't help",
        "part of me",
        "it seems",
        "seems like",
        "seems as",
        "as if",
        "as though",
        "i notice",
        "drawn to",
        "i suspect",
        "tempted",
        "i keep",
        "i half",
        "i almost",
    ];
    let lower = line.to_lowercase().replace('\u{2019}', "'");
    !FRAMES.iter().any(|f| lower.contains(f))
}

/// How many sentences `t` holds.
///
/// Sentence-enders that are followed by more writing. A trailing full stop is
/// the end of the one sentence, not a second one.
pub(super) fn sentences(t: &str) -> usize {
    t.split_inclusive(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .count()
}

/// The reflection call's own check: its one field, held to §6's invariants.
pub(super) fn reflection_call_fault(call: &str) -> Option<String> {
    reflection_fault(&field(call, "said").unwrap_or_default())
}
