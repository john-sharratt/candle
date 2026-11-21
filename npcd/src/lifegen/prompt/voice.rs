//! Gives each lifegen phase the character's own idiom, so that a life reads back
//! to its character in their own voice.
//!
//! The character's anchor is taken whole from the personality the engine already
//! holds for them, so the life and the character work from one description. An
//! author's voice, where one is given, is appended as the explicit statement. The
//! voice demonstrates the second person with a pronoun contrast and no scene,
//! since a model reproduces any image in a prompt as content. A character with
//! neither an anchor nor a voice still gets the plain diary framing rather than
//! an empty heading.

use super::*;

/// The character's own idiom, drawn from the personality they are a life for.
///
/// # Why a life needs the personality's voice and not just its facts
///
/// These documents are **injected into a character's prefix** — they are what
/// that character remembers, read back to them as their own memory. A memory
/// written in a neutral literary register is a memory in somebody else's words,
/// and stuffing a few thousand tokens of somebody else's words into a prefix
/// teaches the model to answer in them.
///
/// So the seed is not enough. A seed says a character was a shot-firer in a
/// quarry; the personality says he counts down out loud, does the sums where
/// people can check them, and never hurries. The first is biography and the
/// second is how the diary has to sound.
///
/// The anchor is taken whole because it is the same text the engine already
/// makes resident for that character, so a life written against it and a
/// character speaking from it are working from one description rather than two
/// that can drift. `personality.voice`, where an author wrote one, is appended
/// because it says explicitly what the anchor only demonstrates.
pub fn voice_of(personality: &Value) -> String {
    let mut out = String::new();
    if let Some(a) = personality.get("anchor").and_then(|v| v.as_str()) {
        out.push_str(a.trim());
    }
    if let Some(v) = personality
        .get("personality")
        .and_then(|p| p.get("voice"))
        .and_then(|v| v.as_str())
    {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str("How they speak: ");
        out.push_str(v.trim());
    }
    out
}

/// The diary instruction, wrapped around whatever [`voice_of`] found.
///
/// Empty when the personality carries neither an anchor nor a voice, so a life
/// for a character with no personality behind it still generates — plainly,
/// rather than against an empty heading that reads as a missing section.
pub(super) fn in_their_voice(voice: &str) -> String {
    if voice.trim().is_empty() {
        return String::new();
    }
    format!(
        "\n\nWHOSE VOICE THIS IS IN\n{voice}\n\n\
         Write it the way THIS person would write it in a diary — their vocabulary, their \
         rhythm, the things they notice and the things they would not bother to say. A \
         soldier who counts and a duelist who does not explain herself do not describe the \
         same afternoon the same way.\n\n\
         This is the character remembering, not somebody describing the character. Never \
         explain them from outside, never name their traits, and never write a sentence they \
         would not have written."
    )
}

/// How the daemon addresses the writer in every phase.
/// The voice every phase writes in.
///
/// # The example used to be a scene, and models wrote the scene
///
/// This demonstrated second person with a sentence: *"You came back to the yard
/// and the granary was gone."* It is a good sentence, which is the problem — a
/// yard and a burnt granary are an image, and an image in a prompt is content.
/// Two different models, a 9 B and a 14 B, both reproduced it word for word: a
/// Vietnamese duelist's life story opened on that yard, and so did a Zenling
/// combat drone's, whose entire existence is eleven kilometres of desert and
/// which has no yard, no granary and no capacity to miss one.
///
/// The illustration is now grammatical rather than pictorial — a pronoun
/// contrast with no scene attached, so there is nothing in it worth stealing.
/// Anything demonstrating style to a model has to be inert as content, because
/// a model cannot tell the two apart from position alone.
pub(super) const VOICE: &str = "\
You are writing one character's diary — their own account of their own life, \
which a game engine will read back to them as memory. It is not a biography and \
there is no narrator: the only person here is the one whose life it is.

Write in the second person, the way a diary written to oneself reads: \
\"You waited\" — never \"She waited\", and never the character's name as the \
subject. Plain past tense. No headings inside the prose, no bullet lists, no \
commentary about the writing.

Write what they would actually set down. A diary is uneven — it lingers on what \
mattered to THEM and passes over what did not, states without justifying, and \
leaves out what they would consider obvious. It does not summarise a person from \
outside or name their qualities; someone writing about their own year does not \
call themselves patient.

Write only this character's life. Nothing from any example, instruction or \
other person's record above belongs in the prose.";
