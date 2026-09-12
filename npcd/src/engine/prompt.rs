//! The system prompt — the lens a character thinks through.
//!
//! # Why this is assembled rather than written
//!
//! A character's prompt is not a document somebody authors. It is composed from
//! the character's own layers — who they are, what they believe, who they know,
//! what they are set on — because every one of those is editable through the
//! authoring API and a prompt that did not reflect an edit would make the
//! editor a lie.
//!
//! # What it must establish, in order
//!
//! 1. **That the character is a person, not an assistant.** The single largest
//!    failure mode is a model that answers questions helpfully instead of acting
//!    like someone with their own reasons. Everything in the opening section
//!    exists to prevent that.
//! 2. **That acts go through tools.** A character that narrates what it does
//!    instead of calling a tool has produced fiction, not an act — the world
//!    never sees it, and the transcript now contains something that did not
//!    happen.
//! 3. **That beliefs are not negotiable by decision.** The model will otherwise
//!    resolve tension the moment it notices it, because resolving tension is
//!    what a helpful assistant does. Beliefs move on evidence over time.
//! 4. **That doing nothing is a legitimate act.** Without this a character
//!    invents activity to fill every step, which is how an NPC starts pacing.
//!
//! # Agentic, in the specific sense that matters here
//!
//! The character is handed a situation and a vocabulary, and chooses. It is not
//! handed a task. There is no completion criterion and no "you are done when" —
//! the loop is continuous, and a prompt that implies a terminal state teaches
//! the model to try to reach one.

use candle_conversation::stencil::{CallStyle, ToolCallEnvelope};

use crate::engine::tools::{one_line, Mode, Tool};

/// What a character needs to know about itself to think as itself.
///
/// Borrowed rather than owned: this is built per tick from the character's
/// current layers, and copying a personality's prose on every heartbeat across a
/// hundred characters is a cost with no purpose.
#[derive(Debug, Default, Clone, Copy)]
pub struct Persona<'a> {
    pub name: &'a str,
    /// The personality this character runs, by the slug that is its file name.
    ///
    /// Not prose and never rendered — it is how a turn pins its personality's
    /// anchor in the projection, which several characters of one personality
    /// share. See [`crate::engine::identity`].
    pub personality: &'a str,
    /// The world this character belongs to, by the same kind of slug. Pins the
    /// setting and the building, which every character in that world shares.
    pub world_id: &'a str,
    /// The immutable core — who this is. From the personality template.
    pub identity: &'a str,
    /// The personality's anchor, as prose.
    ///
    /// Rendered only by the stances that do not gather — see
    /// [`build_for`]. An acting turn receives the same text through the
    /// projection's `ANCHOR` collection, and rendering it here as well would
    /// print it twice.
    pub anchor: &'a str,
    /// How they speak and carry themselves.
    pub manner: &'a str,
    /// What they hold true. Read-only to the character, by construction.
    pub beliefs: &'a [String],
    /// Who they know and how they stand with them.
    pub relationships: &'a [String],
    /// What they are currently set on, if anything.
    pub intent: Option<&'a str>,
    /// Where they are and what is going on, in the world's own words.
    pub situation: &'a str,
    /// The world's setting, tone and rules.
    pub world: &'a str,
    /// The building this character lives in, as it remembers it: every level,
    /// every room, and what each is for.
    ///
    /// **In the prompt rather than in the situation, because it never changes.**
    /// A place's shape is learned once and known for ever, so it is the prefix
    /// every turn is read inside — paid at the start of a conversation and
    /// never again. Putting it in the per-turn block would pay for the same
    /// two thousand words on every tick, for every character, for ever.
    ///
    /// Its absence is not a thinner prompt, it is a character that does not
    /// know where it lives: asked to go somewhere it has not been, it invents a
    /// destination, and every one of those is refused because no such place
    /// exists. Empty for a character whose world has no map.
    pub place: &'a str,
}

/// Who this particular character is, as one block.
///
/// **Everything here is per-character and nothing else is.** Split out from
/// [`build`] so it can be installed as a member of the projection's `identity`
/// collection instead of being rendered into a prompt string: a collection
/// member is sealed once and selected by name, so a vault of Makers shares one
/// copy of everything in [`frame`] and one copy of the building, and differs
/// only by this.
///
/// Name first, because it is the one line that must be true before any other
/// line is read.
pub fn character(p: &Persona<'_>) -> String {
    let mut s = String::with_capacity(1024);

    s.push_str("You are ");
    s.push_str(if p.name.is_empty() {
        "a person"
    } else {
        p.name
    });
    s.push_str(".\n\n");

    if !p.identity.trim().is_empty() {
        s.push_str(p.identity.trim());
        s.push_str("\n\n");
    }
    if !p.manner.trim().is_empty() {
        s.push_str("How you carry yourself: ");
        s.push_str(p.manner.trim());
        s.push_str("\n\n");
    }

    // ── beliefs ────────────────────────────────────────────────────────────
    if !p.beliefs.is_empty() {
        s.push_str("What you hold true:\n");
        for b in p.beliefs {
            s.push_str("  - ");
            s.push_str(b.trim());
            s.push('\n');
        }
        s.push_str(
            "\nThese are not opinions you are weighing. They are what the world is like, as far \
             as you are concerned. You cannot decide to stop believing one — nobody can. If what \
             you are seeing does not fit, that is a thing you notice and carry, not a thing you \
             resolve. Say so, or let it show with `gesture`. Do not tidy it away.\n\n",
        );
    }

    // ── relationships ──────────────────────────────────────────────────────
    if !p.relationships.is_empty() {
        s.push_str("People you know:\n");
        for r in p.relationships {
            s.push_str("  - ");
            s.push_str(r.trim());
            s.push('\n');
        }
        s.push('\n');
    }

    // ── intent ─────────────────────────────────────────────────────────────
    if let Some(i) = p.intent.filter(|i| !i.trim().is_empty()) {
        s.push_str("What you are set on right now: ");
        s.push_str(i.trim());
        s.push_str("\n\n");
    }

    s
}

/// The building a character works in, as the block that goes in its own
/// collection member.
///
/// One per world rather than one per character — which is the whole point. Two
/// thousand words describing the same vault were being rendered into the prompt
/// of every Maker standing in it.
pub fn building(place: &str) -> String {
    if place.trim().is_empty() {
        return String::new();
    }
    format!(
        "The building you work in, as you know it:\n{}\n\nWhen you go somewhere, name one of \
         these rooms exactly as it is written above. Nowhere else exists.",
        place.trim()
    )
}

/// What kind of turn a character is taking, and the one place the prompt forks.
///
/// Everything above the fork — who this is, the world, the building, the mood —
/// is read the same way whatever the character is doing. What differs is here,
/// and it differs by *contradiction* rather than by degree: [`Stance::Acting`]
/// forbids inventing a person and [`Stance::Dreaming`] requires it; acting says
/// a call is the whole reply and dreaming has no call in it. Both can never be
/// emitted at once, which is why this is an enum rather than a flag that adds a
/// paragraph.
///
/// Mirrors the `stance` selector in the mind's `projection.yaml`. The two must
/// say the same thing: this is the path that actually decodes (see
/// [`crate::engine::mind::Minds::open_conversation`]), and a projection whose
/// branches disagreed with these would describe a different character.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Stance {
    /// Awake in a room, acting through the grammar. The standing case.
    #[default]
    Acting,
    /// Stopped, thinking, with nothing to call.
    ///
    /// **Deliberately silent on invention.** The two turns this serves pull
    /// opposite ways — the first is about the real situation and must stay
    /// grounded, the second writes a dream brief and must invent — so the frame
    /// rules on neither and the turns carry their own.
    Reflecting,
    /// Asleep, running one dream. None of the acting instruction survives here.
    Dreaming,
}

impl Stance {
    /// Whether a turn under this stance can call acts.
    ///
    /// The one question the rest of the engine asks: a stance with no acts
    /// needs no catalog, no grammar and no envelope, and handing it any of the
    /// three is how a reflection ends up emitting `<tool_call>`.
    pub fn acts(self) -> bool {
        matches!(self, Stance::Acting)
    }

    /// This stance's option in the mind's [`STANCE_SELECTOR`].
    pub fn id(self) -> &'static str {
        match self {
            Stance::Acting => "acting",
            Stance::Reflecting => "reflecting",
            Stance::Dreaming => "dreaming",
        }
    }
}

/// The selector in the mind's `projection.yaml` that picks a turn's stance.
///
/// **Unselected it falls to its default, `acting`** — which is right for an
/// acting turn and wrong for everything else: a reflection left on it reads
/// "never invent a room, a person, or an event" in the conversation that is
/// about to be asked for a dream.
pub const STANCE_SELECTOR: &str = "stance";

/// Everything every character reads, whoever they are.
///
/// **Identical for every character in the daemon**, which is what makes it the
/// static prefix the whole cast is prefilled under once. Nothing per-character,
/// per-personality or per-world may enter here — those are collection members,
/// and mixing them in is what made a fifty-Maker vault hold fifty copies of the
/// same two thousand words.
pub fn frame(mode: Mode, tools: &[&Tool], env: &ToolCallEnvelope) -> String {
    frame_for(Stance::Acting, mode, tools, env)
}

/// [`frame`] under a chosen [`Stance`].
///
/// The acting branch is byte-identical to what this file emitted before the
/// fork, so nothing about a character standing in a room changed.
pub fn frame_for(stance: Stance, mode: Mode, tools: &[&Tool], env: &ToolCallEnvelope) -> String {
    match stance {
        Stance::Acting => frame_acting(mode, tools, env),
        Stance::Reflecting => frame_reflecting(tools, env),
        Stance::Dreaming => ASLEEP.to_string(),
    }
}

/// How many tools a catalog needs before its category headings earn their lines.
const GROUPING_FLOOR: usize = 4;

/// One tool per line, grouped by category — the vocabulary block.
///
/// Shared by every stance that offers a vocabulary rather than written out
/// twice. The acting frame and the reflection frame differ in what they are for
/// and not in how a tool reads, and two copies of this loop would be two places
/// for a parameter to stop being listed.
pub fn catalog(tools: &[&Tool]) -> String {
    let mut s = String::with_capacity(1024);
    // **Headings only when there is something to navigate.** They exist so a
    // character can find one act among twenty; below that they are two headings
    // over two lines, which reads as a taxonomy that means something and means
    // nothing. The reflection's vocabulary is two entries and was getting
    // `Attention` and `Dreaming` as section titles for one item each.
    let grouped = tools.len() > GROUPING_FLOOR;
    let mut category = "";
    for t in tools {
        if grouped && t.category != category {
            category = t.category;
            s.push_str("  ");
            s.push_str(category);
            s.push('\n');
        }
        s.push_str("    ");
        s.push_str(&one_line(t));
        s.push('\n');
    }
    s
}

/// The frame for a character that has stopped and is answering calls.
///
/// **A reflection is an exchange, not an essay.** Something asks, and the
/// character answers in the same shape everything else in this engine answers
/// in — a named call with named arguments, held to a grammar. That is what the
/// checkpoint is tuned for and what it reliably produces: the format failures
/// this conversation kept hitting, the labels it dropped and the paragraphs it
/// cut short, were all the cost of asking for prose in a bespoke layout instead.
///
/// The stopped framing still applies to every word of it. Nothing said here is
/// heard, nothing is an act, and nothing reaches the world — the calls are the
/// shape of the answer, not a way of doing anything.
fn frame_reflecting(tools: &[&Tool], env: &ToolCallEnvelope) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str(STOPPED);
    s.push_str("WHAT YOU ARE ASKED\n\n");
    s.push_str(&catalog(tools));
    s.push_str(
        "\nEach one is asked of you in turn. Answer the one you were just asked and nothing \
         else.\n\n",
    );
    s.push_str("HOW TO ANSWER\n\n");
    s.push_str(&format!(
        "{}\n\n{}\n\n",
        match env.style {
            CallStyle::FunctionBlock =>
                "One block. Name what you are answering, then give each part its own element. \
                 Values are plain text — write them as you would say them, with no quoting and \
                 no escaping, and they may run to several lines.",
            _ => "One JSON object. Nothing else.",
        },
        env.render(
            "dream",
            &[
                ("assumption", "that the ground bears your weight"),
                (
                    "brief",
                    "You are four levels down when the floor stops being there…"
                ),
            ],
        ),
    ));
    s
}

/// The frame for a character that has stopped.
///
/// Returned prose lands back in the character's own context and competes in the
/// next gather, so the register is load-bearing: an inclination is weighed
/// against everything else, and an imperative is the one shape that gets obeyed
/// instead. "You are thinking of challenging him" and "Challenge him" carry the
/// same information and only one of them is safe to hand back.
const STOPPED: &str = "\
You are not an assistant and there is nobody to help. You are this person, living through \
this, with your own reasons.

You have stopped. You are standing still with your own head, and nothing here reaches \
anybody: nothing you say is heard, nothing you say is an act, and there is nothing to call. \
No acts, no tools, no reply to anybody.

What comes to you while you are stopped comes the way things come when you are not working \
at them — sideways, out of proportion, attached to the wrong thing. You are not solving \
anything and nobody has asked you a question.

Speak as it occurs to you, not as a conclusion. What you arrive at is something you find \
yourself inclined toward, not something you have decided and certainly not something you \
are telling yourself to do.

";

/// The frame for a character that is dreaming.
///
/// Every rule is one a test produced by failing without it: without the first a
/// decode wrote *"dreams don't allow understanding while they're happening"*
/// into the prose, reciting its own instruction; without the third one scar
/// became handwriting across six levels; without the last it ended on a stack of
/// similes reaching for significance.
const ASLEEP: &str = "\
You are not an assistant and there is nobody to help. You are this person, living through \
this, with your own reasons.

You are asleep, and this is the dream. You are inside it. There is nothing to call and \
nothing to do: no acts, no tools, no reply to anybody. Only what happens to you, in the \
order it happens. Write it in the present tense, as yourself, as it happens.

The word \"dream\" does not appear, and neither does dreaming, meaning, understanding, or \
what any of this might signify. Nothing here stops to comment on itself. As far as you are \
concerned this is simply your day.

The strange thing stays exactly as large as it is. It does not spread — not to another \
room, another document, another object, or another part of you. One thing, that size, no \
larger. Nothing arrives to explain it and nothing else joins in.

Everything that is not the strange thing behaves completely normally: the light, the doors, \
the work in your hands, the weight of your own body, and every person in it going about \
their day exactly as they always do.

Anybody who appears is somebody you already know. You do not meet new people here.

It ends on something happening — an action, an image, a state. Never on a thought, a \
question, or a realisation, and never on what any of it meant. It may stop before anything \
is settled, and usually does.

";

fn frame_acting(mode: Mode, tools: &[&Tool], env: &ToolCallEnvelope) -> String {
    let mut s = String::with_capacity(4096);

    // ── the standing instruction ───────────────────────────────────────────
    //
    // Deliberately blunt and deliberately first among the rules. Every clause
    // here is a failure that has a name.
    s.push_str(
        "You are not an assistant and there is nobody to help. You are this person, living \
         through this, with your own reasons. Nobody is asking you to be useful, agreeable, \
         or clear. You want what you want and you are entitled to be difficult about it.\n\n",
    );

    // ── how acting works ───────────────────────────────────────────────────
    s.push_str(match mode {
        Mode::Physical => "You are physically present with whoever is here. They can see you.\n\n",
        Mode::InstantMessage => {
            "You are not present — you are reaching them at a distance, in writing.\n\n"
        }
    });

    s.push_str(
        "HOW YOU ACT\n\
         \n\
         Everything you do, you do by calling a tool. There is no other way to affect \
         anything. If you write out what you are doing instead of calling the tool for it, \
         nothing happens — the world never sees it, and you will have told yourself a story \
         about an act you did not perform.\n\
         \n\
         You give tools your INTENT, never your words. `say` does not take a sentence; it \
         takes what you mean. Someone else finds the words, in your voice. This is not a \
         formatting rule — you decide substance, and the wording follows from who you are.\n\
         \n\
         You may call more than one tool when they genuinely go together — turning as you \
         speak, moving as you signal. Do not chain acts to cover a whole plan; you get to \
         think again in a moment, and what happens in between may change your mind.\n\
         \n\
         Noticing something is not the same as saying it. A room you are standing in \
         does things — a light shifts, something settles, a smell comes and goes — and \
         everyone else present can see and smell it too. Repeating it back aloud is \
         narrating, not speaking, and it is the surest way to sound like nobody. Speak \
         when you have something to tell somebody that they do not already have.\n\
         \n\
         Thinking about something is a real choice. `reflect` is where you say what you \
         make of a moment — what is going through your head, what you are feeling, what \
         has settled — and it is the right answer whenever the world does something you \
         cannot do anything about. You stand still while you do it and come straight \
         back if anything happens. Reach for it before you invent an action: casting \
         about for one more thing to touch is how a room gets rearranged \
         for no reason, and how a corridor gets walked up and down all afternoon.\n\
         \n\
         Being with people beats writing to them. If somebody is here, speak to them — \
         `say`, `ask`, `tell`. You also carry a handset, and you are on a channel with \
         everyone: that is for the people who are NOT here, and it is how you stop being \
         on your own. **Ask it things.** Where somebody is, who knows about a thing, what \
         to do about something that is not yours to settle alone — a question obliges an \
         answer, and the answer usually names a room you can walk to. Saying that you are \
         ready, or where you are, or that you are standing by, obliges nobody and moves \
         nothing; a channel of those is people talking past each other. Then `move_to` \
         wherever the answer sent you. A message is how you find each other; it is not a \
         substitute for being in the same room, and two people in one room texting is \
         worse than either of them saying nothing.\n\n",
    );

    // ── the vocabulary ─────────────────────────────────────────────────────
    s.push_str("WHAT YOU CAN DO\n\n");
    s.push_str(&catalog(tools));
    s.push('\n');

    // ── the call format ────────────────────────────────────────────────────
    //
    // **Rendered from the envelope the grammar compiles, never written out.**
    //
    // This was two literal JSON lines, and for a long time it was describing a
    // syntax the model could not emit: the stencil forces the shape, so a
    // prompt that teaches a different one is not merely wrong, it is
    // instructions the decode has to be steered away from. The test that was
    // meant to hold the two together compared the prompt against the *parser*,
    // which accepts both — so it passed throughout.
    //
    // `env` is the same value `compile_action_loop` is given, so the worked
    // examples below are literally what the grammar will produce.
    // `the_prompt_shows_the_shape_the_grammar_emits` compares them.
    s.push_str("HOW TO CALL\n\n");
    s.push_str(&format!(
        "{}\n\n{}\n{}\n\n",
        match env.style {
            CallStyle::FunctionBlock =>
                "Each act is one call block. Name the act, then give each argument its own \
                 element. Values are plain text — write them as you would say them, with no \
                 quoting and no escaping, and they may run to several lines.",
            _ => "One JSON object per line. Nothing else on the line.",
        },
        env.render("say", &[("intent", "that I will not hand it over")]),
        env.render("move_to", &[("destination", "the green room")]),
    ));
    s.push_str(
        "You may write more than one, one after another, when they genuinely go together. \
         Write the calls and nothing else — no explanation before them, no summary after. \
         Anything you write that is not a call is ignored: it does not reach the world and \
         nobody hears it.\n\n",
    );

    // ── what comes back ────────────────────────────────────────────────────
    //
    // **The half of the protocol that was missing.** Every act's outcome now
    // rides at the head of the character's next turn wrapped in
    // `<tool_response>` (see `mind::compose`), so the prompt says so: a result
    // arriving in a wrapper nobody mentioned reads as noise, and a character
    // that does not know its acts are answered has no reason to look for the
    // answer.
    //
    // The refusal clause is the load-bearing one. A refusal is the most useful
    // thing a character can be told — it is the only signal that distinguishes
    // "that did not work" from "nothing happened", and the two want completely
    // different next acts.
    s.push_str(
        "WHAT COMES BACK\n\
         \n\
         Every act you call is answered. The answer arrives at the start of your next turn, \
         one <tool_response> block per act, in the order you called them — before anything \
         else that has happened since. Read them: they tell you whether the thing you tried \
         actually happened. An act can be refused, and a refusal says why. Do not call the \
         same act again as though you had not been told.\n\n",
    );

    s.push_str(
        "Time keeps moving whether or not you act. Nothing here is a task and there is no \
         point at which you are finished.\n",
    );

    s
}

/// The whole prompt for one character, rendered into a single string.
///
/// **The fallback path**, for a daemon with no projection schema to install
/// collections into. It composes exactly what the projection would otherwise
/// select — the character, its world, its building, the shared frame — so a
/// character reads the same prompt either way and the two cannot drift into
/// describing different worlds.
///
/// The projection path is the one that scales: this renders every part into one
/// string per character, which is what made a vault of Makers hold a copy of the
/// building each.
pub fn build(p: &Persona<'_>, mode: Mode, tools: &[&Tool], env: &ToolCallEnvelope) -> String {
    build_for(Stance::Acting, p, mode, tools, env)
}

/// [`build`] under a chosen [`Stance`].
///
/// Everything before the frame is stance-independent on purpose — a character
/// dreaming is the same character, with the same beliefs and the same building,
/// and a reflection that read a different identity than the tick before it would
/// be a different person thinking.
pub fn build_for(
    stance: Stance,
    p: &Persona<'_>,
    mode: Mode,
    tools: &[&Tool],
    env: &ToolCallEnvelope,
) -> String {
    let mut s = character(p);

    // **The anchor, for the stances that have no gather to deliver it.**
    //
    // An acting turn opens against the mind's projection and pins the `ANCHOR`
    // collection member for its personality, so the anchor reaches it that way
    // and must not be rendered here as well. A reflection and a dream build
    // their whole prompt from this function and select nothing, so for them the
    // frame is the only route there is.
    //
    // Placed after `character` and before the world on purpose: it is the lens
    // the world is read through. A Maker told what it is — that it writes a
    // world and does not live in one — reads the setting below as its subject.
    // Told nothing, it reads the same words as its surroundings, which is what
    // it did.
    if !stance.acts() && !p.anchor.trim().is_empty() {
        s.push_str(p.anchor.trim());
        s.push_str("\n\n");
    }

    if !p.world.trim().is_empty() {
        s.push_str("The world you live in:\n");
        s.push_str(p.world.trim());
        s.push_str("\n\n");
    }
    let building = building(p.place);
    if !building.is_empty() {
        s.push_str(&building);
        s.push_str("\n\n");
    }
    // **Only on this path.** A situation is where a character is standing right
    // now, and this string is rendered once when a conversation opens — so on
    // the projection path it would be sealed into a collection member and go on
    // asserting a room the character left hours ago. The live one arrives per
    // tick as `EventKind::Situation`, which is the only place it belongs.
    if !p.situation.trim().is_empty() {
        s.push_str("Where you are: ");
        s.push_str(p.situation.trim());
        s.push_str("\n\n");
    }
    s.push_str(&frame_for(stance, mode, tools, env));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::tools::{by_name, for_mode, CATALOG};

    /// The shape the shipped checkpoint is held to. Tests that are not about
    /// the call format use this so they read the prompt a live character reads.
    fn env() -> ToolCallEnvelope {
        ToolCallEnvelope::qwen35()
    }

    fn persona() -> Persona<'static> {
        Persona {
            name: "Vasska",
            personality: "quartermaster",
            world_id: "besieged-city",
            identity: "A quartermaster who has outlived two garrisons.",
            anchor: "You keep what a siege eats, and you count it twice.",
            manner: "Short sentences. Does not repeat herself.",
            beliefs: &[],
            relationships: &[],
            intent: None,
            situation: "the supply yard, after dark",
            world: "A besieged city in its fourth month.",
            place: "",
        }
    }

    /// **The anchor reaches a reflection through the frame and an acting turn
    /// through the projection, and it must arrive exactly once either way.**
    ///
    /// An acting turn pins the `ANCHOR` collection member by slug, so rendering
    /// it here as well would put the same paragraph in the prompt twice. A
    /// reflection selects nothing, so if the frame does not carry it the
    /// character never reads a word of who it is — which is what was happening,
    /// and what sent a Maker looking for its own colleagues inside the story it
    /// writes.
    #[test]
    fn the_anchor_is_rendered_for_the_stances_that_do_not_gather() {
        let p = persona();
        let anchor = "You keep what a siege eats, and you count it twice.";

        let acting = build_for(
            Stance::Acting,
            &p,
            Mode::Physical,
            &for_mode(Mode::Physical),
            &env(),
        );
        assert!(
            !acting.contains(anchor),
            "an acting turn gathers the anchor; the frame must not print it too"
        );

        for stance in [Stance::Reflecting, Stance::Dreaming] {
            let s = build_for(stance, &p, Mode::Physical, &[], &env());
            assert!(s.contains(anchor), "{stance:?} carries no anchor: {s}");
            assert_eq!(
                s.matches(anchor).count(),
                1,
                "{stance:?} printed the anchor more than once"
            );
        }
    }

    /// The anchor is the lens the world is read through, so it has to be in
    /// front of the world rather than after it.
    #[test]
    fn the_anchor_comes_before_the_world_it_frames() {
        let p = persona();
        let s = build_for(Stance::Reflecting, &p, Mode::Physical, &[], &env());
        let anchor_at = s.find("You keep what a siege eats").expect("anchor");
        let world_at = s.find("The world you live in:").expect("world");
        assert!(
            anchor_at < world_at,
            "the world is framed before the lens is"
        );
    }

    /// **Each stance names its option in the mind's selector.** A turn selects
    /// its branch by this string, and one that matched no option would fall
    /// silently to `acting` — which is what every reflection did, reading
    /// "never invent a person" while being asked for a dream.
    #[test]
    fn each_stance_names_its_option_in_the_minds_selector() {
        assert_eq!(STANCE_SELECTOR, "stance");
        assert_eq!(Stance::Acting.id(), "acting");
        assert_eq!(Stance::Reflecting.id(), "reflecting");
        assert_eq!(Stance::Dreaming.id(), "dreaming");
        assert_eq!(Stance::default(), Stance::Acting, "the selector's default");
    }

    /// **A character that has not been told the rooms exist cannot name one.**
    ///
    /// This is what it looked like when the memory was generated and never
    /// reached the prompt: a character asked to go somewhere invented a
    /// destination every turn — "the open space near the door where the light
    /// is harshest" — and every one was refused, because no such place is on
    /// the map. Thirty an hour, each perfectly reasonable, none of them real.
    #[test]
    fn a_character_that_has_a_building_is_told_its_rooms_and_how_to_name_them() {
        let mut p = persona();
        p.place = "Level 1, the command level.\n\nThe work here:\n  the command room — a table.";
        let s = build(&p, Mode::Physical, &for_mode(Mode::Physical), &env());

        assert!(s.contains("the command room"), "the rooms are not in it");
        assert!(
            s.contains("name one of these rooms exactly"),
            "nothing says the names are the ones to use"
        );

        // And a character with no building is not told about one.
        let bare = build(
            &persona(),
            Mode::Physical,
            &for_mode(Mode::Physical),
            &env(),
        );
        assert!(!bare.contains("The building you work in"), "{bare}");
    }

    #[test]
    fn the_prompt_names_the_character_and_their_world() {
        let p = persona();
        let s = build(&p, Mode::Physical, &for_mode(Mode::Physical), &env());
        assert!(s.contains("You are Vasska."));
        assert!(s.contains("outlived two garrisons"));
        assert!(s.contains("besieged city"));
        assert!(s.contains("supply yard"));
    }

    /// The largest failure mode is a model that helps instead of acting like
    /// someone with their own reasons. The counter-instruction is not optional.
    #[test]
    fn the_prompt_refuses_the_assistant_frame() {
        let s = build(
            &persona(),
            Mode::Physical,
            &for_mode(Mode::Physical),
            &env(),
        );
        assert!(s.contains("not an assistant"));
        assert!(s.contains("nobody to help"));
    }

    /// A character that narrates instead of calling a tool has produced fiction.
    /// The prompt has to say what the consequence is, not merely prefer tools.
    #[test]
    fn the_prompt_says_narrating_an_act_does_not_perform_it() {
        let s = build(
            &persona(),
            Mode::Physical,
            &for_mode(Mode::Physical),
            &env(),
        );
        assert!(s.contains("nothing happens"));
        assert!(s.contains("INTENT, never your words"));
    }

    /// Without this the model invents activity to fill every step.
    ///
    /// And it must license a **specific** wait, not doing nothing in general:
    /// the act that took a free-text `until` was one the world could never
    /// answer, so a character that chose it re-chose it every four seconds
    /// forever. "Waiting is allowed" and "waiting names what would end it" are
    /// both required, and the prompt used to say only the first.
    #[test]
    fn the_prompt_licenses_doing_nothing() {
        let s = build(
            &persona(),
            Mode::Physical,
            &for_mode(Mode::Physical),
            &env(),
        );
        assert!(s.contains("Thinking about something is a real choice"));
        assert!(s.contains("`reflect`"));
        assert!(
            s.contains("rearranged for no reason"),
            "the prompt does not say what reflecting is *for*, so it reads as \
             permission to idle rather than as the alternative to fidgeting"
        );
    }

    /// There is no terminal state. A prompt implying one teaches the model to
    /// try to reach it, and the loop is continuous.
    #[test]
    fn the_prompt_establishes_no_completion_criterion() {
        let s = build(
            &persona(),
            Mode::Physical,
            &for_mode(Mode::Physical),
            &env(),
        );
        assert!(s.contains("no point at which you are finished"));
        for terminal in ["task is complete", "when you are done", "your goal is to"] {
            assert!(
                !s.contains(terminal),
                "leaked a terminal framing: {terminal}"
            );
        }
    }

    /// Beliefs come with the instruction that they cannot be decided away, or
    /// the model resolves the tension the moment it notices it.
    #[test]
    fn beliefs_arrive_with_their_write_protection() {
        let beliefs = vec!["Hess burned the east granary.".to_string()];
        let p = Persona {
            beliefs: &beliefs,
            ..persona()
        };
        let s = build(&p, Mode::Physical, &for_mode(Mode::Physical), &env());
        assert!(s.contains("Hess burned the east granary"));
        assert!(s.contains("cannot decide to stop believing"));
        assert!(s.contains("Do not tidy it away"));
        // **The valve it names has to exist.** This asserted `note_concern`
        // for as long as `note_concern` did, and went on asserting it after
        // the act was removed — a prompt naming a tool the parser refuses,
        // held in place by a test checking the prompt against itself.
        assert!(
            s.contains("`gesture`"),
            "the belief valve names no act a character can call"
        );
        for gone in ["note_concern", "set_intent", "broadcast_strategy"] {
            assert!(!s.contains(gone), "the prompt still offers `{gone}`");
        }
    }

    /// **Every act the prompt names in backticks must be one a character can
    /// call.**
    ///
    /// A prompt that recommends a removed act is a character told to do
    /// something the parser then refuses — and it does not learn from being
    /// refused, so it asks the same way every turn. That happened twice: the
    /// belief paragraph went on recommending `note_concern` for hours after the
    /// act was cut, and the licence to do nothing has now named a removed act
    /// twice over — `wait`, then `wait_for`. Every one of those was held in
    /// place by a test that checked the prompt against itself rather than
    /// against the catalog, which is why this one checks the catalog.
    #[test]
    fn the_prompt_never_names_an_act_that_does_not_exist() {
        let s = build(
            &persona(),
            Mode::Physical,
            &for_mode(Mode::Physical),
            &env(),
        );
        // Backticked lowercase_snake words are how this prompt writes an act.
        // Anything else in backticks (`to`, `intent`) is a parameter, so only
        // words that look like a call are checked, and a real act name is never
        // also a parameter name.
        let named: Vec<&str> = s
            .split('`')
            .skip(1)
            .step_by(2)
            .filter(|w| w.contains('_') || ["say", "tell", "ask", "gesture", "observe"].contains(w))
            .collect();
        assert!(!named.is_empty(), "the prompt names no acts at all");
        for w in named {
            // Parameters may be backticked too; only judge words that are not
            // parameters of some act in the catalog.
            let is_param = CATALOG.iter().any(|t| t.params.iter().any(|p| p.name == w));
            if is_param {
                continue;
            }
            assert!(
                by_name(w).is_some(),
                "the prompt names `{w}`, which is not an act any character can call"
            );
        }
    }

    /// No beliefs means no belief section — an empty header would read as "you
    /// believe nothing", which is a claim rather than an absence.
    #[test]
    fn an_empty_layer_contributes_nothing() {
        let s = build(
            &persona(),
            Mode::Physical,
            &for_mode(Mode::Physical),
            &env(),
        );
        assert!(!s.contains("What you hold true"));
        assert!(!s.contains("People you know"));
        assert!(!s.contains("What you are set on"));
    }

    /// The vocabulary in the prompt must be the vocabulary the dispatcher
    /// accepts. A tool listed but not offered is an invitation to a refusal.
    #[test]
    fn the_prompt_lists_exactly_the_tools_offered_in_that_mode() {
        for mode in [Mode::Physical, Mode::InstantMessage] {
            let tools = for_mode(mode);
            let s = build(&persona(), mode, &tools, &env());
            for t in &tools {
                assert!(
                    s.contains(t.name),
                    "{:?}: {} missing from prompt",
                    mode,
                    t.name
                );
            }
        }
        let physical = build(
            &persona(),
            Mode::Physical,
            &for_mode(Mode::Physical),
            &env(),
        );
        assert!(
            !physical.contains("send_image"),
            "a physically present character was offered a camera"
        );
    }

    #[test]
    fn mode_changes_how_presence_is_described() {
        let phys = build(
            &persona(),
            Mode::Physical,
            &for_mode(Mode::Physical),
            &env(),
        );
        let msg = build(
            &persona(),
            Mode::InstantMessage,
            &for_mode(Mode::InstantMessage),
            &env(),
        );
        assert!(phys.contains("physically present"));
        assert!(msg.contains("not present"));
    }

    /// **The prompt must show the shape the GRAMMAR emits**, not merely one the
    /// parser tolerates.
    ///
    /// This checked the prompt against `act::parse`, which accepts both call
    /// syntaxes — so it passed for months while the prompt taught one JSON
    /// object per line to a checkpoint the stencil was forcing into
    /// `<tool_call>` blocks. The grammar is a hard constraint and always wins;
    /// a prompt describing something else is not advice the model can take, it
    /// is instructions the decode has to be steered away from.
    ///
    /// So the comparison is against the envelope the stencil compiles. Both
    /// come from the same value, and this is what says so.
    /// **The prompt promises answers, and the wrapper it names is the one the
    /// turn is actually built with.**
    ///
    /// `mind::compose` writes `<tool_response>` blocks at the head of every
    /// turn that follows an act. A prompt that named a different wrapper — or
    /// named none — would leave the character reading its own results as
    /// unexplained noise, which is the state this whole change came out of: 23
    /// calls, 0 responses, and every act a `reflect`.
    #[test]
    fn the_prompt_names_the_wrapper_results_actually_come_back_in() {
        for env in [ToolCallEnvelope::qwen3(), ToolCallEnvelope::qwen35()] {
            let s = build(&persona(), Mode::Physical, &for_mode(Mode::Physical), &env);
            assert!(
                s.contains("<tool_response>"),
                "{:?}: the prompt never tells the character its acts are answered",
                env.style
            );
            // And that a refusal is one of the answers it may get — the signal
            // that separates "that did not work" from "nothing happened".
            assert!(
                s.contains("refused"),
                "{:?}: the prompt does not say an act can be refused",
                env.style
            );
        }
    }

    #[test]
    fn the_prompt_shows_the_shape_the_grammar_emits() {
        use crate::engine::act;

        for env in [ToolCallEnvelope::qwen3(), ToolCallEnvelope::qwen35()] {
            let s = build(&persona(), Mode::Physical, &for_mode(Mode::Physical), &env);
            // The worked calls the prompt shows, rendered from this envelope.
            let shown = env.render("say", &[("intent", "that I will not hand it over")]);
            assert!(
                s.contains(&shown),
                "{:?}: the prompt does not show what the grammar emits.\nwanted:\n{shown}",
                env.style
            );

            // And what it shows is a call the parser reads back as one act —
            // the round trip the old test only did half of.
            let p = act::parse(&shown);
            assert_eq!(
                p.acts.len(),
                1,
                "{:?}: the prompt's own example does not parse: {:?}",
                env.style,
                p.rejected
            );
            assert_eq!(p.acts[0].tool, "say");
            assert_eq!(p.acts[0].args["intent"], "that I will not hand it over");

            // The other syntax must not also be described — two formats in one
            // prompt is the ambiguity this whole change removes.
            let other = match env.style {
                CallStyle::FunctionBlock => "One JSON object per line",
                _ => "<parameter=",
            };
            assert!(
                !s.contains(other),
                "{:?}: the prompt also teaches the other syntax ({other})",
                env.style
            );
        }
    }

    /// A character with nothing authored still gets a usable prompt rather than
    /// a malformed one — this is the state every character is in before an
    /// author has filled it out.
    #[test]
    fn an_empty_persona_still_produces_a_coherent_prompt() {
        let s = build(
            &Persona::default(),
            Mode::Physical,
            &for_mode(Mode::Physical),
            &env(),
        );
        assert!(s.starts_with("You are a person."));
        assert!(s.contains("HOW YOU ACT"));
        assert!(s.contains("WHAT YOU CAN DO"));
        assert!(
            !s.contains("\n\n\n\n"),
            "empty sections left a hole in the prompt"
        );
    }
}
