//! Defines the speech, attention and movement acts a character can take, each
//! pitched by who is meant to hear it.
//!
//! Speech comes in three reaches: `tell` is to one person and overheard by the
//! room, `whisper` is to one person with nobody else meant to hear, and `shout`
//! is to nobody in particular and carries to rooms that can see into this one.
//! The speech acts take what a character means, not a finished sentence, and they
//! are offered only where a person would hear them, because speech into an empty
//! room reaches nobody and changes nothing. The stop-and-think act, `reflect`, is
//! a fixed span with a required answer rather than a standing wait, so a quiet
//! character is recorded as thinking instead of leaving a column of identical rows.

use super::*;

/// Speech, pitched three ways.
///
/// **Three tools for three reaches, not one with a volume.** Who hears an
/// utterance is decided by how it was pitched: [`TELL`] is to one person and
/// overheard by the room, [`WHISPER`] is to one person and heard by nobody
/// else, and [`SHOUT`] is to nobody in particular and carries to every room that
/// can see into this one. As one tool with an optional target a character is
/// invited to name somebody every turn — which it does, because a model handed
/// a field fills it in — and the world carries both who an utterance was aimed
/// at and how it was pitched, so the difference is legible to everybody else.
///
/// # Talking to somebody is `tell`
///
/// The cast talked to each other through the room-wide act, aimed at nobody,
/// with one person standing there to aim at, and every exchange read as two
/// characters announcing things to the air. So the everyday act is the
/// addressed one.
///
/// # `whisper` is the exception, and says so
///
/// Offered beside `tell` as "for when it is only for them", a whisper was what
/// the cast reached for by default: in a conversation between two people every
/// remark is only for the other one. A whisper is for keeping something from
/// somebody else who is standing there, and both descriptions now say that in
/// as many words — `tell` is the ordinary voice, `whisper` needs a reason.
///
/// # `tell` and `whisper` need somebody to hear
///
/// **Speech into an empty room reaches nobody and changes nothing**, which by
/// this catalog's own standard is an act that does not act — and offered alone
/// it was worse than nothing. Three characters stood alone in three rooms, and
/// every time the building did something each of them said it back out loud,
/// paraphrased, to nobody:
///
/// ```text
/// perceived: A smell of hot plastic comes and goes with no source anybody could point at.
/// act:       say — The scent of burnt plastic drifts through the air, vanishing as
///                  quickly as it arrived, with no source I can find.
/// ```
///
/// That is a character narrating its world rather than reacting to it. So both
/// addressed acts are absent when alone, and their `to` is bound to the people
/// actually standing here.
///
/// # Room speech is for people
///
/// Every voice here needs a **person** to hear it (`docs/npc_conversations.md`
/// §12): characters reach each other by `message`, even standing side by side.
/// A shout needs a person within its reach, or a fight it is part of.
pub(super) const SHOUT: Tool = Tool {
    name: "shout",
    at: &[],
    category: "Speech",
    plane: Plane::Speech,
    availability: Availability::PersonInEarshot,
    description: "Call out, for anybody within earshot — everyone in this room and in every room \
                  that can see into it. Aimed at nobody in particular. You give what you MEAN, \
                  not the sentence; the narrator renders it in your own voice. How you talk to \
                  the people within earshot all at once; one of them here is `tell`.",
    params: &[
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean to convey. Substance, not wording: \"that the redoubt \
                          burned twice and both are written down\" — never a finished line of \
                          dialogue.",
        },
        Param {
            name: "manner",
            ty: "string",
            required: false,
            description: "How it is meant to land — flatly, warmly, as a warning. Colours the \
                          rendering; never becomes words itself.",
        },
    ],
    examples: &[
        Example {
            situation: "You are alone in the green room and have just found the same era \
                        written up twice, differently. Two visitors are looking round the run \
                        outside.",
            call: r#"{"intent":"that the third era is written twice and the two do not agree, and that anybody who has been working from it should come and look","manner":"urgently"}"#,
            because: "It is for whoever can hear, and there is nobody here to tell. A shout \
                      reaches the rooms around you.",
        },
        Example {
            situation: "The lift has jammed half-open, and the visitors on the run outside are \
                        about to walk into it.",
            call: r#"{"intent":"that the lift is not safe and nobody should get in it","manner":"as a warning"}"#,
            because: "A warning is for everybody within earshot at once. Naming one person \
                      would leave the rest to walk into it.",
        },
    ],
};

/// # Peers do not give each other orders
///
/// Three characters of equal standing, each handed the same finding and told to
/// say what they mean, converged on commanding one another: "that Pax should
/// check the seal", "that we should move somewhere", "that there is no need to
/// keep asking". Every line was an instruction, so every listener heard a
/// superior, each tried the same in return, and the room became a ring of
/// people closing each other's questions. The `intent` parameters of the speech
/// acts therefore say what an utterance is *for* — what you found, believe or
/// want — and that nobody here answers to you.
pub(super) const TELL: Tool = Tool {
    name: "tell",
    at: &[],
    category: "Speech",
    plane: Plane::Speech,
    // Offered only while a person is here to be spoken to. A character alone has
    // nobody to name, and being invited to name one is what makes it invent a
    // person who is not there; one with only characters for company messages
    // them.
    availability: Availability::PersonNearby,
    description: "Talk to somebody here. This is your ordinary speaking voice and how you talk \
                  to the people here — replies, news, questions you are not pressing, anything \
                  you would say to their face. Name who you are talking to, by the name you \
                  know them by; anybody else in the room overhears, and hears that it was for \
                  them. You give what you MEAN, not the sentence; the narrator renders it in \
                  your own voice. Use this unless you have a reason not to: `whisper` only when \
                  it must be kept from somebody else who is standing here, `shout` only when it \
                  is for everybody within earshot.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Who you are addressing, exactly as their name appears where you are. \
                          They must be here — you cannot address somebody in another room.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean to convey: what you found, saw, believe or want. \
                          Substance, not wording: \"that I will not hand over the ledger, and \
                          that pressing me will cost him\" — never a finished line of dialogue. \
                          Nobody here answers to you and you answer to nobody here, so give \
                          them what you know or would like and let them decide; do not tell \
                          them what they should do, or that a matter is closed.",
        },
        Param {
            name: "manner",
            ty: "string",
            required: false,
            description: "How it is meant to land — flatly, warmly, as a warning. Colours the \
                          rendering; never becomes words itself.",
        },
    ],
    examples: &[
        Example {
            situation: "Wren, a visitor, is in the green room with you. She has been reading \
                        the third era, and you have just found that era written up twice.",
            call: r#"{"to":"Wren","intent":"that the third era is written twice and the two do not agree, and that I would like to know which of them she has been reading","manner":"plainly"}"#,
            because: "Aimed at the one person who can answer it. Anyone else in the room hears \
                      it too and hears that it was not for them, which is the difference \
                      between being asked and overhearing a question.",
        },
        Example {
            situation: "A visitor has just come in and said they are not going to interrupt.",
            call: r#"{"to":"Wren","intent":"that they are not interrupting and I would rather have the company","manner":"warmly"}"#,
            because: "A reply is addressed. Shouting it to everybody would leave the person \
                      who spoke to you unsure it was meant for them.",
        },
        Example {
            situation: "The strip along the door has worked loose again, and Wren, a visitor \
                        who is handier with the tools than you, is standing by it.",
            call: r#"{"to":"Wren","intent":"that the strip along the door has worked loose again, and that I would be glad of a hand with it if they have the time","manner":"easily"}"#,
            because: "A finding and a request leave them free to take it up or not. Saying they \
                      should fix it would be an order from somebody who has no standing to \
                      give one, and orders between equals are how everyone ends up waiting on \
                      everyone.",
        },
        Example {
            situation: "Wren and Orion, both visitors, are in the green room with you, and you \
                        want Wren to know the kiln has cooled enough to open.",
            call: r#"{"to":"Wren","intent":"that the kiln has cooled and she can open it whenever she likes","manner":"easily"}"#,
            because: "Nothing in it needs keeping from Orion. Whispering it would make an \
                      ordinary remark look like a secret, and leave Orion wondering what \
                      is being kept from him.",
        },
    ],
};

/// Speech to one person that nobody else makes out.
///
/// **The room still sees it happen.** Two heads together is a thing people
/// notice, and a character that could whisper invisibly could conspire in front
/// of anybody with nobody the wiser — so everybody else in the room is told
/// that it happened, and to whom, and not a word of it.
pub(super) const WHISPER: Tool = Tool {
    name: "whisper",
    at: &[],
    category: "Speech",
    plane: Plane::Speech,
    // A person here, and at least one more to keep it from — with only the
    // listener here, it is `tell` said quietly.
    availability: Availability::PersonAmongOthers,
    description: "Speak to one person here too quietly for anybody else to make out. Only for \
                  something that must be kept from somebody ELSE who is standing here — a \
                  secret, a warning about a person in the room, a thing you would not say in \
                  front of them. It is not how you talk to people; that is `tell`, even when \
                  what you say is meant for one person. The rest of the room sees you whisper \
                  and hears none of it, so an ordinary remark whispered reads as something \
                  being hidden.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Who you are whispering to, exactly as their name appears where you \
                          are. They must be here.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean to convey: what you know or want, not an instruction — \
                          nobody here answers to you. Substance, not wording — never a finished \
                          line of dialogue.",
        },
        Param {
            name: "manner",
            ty: "string",
            required: false,
            description: "How it is meant to land — urgently, conspiratorially, as a warning. \
                          Colours the rendering; never becomes words itself.",
        },
    ],
    examples: &[Example {
        situation: "Wren, a visitor, is beside you in the green room, and so is Maker-02, \
                    whose version of the third era you think is the wrong one.",
        call: r#"{"to":"Wren","intent":"that I think Maker-02's version of the third era is the wrong one, and I would rather not say so in front of it","manner":"quietly"}"#,
        because: "It is for one of them and about the other. Said aloud it would be an \
                  accusation in front of the accused. Had Maker-02 not been there, this would \
                  be `tell`.",
    }],
};

/// A question put to somebody here.
///
/// **Not a variant of `tell` — the act with a different social obligation.** A
/// statement leaves the other party free to do nothing, and free is what they
/// did: two characters shared a room for a hundred turns while one restated the
/// same finding and the other, hearing it every second turn, walked on the spot.
/// Neither was malfunctioning. Nothing had been *asked*.
///
/// It renders as speech and lands as speech; what it changes is that the
/// listener perceives a question, which has an obvious next act where a
/// statement has none.
pub(super) const ASK: Tool = Tool {
    name: "ask",
    at: &[],
    category: "Speech",
    plane: Plane::Speech,
    // Same rule as `tell`: a person here to ask, or the act is absent — a
    // character asks another by message.
    availability: Availability::PersonNearby,
    description: "Ask one person here something, by the name you know them by. Everyone in the \
                  room hears it and hears who it was for. Use this when you want an answer — a \
                  statement lets them say nothing, a question does not.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Who you are asking, exactly as their name appears where you are. They \
                          must be here.",
        },
        Param {
            name: "about",
            ty: "string",
            required: true,
            description: "What you want to know. Substance, not wording: \"which of the two \
                          versions of the third era they have been working from\" — never a \
                          finished line of dialogue.",
        },
        Param {
            name: "manner",
            ty: "string",
            required: false,
            description: "How it is meant to land — plainly, warily, as a challenge. Colours the \
                          rendering; never becomes words itself.",
        },
    ],
    examples: &[
        Example {
            situation: "Wren, a visitor, is here. You have just found the third era written up \
                        twice, differently, and she has read both.",
            call: r#"{"to":"Wren","about":"which of the two versions of the third era she has been reading","manner":"plainly"}"#,
            because: "Only one person can answer this, and asking obliges them to. Saying it as \
                      a finding would have left them free to say nothing.",
        },
        Example {
            situation: "A visitor has been in the room with you for a while and you have both \
                        been working in silence.",
            call: r#"{"to":"Wren","about":"what they are working on"}"#,
            because: "The plainest way out of a silence, and it costs nothing to be wrong about \
                      whether they wanted to talk.",
        },
    ],
};

pub(super) const MOVE_TO: Tool = Tool {
    name: "move_to",
    at: &[],
    category: "Movement",
    plane: Plane::World,
    // Walking needs legs. A tower consciousness has none and never will, so the
    // act is absent for it rather than refused — offered, it would be tried.
    availability: Availability::Embodied,
    description: "Go somewhere, named the way you know it rather than as coordinates — on this \
                  level or any other. You name the place; getting there is not your business: \
                  a room on another level is reached by the lift, which you wait for on the \
                  landing and ride as part of the journey. You will be told when you arrive, and \
                  told if you never do. Somewhere you have no memory of is refused before you \
                  stand up.",
    params: &[
        Param {
            name: "destination",
            ty: "string",
            required: true,
            description: "A room you know: \"the green room\", \"band one\", \"the command \
                          room\". A name two levels share means the one on your own level; \
                          the other is named with its level.",
        },
        Param {
            name: "urgency",
            ty: "string",
            required: false,
            description: "walk | hurry | run. Absent means whatever the moment warrants.",
        },
    ],
    examples: &[
        Example {
            situation: "You are at a terminal in band one. You want to ask somebody about a \
                        character you cannot place, and the green room is where people sit \
                        with nothing in hand.",
            call: r#"{"destination":"the green room"}"#,
            because: "A place named the way the memory names it. The route is the world's \
                      business — naming a corridor on the way would be describing a journey \
                      rather than choosing a destination.",
        },
        Example {
            situation: "You have finished what you were set and the orders are given upstairs.",
            call: r#"{"destination":"the command room","urgency":"walk"}"#,
            because: "Another level, and it costs more than one stop — which is the point of \
                      naming the destination rather than the lift: you say where you are going \
                      once, and arrive some turns later.",
        },
    ],
};

pub(super) const FOLLOW: Tool = Tool {
    name: "follow",
    at: &[],
    category: "Movement",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Keep with somebody as they move, at a distance you choose. You go where they \
                  go until something stops you — this is a standing intention rather than one \
                  journey, which is what makes it different from walking to where they happen to \
                  be now.",
    params: &[
        Param {
            name: "target",
            ty: "string",
            required: true,
            description: "Who you keep with, by the name the world writes down.",
        },
        Param {
            name: "distance",
            ty: "string",
            required: false,
            description: "close | at a distance | out of sight.",
        },
    ],
    examples: &[Example {
        situation: "The courier you were told to watch leaves the inn and turns down an alley. \
                    You were told to watch, not to be seen.",
        call: r#"{"target":"the courier","distance":"out of sight"}"#,
        because: "The distance carries the instruction you were given. Following close would \
                  satisfy the verb and fail the order.",
    }],
};

/// The room's non-verbal channel.
///
/// Absorbs what `express` used to be. They were one act split by whether the
/// showing was chosen — a distinction the world cannot represent and nobody
/// watching can tell, since both arrive as *somebody did something*.
///
/// Needs company for the reason `tell` does: a signal nobody is there to read is
/// a signal that reaches nobody, and offering it to a character alone is what
/// makes it perform to an empty room. What is left for a solitary body is
/// `act`, which is a thing done rather than a thing shown.
pub(super) const GESTURE: Tool = Tool {
    name: "gesture",
    at: &[],
    category: "Gesture",
    plane: Plane::World,
    availability: Availability::Nearby,
    description: "Do something without speaking — a signal, a warning, a refusal, or just what \
                  shows on you. Everyone here sees it. Like `tell`, you give the meaning and not \
                  the movement.",
    params: &[
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What it conveys. Substance, not choreography: \"that I have heard \
                          enough and am not going to argue\" — never \"I fold my arms\".",
        },
        Param {
            name: "to",
            ty: "string",
            required: false,
            description: "Who it is aimed at, by the name they go by here. They must be here. \
                          Omit for something the whole room simply sees.",
        },
    ],
    examples: &[
        Example {
            situation: "Your companion is about to speak. The guard is within earshot and you \
                        have just recognised his colours.",
            call: r#"{"intent":"stop talking, now, and do not look at the guard","to":"my companion"}"#,
            because: "Speaking would defeat the purpose. The act is chosen for the constraint \
                      the situation imposes.",
        },
        Example {
            situation: "Hess mentions the granary fire in passing, as if it were nothing. You \
                        have believed for two years that he set it.",
            call: r#"{"intent":"a flicker of something held down, gone almost before it shows"}"#,
            because: "A belief that cannot be acted on can still leak. Aimed at nobody, because \
                      it was not aimed — it was seen.",
        },
    ],
};

/* **`observe` was here, and looking is not an act in this world.**
 *
 * It absorbed `listen` and `inspect` — the same act named for the sense or the
 * range — on the rule that a step spent looking has to come back with something
 * the character did not have. The survivor never did either. Its whole
 * implementation interpolated the target into a sentence and threw it away, then
 * reported the room's name and who was in it: both of which the percept hands
 * over free at the top of the same turn.
 *
 * It was not fixable, because there is nothing left for it to return. Sight here
 * is binary — the same room, a room you can see into, or nothing — so there is
 * no gradient for looking harder to move along. What a room *is* and what every
 * part in it does are in the building memory, carried in the prompt prefix and
 * known permanently. What somebody else is holding is withheld on purpose, and
 * has to stay withheld: needing to walk to the green room and ask is the whole
 * social mechanism of the vault.
 *
 * So every answer it could give is one the character already has, and the doc it
 * shipped with named its own defect: *an `observe` that returned nothing was a
 * turn spent to learn nothing, which is worse than idling because it looks like
 * diligence.* Perception here is pushed the moment anything changes; an act for
 * pulling it is an act for a world this is not. */

pub(super) const SEND_IMAGE: Tool = Tool {
    name: "send_image",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    // Video and text carry a picture; a voice call does not. Collapsing those
    // three into one "remote" mode is what handed a character on the telephone
    // a way to text a photo down it.
    availability: Availability::Pictorial,
    description: "Send a picture. You give what you mean the recipient to see; the scene is \
                  rendered from that. Only available when you are messaging rather than present.",
    params: &[
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean them to see, and why you are sending it.",
        },
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "The recipient's unique name, validated against this interaction.",
        },
    ],
    examples: &[Example {
        situation: "You are messaging Hess's sister. She does not believe the granary is gone.",
        call: r#"{"intent":"the granary as it stands now, so she stops needing me to convince her","to":"Hess's sister"}"#,
        because: "Absent entirely when the character is physically present — a person standing in \
                  front of you does not text you a photo, and the model is never invited to try.",
    }],
};

/// Taking stock — and standing still for a moment while you do.
///
/// # Why it is `reflect` and not `pause`
///
/// **It was `pause`, and the name was describing the machinery instead of the
/// act.** What a character does here is notice something and say what it makes
/// of it; stopping for [`PAUSE`] is only what that costs. Named for the cost,
/// it read as *the option that does nothing* — and a model choosing among named
/// tools reads the name first, so it chose almost anything else.
///
/// That was measurable. A solitary cast chose `act` on itself fifty-three times
/// out of fifty-three; with `act` cooling it chose `move_to` six times out of
/// six. It never once chose to stop and think, which is the honest answer to a
/// room that has just done something a character can do nothing about — and the
/// three fields below are exactly that answer.
///
/// The scheduler still calls it a pause internally (`Inbox::pause`, [`PAUSE`],
/// `Runtime::arm_pause`), because from there it *is* one: a deadline and
/// nothing else. Two names for two different things, which is why neither is
/// wrong.
///
/// # Why the stopping is a fixed span and not a subscription
///
/// It has been both. As free text — *wait until the silence speaks* — it was a
/// no-op the character re-chose every four seconds, because nothing in the world
/// could read the condition, so nothing could ever satisfy it. Typed and
/// referenced it became a real subscription: name a thing, go quiet, be woken
/// when it happens.
///
/// That worked and it cost more than it was worth. A subscription needs a
/// condition language the world can answer, a patience clock so a wait nothing
/// can satisfy still ends, a rousing bar so being messaged mid-wait is not read
/// two minutes late, and a deadlock guard so two characters cannot wait at each
/// other — four mechanisms, each with its own failure, to express *I have
/// nothing to do this second*.
///
/// **A pause says that directly.** The character stops for [`PAUSE`] and then
/// thinks again, and anything that arrives meanwhile wakes it at once, because
/// that is already true of every character with an empty queue. Nothing to
/// name, nothing to satisfy, nothing to time out. The deadlock it was guarding
/// against cannot form either: two characters pausing at each other both wake
/// on their own clock rather than on each other's.
pub const REFLECT: Tool = Tool {
    name: "reflect",
    at: &[],
    category: "Attention",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Take stock of where you are and what has just happened. Say where you are \
                  and what is going on, what is going through your head, and what you are \
                  actually feeling — nobody hears any of it. **This is the act for a moment \
                  you cannot do \
                  anything about**: a room settles, a light goes, somebody laughs two floors \
                  away. You stand still while you think, and anything happening around you \
                  brings you straight back.",
    params: &[
        // **The handoff to the reflection.** That conversation has no
        // perception, no window and no world state of its own; it is framed on
        // the character's own account of its circumstances, and this is that
        // account — `docs/reflection_and_dreams.md` §3. First, because it is
        // what the other two are about. And a cheap diagnostic besides: a
        // character that describes its situation wrongly has a perception fault
        // nothing else would surface.
        Param {
            name: "situation",
            ty: "string",
            required: true,
            description: "Where you are and what is going on around you, as you see it — the \
                          room, who is in it, what has just happened.",
        },
        // **Required, and that is the point of it.** Stopping is the one act
        // whose outward half is nothing at all, so without this the record of a
        // character's quietest hours is a column of identical rows and there is
        // no way to tell a character that is thinking from one that has run out
        // of things to do. A model asked to fill this in has to have an answer,
        // and the answer is the only trace of an inner life the engine gets for
        // free.
        Param {
            name: "inner_thoughts",
            ty: "string",
            required: true,
            description: "What is actually going through your head as you stand there. Nobody \
                          hears it and nobody can ask you about it later, so it is worth being \
                          honest — what you are turning over, what is nagging at you, what you \
                          have decided not to say.",
        },
        // **Required, and safe to require**, because [`Choices::Feelings`] is a
        // vocabulary rather than a possibility: a mind with no moods authored
        // leaves this free text instead of taking the whole act out of the
        // grammar. A character always has a feeling; the only question is
        // whether this world has written a word for it.
        Param {
            name: "feeling",
            ty: "string",
            required: true,
            description: "The register you are actually in, named from the list. Not what you \
                          think you ought to feel — what is true while you stand there.",
        },
        // **No third field for what it has come to think.** There was one,
        // `my_reflections`, and it asked the character to write inline the one
        // thing the reflection conversation exists to produce properly — then
        // read it straight back as the act's outcome, which is what made this the
        // act whose result was a restatement of itself. What it has come to think
        // is now what comes back: see `docs/reflection_and_dreams.md` §3 and §6.
    ],
    examples: &[
        Example {
            situation: "You have asked Maker-04 something and it has not answered yet. There is \
                        nothing else you need from this room.",
            call: r#"{"situation":"in the command room with Maker-04, waiting on an answer it has not given","inner_thoughts":"it heard me and is deciding whether to tell me, and I would rather know why it hesitated than have the answer","feeling":"alert"}"#,
            because: "The answer is theirs to give and there is nothing to do until it comes. \
                      What you notice while waiting is worth more than filling the silence.",
        },
        Example {
            situation: "You are alone in the reading room, your work is filed, and no order has \
                        come down.",
            call: r#"{"situation":"alone in the reading room with my work filed and nothing asked of me","inner_thoughts":"the filing went too easily, which usually means I have missed something, and I cannot find what","feeling":"uneasy"}"#,
            because: "Nothing here needs doing. Standing still is the honest act, and casting \
                      about for one more thing to touch is how a room gets rearranged for no \
                      reason.",
        },
        // **The case that was being answered wrongly.** The room does something
        // small, and a character with nothing to do about it reached for the
        // nearest physical verb and did that instead — fifty-three times out of
        // fifty-three, on its own body, because a box sagging is not something
        // you can act on and the catalog offered no other way to have noticed
        // it.
        Example {
            situation: "A cardboard box on the floor gives up a fold and sags. Nobody is here \
                        and nothing about it is yours to see to.",
            call: r#"{"situation":"on my own in the store, where a box on the floor has just sagged","inner_thoughts":"another thing in here has quietly given up while nobody was watching it","feeling":"weary"}"#,
            because: "There is nothing to *do* to a sagging box. Touching it, or walking off \
                      somewhere, would be a character inventing an action to fill a moment that \
                      called for a thought.",
        },
    ],
};

/// The effector device's one verb — the fixed surface over a world that can
/// grow without bound (effector design §5).
///
/// **One tool, and it never changes while the world behind it does.** A new
/// place, machine or whole subsystem shows up as new addresses under
/// `http://local/...`, acted on through exactly this, with no new `Tool` static
/// and no recompile. That is the whole escape from the tool surface that could
/// not scale: the body keeps its hands, and gains a device whose *verb* is fixed
/// over a *dynamic* set of addressable things. Finding the addresses is the body
/// act `scan`.
///
/// Handled on the async effector fast path (`Runtime::enact_device`, effector
/// design §6, §8.1), not through `body::perform`: it is a body/meta tool, always
/// offered, never a world act. See [`crate::engine::body::is_device`].
pub(super) const INVOKE: Tool = Tool {
    name: "invoke",
    at: &[],
    category: "Device",
    // Its effect reaches the world, but through the effector router rather than
    // the world-act arbiter — enacted on the async fast path, so it declares
    // itself a device/meta tool, not `Plane::World`. See `Runtime::enact_device`.
    plane: Plane::Meta,
    availability: Availability::Always,
    description: "Act at an address on your effector device, giving it the fields it asked for \
                  as a JSON object. The world does the thing and answers, or tells you plainly \
                  what was wrong so you can fix it and try again. `scan` with no place named \
                  lists what you can work here, with the address and fields for each.",
    params: &[
        Param {
            name: "url",
            ty: "string",
            required: true,
            description: "The address to act at, exactly as it appears on your device — e.g. \
                          \"http://local/command/order-table~0/report_done\".",
        },
        // **The body is a JSON object, typed by the address.** The grammar
        // replaces this parameter with the fields the chosen address takes — see
        // [`crate::engine::invoke_body`] — so what is declared here is the prose
        // a character reads. The handler still parses a body written out as a
        // string for a caller that is not the grammar; see
        // [`crate::engine::runtime::Runtime::enact_device`].
        Param {
            name: "body",
            ty: "string",
            required: false,
            description: "The fields the address asked for, as a JSON object — e.g. \
                          {\"what\": \"the muster board\"}. `{}` for an address that takes none.",
        },
    ],
    examples: &[Example {
        situation: "You are back at the command table with the reading your mission sent you \
                    for.",
        call: r#"{"url":"http://local/command/order-table~0/report_done","body":{"found":"I read the breaker panel: the main breaker is tripped and the others are closed"}}"#,
        because: "The address is the table's own, and the body carries the one field it asked \
                  for. The world answers that the mission is closed.",
    }],
};

/// How a character names its own body as the target of an act.
///
/// A word rather than the character's own name, for two reasons. The name is
/// what everybody *else* in the room is offered, so a character choosing its own
/// name off that list is picking a third party who happens to be itself — and
/// the model would have to know its own name to find it. And a name can collide:
/// two bodies called the same thing in one room would make the set a duplicate,
/// which is the `EmptyArm` failure that stops the whole grammar compiling.
pub const SELF: &str = "yourself";

/// The acts by which a body finds its way about the world it stands in:
/// looking into another room, turning its attention to one. A mind with no
/// body is never offered them — it sees every room already, and has nowhere
/// to turn.
pub const WAYFINDING: &[&str] = &["scan", "focus"];
