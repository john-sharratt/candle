//! Defines the acts by which a character speaks to people who are not in the room, and the threads those acts reply on.
//!
//! The phone acts are separate from room speech, since a character texting from a room is not heard by it. They bind to threads on the character's phone, and a character without a phone has none of them, so being out of contact is a state the world can impose rather than a flag. A standing channel every character joins on arrival is the one act that can reach someone the character cannot see, and this is what lets a character name where it is and find others by talking. `reply` answers only the conversations where somebody else spoke last, because a message to a person's name would otherwise take a group question into a private thread. Group messaging is the same act as a direct message with more members, so there is no separate group mechanism.

use super::*;

/// Start a conversation with somebody who is not here.
pub const REACH_OUT: Tool = Tool {
    name: "reach_out",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Get hold of somebody you have no conversation with yet, and start one. You can \
                  do this from anywhere and while doing something else — that is what carrying a \
                  handset is for. Somebody you already have a thread with is `message`, not this.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Who you are reaching.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you are opening with, and what you want. Substance, not the \
                          message itself.",
        },
    ],
    examples: &[Example {
        situation: "You have not properly spoken to somebody in a long time and the reason has \
                    stopped being a reason.",
        call: r#"{"to":"Wren","intent":"that I have been the one not writing, and would like to fix it rather than explain it"}"#,
        because: "Speaking first is an act somebody has to take. A character that can only ever \
                  answer is a presence, not a person.",
    }],
};

// The phone
//
// **Not the room, and not a mode.** These are the acts of a handset a character
// carries: several conversations at once, to people who are not here, at a time
// that is not now. A character texts while standing in a room and the room
// hears none of it — which is why these are separate acts rather than `tell` and
// `shout` pointed somewhere else.
//
// None of them declares an availability. They bind to the threads on the phone,
// and a character without a phone has none, so the ordinary empty-set rule takes
// every one of them out of the grammar. Being out of contact is therefore a
// thing the world can *do* to somebody — take the handset — rather than a flag
// anybody has to remember to set.

/// Say something on one of your threads.
///
/// # The one act that can reach somebody you cannot see
///
/// Every world here gives its cast a standing channel — `sim::phone::CHANNEL` —
/// which every character joins on arrival without choosing to. This is the act
/// that speaks on it, and it needed no changes to become that: a channel is a
/// thread with everybody on it, so it arrives in `Choices::Threads` beside the
/// private ones and binds the same way.
///
/// That matters more than it sounds. A character alone could previously reach
/// nobody it could not already see: the percept names who is in this room, and
/// `move_to` offers rooms with no indication of who is in any of them, so the
/// building was a list of identical doors. A character that says where it is
/// has handed everybody else a room name with a person in it, which is the
/// fact that was missing — and it is why the channel is left general rather
/// than made into a find-people feature. Finding people is downstream of
/// ordinary talk, not a separate mechanism.
///
/// # How characters reach each other
///
/// Room speech is for people (`docs/npc_conversations.md` §12): `tell`, `ask`
/// and `whisper` appear only with a person in the room, so this is how a
/// character reaches another character, even one standing beside it. `message`
/// is `Always`, and has to be. A private message is not written on the thread
/// here: it becomes a purpose for the character's conversation with that
/// person, which says it in its own words (§10).
pub const MESSAGE: Tool = Tool {
    name: "message",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Send something on one of your conversations. Say first how it goes — \
                  `private`, to one person alone, or `channel`, to one of your channels or \
                  groups — and then which one. It reaches everybody on that conversation and \
                  nobody else — not the room you are standing in, and not people on your other \
                  threads. They will see it whenever they next look, which may not be now. As \
                  with speaking, you give what you MEAN and the wording follows.\n\
                  \n\
                  **Each channel is for something**, and your channels are listed with what \
                  each one is for: say a thing on the channel whose description fits it. To \
                  answer something said to you, use `reply`, which goes back where it was \
                  said.\n\
                  \n\
                  **How you reach another character, even one standing beside you.** Speak \
                  aloud only to the people here. The channel reaches everyone at once, so it is for anything \
                  that concerns whoever happens to be listening: a question you need an answer \
                  to, something the others need to know before they act, a decision that is not \
                  yours alone.\n\
                  \n\
                  **Ask rather than announce.** A message that reports your own status — where \
                  you are, that you are ready, that you are standing by — gives nobody a reason \
                  to reply and nothing to do about it, and a channel full of those is a room of \
                  people talking past each other. A question obliges an answer. If you want to \
                  be with somebody, ask where they are and go there; do not announce that you \
                  are ready and wait.",
    params: &[
        Param {
            name: "via",
            ty: "string",
            required: true,
            description: "How it goes: `private`, to one person alone, or `channel`, to one of \
                          your channels or groups — your channels are listed with what each is \
                          for.",
        },
        Param {
            name: "to",
            ty: "string",
            required: false,
            description: "Which one: the person's name for `private`, the channel's or the \
                          group's for `channel`.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean to get across. Substance, not the message itself.",
        },
    ],
    examples: &[
        Example {
            situation: "You are two rooms away from somebody who is waiting on a thing you have \
                        just finished.",
            call: r#"{"via":"private","to":"Soren","intent":"that it is done and I am not walking it over tonight"}"#,
            because: "Reaching somebody who is not here is what the handset is for. Walking over \
                      is `move_to` and costs the evening.",
        },
        Example {
            situation: "Three of you are working the same span and one of them has just found \
                        something that changes what the other two are doing.",
            call: r#"{"via":"channel","to":"the eastern sweep","intent":"stop dating anything until we have settled the boundary"}"#,
            because: "One send, everybody on the thread. A group is a thread with more people on \
                      it, not a different kind of act.",
        },
        Example {
            situation: "You have found something the survey turned up, and among your channels \
                        is #research, for findings and open questions.",
            call: r##"{"via":"channel","to":"#research","intent":"that the seam under the east wing runs further than the map shows"}"##,
            because: "A channel is for what its description says. A finding goes where findings \
                      are kept, not to whoever you last spoke to.",
        },
        // **The case a solitary character had no answer to.** Nobody here, and
        // no way to learn that anybody exists anywhere — the percept names who
        // is in this room, and `move_to` offers rooms with no sign of who is in
        // any of them, so the building is a list of identical doors. The
        // question is the act that was missing.
        Example {
            situation: "You have been alone in the chronicle for an hour and you do not know \
                        where anybody else is.",
            call: r#"{"via":"channel","to":"the channel","intent":"where each of you is working, because I want to bring the third era to whoever is nearest"}"#,
            because: "**Asking is what finds people.** Nothing else tells you which of seventy \
                      rooms has somebody in it, and an answer names one you can walk to. \
                      Reporting your own position instead leaves everybody informed and \
                      stationary.",
        },
        // **The failure this example is here to prevent**, taken from a live
        // feed: two characters spent an afternoon agreeing to meet and never
        // met. Seven of twenty-seven messages were the words "I'm standing by",
        // and the pair oscillated between the same two rooms, each walking to
        // where it guessed the other was. Nothing they sent each other named a
        // place, because none of them ever asked.
        Example {
            situation: "Somebody has said they want to work through something with you, and you \
                        do not know which room they are in.",
            call: r#"{"via":"channel","to":"the channel","intent":"which room you are in, so I can come to you rather than both of us moving"}"#,
            because: "One of you has to name a place or you will cross. \"I am ready\" from both \
                      sides is two people waiting; a question has an answer, and the answer is \
                      somewhere to walk.",
        },
    ],
};

/// Answer a conversation that is waiting on you, on that conversation.
///
/// # Why it is not `message`
///
/// It sends exactly what `message` sends, on the thread it names; what differs
/// is the set it names one from ([`crate::engine::tools::Choices::Awaiting`]).
/// Somebody on a group with this character is on a direct thread with it too,
/// under the same name, and `message` took a person's name for the direct
/// thread — so a question asked on a Discord channel was answered, every time,
/// in private. Bound to the conversations where somebody else spoke last, the
/// answer goes where the question was asked, and a thread nobody is waiting
/// on is not one it can be sent to.
pub const REPLY: Tool = Tool {
    name: "reply",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Answer somebody who is waiting on you, on the conversation they said it on. \
                  Every conversation offered here has somebody else's word last — a question, \
                  a request, news — and the answer reaches everybody on it, as `message` does. \
                  When somebody asks you something on a group or a channel, this is how you \
                  answer them there, where everybody who saw the question sees the answer. As \
                  with speaking, you give what you MEAN and the wording follows.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "The conversation you are answering — the one the thing you are \
                          answering was said on.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean to get across. Substance, not the message itself.",
        },
    ],
    examples: &[
        Example {
            situation: "Somebody asked you on the server's channel what you can see from the \
                        watch room.",
            call: r#"{"to":"zenworld #general","intent":"that the perimeter is quiet and the flicker was the bench light"}"#,
            because: "Asked on the channel, answered on the channel: everybody who saw the \
                      question sees the answer. Sent to the asker's own name it would go to \
                      your private conversation with them instead.",
        },
        Example {
            situation: "Soren messaged you privately to ask whether the survey is finished.",
            call: r#"{"to":"Soren","intent":"that it is finished and filed"}"#,
            because: "A private question has its private conversation waiting on you, so that \
                      is the one to answer on.",
        },
    ],
};

/// Bring somebody onto one of your threads.
///
/// **This is the whole of group messaging.** A thread with two people on it is a
/// direct message and one with three is a group; the difference is the
/// membership, not a mode, a second act, or a branch in the catalogue.
pub const INVITE: Tool = Tool {
    name: "invite",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Bring somebody onto one of your conversations. From then on they see what is \
                  said on it and can answer; what was said before they arrived is yours to \
                  repeat or not. Two people is a message and three is a group — this is how one \
                  becomes the other.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Which conversation you are bringing them onto.",
        },
        Param {
            name: "who",
            ty: "string",
            required: true,
            description: "Who you are bringing in.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: false,
            description: "Why, if it is not obvious. Everybody on the thread sees it.",
        },
    ],
    examples: &[Example {
        situation: "You and one other have been working out something that turns on a third \
                    person's part of the record, and relaying it back and forth is losing detail.",
        call: r#"{"to":"Soren","who":"Perrin Vastwood","intent":"that the dates are theirs and we have been guessing"}"#,
        because: "Bringing them on beats relaying. It also means the next thing said reaches all \
                  three without anybody having to remember to pass it along.",
    }],
};

/// Start a group outright.
pub const OPEN_GROUP: Tool = Tool {
    name: "open_group",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Start a named conversation with several people at once, rather than growing one \
                  from a pair. Use it when the thing concerns a set of people from the start — \
                  everybody on it sees everything said on it.",
    params: &[
        Param {
            name: "called",
            ty: "string",
            required: true,
            description: "What the group is called. Everybody on it sees this name, so it should \
                          say what the group is for.",
        },
        Param {
            name: "with",
            ty: "string",
            required: true,
            description: "Who is on it, by name. Several, separated by commas.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: false,
            description: "What to open with — the thing that makes it clear why everybody has \
                          been put on one conversation. Leave it out to open the group silently.",
        },
    ],
    examples: &[Example {
        situation: "Four people are each about to date the same span differently and none of them \
                    knows the others are doing it.",
        call: r#"{"called":"the boundary","with":"Soren, Perrin Vastwood, Orion Vance","intent":"that nobody should date anything until we agree where it falls"}"#,
        because: "A standing thread is a reason for four people to stay in step that no single \
                  message between two of them can be.",
    }],
};
