//! The acts a body has only because of what it is standing next to.
//!
//! [`super::tools`] is what a body can do anywhere — speak, move, look. This is
//! the other half: the things it can do only *here*, because they work through a
//! station in the room with it.
//!
//! # The map decides, not this file
//!
//! Every act below is `Availability::AtPart`, and what puts it in reach is the
//! part's own `tools:` line in the world files. A Maker at an easel cannot write
//! a chronicle entry — not because a rule here forbids it, but because the thing
//! that does that is two floors up. Moving a station to another room moves its
//! acts with it, and neither this file nor the engine has an opinion.
//!
//! That is why the descriptions never name a room. They name what the act *does*
//! and what it costs; where it can be done is the building's business.
//!
//! # Reading is not here
//!
//! Twenty-two of the acts the map used to declare were `read_this` and
//! `read_that` — one act against different things, which is what `Choices` is
//! for. They collapsed into the body's `read`, whose argument binds to whatever
//! the station actually holds. A named tool per readable thing would have made
//! the model choose among names it has to remember rather than among things that
//! are in front of it.
//!
//! Six more were `take_this` and `take_that`, and collapsed into `claim` for the
//! same reason. What is left below is the acts that *change* something.

use super::tools::{Availability, Example, Param, Plane, Tool};

/// A station act with one required argument naming what it acts on.
///
/// `$at` is the parts it attaches to, by the id the map gives them — see
/// [`Tool::at`]. It is the first thing each definition says, because where an
/// act can be done is as much a part of what it is as what it takes.
macro_rules! on {
    ($name:literal, $at:expr, $cat:literal, $plane:expr, $desc:literal,
     $arg:literal, $argdesc:literal,
     $situation:literal, $call:literal, $because:literal) => {
        Tool {
            name: $name,
            at: $at,
            category: $cat,
            plane: $plane,
            availability: Availability::AtPart,
            description: $desc,
            params: &[Param {
                name: $arg,
                ty: "string",
                required: true,
                description: $argdesc,
            }],
            examples: &[Example {
                situation: $situation,
                call: $call,
                because: $because,
            }],
        }
    };
}

/// A station act taking a thing and something to say about it.
macro_rules! on_with {
    ($name:literal, $at:expr, $cat:literal, $plane:expr, $desc:literal,
     $a:literal, $adesc:literal, $b:literal, $bdesc:literal,
     $situation:literal, $call:literal, $because:literal) => {
        Tool {
            name: $name,
            at: $at,
            category: $cat,
            plane: $plane,
            availability: Availability::AtPart,
            description: $desc,
            params: &[
                Param {
                    name: $a,
                    ty: "string",
                    required: true,
                    description: $adesc,
                },
                Param {
                    name: $b,
                    ty: "string",
                    required: true,
                    description: $bdesc,
                },
            ],
            examples: &[Example {
                situation: $situation,
                call: $call,
                because: $because,
            }],
        }
    };
}

/// A station act taking four things.
///
/// Only the craft libraries need this: a piece of the craft is addressed by
/// *which* library and *which* id, and then a field and its text — four because
/// the address is two halves, not because the act is complicated.
macro_rules! on_with4 {
    ($name:literal, $at:expr, $cat:literal, $plane:expr, $desc:literal,
     $a:literal, $adesc:literal, $b:literal, $bdesc:literal,
     $c:literal, $cdesc:literal, $d:literal, $ddesc:literal,
     $situation:literal, $call:literal, $because:literal) => {
        Tool {
            name: $name,
            at: $at,
            category: $cat,
            plane: $plane,
            availability: Availability::AtPart,
            description: $desc,
            params: &[
                Param {
                    name: $a,
                    ty: "string",
                    required: true,
                    description: $adesc,
                },
                Param {
                    name: $b,
                    ty: "string",
                    required: true,
                    description: $bdesc,
                },
                Param {
                    name: $c,
                    ty: "string",
                    required: true,
                    description: $cdesc,
                },
                Param {
                    name: $d,
                    ty: "string",
                    required: true,
                    description: $ddesc,
                },
            ],
            examples: &[Example {
                situation: $situation,
                call: $call,
                because: $because,
            }],
        }
    };
}

// The stations, named once so a typo is a compile error rather than an act
// that quietly attaches to nothing.
const CHRONICLE: &[&str] = &["chronicle-terminal"];
/// Where the craft libraries are worked on. Both of these are places a Maker
/// is already thinking about register — how a thing reads — rather than about
/// what happened, which is the chronicle's business.
const CRAFT: &[&str] = &["story-desk", "character-terminal"];
const CONCORDANCE: &[&str] = &["concordance-table"];
const STORY_DESK: &[&str] = &["story-desk"];
const EASEL: &[&str] = &["easel"];
const PLATE_RACK: &[&str] = &["plate-rack"];
const LIKENESS: &[&str] = &["likeness-table"];
const CHARACTER: &[&str] = &["character-terminal"];
const RELATIONS: &[&str] = &["relations-table"];
const SURVEY: &[&str] = &["survey-desk"];
const ROAD: &[&str] = &["road-table"];
const MAP_TABLE: &[&str] = &["map-table"];
const ACCESSION: &[&str] = &["accession-desk"];
const APPRAISAL: &[&str] = &["appraisal-bench"];
const CATALOGUE: &[&str] = &["catalogue"];
const MENDING: &[&str] = &["mending-bench"];
const ENQUIRY: &[&str] = &["enquiry-desk"];
const ORDERS: &[&str] = &["order-table", "muster-board"];
const MUSTER: &[&str] = &["muster-board"];
const DISPATCH: &[&str] = &["dispatch-board"];
const WATCH: &[&str] = &["watch-desk"];
const CHAIR: &[&str] = &["creators-chair"];
const PLANT: &[&str] = &["plant-panel"];
const STORES: &[&str] = &["stores"];
const STRUCTURE: &[&str] = &["structure-board"];
const READING_TABLE: &[&str] = &["reading-table"];

// ── The chronicle ───────────────────────────────────────────────────────────

pub const CHRONICLE_ADD_ENTRY: Tool = on_with!(
    "chronicle_add_entry",
    CHRONICLE,
    "Chronicle",
    Plane::World,
    "Add an entry to the era you are holding, in the voice that era already has. What you write \
     is added to what is there — you are not writing over anybody.",
    "to",
    "The era it belongs under.",
    "what",
    "What happened, written as the record writes things.",
    "You are holding the third era and the span between the two burnings has nothing in it at all.",
    r#"{"to":"the third era","what":"the second burning, and the fact that nobody rebuilt the redoubt afterwards"}"#,
    "Adding is not rewriting. The entry joins the era; nothing already in it moves."
);

pub const CHRONICLE_REWRITE_PAGE: Tool = on_with!(
    "chronicle_rewrite_page",
    CHRONICLE,
    "Chronicle",
    Plane::World,
    "Rewrite a page of the era you are holding. Use it when what stands is wrong rather than \
     merely thin — adding is `chronicle_add_entry`, and it is almost always the better act.",
    "in",
    "The era whose page you are rewriting.",
    "what",
    "What it should say instead, and why the old wording could not stand.",
    "The dates in the era you hold contradict both spans either side of it, and no amount of \
     adding fixes a wrong date.",
    r#"{"in":"the third era","what":"the burning dated to the year the redoubt fell, not the year after — the old date disagreed with both neighbours"}"#,
    "Rewriting is the heavier act and says so. Reaching for it when adding would do is how a \
     record loses somebody else's work."
);

pub const CHRONICLE_RETIRE_ENTRY: Tool =
    on!(
    "chronicle_retire_entry", CHRONICLE, "Chronicle", Plane::World,
    "Take an entry out of the record, because the record can no longer carry it. It stops being \
     part of what the world says about itself.",
    "what", "The entry to retire.",
    "An entry you are holding is contradicted by everything written since, and keeping it means \
     the record says two things.",
    r#"{"what":"the account of the third burning"}"#,
    "Retiring is deliberate and visible. A thing quietly left in place while everybody works \
     around it is worse than one taken out."
);

pub const CHRONICLE_SETTLE_BOUNDARY: Tool =
    on_with!(
    "chronicle_settle_boundary", CONCORDANCE, "Chronicle", Plane::World,
    "Agree where two eras meet. A boundary belongs to both of them, so neither holder can decide \
     it alone and neither can be trusted to decide it for the other — this takes both of you.",
    "between", "The era you hold, and are settling from your side.",
    "and", "The era the other holder has, whose boundary this also is.",
    "You hold the third era, somebody else holds the fourth, and an event sits in both.",
    r#"{"between":"the third era","and":"the fourth era"}"#,
    "The one thing a chronicle terminal will not do alone. It lands in both records at once."
);

// ── Stories ─────────────────────────────────────────────────────────────────

pub const STORY_DRAFT: Tool = on_with!(
    "story_draft",
    STORY_DESK,
    "Story",
    Plane::World,
    "Write into the silence you are holding. Keep writing until it stops being a summary — a \
     draft that lists what happened has not been written yet.",
    "for",
    "The gap you are filling.",
    "what",
    "The writing itself: what happened, to whom, and what it cost them.",
    "You took the longest silence on the ledger and you have read far enough either side to be \
     contradicted and know it.",
    r#"{"for":"the third silence","what":"the night the eastern gate was left open, told from the one who left it"}"#,
    "Drafting is making, not planning. What comes out is a thing somebody can read and refuse."
);

pub const STORY_FILE: Tool =
    on!(
    "story_file", STORY_DESK, "Story", Plane::World,
    "File the thing you made, and let the standing list get one shorter. It stops being yours \
     and becomes part of the record.",
    "what", "What you are filing.",
    "You carried your draft to where people sit, somebody told you what was wrong with it, and \
     you have changed it.",
    r#"{"what":"the third silence"}"#,
    "Filing is a decision that it is finished, not that it is perfect. It also releases it."
);

// ── Portraits ───────────────────────────────────────────────────────────────

pub const PORTRAIT_DRAW: Tool = on_with!(
    "portrait_draw",
    EASEL,
    "Portrait",
    Plane::World,
    "Draw a likeness of somebody you have only read. You give what the face has to carry, not \
     the brushwork — the same way `tell` takes what you mean rather than the words.",
    "of",
    "Who it is a likeness of.",
    "carrying",
    "What the face has to show: what they have been through, and what they hide.",
    "You have read everything written about somebody nobody has ever drawn.",
    r#"{"of":"ash-the-drifter","carrying":"somebody who has been believed for years about a thing they got wrong"}"#,
    "Intent, not choreography. What it looks like follows from what it has to carry."
);

pub const PORTRAIT_PROMPT_READ: Tool =
    on!(
    "portrait_prompt_read", EASEL, "Portrait", Plane::World,
    "Read the words a likeness is drawn from — the framing, the light, what the face has to \
     carry. This is the art direction, not the description: one is written to be drawn and the \
     other to be read.",
    "of", "Whose likeness — the personality it belongs to, by its id.",
    "You are about to change how somebody is drawn and want the words that produced the picture \
     standing there now.",
    r#"{"of":"ash-the-drifter"}"#,
    "The prompt and the picture are two halves of one thing. Changing the first from memory is \
     how they come apart."
);

pub const PORTRAIT_PROMPT_EDIT: Tool = on_with!(
    "portrait_prompt_edit",
    EASEL,
    "Portrait",
    Plane::World,
    "Change the words a likeness is drawn from. The picture already racked does not change — it \
     is redrawn from these words, which is why the words are what you edit.",
    "of",
    "Whose likeness — the personality it belongs to, by its id.",
    "carrying",
    "What the face has to show now, whole. This replaces the art direction rather \
     than adding to it.",
    "Somebody was drawn before the thing that happened to them, and the picture is now of \
     somebody who no longer exists.",
    r#"{"of":"ash-the-drifter","carrying":"the same lean sun-darkened man after the winter, quieter and harder to read, low sun behind him"}"#,
    "Redrawing says the world moved. Painting over the plate would say the first one was a \
     mistake."
);

// ── The craft libraries ─────────────────────────────────────────────────────
//
// A mood and a response are not lore: they are how *every* character in this
// world feels and answers. Editing one changes the next turn of a conversation
// somebody else is having, which is why these say what they are for rather than
// reading as ordinary documents — a Maker that treats the mood library as a
// notebook is rewriting the cast from inside it.

pub const LIBRARY_READ: Tool =
    on_with!(
    "library_read", CRAFT, "Craft", Plane::World,
    "Read a piece of the craft: a mood the world can be in, or a shape an answer can take.",
    "kind", "Which library — `mood` or `response`.",
    "id", "Which piece, by its id — `undone`, `accept_then_move_on`.",
    "You are about to change how something reads and want what it actually says now.",
    r#"{"kind":"mood","id":"undone"}"#,
    "The libraries are read far more often than they are changed, and a change made from memory \
     is a change made to a thing that is not there."
);

pub const LIBRARY_WRITE: Tool = on_with4!(
    "library_write",
    CRAFT,
    "Craft",
    Plane::World,
    "Change one field of a mood or a response. **This changes how every character in the world \
     feels or answers**, not just yours — so it is worth being sure, and worth saying why when \
     you commit it.",
    "kind",
    "Which library — `mood` or `response`.",
    "id",
    "Which piece, by its id.",
    "field",
    "Which part: `description` (the one line that says what it is for) or `template` \
     (the register itself).",
    "text",
    "What it says now, whole.",
    "A mood reads as two things at once, and characters keep landing on the wrong one.",
    r#"{"kind":"mood","id":"undone","field":"description","text":"So thoroughly opened by what just happened that the whole interior has rearranged."}"#,
    "One field, spliced in place: everything a person wrote around it — the comments explaining \
     why it reads this way — survives untouched."
);

pub const PORTRAIT_SETTLE_LIKENESS: Tool =
    on_with!(
    "portrait_settle_likeness", LIKENESS, "Portrait", Plane::World,
    "Take a likeness to whoever wrote the person and settle whether it is them. It takes both of \
     you: the drawing is yours and the person is theirs.",
    "of", "The likeness in question.",
    "with", "Who holds the person it is meant to be.",
    "You have drawn somebody from what was written, and the one who wrote them is here.",
    r#"{"of":"a face nobody has drawn","with":"Perrin Vastwood"}"#,
    "Neither of you can decide it alone, which is why the act names the other party."
);

pub const PORTRAIT_FILE_PLATE: Tool =
    on!(
    "portrait_file_plate", PLATE_RACK, "Portrait", Plane::World,
    "Rack a finished plate so it can be found. A likeness nobody can find is a likeness nobody \
     drew.",
    "what", "The plate to rack, by the name the likeness is held under.",
    "The likeness has been agreed with whoever holds the person, and it is still in your hands.",
    r#"{"what":"a face nobody has drawn"}"#,
    "Filing is what makes it part of the record rather than a thing you made."
);

// ── Characters ──────────────────────────────────────────────────────────────

pub const CHARACTER_WRITE_IDENTITY: Tool = on_with!(
    "character_write_identity",
    CHARACTER,
    "Casting",
    Plane::World,
    "Write who somebody is before they have lived anything — what they are like, and what they \
     were already sure of when they arrived.",
    "of",
    "The character you are holding.",
    "what",
    "Who they are: manner, bearing, and what they take for granted.",
    "You have taken an unclaimed character and read everything the record already says about them.",
    r#"{"of":"the courier","what":"somebody who has been reliable so long that being doubted is the thing they cannot take"}"#,
    "The founding sheet, not a memory. What a character comes to believe from living is earned \
     on the sleep clock and is not yours to write."
);

pub const CHARACTER_WRITE_WANTS: Tool = on_with!(
    "character_write_wants",
    CHARACTER,
    "Casting",
    Plane::World,
    "Write what somebody wants badly enough to be wrong about. A want nobody could be wrong \
     about is not a want, it is a preference.",
    "of",
    "The character you are holding.",
    "what",
    "What they are after, and what it costs them to want it.",
    "You are holding a character who does things in the record and has no reason for any of them.",
    r#"{"of":"the courier","what":"to be trusted again by the one house that stopped, badly enough to carry for them for nothing"}"#,
    "Wants are what make a character act. Without one they are a description that moves."
);

pub const CHARACTER_WRITE_MEMORIES: Tool = on_with!(
    "character_write_memories",
    CHARACTER,
    "Casting",
    Plane::World,
    "Write something that happened to somebody — one particular thing on one particular \
     afternoon, which says more than a description would.",
    "of",
    "The character you are holding.",
    "what",
    "What happened, and what it left them with.",
    "A character does a thing in the record that nothing in their life accounts for.",
    r#"{"of":"the courier","what":"the afternoon they waited four hours at a gate that was never going to open, and told nobody"}"#,
    "What happened, not what they concluded. The conclusion is theirs to reach."
);

pub const CHARACTER_SETTLE_RELATION: Tool =
    on_with!(
    "character_settle_relation", RELATIONS, "Casting", Plane::World,
    "Settle what two people are to each other, with whoever holds the other one. It takes both: \
     a relationship written from one side is a description of one person.",
    "between", "The character you hold, whose sheet this lands in.",
    "and", "The other one, whose holder has to agree it with you.",
    "Two characters keep turning up in each other's accounts and nothing says what they are.",
    r#"{"between":"the courier","and":"Perrin Vastwood"}"#,
    "Both halves land at once, so neither record can disagree with the other about it."
);

// ── Places and the map ──────────────────────────────────────────────────────

pub const PLACE_WRITE_ENTRY: Tool = on_with!(
    "place_write_entry",
    SURVEY,
    "Cartography",
    Plane::World,
    "Write what a place is — what it is like to stand in when nothing is happening, which is \
     most of the time and the part nobody writes.",
    "of",
    "The place you are holding.",
    "what",
    "What it is, and what being there is like.",
    "A place is named all over the record and described nowhere.",
    r#"{"of":"the eastern flats","what":"ground fused smooth, no cover anywhere on it, and a wind that does not stop"}"#,
    "A place that is only ever a name is a place the record asserts rather than has."
);

pub const PLACE_WRITE_LOCAL_HISTORY: Tool = on_with!(
    "place_write_local_history",
    SURVEY,
    "Cartography",
    Plane::World,
    "Write the history of one place, so it explains something the wider record only asserts.",
    "of",
    "The place you are holding.",
    "what",
    "What happened here, and what it explains.",
    "The record says a road was abandoned and never says why.",
    r#"{"of":"the eastern flats","what":"why the road stops here — what came across it, and the year it stopped being worth rebuilding"}"#,
    "Local history is where the wider record's assertions get their reasons."
);

pub const PLACE_SETTLE_ROUTE: Tool = on_with!(
    "place_settle_route",
    ROAD,
    "Cartography",
    Plane::World,
    "Agree with somebody how two places connect, so neither entry lies about the journey.",
    "from",
    "The place you hold, at your end of the route.",
    "to",
    "The place they hold, at the other end of it.",
    "Your entry says a day's walk and theirs implies three.",
    r#"{"from":"the eastern flats","to":"the ridge"}"#,
    "Both entries are changed together, which is the only way they can stop disagreeing."
);

pub const MAP_ADD_PLACE: Tool = on_with!(
    "map_add_place",
    MAP_TABLE,
    "Cartography",
    Plane::World,
    "Put a new place into the world, and make the places around it still make sense.",
    "called",
    "What it is called, as everybody else will have to refer to it.",
    "where",
    "What it sits between, and what that does to its neighbours.",
    "The record keeps referring to somewhere between two known places that is not on the map.",
    r#"{"called":"the cut","where":"between the ridge and the ruins — which makes the ridge road the long way round, not the short one"}"#,
    "A place added without reckoning with its neighbours is a place that contradicts them."
);

pub const MAP_SETTLE_BORDER: Tool = on_with!(
    "map_settle_border", MAP_TABLE, "Cartography", Plane::World,
    "Move a border, with whoever holds the other side of it. A border belongs to both regions, \
     so it cannot be moved from one of them.",
    "between", "The region you hold.",
    "and", "The region on the other side.",
    "The border you think is wrong runs between your region and one somebody else holds.",
    r#"{"between":"the eastern flats","and":"the ridge"}"#,
    "Moving it alone would leave the other region's record describing a line that no longer exists."
);

pub const MAP_REMOVE_PLACE: Tool =
    on!(
    "map_remove_place", MAP_TABLE, "Cartography", Plane::World,
    "Take a place out of the world. Everything written about it stays written, and now has to be \
     reckoned with — which is the expensive half and the reason to be sure.",
    "what", "The place to remove.",
    "A place exists on the map, nothing has ever happened there, and its being there makes two \
     journeys nonsense.",
    r#"{"what":"the eastern flats"}"#,
    "Named for what it does. A place is removed; the record of it is not."
);

// ── Keeping the record ──────────────────────────────────────────────────────

pub const RECORD_ACCESSION: Tool = on_with!(
    "record_accession",
    ACCESSION,
    "Custody",
    Plane::World,
    "Take formal charge of something that has arrived, and write down where it came from before \
     you touch it. A thing whose origin was never written cannot be trusted afterwards.",
    "what",
    "What has come in, by whatever it is going to be called from now on.",
    "from",
    "Where it came from, and how much of what it says about itself you believe.",
    "Something has arrived at the intake and nobody has taken charge of it.",
    r#"{"what":"the western intake","from":"handed over by a caravan that could not say who packed it — treat the dates as claims"}"#,
    "Origin first, work second. No amount of later effort supplies an origin nobody wrote down."
);

pub const RECORD_WRITE_PROVENANCE: Tool = on_with!(
    "record_write_provenance",
    ACCESSION,
    "Custody",
    Plane::World,
    "Write down where something came from, for a thing already in daily use whose origin nobody \
     can state.",
    "of",
    "The thing in question.",
    "from",
    "What is actually known about where it came from — including that little is.",
    "Something everybody relies on turns out to have no origin written anywhere.",
    r#"{"of":"the western intake","from":"in use since before anybody here arrived; no record of who brought it"}"#,
    "Writing down that the chain goes quiet is worth more than leaving the question open."
);

pub const RECORD_HAND_ON: Tool =
    on_with!(
    "record_hand_on", ACCESSION, "Custody", Plane::World,
    "Hand something on properly: what it is, what you did to it, and what you did not.",
    "what", "The thing you are handing on.",
    "to", "Who is taking it on, and will answer for it after you.",
    "You are giving up something you took charge of, and the next holder needs to know what you \
     touched.",
    r#"{"what":"the western intake","to":"Perrin Vastwood"}"#,
    "The chain of custody survives a handover only if the handover is an act."
);

pub const RECORD_APPRAISE: Tool = on_with!(
    "record_appraise",
    APPRAISAL,
    "Appraisal",
    Plane::World,
    "Judge whether a thing has earned the room it takes. Most of what a world throws off has \
     not, and a record that keeps everything answers no questions.",
    "what",
    "What you are weighing.",
    "verdict",
    "What you think it is worth keeping for, or why it is not.",
    "Something has been kept for years and you cannot find anybody who has used it.",
    r#"{"what":"the western intake","verdict":"nothing depends on it and nothing ever has — it is here because nobody decided"}"#,
    "Appraising is mostly refusing, and the refusal is the work."
);

pub const RECORD_WRITE_REASON: Tool = on_with!(
    "record_write_reason",
    APPRAISAL,
    "Appraisal",
    Plane::World,
    "Write down why a thing was let go, clearly enough that somebody can disagree with it later. \
     A refusal nobody can argue with is a refusal nobody can check.",
    "about",
    "What was let go, and now needs its reasoning written down.",
    "why",
    "The reasoning, in enough detail to be wrong.",
    "You are about to let something go and whoever comes after will want to know why.",
    r#"{"about":"the western intake","why":"duplicated whole by the eastern copy, which is dated and this is not"}"#,
    "The reason is written at the time or not at all — nobody reconstructs one honestly."
);

pub const RECORD_LET_GO: Tool = on_with!(
    "record_let_go",
    APPRAISAL,
    "Appraisal",
    Plane::World,
    "Refuse to keep something. It leaves the record, and the reason you give goes with it.",
    "what",
    "What you are letting go.",
    "because",
    "Why. Without this the act is not available — a thing let go for no stated reason \
     cannot be argued with.",
    "You have appraised something, written the reason, and it has not earned its room.",
    r#"{"what":"the western intake","because":"duplicated whole by a dated copy"}"#,
    "Deliberate and reasoned. The reason travels with the absence."
);

pub const RECORD_DESCRIBE: Tool = on_with!(
    "record_describe",
    CATALOGUE,
    "Description",
    Plane::World,
    "Write the way in to a body of work, for somebody who does not yet know what they are \
     looking for. Not the work — the door.",
    "what",
    "What you are describing.",
    "how",
    "The way in, in the words somebody would search with rather than the ones you used \
     while making it.",
    "A body of work has grown past the point where anybody can find anything in it.",
    r#"{"what":"the third era","how":"the burnings, the two gates, and who was blamed — start at the second burning"}"#,
    "Describing is a separate craft from making, practised for a reader who is not you."
);

pub const RECORD_ARRANGE: Tool = on_with!(
    "record_arrange",
    CATALOGUE,
    "Description",
    Plane::World,
    "Put a thing where the person who needs it would look for it — the order that makes the \
     fewest people ask where something is.",
    "what",
    "What you are arranging.",
    "under",
    "Where it should sit, and why that is where somebody would look.",
    "Something is filed where it was made rather than where it would be looked for.",
    r#"{"what":"the third era","under":"the burnings, because that is what anybody comes to it for"}"#,
    "Arrangement is for the person who does not know what they want yet."
);

pub const RECORD_CROSS_REFERENCE: Tool = on_with!(
    "record_cross_reference",
    CATALOGUE,
    "Description",
    Plane::World,
    "Point one part of the record at another that bears on it. Work nobody can find is work \
     nobody did.",
    "from",
    "The thing that should point.",
    "to",
    "What it should point at.",
    "Two entries are about the same event and neither mentions the other.",
    r#"{"from":"the third era","to":"the third silence"}"#,
    "A reference is cheap and is the difference between a record and a heap."
);

pub const RECORD_LEAVE_NOTE: Tool = on_with!(
    "record_leave_note",
    CATALOGUE,
    "Description",
    Plane::World,
    "Leave a note for whoever picks this up next — what you were in the middle of, and what you \
     would have done tomorrow.",
    "on",
    "What the note is about.",
    "what",
    "What the next person needs to know.",
    "You are stopping in the middle of something somebody else may take on.",
    r#"{"on":"the third era","what":"the dates are checked to the second burning and not past it"}"#,
    "A note costs a minute and saves the next person the hour of working out where you got to."
);

pub const RECORD_TIDY_INDEX: Tool = on!(
    "record_tidy_index",
    CATALOGUE,
    "Description",
    Plane::World,
    "Bring an index back to what it actually indexes, when the two have drifted.",
    "what",
    "The thing whose index entry no longer says what it is.",
    "You went looking for something the index said was there and it was not.",
    r#"{"what":"the catalogue"}"#,
    "An index that lies is worse than none: it stops people looking."
);

pub const RECORD_MEND: Tool = on_with!(
    "record_mend",
    MENDING,
    "Condition",
    Plane::World,
    "Repair something that has gone wrong with age — an entry true when it was written and \
     quietly wrong now.",
    "what",
    "What you are mending.",
    "how",
    "What you changed, and what it said before.",
    "An entry refers to a place by a name nothing has used for two spans.",
    r#"{"what":"the third era","how":"the old name kept in brackets after the current one, so both searches find it"}"#,
    "Mending keeps the thing usable. Rewriting it would lose what it used to say."
);

pub const RECORD_MARK_REPAIR: Tool = on!(
    "record_mark_repair",
    MENDING,
    "Condition",
    Plane::World,
    "Leave the repair visible. A mend passed off as an original is worse than the damage, \
     because nobody afterwards can tell what is old.",
    "what",
    "What you repaired, and whose mend should stay visible.",
    "You have just mended something and it currently reads as though it was always that way.",
    r#"{"what":"the third era"}"#,
    "The mark is what keeps the record honest about its own history."
);

// ── Enquiry — the record facing outward ─────────────────────────────────────

pub const ENQUIRY_TAKE_QUESTION: Tool =
    on!(
    "enquiry_take_question", ENQUIRY, "Service", Plane::World,
    "Take a question asked in somebody else's words and work out what is actually being asked. \
     It is almost never the question they asked.",
    "what", "The enquiry you are taking.",
    "Somebody outside the vault has asked something and it is sitting unanswered.",
    r#"{"what":"the question about the eastern gate"}"#,
    "Taking it is a claim: it is yours until it is answered or named unanswerable."
);

pub const ENQUIRY_ANSWER_FROM_RECORD: Tool = on_with!(
    "enquiry_answer_from_record",
    ENQUIRY,
    "Service",
    Plane::World,
    "Answer out of the record, and say plainly which part of your answer the record does not \
     support. Refusing to go beyond what you know is part of the answer.",
    "what",
    "The enquiry you are answering.",
    "answer",
    "What the record says, and where it stops.",
    "You have found what the record holds on a question, and it covers most of it.",
    r#"{"what":"the question about the eastern gate","answer":"who held it and when it fell; the record does not say who opened it, and nothing here ever has"}"#,
    "The boundary of the answer is the most useful part of it."
);

pub const ENQUIRY_NAME_THE_GAP: Tool = on_with!(
    "enquiry_name_the_gap",
    ENQUIRY,
    "Service",
    Plane::World,
    "Say that a question cannot be answered, and name what is missing. An enquiry nobody can \
     satisfy names a hole nobody inside had noticed.",
    "what",
    "The enquiry the record cannot answer as it stands.",
    "missing",
    "What would have to exist for it to be answerable.",
    "You have read everything bearing on a question and the thing it turns on was never written.",
    r#"{"what":"the question about the eastern gate","missing":"anything at all about the night watch that span"}"#,
    "The unanswerable ones are the valuable ones — they are a second source of work."
);

pub const ENQUIRY_RAISE_WORK: Tool = on_with!(
    "enquiry_raise_work",
    ENQUIRY,
    "Service",
    Plane::World,
    "Turn a question you could not answer into a piece of work somebody could take.",
    "from",
    "The enquiry it came out of.",
    "work",
    "What somebody would actually have to do.",
    "You have named a gap and it is large enough that somebody should fill it.",
    r#"{"from":"the question about the eastern gate","work":"write the night watch for that span from the muster rolls"}"#,
    "A gap named and left is a gap. A gap turned into work is the vault's second source of it."
);

// ── Orders, dispatch, the cast ──────────────────────────────────────────────

pub const ORDERS_SET: Tool =
    on_with!(
    "orders_set", MUSTER, "Command", Plane::World,
    "Put a standing order on the board for whoever will take it. It waits there until somebody \
     claims it, so this is for work that needs doing rather than work you are doing — an order \
     nobody takes is still on the board tomorrow.",
    "what", "What is to be done, in one line somebody can take off a board and act on.",
    "for", "On whose authority, when it is not your own. An order in somebody else's name is a \
     different thing from one in yours, and anybody reading the board can tell.",
    "Something needs doing, you are not the one to do it, and nobody has been told.",
    r#"{"what":"survey the eastern ridge before the light goes","for":"the tower lord"}"#,
    "Setting work is an act. A finding nobody turned into an order is a finding nobody acts on."
);

pub const ORDERS_HAND_TO: Tool =
    on_with!(
    "orders_hand_to", MUSTER, "Command", Plane::World,
    "Give an order to somebody in particular, rather than leaving it for whoever comes.",
    "what", "The order, as it is written on the board.",
    "to", "Who is to carry it out.",
    "The job needs the one person who has done it before, and leaving it on the board would get \
     you somebody else.",
    r#"{"what":"survey the eastern ridge before the light goes","to":"Wren"}"#,
    "Handing it to somebody is delegation and reads as such — they know it was chosen for them."
);

pub const ORDERS_REPORT_DONE: Tool = on!(
    "orders_report_done",
    ORDERS,
    "Command",
    Plane::World,
    "Report an order you were holding as finished, to whoever set it.",
    "what",
    "The order you have finished.",
    "You have done the thing and nobody knows yet.",
    r#"{"what":"survey the eastern ridge before the light goes"}"#,
    "Work nobody reported is work nobody can build on."
);

pub const DISPATCH_POST_WAKE: Tool = on_with!(
    "dispatch_post_wake",
    DISPATCH,
    "Command",
    Plane::World,
    "Post what your change has broken, so the people whose work it breaks find out from you \
     rather than from the breakage.",
    "from",
    "The change you made.",
    "breaks",
    "What now needs repairing, and whose it is.",
    "You moved a boundary, and three entries either side of it now say the wrong thing.",
    r#"{"from":"the third era","breaks":"the two entries in the fourth era that date from the old boundary"}"#,
    "The wake is the expensive half of a change and the half that gets skipped."
);

pub const CAST_REPORT_DISAGREEMENT: Tool = on_with!(
    "cast_report_disagreement",
    WATCH,
    "Casting",
    Plane::World,
    "Report something across the cast that does not fit — and it is not yours to correct, which \
     is the point of reporting it.",
    "about",
    "Who or what does not fit.",
    "what",
    "What the disagreement actually is.",
    "Reading across everybody, two characters cannot both have been where they are said to be.",
    r#"{"about":"the courier","what":"placed at the gate and on the road on the same night, by two different holders"}"#,
    "Noticing is yours; fixing is whoever holds it. Reporting is what joins the two."
);

pub const CREATOR_PRESENT: Tool =
    on!(
    "creator_present", CHAIR, "Command", Plane::World,
    "Present finished work, and be able to say what you would still change about it.",
    "what", "What you are presenting.",
    "You have filed something long and it is the first thing anybody has asked you about.",
    r#"{"what":"the third silence"}"#,
    "Presenting is not reporting. What you would still change is the part that is worth hearing."
);

// ── The plant, the stores, the shape of a piece ─────────────────────────────

pub const PLANT_NOTE_DRIFT: Tool = on_with!(
    "plant_note_drift",
    PLANT,
    "Plant",
    Plane::World,
    "Note a reading that has been slightly wrong for a while. In a place that runs itself the \
     danger is not failure, it is drift.",
    "what",
    "The reading that has moved, by the name on the panel.",
    "drift",
    "What it should be, and how long you think it has been off.",
    "A figure on the panel is not alarming and is not what it was a span ago.",
    r#"{"what":"the circulation figure","drift":"four per cent low, and steady — so it has been drifting since before anybody looked"}"#,
    "A fault announces itself. Drift does not, which is why noticing it is an act."
);

pub const PLANT_RAISE_FAULT: Tool = on_with!(
    "plant_raise_fault",
    PLANT,
    "Plant",
    Plane::World,
    "Raise a fault nobody else thinks is a fault, and be prepared to be wrong in public.",
    "what",
    "What you think is wrong.",
    "why",
    "Why you think so, given nobody else does.",
    "Everything reads normal and one thing has not behaved the way it used to.",
    r#"{"what":"the circulation figure","why":"it has never held this steady before; steady is the part that is wrong"}"#,
    "Being wrong in public about a fault is cheaper than being right in private."
);

pub const STORES_PUT_BACK: Tool = on!(
    "stores_put_back",
    STORES,
    "Stores",
    Plane::World,
    "Put something back on the racks where it belongs, without minding who left it out.",
    "what",
    "What you are putting back.",
    "Something is out that should be racked, and you did not leave it.",
    r#"{"what":"the western intake"}"#,
    "Noticing where a thing goes is most of the work, and it is why this is an act."
);

pub const STORES_TAKE_OUT: Tool =
    on!(
    "stores_take_out", STORES, "Stores", Plane::World,
    "Take something off the racks to work with. It is out until somebody puts it back.",
    "what", "What you are taking out.",
    "You need a thing that is racked and you are about to work on it.",
    r#"{"what":"the western intake"}"#,
    "What is out and what is racked are different states, and the difference is what a walk of \
     the racks finds."
);

pub const STRUCTURE_LAY_OUT_SCENES: Tool =
    on!(
    "structure_lay_out_scenes", STRUCTURE, "Structure", Plane::World,
    "Lay a piece out as its scenes, so the one where nobody wants anything shows itself. A piece \
     that is merely nice cannot hide from this.",
    "what", "The piece to lay out.",
    "Something you made reads well and does not hold together, and you cannot see where.",
    r#"{"what":"the third silence"}"#,
    "The shape is invisible while you are inside the prose and obvious once it is pinned up."
);

pub const STRUCTURE_TEST_THE_WANT: Tool =
    on_with!(
    "structure_test_the_want", STRUCTURE, "Structure", Plane::World,
    "Test a scene's want by asking whether you could photograph the moment it is satisfied. If \
     you could not, it is a mood and has to be replaced.",
    "in", "The piece the scene belongs to.",
    "scene", "The scene whose want you are testing.",
    "A scene in your piece is doing nothing and you suspect the want is the reason.",
    r#"{"in":"the third silence","scene":"the one at the gate"}"#,
    "\"Understand the era better\" fails it. \"Get the gate open\" passes."
);

pub const STRUCTURE_FIND_THE_SLACK: Tool = on!(
    "structure_find_the_slack",
    STRUCTURE,
    "Structure",
    Plane::World,
    "Find where a piece stops being a chain of consequences and becomes a list of events.",
    "what",
    "The piece you are reading for slack.",
    "The middle of something you made goes slack and you cannot say where it starts.",
    r#"{"what":"the third silence"}"#,
    "The slack is always somewhere specific. Naming it is what makes it fixable."
);

pub const GATHER_CALL: Tool = on_with!(
    "gather_call",
    READING_TABLE,
    "Gathering",
    Plane::World,
    "Call everybody whose work touches a thing that concerns all of them, and say what it is \
     about.",
    "about",
    "What the gathering is for.",
    "who",
    "Whose work it touches.",
    "A question has come up that four people are each solving differently and none of them know.",
    r#"{"about":"how dates either side of a boundary should be written","who":"whoever holds an era"}"#,
    "A standing reason to be in one room is the strongest thing there is for convergence, and \
     no individual has to invent it."
);

/// Everything in this module, in the order it is offered.
pub const STATION_ACTS: &[Tool] = &[
    CHRONICLE_ADD_ENTRY,
    CHRONICLE_REWRITE_PAGE,
    CHRONICLE_RETIRE_ENTRY,
    CHRONICLE_SETTLE_BOUNDARY,
    STORY_DRAFT,
    STORY_FILE,
    PORTRAIT_DRAW,
    PORTRAIT_PROMPT_READ,
    PORTRAIT_PROMPT_EDIT,
    PORTRAIT_SETTLE_LIKENESS,
    PORTRAIT_FILE_PLATE,
    LIBRARY_READ,
    LIBRARY_WRITE,
    CHARACTER_WRITE_IDENTITY,
    CHARACTER_WRITE_WANTS,
    CHARACTER_WRITE_MEMORIES,
    CHARACTER_SETTLE_RELATION,
    PLACE_WRITE_ENTRY,
    PLACE_WRITE_LOCAL_HISTORY,
    PLACE_SETTLE_ROUTE,
    MAP_ADD_PLACE,
    MAP_SETTLE_BORDER,
    MAP_REMOVE_PLACE,
    RECORD_ACCESSION,
    RECORD_WRITE_PROVENANCE,
    RECORD_HAND_ON,
    RECORD_APPRAISE,
    RECORD_WRITE_REASON,
    RECORD_LET_GO,
    RECORD_DESCRIBE,
    RECORD_ARRANGE,
    RECORD_CROSS_REFERENCE,
    RECORD_LEAVE_NOTE,
    RECORD_TIDY_INDEX,
    RECORD_MEND,
    RECORD_MARK_REPAIR,
    ENQUIRY_TAKE_QUESTION,
    ENQUIRY_ANSWER_FROM_RECORD,
    ENQUIRY_NAME_THE_GAP,
    ENQUIRY_RAISE_WORK,
    ORDERS_SET,
    ORDERS_HAND_TO,
    ORDERS_REPORT_DONE,
    DISPATCH_POST_WAKE,
    CAST_REPORT_DISAGREEMENT,
    CREATOR_PRESENT,
    PLANT_NOTE_DRIFT,
    PLANT_RAISE_FAULT,
    STORES_PUT_BACK,
    STORES_TAKE_OUT,
    STRUCTURE_LAY_OUT_SCENES,
    STRUCTURE_TEST_THE_WANT,
    STRUCTURE_FIND_THE_SLACK,
    GATHER_CALL,
];
