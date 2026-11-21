//! The durable record of one character: its identity, its configuration, and what a
//! restart must remember.
//!
//! Values a running engine derives from the world are kept out of the record, so a
//! resting character writes nothing new. The payload holds what an author set,
//! plus the last place the character stood, its register, and its suspended state,
//! which the engine cannot rebuild on its own. A record written before the
//! authoring fields existed still decodes, with those fields unwritten. Deletion is
//! a superseding record marked tombstoned, so the id stays taken.

use super::*;

/// JSON payload for a [`RecordType::Npc`] record — one character, as it
/// durably is. One entry per `npc_id`, last-writer-wins on replay.
///
/// # Durable configuration only
///
/// This is deliberately *not* the whole `Npc` object the API serves. The wire
/// object (`docs/npc_api_gui_design.md` §10) also carries values that move
/// every tick — `tick.last_tick_ms`, `tick.pending_events`, `monitor.overlap`,
/// `monitor.band` — and one that is computed per caller (`access`, which is a
/// function of `owner_id` and who is asking).
///
/// Storing those would mean a record per tick per character: a write
/// amplification that turns an idle population into continuous log growth, to
/// persist numbers that are meaningless a second later and are recomputed at
/// load anyway. Keeping them out is what lets the write policy be
/// *on change to durable state*, which for a resting character is never.
///
/// So: what an author set, plus what the engine must remember across a restart.
/// Nothing that a running engine can derive.
///
/// `PartialEq` but not `Eq`: `salience_gate` is an `f32`, and a threshold read
/// from a slider is a float whether or not that is convenient here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NpcPayload {
    /// Decimal-string id on the wire, `u64` here — the same treatment
    /// `timeline_id` gets, and for the same reason: JSON numbers above 2^53
    /// are not safely representable in a browser.
    pub npc_id: u64,
    /// The account that owns this character. Ownership is authorization
    /// (§8.2), so this is the field every access check reads.
    ///
    /// A string, not a `u64`, because that is what an account id *is* — the
    /// normative object in §10 gives it as `"u_8812"`, a prefixed handle rather
    /// than a number. `npc_id` beside it is a genuine `u64` written as a
    /// decimal string; the two look alike on the wire and are not the same kind
    /// of thing.
    pub owner_id: String,
    /// Monotonic per-character edit counter. Replay is in log order, so the
    /// last record already wins; this makes that deterministic when a
    /// compaction re-emits several characters in one pass, and gives an edit a
    /// stable identity to log.
    pub revision: u64,
    pub created_ms: u64,
    pub updated_ms: u64,
    /// `active` | `idle` | `asleep` | `suspended` | `tombstoned`. A deleted
    /// character is `tombstoned` and keeps its record: the id must stay taken,
    /// and the acts it already committed still name it.
    pub state: String,
    pub name: String,
    /// The authored world and personality this character belongs to, named by
    /// the slug that IS their file name — `"battle-cities"`, `"commander"`.
    ///
    /// These were `u64`s, from a design where both were numbered rows. They are
    /// files now (`worlds/<id>.yaml`, `personalities/<id>.yaml`), and a file's
    /// identity is its name; a number beside it would be a second identity for
    /// the same thing, needing a table to reconcile them and free to disagree.
    /// Validity is `registry::id::check` — the same rule that decides what may
    /// become a file name, so a reference can always be resolved to one.
    pub world_id: String,
    pub personality_id: String,
    /// Omitted from default listings (§8.3). Never counted in a total — the
    /// count is what gives a hidden character away.
    pub hidden: bool,
    /// Idle metabolism in milliseconds, and the salience level below which an
    /// event does not wake the character. Authored configuration, not a
    /// measurement — the *live* tick figures are excluded, see above.
    pub heartbeat_ms: u64,
    pub salience_gate: f32,
    pub tags: Vec<String>,
    pub persona_description: String,
    /// `authored` | `generated`.
    pub persona_origin: String,
    /// Absent until a portrait exists.
    pub portrait_image_id: Option<String>,
    /// `uploaded` | `generated`; absent with the image.
    pub portrait_origin: Option<String>,

    /// Where this character was last standing, as `area/node`.
    ///
    /// **The one piece of world state on this record, and it earns its place by
    /// the rule stated above**: it is not something a running engine can derive.
    /// The world — who is standing where, what has just happened — is held in
    /// RAM by the daemon and has no persistence of its own, so on a restart
    /// every character re-entered its world at the arrival door regardless of
    /// where it had walked to. Two Makers who had spent an hour finding each
    /// other were returned to the front room as strangers, while their
    /// conversation history said otherwise.
    ///
    /// Position only. A journey in flight is *not* here: `Actor::at` is by
    /// definition the last place actually reached, so nothing is lost by
    /// dropping the rest of the route, and a character re-forms the intention
    /// anyway. What a character is holding is not here either — nothing claims a
    /// station yet, and a field for it would be a guess at a shape.
    ///
    /// `None` for a character that has never been embodied, or whose world has
    /// no map. Written by daemons from before a world kept its bodies' places
    /// as its own state records (`docs/replicated_substrate.md` §9.1), and read
    /// now only to take such a world in, once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,

    /// The register this character last said it was in, by mood id.
    ///
    /// **The second piece of lived state here, and it earns its place the same
    /// way `at` does**: a running engine cannot derive it. A mood is what a
    /// character *is* between one thought and the next, and holding it only in
    /// RAM meant a restart returned everybody to no register at all — so a
    /// character that had spent the afternoon getting angrier came back with no
    /// sign of it, while its conversation history said otherwise.
    ///
    /// Written when a character names one for itself and it has changed. Not an
    /// authoring act: it bumps no revision and moves no `updated_ms`, for the
    /// reason `at` does not — those describe what somebody *edited*, and a
    /// roster that sorted by "recently edited" would report whoever was feeling
    /// something.
    ///
    /// `None` for a character that has never said. Absent rather than a
    /// default register, because "nobody has asked it yet" and "it is calm" are
    /// different facts and only one of them is true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mood: Option<String>,

    /// The state a suspension of the whole cast found this character in, while
    /// that suspension holds it `suspended`.
    ///
    /// `suspended` is what stops a character's turns, and suspending the whole
    /// cast sets it on every character it reaches — over whatever state each
    /// one's owner had set. This is that state, so resuming the whole cast
    /// puts each one back where it was. Written only by that suspension; any
    /// other change of state, an owner's own suspend among them, clears it, so
    /// a character its owner suspended stays suspended when the cast resumes.
    ///
    /// Absent on every other record, so a record no such suspension holds
    /// carries the same bytes it always did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suspended_from: Option<String>,

    /* ── the authoring plane (§16)──────────────────────────────────────────
     *
     * What an operator says a character believes, who they know, what they are
     * trying to do, and the dials their affect starts at. Every write here is
     * an authoring act — `origin: "authored"` — as distinct from what the
     * evidence process later earns.
     *
     * Carried on the character's own record rather than in record types of
     * their own, for three reasons. It is operator-scale: tens of entries, not
     * the thousands an engine would accumulate. It supersedes with the
     * character it belongs to, which is the lifetime it actually has. And the
     * write path is the one already here, already tested, already handled by
     * compaction — where a new header-keyed type would need all three again.
     *
     * The engine's own belief traffic is a different problem with a different
     * volume, and will want its own records. This is not that; this is what
     * somebody types when they build a world. */
    /* The five authoring-plane layers below carry `#[serde(default)]`, and that
     * is a statement about meaning rather than a compatibility shim.
     *
     * A character nobody has written beliefs for **holds none** — which is
     * exactly what `Npcs::create` writes, an empty vector — so an absent field
     * and an empty one denote the same character. Without the default, a record
     * written before these layers existed fails to decode and the walker skips
     * it, which is how a substrate holding thirteen characters reported a cast
     * of zero: not a corrupt log, a record whose reader had learned five new
     * required questions and refused anything that could not answer them.
     *
     * The distinction worth keeping is between "absent" and "measured zero",
     * and it does not apply here: these are authored content, and unwritten
     * authored content is empty. Where a *measurement* is missing the engine
     * still reports `null` rather than defaulting it — see `crate::engine`. */
    /// Beliefs an operator wrote. Never written by the action plane — a tool
    /// declaring `beliefs` in `writes_layers` is refused at registration.
    #[serde(default)]
    pub beliefs: Vec<AuthoredBelief>,
    /// Who this character knows, and how they hold them.
    #[serde(default)]
    pub relationships: Vec<AuthoredRelationship>,
    /// What they are trying to do, as a tree.
    #[serde(default)]
    pub agency: Vec<AuthoredStrategy>,
    /// The affect dials an operator set.
    #[serde(default)]
    pub modulation: Modulation,
}

/// One authored belief.
///
/// Only what an operator states. `disconfirmation`, whether it is under
/// pressure, and its confidence history are all measurements the evidence
/// process makes, so they are not stored here and are absent on the wire until
/// something has measured them — the same rule the live tick fields follow.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuthoredBelief {
    pub belief_id: String,
    pub statement: String,
    /// How strongly it is held, 0..1.
    pub confidence: f32,
    /// How much contrary evidence it takes to break it, 0..1.
    pub threshold: f32,
}

/// One authored relationship.
///
/// A relationship is a calibration trajectory and is meant to move easily —
/// §16 notes it carries none of the write-protection beliefs do, on either
/// plane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuthoredRelationship {
    pub entity_id: String,
    /// What this character calls them.
    pub display: String,
    pub trust: f32,
    pub affect: f32,
    pub familiarity: f32,
    pub notes: String,
}

/// One authored strategy, in a tree by `parent_id`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuthoredStrategy {
    pub strategy_id: String,
    pub statement: String,
    /// `None` for a root strategy.
    pub parent_id: Option<String>,
    /// `active` | `finished` | `abandoned`.
    pub state: String,
}

/// The affect dials, as set. Each is −1..1 except `threat` and `curiosity`,
/// which are 0..1 — bounds the daemon checks, not this record.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Modulation {
    pub affect: f32,
    pub threat: f32,
    pub curiosity: f32,
}

impl Default for Modulation {
    /// Neutral. A character nobody has tuned is not sad, not threatened, and
    /// mildly curious — which is what an author who has not touched the dials
    /// means, and is a real position rather than an absence.
    fn default() -> Self {
        Self {
            affect: 0.0,
            threat: 0.0,
            curiosity: 0.5,
        }
    }
}

impl NpcPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("NpcPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("Npc JSON parse: {e}")))
    }

    /// Whether this record marks the character as deleted. A tombstoned
    /// character is still a record — the id stays taken and its committed acts
    /// still name it — but it is gone from every listing.
    pub fn is_tombstoned(&self) -> bool {
        self.state == "tombstoned"
    }
}
