//! Defines the JSON form a character takes on the wire and the base-36 form its
//! id takes.
//!
//! Ids are `u64` and travel as base-36 strings, because a JSON number above
//! 2^53 is rounded by every browser that parses it. Engine measurements are sent
//! as null rather than zero, since a zero would claim a measurement nobody made.
//! Floats are rounded before they are widened, so a value typed as 0.42 is not
//! sent back as 0.41999998688697815. The filter is applied here, not in the
//! browser, so a filter control cannot silently fail to exclude what it names.

use super::*;

/// What the roster's filter bar asks for.
///
/// Applied here rather than in the browser because the page fetches a listing
/// and would otherwise narrow only what it happens to hold — and because a
/// filter control that silently does nothing is worse than no control at all:
/// the reader believes they have excluded something.
#[derive(Debug, Default)]
pub struct Filter<'a> {
    /// Exact tag match. Tags are chosen from a small authored set, so a
    /// substring match here would be surprising rather than helpful.
    pub tag: Option<&'a str>,
    /// One of [`STATES`]. `any` (or absent) means no state filter.
    pub state: Option<&'a str>,
    pub world_id: Option<&'a str>,
    /// Free text over the name and the persona — case-insensitive substring.
    pub q: Option<&'a str>,
    pub include_hidden: bool,
}

impl Filter<'_> {
    pub(super) fn matches(&self, n: &NpcPayload) -> bool {
        if let Some(t) = self.tag.filter(|t| !t.is_empty()) {
            if !n.tags.iter().any(|x| x == t) {
                return false;
            }
        }
        if let Some(s) = self.state.filter(|s| !s.is_empty() && *s != "any") {
            if n.state != s {
                return false;
            }
        }
        if let Some(w) = self.world_id.filter(|w| !w.is_empty()) {
            if n.world_id != w {
                return false;
            }
        }
        if let Some(q) = self.q.map(str::trim).filter(|q| !q.is_empty()) {
            // Lowercased on both sides: somebody searching "varek" means the
            // character called "Varek".
            let q = q.to_lowercase();
            let hit = n.name.to_lowercase().contains(&q)
                || n.persona_description.to_lowercase().contains(&q)
                || n.tags.iter().any(|t| t.to_lowercase().contains(&q));
            if !hit {
                return false;
            }
        }
        true
    }
}

/// Write a character id in base-36 (`0-9a-z`), lowercase.
///
/// The same reason and the same alphabet as a world instance id's ordinal
/// (`npc_map::instance::base36`): this id is bandied around every page that
/// lists a character and every route that names one, so it is spent in a
/// human's read and a console's URL far more than the twenty digits a minted
/// hash needs in decimal — a real id is `6817662845163923144`, base-36 is
/// `1gtpgs1gobgkw`. It is still carried as a JSON **string**, never a bare
/// number: a `u64` this wide already sits well past 2^53, so a bare JSON
/// number would round in any JavaScript reader before the encoding even
/// mattered — see [`wire`] and [`npc_id_of_wire`].
pub fn npc_id_wire(mut n: u64) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if n == 0 {
        return "0".to_string();
    }
    let mut buf = Vec::new();
    while n > 0 {
        buf.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    buf.reverse();
    // Every byte is an ASCII digit from `DIGITS`, so this is always valid UTF-8.
    String::from_utf8(buf).expect("base-36 digits are ASCII")
}

/// The inverse of [`npc_id_wire`]: read a character id back off the wire.
///
/// `None` for anything that is not a valid base-36 numeral — an empty string,
/// a character outside `0-9a-z` (read case-insensitively, so a hand-typed
/// uppercase id still resolves), or digits that overflow `u64`. Every route
/// that used to treat a failed decimal parse as "no such character" treats a
/// `None` here the same way.
///
/// **There is no version tag distinguishing this from the decimal `to_string`
/// this format replaced**, because every decimal numeral is also a valid
/// base-36 one — a caller holding a pre-migration decimal id decodes it here
/// as a base-36 number instead of getting a clean parse failure. No shim
/// carries that old format forward (`AGENTS.md`: no backward compatibility in
/// this pre-publication codebase), so the residual risk is bounded by how the
/// id itself is minted rather than guarded against here: character ids are
/// drawn across the full `u64` range, so a real id's base-36 form is 13
/// characters wide and the odds every one of them lands on a digit — the only
/// way it would still read as a plausible decimal number — are
/// `(10/36)^13`, on the order of one in ten million. A short, hand-authored
/// test id (`npc_id: 5`) does not carry that guarantee, so treat any
/// deliberately small id as decimal-shaped and pick a large one instead.
pub fn npc_id_of_wire(s: &str) -> Option<u64> {
    u64::from_str_radix(s, 36).ok()
}

/// A `#[serde(serialize_with = "npcs::npc_id_wire_serde")]` for a struct field
/// that is a character id — [`npc_id_wire`] in the shape `serde` wants of a
/// field serializer, for a wire type derived rather than built by hand.
pub fn npc_id_wire_serde<S: serde::Serializer>(id: &u64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&npc_id_wire(*id))
}

/// One character as §10 defines it on the wire.
///
/// Ids cross as base-36 **strings** ([`npc_id_wire`]): they are `u64`, and a
/// JSON number above 2^53 is silently rounded by every browser that parses
/// it, so the string is mandatory with or without the encoding underneath it.
pub(super) fn wire(n: &NpcPayload, caller: &str) -> Value {
    json!({
        "npc_id": npc_id_wire(n.npc_id),
        "name": n.name,
        "world_id": n.world_id,
        "personality_id": n.personality_id,
        "state": n.state,
        "tick": {
            "heartbeat_ms": n.heartbeat_ms,
            // Rounded on the way out. `json!` widens the stored `f32` to `f64`,
            // and 0.42f32 widens to 0.41999998688697815 — a gate somebody typed
            // as "0.42" coming back as noise, which looks like the daemon
            // changed it.
            "salience_gate": (f64::from(n.salience_gate) * 1_000.0).round() / 1_000.0,
            // Live values, and this daemon runs no engine, so they are absent
            // rather than zero — a character with `pending_events: 0` reads as
            // measured and idle, which is a claim nothing here can make.
            "last_tick_ms": Value::Null,
            "pending_events": Value::Null,
        },
        // Same reason: the monitor is an engine measurement.
        "monitor": Value::Null,
        // The register it last said it was in, by mood id.
        //
        // **Null, never a default register.** A character nobody has asked yet
        // and one that is calm are different facts, and only one of them is a
        // measurement — the same rule the live tick figures above follow.
        "mood": n.mood,
        "modulation": {
            "affect": round3(n.modulation.affect),
            "threat": round3(n.modulation.threat),
            "curiosity": round3(n.modulation.curiosity),
        },
        "owner_id": n.owner_id,
        "access": if n.owner_id == caller { "owner" } else { "viewer" },
        "hidden": n.hidden,
        "tags": n.tags,
        "portrait": n.portrait_image_id.as_ref().map(|id| json!({
            "image_id": id,
            "origin": n.portrait_origin.clone().unwrap_or_else(|| "generated".to_string()),
        })),
        "persona": { "description": n.persona_description, "origin": n.persona_origin },
        "created_ms": n.created_ms,
        "updated_ms": n.updated_ms,
        "revision": n.revision,
    })
}

/// An `f32` on the wire, without the widening noise.
///
/// `json!` widens to `f64`, and `0.42f32` widens to `0.41999998688697815` — a
/// dial somebody typed as 0.42 coming back as noise, which reads as the daemon
/// having changed it.
pub(super) fn round3(v: f32) -> f64 {
    (f64::from(v) * 1_000.0).round() / 1_000.0
}

/// The authoring plane, as the console reads it.
///
/// The engine's measurements are **absent**, not zero. A belief has no
/// `disconfirmation` until something has weighed evidence against it, and a
/// strategy has no `salience` until something has scored it — reporting either
/// as 0 would be a measurement this daemon has not made.
pub fn beliefs_wire(n: &NpcPayload) -> Value {
    json!({ "beliefs": n.beliefs.iter().map(|b| json!({
        "belief_id": b.belief_id,
        "statement": b.statement,
        "confidence": round3(b.confidence),
        "threshold": round3(b.threshold),
        "origin": "authored",
        "disconfirmation": Value::Null,
        "under_pressure": Value::Null,
        "history": Value::Null,
    })).collect::<Vec<_>>() })
}

pub fn relationships_wire(n: &NpcPayload) -> Value {
    json!({ "relationships": n.relationships.iter().map(|r| json!({
        "entity_id": r.entity_id,
        "display": r.display,
        "trust": round3(r.trust),
        "affect": round3(r.affect),
        "familiarity": round3(r.familiarity),
        "notes": r.notes,
        "origin": "authored",
    })).collect::<Vec<_>>() })
}

pub fn agency_wire(n: &NpcPayload) -> Value {
    json!({ "agency": n.agency.iter().map(|a| json!({
        "strategy_id": a.strategy_id,
        "statement": a.statement,
        "parent_id": a.parent_id,
        "state": a.state,
        "origin": "authored",
        // Scored by the engine against what is happening; nothing here has.
        "salience": Value::Null,
        "progress_notes": Value::Null,
    })).collect::<Vec<_>>() })
}

pub fn modulation_wire(n: &NpcPayload) -> Value {
    json!({
        "affect": round3(n.modulation.affect),
        "threat": round3(n.modulation.threat),
        "curiosity": round3(n.modulation.curiosity),
    })
}

/// Ids arrive as decimal strings; accept a number too, since a hand-written
/// request is the common way to hit this.
/// A reference to an authored world or personality: the slug that is its file
/// name.
///
/// Validated with the registry's own gate rather than a looser rule of its own.
/// The two must agree, because the point of storing the slug is that it always
/// resolves to a file — a reference this accepted and `registry::id::check`
/// would refuse is a durable record naming something that can never exist.
///
/// Whether the file is actually *there* is a different question, answered where
/// the registries are (`api::create_npc`); this is the shape check.
pub(super) fn ref_id(v: Option<&Value>) -> Option<String> {
    let s = v?.as_str()?;
    id::check(s).ok().map(|()| s.to_string())
}
