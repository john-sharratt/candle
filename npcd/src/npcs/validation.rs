//! Holds the bounds and state rules every field of a character record must
//! meet.
//!
//! A value is within its bound, refused with the field's name, or left
//! unchanged when absent, so a partial edit never resets an unrelated field.
//! Bounds exist because every record is rewritten whole on each save: a layer
//! that could grow without limit would make every later write, compaction and
//! maintenance pass more expensive. Names are cleaned of control characters so
//! they cannot reach a record or a page, and duplicate tags are dropped rather
//! than failing the edit.

use super::*;

/// States a character may be in, mirroring §10 of the design doc. A state off
/// this list is a typo or a version skew, and either way not something to write
/// into a durable record.
pub const STATES: [&str; 5] = ["active", "idle", "asleep", SUSPENDED, "tombstoned"];

/// The state that stops a character's turns: a suspended character takes no
/// turn of any kind until its state is changed again (`crate::suspend`). Every
/// other state is its owner's word for it and changes nothing about how it
/// runs.
pub const SUSPENDED: &str = "suspended";

/// The state a new character starts in. Idle, not active: a character that has
/// never been ticked is not doing anything, and saying `active` would put it in
/// every "who is thinking" count before it has thought.
pub(super) const INITIAL_STATE: &str = "idle";

/// Default idle metabolism. Salience raises it; this is the resting rate.
pub(super) const DEFAULT_HEARTBEAT_MS: u64 = 30_000;

pub(super) const DEFAULT_SALIENCE_GATE: f32 = 0.42;

/// A name has to fit in a record and in a sentence. Long enough for a title
/// ("The Toll-keeper of the North Gate"), short enough not to be prose.
pub(super) const MAX_NAME: usize = 120;

pub(super) const MAX_PERSONA: usize = 8_000;

pub(super) const MAX_TAGS: usize = 32;

pub(super) const MAX_TAG: usize = 48;

/// How many entries each authoring-plane layer may hold.
///
/// **These are the only unbounded collections on the record, and the record is
/// rewritten whole on every edit.** Each layer is keyed by an id, so revising an
/// entry replaces it — but a *new* id appended, and nothing stopped that. A
/// character accumulating beliefs one save at a time grows a record that every
/// subsequent write copies, every compaction carries forward, and every
/// maintenance pass relocates.
///
/// The figure is operator scale, which is what this plane is for: §16's
/// authoring layers are "tens of entries, not the thousands an engine would
/// accumulate". A hundred and twenty-eight beliefs is far past any character
/// somebody has actually written and still bounds the record.
///
/// The cap is on *appending*. An entry that already exists can always be
/// revised or deleted, so reaching the limit never leaves a character stuck
/// with content it cannot edit its way out of.
pub(super) const MAX_BELIEFS: usize = 128;

pub(super) const MAX_RELATIONSHIPS: usize = 128;

pub(super) const MAX_STRATEGIES: usize = 128;

/// A field that must be 0..1, or the value it already had.
///
/// Absent means unchanged, never zero: a `PUT` that sets one dial must not
/// silently reset the other two, which is the whole reason these take the
/// current value rather than a default.
pub(super) fn unit(body: &Value, key: &str, current: f32) -> Result<f32, NpcError> {
    bounded(body, key, current, 0.0, 1.0)
}

/// A field that must be −1..1, on the same terms.
pub(super) fn signed(body: &Value, key: &str, current: f32) -> Result<f32, NpcError> {
    bounded(body, key, current, -1.0, 1.0)
}

pub(super) fn bounded(
    body: &Value,
    key: &str,
    current: f32,
    lo: f64,
    hi: f64,
) -> Result<f32, NpcError> {
    let Some(v) = body.get(key) else {
        return Ok(current);
    };
    let n = v.as_f64().ok_or(NpcError::Invalid(leak(key)))?;
    if !n.is_finite() || !(lo..=hi).contains(&n) {
        return Err(NpcError::Invalid(leak(key)));
    }
    Ok(n as f32)
}

/// The longest one line of authored text may be. Generous for a statement or a
/// note, short of a way to fill a disk one save at a time.
///
/// This was `MAX_PROMPT_CHARS`, belonging to the simulated environment's system
/// prompt and borrowed by the helper below. That feature is gone; the bound is
/// still needed here, so it is named for what it actually guards.
pub(super) const MAX_LINE_CHARS: usize = 8_000;

/// One line of authored text, trimmed and bounded.
pub(super) fn clean_line(v: &Value, key: &'static str) -> Result<String, NpcError> {
    let s = v.as_str().ok_or(NpcError::Invalid(key))?.trim();
    if s.chars().count() > MAX_LINE_CHARS {
        return Err(NpcError::Invalid(key));
    }
    Ok(s.to_string())
}

/// `NpcError::Invalid` names the field in a `&'static str`, and these come from
/// a runtime key. The set is closed and small, so it is matched rather than
/// leaked — a `Box::leak` here would grow the binary's heap by one string per
/// bad request, for ever.
pub(super) fn leak(key: &str) -> &'static str {
    match key {
        "confidence" => "confidence",
        "threshold" => "threshold",
        "trust" => "trust",
        "affect" => "affect",
        "familiarity" => "familiarity",
        "threat" => "threat",
        "curiosity" => "curiosity",
        _ => "value",
    }
}

pub(super) fn clean_name(v: Option<&Value>) -> Result<String, NpcError> {
    let s = v
        .and_then(Value::as_str)
        .ok_or(NpcError::Invalid("name"))?
        .trim();
    if s.is_empty() || s.chars().count() > MAX_NAME {
        return Err(NpcError::Invalid("name"));
    }
    // Control characters would survive into a record and out onto a page.
    if s.chars().any(|c| c.is_control()) {
        return Err(NpcError::Invalid("name"));
    }
    Ok(s.to_string())
}

pub(super) fn clean_persona(v: Option<&Value>) -> Result<String, NpcError> {
    let s = v.and_then(Value::as_str).unwrap_or("").trim();
    if s.chars().count() > MAX_PERSONA {
        return Err(NpcError::Invalid("persona_description"));
    }
    Ok(s.to_string())
}

pub(super) fn clean_tags(v: Option<&Value>) -> Result<Vec<String>, NpcError> {
    let Some(arr) = v.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    if arr.len() > MAX_TAGS {
        return Err(NpcError::Invalid("tags"));
    }
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        let s = t.as_str().ok_or(NpcError::Invalid("tags"))?.trim();
        if s.is_empty() || s.chars().count() > MAX_TAG || s.chars().any(|c| c.is_control()) {
            return Err(NpcError::Invalid("tags"));
        }
        // Duplicates are dropped rather than refused: a repeated tag is a
        // slip, not an error worth losing an edit over.
        if !out.contains(&s.to_string()) {
            out.push(s.to_string());
        }
    }
    Ok(out)
}
