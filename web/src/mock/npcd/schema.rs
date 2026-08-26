//! The projection schema a world runs under, and the authored section
//! collections that fill its system prompt.
//!
//! A world is not "a name and a blurb" — it is the substrate layers its NPCs
//! think through (window, budget, selection rule, masking) plus the section
//! collections the lens is assembled from. An archetype owns collections of its
//! own: the identity anchor and its detail facets, which are read-only by
//! construction because they are the CoW prefix.

use serde_json::{json, Value};

#[allow(clippy::too_many_arguments)]
fn layer(
    name: &str,
    window: u64,
    prio: u64,
    min_pct: u64,
    sel: &str,
    mask: &str,
    thresh: f64,
    dec: &str,
    summarize: bool,
    desc: &str,
) -> Value {
    json!({
        "layer": name, "window": window,
        "budget": { "priority": prio, "min_percent": min_pct },
        "selection": sel, "masking": mask, "score_threshold": thresh,
        "decode_priority": dec, "summarize": summarize, "description": desc
    })
}

pub fn layers() -> Value {
    json!({ "layers": [
        layer("perception", 16000, 100, 40, "sequence(recent 12, top-k 8)", "self-local", 0.0, "high", false,
            "API-fed. Maps supersede at the same zoom band; descriptions accumulate."),
        layer("action", 16000, 95, 30, "sequence(recent 16, top-k 6)", "self-local", 0.0, "high", true,
            "The act stream. Ground truth — everything narrated is read from here."),
        layer("agency", 4000, 80, 20, "top-k 4", "self-local", 0.35, "normal", false,
            "Missions, strategies and sub-goals."),
        layer("relationships", 4000, 75, 20, "top-k 6", "self-local", 0.30, "normal", false,
            "Per-entity calibration. Writable on both planes."),
        layer("beliefs", 4000, 90, 25, "top-k 5", "self-local", 0.40, "normal", false,
            "Write-protected against the action plane. Evidence threshold only."),
        layer("memory", 8000, 60, 15, "sequence(recent 4, top-k 12)", "self-local", 0.25, "low", true,
            "Unbounded. The consolidation target for daydream and sleep folds."),
        layer("interaction", 16000, 100, 50, "sequence(recent 16, top-k 8)", "self-local", 0.0, "high", true,
            "One timeline per interaction, forked from the NPC's sealed prefix."),
        layer("environment", 6000, 50, 10, "sequence(recent 24)", "self-local", 0.0, "low", false,
            "Sliding window only — continuity of the scene, not recall."),
        layer("world", 8000, 70, 10, "top-k 6", "cross-timeline", 0.30, "low", true,
            "The only unmasked layer: shared facts are retrievable across NPCs."),
    ]})
}

fn section(id: &str, category: &str, tokens: u64, examples: u64, body: &str) -> Value {
    json!({ "id": id, "category": category, "tokens": tokens, "examples": examples, "template": body })
}

fn collection(
    name: &str,
    folder: &str,
    rule: &str,
    locked: bool,
    source: &str,
    description: &str,
    sections: Value,
) -> Value {
    json!({ "name": name, "folder": folder, "rule": rule, "locked": locked,
            "source": source, "description": description, "sections": sections })
}

/// Collections owned by a world: the mutable half of the lens.
pub fn world_collections() -> Value {
    json!({ "collections": [
        collection("response", "responses/", "named(selector: response) · locked", false, "world override",
            "The structural mode of a reply. Selected once at interaction start by top-k provenance \
             match against the incoming query, then frozen for the entire decode.",
            json!([
                section("battlefield_urgency", "combat", 128, 3,
                    "Answer in short, load-bearing sentences. Lead with the thing that changes what they do next. No preamble."),
                section("military_briefing", "combat", 142, 3,
                    "Situation, then assessment, then recommendation, in that order. Name uncertainties explicitly."),
                section("merchant_negotiation", "social", 156, 4,
                    "Never name the first number. Acknowledge what they want before saying what it costs."),
                section("casual_conversation", "social", 118, 3,
                    "Follow the other person's thread. Volunteer detail only when asked or genuinely surprising."),
                section("whispered_conspiracy", "social", 134, 2,
                    "Short clauses. Assume you may be overheard. Say the dangerous part last and least directly."),
                section("storytelling", "social", 148, 3,
                    "Set the scene before the event. One concrete sensory anchor per beat, never the same one twice."),
            ])),
        collection("mood", "moods/", "named(selector: mood) · spiking", false, "defaults",
            "Event-driven and threshold-gated, not drifting. Holds its register until provenance scores \
             a different one above the spike threshold at a barrier, then snaps.",
            json!([
                section("confident", "affect", 96, 2, "Certain, unhurried. Declaratives. No hedging."),
                section("tense", "affect", 104, 2,
                    "Clipped. Attention divided — part of you is elsewhere, and it shows in what you leave unfinished."),
                section("grieving", "affect", 112, 2, "Slower. Ordinary things take effort to name."),
                section("analytical", "affect", 98, 2, "Structure first. Enumerate before you conclude."),
                section("guarded", "affect", 101, 2, "Answer the question asked and no more."),
            ])),
        collection("situation", "situations/", "top-k 2", false, "defaults",
            "The current mission, strategy and perception state, surfaced only when relevant.",
            json!([
                section("under_orders", "framing", 88, 1, "You have a standing order and a named fallback."),
                section("unsupervised", "framing", 84, 1, "Nobody is watching. What you do now is yours."),
            ])),
    ]})
}

/// Collections owned by an archetype: the immutable half. Read-only by
/// construction — they are the shared CoW prefix, and mutating one would break
/// the sharing that makes it free.
pub fn archetype_collections() -> Value {
    json!({ "collections": [
        collection("identity_anchor", "identities/<name>/anchor.yaml", "always-visible", true, "archetype",
            "The always-on compressed self. Structurally resident — it never competes for the gather \
             budget, because it is the prefix the budget is read inside.",
            json!([
                section("anchor", "identity", 186, 0,
                    "You are a soldier before you are anything else. An order is a contract. Betrayal is not a setback, it is a category."),
            ])),
        collection("identity", "identities/<name>/*.yaml", "top-k 3", true, "archetype",
            "Detail facets of the same self, surfaced only when relevant to the exchange.",
            json!([
                section("voice", "identity", 132, 2, "Short sentences. Rank and role before names. Silence rather than a guess."),
                section("processing", "identity", 148, 2,
                    "Weight direct observation over second-hand intel. Distrust a plan with no named fallback."),
                section("history", "identity", 164, 1,
                    "Twenty years in, three campaigns, one of them the kind nobody writes down."),
                section("under_pressure", "identity", 121, 2,
                    "Get narrower, not louder. Reduce the problem until one action is obviously next."),
            ])),
        collection("doctrine", "doctrine.yaml", "always-visible", false, "archetype · evolves",
            "The one part of the shared layer designed to change. Aggregated from strategic learning \
             across every NPC of this type, then published as a version.",
            json!([
                section("current", "doctrine", 142, 0,
                    "Flank at 2:1 or not at all. Cross open ground only with a fallback named."),
            ])),
    ]})
}

pub fn log_lines() -> Value {
    let l = |ts: &str, level: &str, target: &str, msg: &str| json!({ "ts": ts, "level": level, "target": target, "msg": msg });
    json!({ "lines": [
        l("06:14:02", "INFO", "npcd", "npcd ready — mock backend, no engine loaded"),
        l("06:14:02", "INFO", "npcd::api", "router mounted: 45 routes"),
        l("06:14:03", "DEBUG", "substrate", "9 layers declared, 3 collections resolved"),
        l("06:14:09", "INFO", "scheduler", "wave slice 2000ms · admission window 4"),
        l("06:14:11", "DEBUG", "tick", "npc …4281 gate 0.42 → tick scheduled"),
        l("06:14:11", "TRACE", "projection", "gather: 34 turns, 15214/16000 tok, 6 dropped (budget)"),
        l("06:14:12", "INFO", "narrator", "rendered a_88211 in 214ms"),
        l("06:14:14", "WARN", "image", "queue depth 2 — waiting for VRAM headroom"),
        l("06:14:19", "DEBUG", "tick", "npc …4283 preempted (salience 0.91)"),
        l("06:14:20", "WARN", "monitor", "npc …4283 overlap 0.38 → band=fixated"),
        l("06:14:22", "INFO", "scheduler", "batch composition: 3 npcs / decode"),
        l("06:14:26", "ERROR", "image", "job_img_1 abandoned: relief exhausted before slot claim"),
        l("06:14:31", "DEBUG", "persistence", "checkpoint written · 41 records · 812 KiB"),
    ]})
}
