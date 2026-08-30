//! The mock world — every `/v1/*` handler reads from here.
//!
//! This exists so the GUI can be built, styled and driven end-to-end before the
//! engine is written. It is deliberately *shaped* like the real thing: the JSON
//! it emits conforms to the object definitions in `docs/npc_api_gui_design.md`
//! §10, ids cross the wire as decimal **strings** (§3), and every timestamped
//! object carries both `at_ms` and `world_ms` (§4).
//!
//! When the engine lands, these functions are replaced by calls into
//! `NpcEngine` and the wire contract does not move.

use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Wall clock in ms. The mock's `world_ms` is derived from a fixed world epoch
/// so narrative time is stable and legible in the UI.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Day 412, 06:14 in the mock world, advancing at 60x wall time.
pub fn world_ms() -> u64 {
    const WORLD_EPOCH: u64 = 412 * 86_400_000 + 6 * 3_600_000 + 14 * 60_000;
    WORLD_EPOCH + (now_ms() % 3_600_000) * 60
}

pub fn user() -> Value {
    json!({
        "user_id": "u_8812",
        "unique_name": "Wren",
        "display": "Johnathan",
        "email": "johnathan.sharratt@gmail.com",
        "avatar_url": null,
        "provider": "google",
        "profile": {
            "description": "Reads people quickly, talks slowly. Ex-surveyor, so \
                            tends to describe places by their edges.",
            "gender": "Male",
            "history": "Grew up on the coast. Came inland for work and stayed.",
            "turn_index": 7,
            "revision": 3
        },
        // Deliberately no `npc_count`. §8.3: hidden characters are never
        // enumerated or counted, and a total of everything an author owns is
        // the one figure that gives them away — subtract it from the roster in
        // front of you. A field here would look like a contract waiting to be
        // honoured, and the honouring would count everything.
        "created_ms": now_ms() - 86_400_000 * 40
    })
}

/// The distinguishing fields of one roster row. Named rather than positional
/// because the interesting differences between these NPCs are two bools, a
/// float and two integers, and a positional call site reduces all of that to
/// `"ticking", 11, "fixated", 0.38, 5_000, false` — unreadable at the exact
/// place the fixture is meant to be read.
struct Row<'a> {
    id: &'a str,
    name: &'a str,
    arch: &'a str,
    state: &'a str,
    pending: u64,
    band: &'a str,
    overlap: f64,
    heartbeat: u64,
    hidden: bool,
    tags: &'a [&'a str],
    desc: &'a str,
}

fn npc(r: Row<'_>) -> Value {
    let Row {
        id,
        name,
        arch,
        state,
        pending,
        band,
        overlap,
        heartbeat,
        hidden,
        tags,
        desc,
    } = r;
    json!({
        "npc_id": id,
        "name": name,
        "world_id": "ardh",
        "personality_id": arch,
        "personality_name": personality_name(arch),
        "state": state,
        "tick": {
            "heartbeat_ms": heartbeat,
            "last_tick_ms": now_ms() - (pending * 900 + 400),
            "pending_events": pending,
            "salience_gate": 0.42
        },
        "environment_enabled": true,
        "monitor": { "overlap": overlap, "band": band },
        "owner_id": "u_8812",
        "access": "owner",
        "hidden": hidden,
        "tags": tags,
        "portrait": { "image_id": format!("img_{id}"), "origin": "generated" },
        "persona": { "description": desc, "origin": "generated" },
        "live_interactions": if state == "active" { 2 } else { 0 },
        "created_ms": now_ms() - 86_400_000 * 12,
        "updated_ms": now_ms() - 4_000
    })
}

pub fn npcs() -> Vec<Value> {
    vec![
        npc(Row {
            id: "10237749914772934281",
            name: "Varek",
            arch: "loyal-soldier",
            state: "active",
            pending: 3,
            band: "healthy",
            overlap: 0.19,
            heartbeat: 30_000,
            hidden: false,
            tags: &["campaign-2", "north"],
            desc: "Fifty-three, a former staff sergeant who now runs the night shift on a \
                   loading dock. Precise about time to the point of rudeness. Comfortable \
                   giving orders, uneasy in conversations with no clear purpose. Keeps a \
                   folding knife he doesn't use.",
        }),
        npc(Row {
            id: "10237749914772934282",
            name: "Ilse",
            arch: "merchant",
            state: "active",
            pending: 0,
            band: "healthy",
            overlap: 0.11,
            heartbeat: 120_000,
            hidden: false,
            tags: &["campaign-2", "market"],
            desc: "Late thirties, runs a stall she inherited and has quietly doubled. \
                   Friendly in a way that is also a negotiation. Remembers every price \
                   anyone ever quoted her.",
        }),
        npc(Row {
            id: "10237749914772934283",
            name: "Hess",
            arch: "commander",
            state: "ticking",
            pending: 11,
            band: "fixated",
            overlap: 0.38,
            heartbeat: 5_000,
            hidden: false,
            tags: &["campaign-2", "north", "command"],
            desc: "Sixty, career officer, recently passed over. Speaks in complete \
                   paragraphs. Has started reading disloyalty into ordinary delays.",
        }),
        npc(Row {
            id: "10237749914772934284",
            name: "Bramble",
            arch: "gardener",
            state: "asleep",
            pending: 0,
            band: "healthy",
            overlap: 0.08,
            heartbeat: 300_000,
            hidden: false,
            tags: &["ambient"],
            desc: "Seventy-one, keeps the allotment behind the church. Cheerful, \
                   digressive, and occasionally stops mid-sentence when something reminds \
                   him of the campaign.",
        }),
        npc(Row {
            id: "10237749914772934285",
            name: "Sable",
            arch: "drifter",
            state: "idle",
            pending: 0,
            band: "healthy",
            overlap: 0.14,
            heartbeat: 90_000,
            hidden: true,
            tags: &["moonlight"],
            desc: "Thirties, no fixed trade, arrives places slightly before she is \
                   expected. Answers questions with questions.",
        }),
        npc(Row {
            id: "10237749914772934286",
            name: "Toll-keeper",
            arch: "drifter",
            state: "suspended",
            pending: 0,
            band: "healthy",
            overlap: 0.05,
            heartbeat: 600_000,
            hidden: false,
            tags: &["north"],
            desc: "Ageless in the way of people who sit in booths. Has opinions about \
                   everyone who crosses, and shares them for a fee.",
        }),
    ]
}

/// Ids here are slugs because that is what a reference IS — the name of the
/// file the document lives in. The fixture used numbered rows; a console that
/// works against `world_id: "1"` and then meets `battle-cities` is a console
/// that was tested against a shape the daemon never serves.
fn personality_name(id: &str) -> &'static str {
    match id {
        "loyal-soldier" => "Loyal Soldier",
        "merchant" => "Merchant",
        "commander" => "Commander",
        "gardener" => "Gardener",
        _ => "Drifter",
    }
}

pub fn worlds() -> Vec<Value> {
    vec![json!({
        "world_id": "ardh",
        "name": "Ardh",
        "public": false,
        "setting": "A kingdom of hill villages on a northern frontier, three years after a war \
                    nobody won. Roads are unsafe after dark. The crown is distant and the \
                    garrisons are underpaid.",
        "npc_count": 6,
        "time": { "world_ms": world_ms(), "scale": 60.0, "paused": false },
        "zoom_bands": ["strategic", "regional", "tactical", "local"],
        "templates": { "responses": "override", "moods": "default" }
    })]
}

pub fn personalities() -> Vec<Value> {
    [
        (
            "loyal-soldier",
            "Betrayal is unforgivable. Orders are a contract, not a request.",
            3,
        ),
        (
            "merchant",
            "Every exchange is a relationship. Price is memory made numeric.",
            1,
        ),
        (
            "commander",
            "Position is read before people are. Loyalty is assessed, not assumed.",
            1,
        ),
        (
            "gardener",
            "Things grow at their own rate. Patience is not passivity.",
            1,
        ),
        ("drifter", "Attachment is a cost. Observation is free.", 2),
    ]
    .iter()
    .map(|(id, anchor, n)| {
        json!({
            "personality_id": id, "name": personality_name(id),
            // `anchor`, matching `personalities/<id>.yaml`. It was
            // `core_identity`, a second name for the same field that only ever
            // appeared on the wire.
            "anchor": anchor,
            "personality": {
                "voice": "Short sentences. Rank and role before names. Silence rather than a guess.",
                "processing": "Weight direct observation over second-hand intel. Distrust a plan \
                               with no named fallback.",
                "under_pressure": "Get narrower, not louder. Reduce the problem until one action \
                                   is obviously next.",
            },
            "npc_count": n, "doctrine_version": 4,
            "doctrine": "Flank at 2:1 or not at all. Cross open ground only with a fallback named."
        })
    })
    .collect()
}

pub fn beliefs(_npc: &str) -> Vec<Value> {
    vec![
        json!({ "belief_id": "hess_word", "statement": "Hess is a man of his word",
                "confidence": 0.72, "threshold": 0.85, "disconfirmation": 0.30,
                "origin": "authored", "under_pressure": true,
                "history": [ {"at_world_ms": world_ms()-86_400_000*40, "confidence": 0.95},
                             {"at_world_ms": world_ms()-86_400_000*20, "confidence": 0.93},
                             {"at_world_ms": world_ms()-86_400_000*5,  "confidence": 0.81},
                             {"at_world_ms": world_ms(),               "confidence": 0.72} ] }),
        json!({ "belief_id": "north_road", "statement": "The northern road is passable in winter",
                "confidence": 0.95, "threshold": 0.60, "disconfirmation": 0.0,
                "origin": "evidence", "under_pressure": false, "history": [] }),
        json!({ "belief_id": "orders", "statement": "An order given badly is still an order",
                "confidence": 0.90, "threshold": 0.95, "disconfirmation": 0.05,
                "origin": "generated", "under_pressure": false, "history": [] }),
    ]
}

pub fn relationships(_npc: &str) -> Vec<Value> {
    vec![
        json!({ "entity_id": "hess", "display": "Commander Hess", "trust": 0.6,
                "affect": 0.2, "familiarity": 0.9,
                "last_contact_world_ms": world_ms() - 3_600_000, "notes": "Chain of command." }),
        json!({ "entity_id": "ilse", "display": "Ilse", "trust": 0.1, "affect": 0.4,
                "familiarity": 0.35, "last_contact_world_ms": world_ms() - 86_400_000,
                "notes": "Sells him tobacco. Overcharges, both know it." }),
        json!({ "entity_id": "wren", "display": "Wren", "trust": 0.0, "affect": 0.05,
                "familiarity": 0.1, "last_contact_world_ms": world_ms() - 600_000,
                "notes": "New. Asks direct questions, which he respects." }),
    ]
}

pub fn agency(_npc: &str) -> Vec<Value> {
    vec![
        json!({ "strategy_id": "hold_ridge", "statement": "Hold the eastern ridge until relieved",
                "state": "active", "parent_id": null,
                "children": ["watch_rotation", "fallback_named"], "salience": 0.88,
                "progress_notes": ["Rotation set", "Fallback to the mill agreed"] }),
        json!({ "strategy_id": "watch_rotation", "statement": "Keep a two-hour watch rotation",
                "state": "active", "parent_id": "hold_ridge", "children": [], "salience": 0.51,
                "progress_notes": [] }),
        json!({ "strategy_id": "fallback_named", "statement": "Name a fallback before dark",
                "state": "finished", "parent_id": "hold_ridge", "children": [], "salience": 0.12,
                "progress_notes": ["The mill"] }),
    ]
}

pub fn layer_summary(_npc: &str) -> Value {
    json!({ "layers": [
        { "layer": "perception",    "turns": 41,   "tokens": 12_400,  "window": 16_000, "resident": 88 },
        { "layer": "action",        "turns": 212,  "tokens": 31_900,  "window": 16_000, "resident": 62 },
        { "layer": "agency",        "turns": 6,    "tokens": 2_100,   "window": 4_000,  "resident": 100 },
        { "layer": "relationships", "turns": 14,   "tokens": 3_800,   "window": 4_000,  "resident": 100 },
        { "layer": "beliefs",       "turns": 9,    "tokens": 2_400,   "window": 4_000,  "resident": 100 },
        { "layer": "memory",        "turns": 4412, "tokens": 918_233, "window": 8_000,  "resident": 61 },
        { "layer": "interaction",   "turns": 88,   "tokens": 19_100,  "window": 16_000, "resident": 74 },
        { "layer": "environment",   "turns": 24,   "tokens": 5_200,   "window": 6_000,  "resident": 100 },
        { "layer": "world",         "turns": 88,   "tokens": 21_000,  "window": 8_000,  "resident": 47 }
    ]})
}

pub fn projection(tick: u64) -> Value {
    json!({
        "tick": tick,
        "budget": { "total": 16_000, "used": 15_214 },
        "system_prompt": {
            "mood": "tense", "mood_spiked_at": tick.saturating_sub(3),
            "template": "battlefield_urgency",
            "sections": ["identity_anchor", "situation", "concerns"]
        },
        "layers": [
            { "layer": "perception",    "gathered": 8,  "available": 41,   "tokens": 4120, "top_score": 0.94 },
            { "layer": "action",        "gathered": 5,  "available": 212,  "tokens": 2010, "top_score": 0.81 },
            { "layer": "beliefs",       "gathered": 3,  "available": 9,    "tokens": 812,  "top_score": 0.88 },
            { "layer": "relationships", "gathered": 2,  "available": 14,   "tokens": 540,  "top_score": 0.77 },
            { "layer": "agency",        "gathered": 1,  "available": 6,    "tokens": 260,  "top_score": 0.69 },
            { "layer": "memory",        "gathered": 11, "available": 4412, "tokens": 3180, "top_score": 0.72 },
            { "layer": "world",         "gathered": 4,  "available": 88,   "tokens": 1290, "top_score": 0.66 }
        ],
        "dropped": [
            { "layer": "memory", "turns": 6, "reason": "budget" },
            { "layer": "world",  "turns": 9, "reason": "threshold" }
        ]
    })
}

pub fn monitor(window: usize) -> Value {
    let pts: Vec<Value> = (0..window)
        .map(|i| {
            let t = i as f64 / window as f64;
            let v = 0.12 + 0.10 * (t * 6.0).sin().abs() + 0.06 * t;
            json!({ "tick": 312 + i as u64, "value": (v * 1000.0).round() / 1000.0 })
        })
        .collect();
    json!({ "band": "healthy", "overlap": pts,
            "thresholds": { "fixated": 0.35, "runaway": 0.55 } })
}

pub fn tools() -> Value {
    let t = |name: &str, cat: &str, desc: &str, src: &str, cal: bool, modes: Value| {
        json!({ "name": name, "category": cat, "description": desc, "source": src,
                "calibrated": cal, "writes_layers": ["action"], "modes": modes })
    };
    let all = json!(["physical", "video_call", "voice_call", "instant_message"]);
    let msg = json!(["video_call", "instant_message"]);
    json!({ "uncalibrated": 1, "tools": [
        t("speak", "speech", "Say something. Carries intent, not words — the narrator renders it.", "generic", true, all.clone()),
        t("send_image", "messaging", "Send a picture to a named interlocutor. Messaging modes only.", "generic", true, msg),
        t("move_to", "movement", "Move to a named location.", "generic", true, all.clone()),
        t("face", "movement", "Turn to face a direction or entity.", "generic", true, all.clone()),
        t("follow", "movement", "Follow an entity.", "generic", true, all.clone()),
        t("flee", "movement", "Break contact and withdraw.", "generic", true, all.clone()),
        t("gesture", "gesture", "Perform a visible gesture.", "generic", true, all.clone()),
        t("express", "gesture", "Show an expression.", "generic", true, all.clone()),
        t("observe", "attention", "Direct attention at something.", "generic", true, all.clone()),
        t("listen", "attention", "Attend to sound.", "generic", true, all.clone()),
        t("inspect", "attention", "Examine an object closely.", "generic", true, all.clone()),
        t("greet", "social", "Acknowledge someone.", "generic", true, all.clone()),
        t("offer", "social", "Offer something.", "generic", true, all.clone()),
        t("refuse", "social", "Decline.", "generic", true, all.clone()),
        t("threaten", "social", "Make a threat.", "generic", true, all.clone()),
        t("note_concern", "internal", "Record a concern. No observable trace.", "generic", true, all.clone()),
        t("set_intent", "internal", "Set a standing intent.", "generic", true, all.clone()),
        t("broadcast_strategy", "internal", "Write upward to the strategy layer.", "generic", true, all.clone()),
        t("wait", "meta", "Do nothing this tick.", "generic", true, all.clone()),
        t("end_interaction", "meta", "Close the interaction.", "generic", true, all.clone()),
        t("open_gate", "extension", "Open a named gate in the world.", "extension", false, all)
    ]})
}

pub fn commands() -> Value {
    let c = |name: &str, group: &str, summary: &str, emits: &str, params: Value, req: Value| {
        json!({ "name": name, "group": group, "summary": summary, "aliases": [],
                "emits": emits, "parameters": params, "required": req })
    };
    json!({ "commands": [
        c("say", "narration", "Speak as yourself", "interaction_event",
          json!({"type":"object","properties":{"text":{"type":"string","description":"What you say"}}}),
          json!(["text"])),
        c("act", "narration", "Perform a physical action", "interaction_event",
          json!({"type":"object","properties":{"action":{"type":"string","description":"What you do"}}}),
          json!(["action"])),
        c("scene", "narration", "Describe the environment", "environment_event",
          json!({"type":"object","properties":{"description":{"type":"string"}}}),
          json!(["description"])),
        c("cue", "narration", "Force the NPC to act (it does not deliberate)", "interaction_event",
          json!({"type":"object","properties":{"character":{"type":"string"},"action":{"type":"string"}}}),
          json!(["action"])),
        c("beat", "narration", "Steer the narration. Operator-only; never shown to a participant.", "interaction_event",
          json!({"type":"object","properties":{"description":{"type":"string"}}}),
          json!(["description"])),
        c("damage", "combat", "Apply damage to the NPC", "perception",
          json!({"type":"object","properties":{
            "amount":{"type":"integer","minimum":1,"maximum":100,"description":"Hit points"},
            "source":{"type":"string","description":"What caused it"},
            "location":{"type":"string","enum":["head","torso","left_arm","right_arm","leg"]},
            "severity":{"type":"number","minimum":0,"maximum":1,"default":0.5}}}),
          json!(["amount"])),
        c("danger", "combat", "Raise the perceived threat level", "perception",
          json!({"type":"object","properties":{"level":{"type":"number","minimum":0,"maximum":1}}}),
          json!(["level"])),
        c("daybreak", "world", "Advance the world clock to dawn", "environment_event",
          json!({"type":"object","properties":{}}), json!([])),
        c("weather", "world", "Change the weather", "environment_event",
          json!({"type":"object","properties":{
            "kind":{"type":"enum","enum":["clear","rain","fog","snow","storm"]},
            "intensity":{"type":"number","minimum":0,"maximum":1,"default":0.5}}}),
          json!(["kind"])),
        c("enter", "world", "Someone enters the scene", "perception",
          json!({"type":"object","properties":{"who":{"type":"string"},"from":{"type":"string"}}}),
          json!(["who"])),
        c("give", "social", "Hand the NPC an object", "perception",
          json!({"type":"object","properties":{"item":{"type":"string"},"from":{"type":"string"}}}),
          json!(["item"])),
        c("open_gate", "extension", "Open a named gate (registered by the game)", "environment_event",
          json!({"type":"object","properties":{"gate_id":{"type":"string"}}}), json!(["gate_id"]))
    ]})
}

// ── interactions ─────────────────────────────────────────────────────────────

static IX_SEQ: Mutex<u64> = Mutex::new(4_471_028_855_119);

pub fn new_interaction_id() -> String {
    let mut g = IX_SEQ.lock().unwrap();
    *g += 1;
    g.to_string()
}

pub fn interaction(ix: &str, npc: &str, mode: &str) -> Value {
    json!({
        "interaction_id": ix,
        "npc_id": npc,
        "mode": mode,
        "interlocutor": { "kind": "operator", "id": "u_8812", "display": "Wren" },
        "state": "live",
        "idle_timeout_secs": match mode { "physical" => 300, "instant_message" => 86_400, _ => 600 },
        "idle_remaining_secs": 612,
        "opened_world_ms": world_ms() - 900_000,
        "opened_ms": now_ms() - 900_000,
        "act_count": 14,
        "narration_count": 5
    })
}

pub fn interactions(npc: &str) -> Vec<Value> {
    vec![
        interaction("4471028855119", npc, "physical"),
        interaction("4471028855120", npc, "instant_message"),
    ]
}

/// A canned act/narration script the SSE stream replays so the console is
/// alive without an engine behind it.
pub fn script() -> Vec<Value> {
    let a = |id: &str, tick: u64, tool: &str, intent: &str, args: Value, obs: Value, text: &str| {
        json!({ "act_id": id, "tick": tick, "tool": tool, "intent": intent, "args": args,
                "observable_in": obs, "committed": true,
                "rendered": if text.is_empty() { Value::Null } else { json!({"text": text}) },
                "world_ms": world_ms(), "at_ms": now_ms() })
    };
    let all = json!(["physical", "video_call", "voice_call", "instant_message"]);
    let vis = json!(["physical", "video_call"]);
    let none = json!([]);
    vec![
        a(
            "a_88210",
            411,
            "face",
            "check the eastern line",
            json!({"dir":"east"}),
            vis.clone(),
            "He glances east.",
        ),
        a(
            "a_88211",
            411,
            "speak",
            "acknowledge Wren, stay watchful",
            json!({"to":"Wren"}),
            all.clone(),
            "Quiet, so far.",
        ),
        json!({ "kind": "tick", "tick": 411, "acts": 2 }),
        json!({ "kind": "narration", "narration_id": "n_5511", "tick": 411,
                "text": "He straightens as you approach, shears still in hand. \"Quiet, so far,\" \
                         he says, and glances east.",
                "covers_acts": ["a_88210","a_88211"], "world_ms": world_ms(), "at_ms": now_ms() }),
        a(
            "a_88212",
            412,
            "observe",
            "read the eastern line",
            json!({"target":"eastern_line"}),
            vis.clone(),
            "He squints east.",
        ),
        a(
            "a_88213",
            412,
            "speak",
            "break off — the line is buckling",
            json!({"to":"Wren"}),
            all,
            "…hold on.",
        ),
        a(
            "a_88214",
            412,
            "move_to",
            "get to the ridge",
            json!({"to":"ridge_east"}),
            vis,
            "He starts moving.",
        ),
        a(
            "a_88215",
            412,
            "broadcast_strategy",
            "the northern road matters more than the ridge",
            json!({}),
            none,
            "",
        ),
        json!({ "kind": "tick", "tick": 412, "acts": 4 }),
        json!({ "kind": "narration", "narration_id": "n_5512", "tick": 412,
                "text": "You ask what he sees; before he can answer he's already moving as the \
                         eastern line buckles. Somewhere below, a horn.",
                "covers_acts": ["a_88212","a_88213","a_88214"], "world_ms": world_ms(), "at_ms": now_ms() }),
    ]
}

pub fn environment(_npc: &str) -> Value {
    json!({
        "enabled": true,
        "window_turns": 24,
        "system_prompt": "You describe what happens around a character in Ardh: a northern \
                          frontier three years after an inconclusive war. Keep to what could be \
                          perceived from where they stand. Never narrate their thoughts or \
                          decide their actions. Change the world slowly and only for a reason.",
        "recent": [
            { "world_ms": world_ms() - 600_000, "text": "Wind off the ridge; the light going amber." },
            { "world_ms": world_ms() - 300_000, "text": "A horn, twice, from below the eastern slope." },
            { "world_ms": world_ms() - 60_000,  "text": "The line east of the mill gives ground." }
        ]
    })
}

pub fn status() -> Value {
    json!({
        "state": "ready",
        "detail": "mock backend — no engine loaded",
        "started_at_ms": now_ms() - 3_600_000,
        "build": "npcd-mock-0.1.0",
        "mode": "server-headless",
        "loading": { "current": "Ready", "progress": 1.0, "completed": [
            "Mock store", "Router", "Web assets" ] }
    })
}

/// A synthetic half-hour in the column shape `npcd::telemetry` serves.
///
/// The real daemon answers this route itself, so this fixture is only reached
/// when the console is served without one behind it. It still has to speak the
/// current shape: a fixture that has drifted from the contract renders a broken
/// page, which is a worse failure than an obviously fake one.
pub fn telemetry() -> Value {
    const N: usize = 900;
    const PERIOD: f64 = 2.0;
    let wave = |i: usize, a: f64, b: f64, f: f64| a + b * ((i as f64) / f).sin();

    let t: Vec<f64> = (0..N).map(|i| i as f64 * PERIOD).collect();
    let used: Vec<f64> = (0..N)
        .map(|i| wave(i, 9_000.0, 1_800.0, 90.0).round())
        .collect();
    let free: Vec<f64> = used.iter().map(|u| 24_576.0 - u).collect();
    let image: Vec<f64> = (0..N)
        .map(|i| if i % 130 < 40 { 640.0 } else { 0.0 })
        .collect();
    let col = |f: &dyn Fn(usize) -> f64| -> Vec<f64> { (0..N).map(f).collect() };

    json!({
        "gpu": { "name": "mock device", "compute_cap": "8.6", "pcie_gen": 3, "pcie_width": 16 },
        "model": {
            "name": "Qwen3-30B-A3B", "quant": "Q6_K",
            "params_total": "30B", "params_active": "3B",
            "repo": "unsloth/Qwen3-30B-A3B-GGUF",
            "filename": "Qwen3-30B-A3B-Q6_K.gguf",
            "bytes": 25_092_532_800_u64
        },
        "host": { "total_mib": 65_457, "free_mib": 41_000, "rss_mib": 820 },
        "sample_period_s": PERIOD,
        "engine_connected": true,
        "image_queue_state": "waiting_for_vram",
        "uptime_s": 4_820,
        "series": {
            "t": t,
            "vram_total_mib": col(&|_| 24_576.0),
            "vram_used_mib": used,
            "vram_free_mib": free,
            "host_total_mib": col(&|_| 65_457.0),
            "host_used_mib": col(&|i| wave(i, 24_000.0, 2_600.0, 140.0).round()),
            "rss_mib": col(&|_| 820.0),
            "weights_mib": col(&|_| 5_400.0),
            "kv_mib": col(&|i| wave(i, 2_200.0, 900.0, 70.0).round()),
            "image_mib": image,
            "decode_tps": col(&|i| wave(i, 430.0, 90.0, 55.0).round()),
            "prefill_tps": col(&|i| wave(i, 1_100.0, 320.0, 33.0).round()),
            "mean_npcs_per_decode": col(&|i| (wave(i, 3.1, 1.1, 61.0) * 10.0).round() / 10.0),
            "max_batch": col(&|_| 6.0),
            "npcs_active": col(&|i| wave(i, 7.0, 4.0, 77.0).round()),
            "ticks_per_sec": col(&|i| (wave(i, 0.45, 0.2, 48.0) * 100.0).round() / 100.0),
            "inbox_depth_p50": col(&|i| wave(i, 1.0, 2.0, 40.0).abs().round()),
            "inbox_depth_p99": col(&|i| wave(i, 11.0, 9.0, 95.0).abs().round()),
            "image_queue_depth": col(&|i| if i % 130 < 40 { 2.0 } else { 0.0 })
        }
    })
}

/// `GET /v1/memory` — the fixture counterpart of `npcd::telemetry::memory`.
pub fn memory() -> Value {
    const MIB: u64 = 1024 * 1024;
    json!({
        "report": null,
        "report_age_ms": null,
        "host_now": {
            "total_bytes": 65_457 * MIB,
            "available_bytes": 41_000 * MIB,
            "free_bytes": 38_200 * MIB
        },
        "process": { "working_set_bytes": 820 * MIB, "virtual_bytes": 41_000 * MIB }
    })
}
