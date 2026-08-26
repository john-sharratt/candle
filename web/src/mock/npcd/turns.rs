//! Turn bodies and their K/V segment vectors, plus the live projection probe.
//!
//! A turn is stored as one continuous block with the intra-turn role boundary
//! baked in — never as two halves that get re-glued, because every separator is
//! itself a materialised piece and splicing one in would corrupt the spacing.
//!
//! The **segment vector** is the engine's complete description of what a turn
//! contributes to the K/V grid. A segment whose `kv` is null is *ethereal*: it
//! was recorded, but is not part of this turn's own grid — the spine
//! materialised it, or a reasoning block was dropped. Surfacing that difference
//! is the whole point of showing the vector rather than the text.

use serde_json::{json, Value};

const USER_START: &str = "<|im_start|>user\n";
const IM_END: &str = "<|im_end|>\n";
const ASSISTANT_START: &str = "<|im_start|>assistant\n";

fn seg(kind: &str, text: Option<&str>, marker: Option<&str>, kv: Option<(u64, u64)>) -> Value {
    json!({
        "kind": kind,
        "text": text,
        "marker": marker,
        "kv": kv.map(|(offset, len)| json!({ "offset": offset, "len": len })),
    })
}

/// One turn: the continuous body plus the segment vector describing it.
pub fn turn(npc: &str, layer: &str, index: u64) -> Value {
    let _ = npc;
    let (user, assistant, thinking) = body_for(layer, index);

    let ul = (user.len() / 4).max(1) as u64;
    let tl = thinking.map(|t| (t.len() / 4).max(1) as u64).unwrap_or(0);
    let al = (assistant.len() / 4).max(1) as u64;

    let mut segments = vec![
        // The turn opener is materialised by the spine, not by this turn — so it
        // is recorded here but carries no K/V of its own.
        seg("glue", None, Some("user_start"), None),
        seg("user", Some(user), None, Some((0, ul))),
        seg("glue", None, Some("im_end"), Some((ul, 2))),
        seg("glue", None, Some("assistant_start"), Some((ul + 2, 3))),
    ];
    let mut cursor = ul + 5;
    if let Some(t) = thinking {
        // A dropped reasoning block: recorded for the operator, absent from K/V.
        segments.push(seg("thinking", Some(t), None, None));
        let _ = tl;
    }
    segments.push(seg("assistant", Some(assistant), None, Some((cursor, al))));
    cursor += al;
    segments.push(seg("glue", None, Some("im_end"), None));
    let _ = cursor;

    let text = format!("{USER_START}{user}{IM_END}{ASSISTANT_START}{assistant}");

    json!({
        "layer": layer,
        "turn": index,
        "text": text,
        "user": user,
        "assistant": assistant,
        "tokens": ul + al,
        "layout": { "segments": segments },
    })
}

fn body_for(layer: &str, index: u64) -> (&'static str, &'static str, Option<&'static str>) {
    let i = (index % 4) as usize;
    match layer {
        "perception" => (
            ["A horn, twice, from the eastern slope.",
             "Wind off the ridge; the light going amber.",
             "The line east of the mill gives ground.",
             "Movement in the treeline — two, maybe three."][i],
            "Registered. Threat weighting raised on the eastern approach.",
            None,
        ),
        "action" => (
            "tick 412",
            ["speak → \"Quiet, so far.\"", "face → east",
             "move_to → ridge_east", "observe → eastern_line"][i],
            Some("The line is giving. Say the reassuring thing, then move — she can follow or not."),
        ),
        "beliefs" => (
            "Evidence review",
            ["Hess countermanded the rotation twice, then denied the second order. \
              Disconfirmation on \"a man of his word\" now 0.30 of a 0.85 threshold.",
             "The northern road held through the thaw. Confidence unchanged at 0.95.",
             "An order given badly is still an order — nothing this week tested it.",
             "No new evidence bearing on any standing belief."][i],
            None,
        ),
        "memory" => (
            "Consolidation fold",
            ["The mill road washed out; the crossing moved a mile north and nobody told the garrison.",
             "Hess countermanded the rotation twice in one week, then denied the second order.",
             "Ilse would not take coin for the tobacco, which meant she wanted something later.",
             "The recruit from the coast asked why the fallback was never written down."][i],
            None,
        ),
        "world" => (
            "World state",
            ["The crown's courier has not come in eleven days.",
             "Tolls on the north road doubled after the thaw.",
             "The garrison at Ardh is three months unpaid.",
             "Winter came early enough that the road froze before it flooded."][i],
            None,
        ),
        _ => (
            "Scope",
            ["The light goes amber and the wind drops.",
             "Rain starts, fine and cold, from the west.",
             "A cart on the mill road, moving faster than a cart should.",
             "Nothing moves for a long while."][i],
            None,
        ),
    }
}

/// The live retrieval probe: prefill a hypothetical message and report what the
/// gather would select for it, scored. This is the instrument that answers the
/// calibration questions the design leaves open — you type, and you see what the
/// character would actually reach for.
pub fn project(npc: &str, text: &str) -> Value {
    let _ = npc;
    let q = text.trim();
    let qn = q.chars().count() as f64;
    // Deterministic pseudo-scoring so the probe is stable while typing rather
    // than jittering under the operator.
    let seed: u64 = q.bytes().map(|b| b as u64).sum::<u64>().max(1);
    let jitter = |k: u64| ((seed.wrapping_mul(k) % 97) as f64) / 97.0;

    let mut tiles: Vec<Value> = Vec::new();
    let mut push =
        |kind: &str, layer: &str, label: &str, base: f64, k: u64, tokens: u64, body: &str| {
            let score = (base + jitter(k) * 380.0) * (1.0 + (qn / 220.0)).min(1.6);
            tiles.push(json!({
                "kind": kind, "layer": layer, "label": label,
                "score": score.round(), "tokens": tokens,
                "selected": score > 520.0, "text": body,
            }));
        };

    push(
        "turn",
        "perception",
        "A horn, twice, from the eastern slope.",
        940.0,
        3,
        96,
        "The signal for ground given, not for contact. Second in an hour.",
    );
    push(
        "belief",
        "beliefs",
        "Hess is a man of his word",
        880.0,
        5,
        128,
        "confidence 0.72 · disconfirmation 0.30 / 0.85 — under pressure",
    );
    push(
        "summary",
        "memory",
        "summary #412 (compresses turns 388–411)",
        720.0,
        7,
        180,
        "The week the rotation was countermanded twice; the fallback was never written down.",
    );
    push(
        "relationship",
        "relationships",
        "Commander Hess",
        610.0,
        11,
        84,
        "trust +0.60 · affect +0.20 · familiarity 0.90 — chain of command",
    );
    push("section", "system", "mood · tense", 540.0, 13, 104,
         "Clipped. Attention divided — part of you is elsewhere, and it shows in what you leave unfinished.");
    push(
        "turn",
        "action",
        "move_to → ridge_east",
        430.0,
        17,
        64,
        "intent: get to the ridge before the line folds",
    );
    push("section", "system", "response · battlefield_urgency", 380.0, 19, 128,
         "Answer in short, load-bearing sentences. Lead with the thing that changes what they do next.");
    push(
        "turn",
        "world",
        "The crown's courier has not come in eleven days.",
        210.0,
        23,
        72,
        "World fact, shared across every character in Ardh.",
    );

    tiles.sort_by(|a, b| {
        b["score"]
            .as_f64()
            .partial_cmp(&a["score"].as_f64())
            .unwrap()
    });

    json!({
        "query_tokens": (qn / 4.0).ceil() as u64,
        "budget": { "total": 16000, "would_use": 15214 },
        "tiles": tiles,
    })
}
