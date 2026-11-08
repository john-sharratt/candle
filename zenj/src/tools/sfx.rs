//! `zenj_sfx`: fire a sampler slot — an air horn, a countdown, a sound Zen
//! made before the night — now, or on the next beat or bar.

use serde_json::{json, Value};

use super::{Answer, Dj, Tool};
use crate::brain::run::Ask;
use crate::conductor::Quant;

pub struct Sfx;

impl Tool for Sfx {
    const NAME: &'static str = "zenj_sfx";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Fire one of rekordbox's sampler slots (1–16) at the party: `now`, or on the next `beat` or `bar` of the track on air, so it lands in time.",
            "inputSchema": {"type": "object", "properties": {
                "slot": {"type": "integer", "minimum": 1, "maximum": 16},
                "when": {"type": "string", "enum": ["now", "beat", "bar"]}
            }, "required": ["slot"]},
        })
    }
}

pub async fn call(dj: &Dj, args: &Value) -> Answer {
    let Some(slot) = args
        .get("slot")
        .and_then(Value::as_u64)
        .filter(|s| (1..=16).contains(s))
    else {
        return Answer::error("`slot` is 1 to 16");
    };
    let when = match args.get("when").and_then(Value::as_str).unwrap_or("bar") {
        "now" => Quant::Now,
        "beat" => Quant::Beat,
        "bar" => Quant::Bar,
        other => return Answer::error(format!("`when` is now, beat or bar, not `{other}`")),
    };
    let slot = slot as u8;
    if dj.asks.send(Ask::Sample { slot, when }).await.is_err() {
        return Answer::error("ZenJ's brain is not running");
    }
    Answer::ok(format!(
        "sample {slot} fires {}",
        match when {
            Quant::Now => "now",
            Quant::Beat => "on the next beat",
            Quant::Bar => "on the next bar",
        }
    ))
}
