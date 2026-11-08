//! `zenj_energy`: more or less energy for a while — the energy curve's
//! target lifted or lowered from now (§5.1).

use serde_json::{json, Value};

use super::{Answer, Dj, Tool};
use crate::brain::run::Ask;

pub struct Energy;

impl Tool for Energy {
    const NAME: &'static str = "zenj_energy";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "More or less energy at the party: lift (`by` above 0) or lower (below 0) the night's target energy from now for `minutes`. 0.15 is a clear step; the target stays between 0 and 1.",
            "inputSchema": {"type": "object", "properties": {
                "by": {"type": "number", "minimum": -0.5, "maximum": 0.5},
                "minutes": {"type": "integer", "minimum": 5, "maximum": 120}
            }, "required": ["by"]},
        })
    }
}

pub async fn call(dj: &Dj, args: &Value) -> Answer {
    let Some(by) = args
        .get("by")
        .and_then(Value::as_f64)
        .filter(|b| (-0.5..=0.5).contains(b))
    else {
        return Answer::error("`by` is between -0.5 and 0.5");
    };
    let minutes = args
        .get("minutes")
        .and_then(Value::as_i64)
        .unwrap_or(30)
        .clamp(5, 120);
    let by = by as f32;
    dj.ask(|reply| Ask::Energy { by, minutes, reply }).await
}
