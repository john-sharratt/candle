//! `view: probe`: how a text ranks the tools on a fresh probe of the base
//! conversation (`POST /v1/substrate/zend/project`) — each tool it scores,
//! strongest first, against the 800 a signature needs, and the other pieces it
//! would pull in. It answers whether a task's prompt would put the tools it
//! names in front of the model, before the task is started. It ranks against
//! the base conversation, whatever conversation it is asked from, and runs a
//! prefill, so it is no cheap read.

use std::sync::Arc;

use serde_json::Value;

use super::projections::bar_at;
use crate::mcp::ops_tools::OpsCall;

/// How many tools, and how many other pieces, an answer names.
const TOOLS: usize = 12;
const OTHERS: usize = 5;

/// The projection `v` of `text` as an answer.
pub fn render(text: &str, v: &Value) -> String {
    let tiles: Vec<&Value> = v
        .get("tiles")
        .and_then(Value::as_array)
        .map(|t| t.iter().collect())
        .unwrap_or_default();
    let is_tool = |t: &&Value| t.get("group").and_then(Value::as_str) == Some("tools");
    let line = |t: &Value| {
        let score = t.get("score").and_then(Value::as_f64).unwrap_or(0.0);
        format!(
            "{} {score:.0}{}",
            t.get("label").and_then(Value::as_str).unwrap_or("?"),
            if t.get("selected").and_then(Value::as_bool) == Some(true) {
                " — shown"
            } else {
                ""
            }
        )
    };
    let tools: Vec<String> = tiles.iter().filter(is_tool).take(TOOLS).map(|t| line(t)).collect();
    let others: Vec<String> = tiles
        .iter()
        .filter(|t| !is_tool(t))
        .take(OTHERS)
        .map(|t| {
            format!(
                "{} {} {:.0}",
                t.get("kind").and_then(Value::as_str).unwrap_or("?"),
                t.get("label").and_then(Value::as_str).unwrap_or(""),
                t.get("score").and_then(Value::as_f64).unwrap_or(0.0)
            )
        })
        .collect();
    let mut out = format!(
        "probe of {} tokens: {:?}\ntools it scores, strongest first (a signature needs {:.0}):",
        v.get("query_tokens").and_then(Value::as_u64).unwrap_or(0),
        text.chars().take(200).collect::<String>(),
        bar_at(0)
    );
    if tools.is_empty() {
        out.push_str("\n  none — it scores no tool above zero");
    }
    for l in tools {
        out.push_str(&format!("\n  {l}"));
    }
    if !others.is_empty() {
        out.push_str("\nother pieces it would pull in:");
        for l in others {
            out.push_str(&format!("\n  {l}"));
        }
    }
    out
}

pub async fn read(call: &OpsCall, text: &str) -> Result<String, String> {
    let session = Arc::clone(&call.session);
    let query = text.to_string();
    let view = tokio::task::spawn_blocking(move || session.substrate_project(&query))
        .await
        .map_err(|e| format!("the probe did not finish: {e}"))?
        .ok_or("the model is still loading; try again shortly")?;
    let v = serde_json::to_value(&view).map_err(|e| e.to_string())?;
    Ok(render(text, &v))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **The tools come first, strongest first, each said shown or not**,
    /// then the other pieces the text would pull in.
    #[test]
    fn a_probe_names_the_tools_it_scores_then_the_rest() {
        let v = json!({"query_tokens": 31, "tiles": [
            {"kind": "section", "group": "tools", "label": "run_command", "score": 1000.0, "selected": true},
            {"kind": "section", "group": "tools", "label": "job_status", "score": 612.0, "selected": false},
            {"kind": "normal", "group": "dialogue", "label": "Codebase Tour", "score": 140.0, "selected": false}
        ]});
        let text = render("call job_status on build:x", &v);
        assert_eq!(
            text,
            "probe of 31 tokens: \"call job_status on build:x\"\ntools it scores, strongest first \
             (a signature needs 800):\n  run_command 1000 — shown\n  job_status 612\nother pieces \
             it would pull in:\n  normal Codebase Tour 140"
        );
        assert!(render("x", &json!({"tiles": []})).contains("none — it scores no tool"));
    }
}
