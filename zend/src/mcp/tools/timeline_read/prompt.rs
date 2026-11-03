//! `view: prompt`: the prompt the conversation would be decoded with now,
//! piece by piece (`GET /v1/conversations/{id}/prompt-tokens`) — the tool
//! signatures in it first, then every piece with its tokens and the score
//! that brought it in. It is computed fresh against the last sealed turn, so
//! it scans the gallery, and only the member holding the conversation open
//! has it.

use host::substrate_views::TimelineDetail;
use serde_json::Value;

use super::head;
use crate::mcp::ops_tools::OpsCall;

/// One piece as its line: its kind, its name, its tokens and its score.
pub fn piece(p: &Value) -> String {
    let score = p.get("score").and_then(Value::as_f64);
    format!(
        "{} {} — {} tokens{}",
        p.get("kind").and_then(Value::as_str).unwrap_or("?"),
        p.get("name").and_then(Value::as_str).unwrap_or(""),
        p.get("token_count").and_then(Value::as_u64).unwrap_or(0),
        match (score, p.get("qualified").and_then(Value::as_bool)) {
            (Some(s), Some(true)) => format!(", scored {s:.0}, cleared the bar"),
            (Some(s), _) => format!(", scored {s:.0}"),
            _ => String::new(),
        }
    )
}

/// The dump `d` as an answer.
pub fn render(t: &TimelineDetail, d: &Value) -> String {
    let pieces: Vec<&Value> = d
        .get("pieces")
        .and_then(Value::as_array)
        .map(|p| p.iter().collect())
        .unwrap_or_default();
    let tools: Vec<&str> = pieces
        .iter()
        .filter_map(|p| p.get("name").and_then(Value::as_str))
        .filter_map(|n| n.strip_prefix("tools/"))
        .filter(|n| *n != "summary")
        .collect();
    let mut out = head(t);
    out.push_str(&format!(
        "\nthe prompt as it would be decoded now: {} tokens in {} pieces\nsignatures in it: {}",
        d.get("token_count").and_then(Value::as_u64).unwrap_or(0),
        pieces.len(),
        if tools.is_empty() {
            "none — the tool names alone".to_string()
        } else {
            tools.join(", ")
        }
    ));
    for p in pieces {
        out.push('\n');
        out.push_str(&piece(p));
    }
    out
}

pub async fn read(call: &OpsCall, t: &TimelineDetail) -> Result<String, String> {
    match call.session.prompt_dump(&t.conv_id).await {
        None => Err("the model is still loading; try again shortly".into()),
        Some(None) => Err(format!(
            "{} is not open on the member that answered, so it has no prompt to show — a \
             conversation's prompt is held where it is being decoded; read its `projections` \
             for what each turn was shown instead",
            t.conv_id
        )),
        Some(Some(dump)) => {
            let v = serde_json::to_value(&dump).map_err(|e| e.to_string())?;
            Ok(render(t, &v))
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::tests::timeline;
    use super::*;

    /// **The signatures in the prompt are named first**, then every piece
    /// with its tokens and its score.
    #[test]
    fn the_prompts_signatures_come_first_then_its_pieces() {
        let d = json!({"token_count": 5000, "pieces": [
            {"kind": "section", "name": "tools/summary", "token_count": 400},
            {"kind": "section", "name": "tools/job_status", "token_count": 300, "score": 912.0, "qualified": true},
            {"kind": "turn", "name": "", "token_count": 80, "score": 120.0, "qualified": false}
        ]});
        let text = render(&timeline(), &d);
        assert!(
            text.contains("5000 tokens in 3 pieces\nsignatures in it: job_status\n"),
            "{text}"
        );
        assert!(text.contains("\nsection tools/job_status — 300 tokens, scored 912, cleared the bar"), "{text}");
        assert!(text.ends_with("\nturn  — 80 tokens, scored 120"), "{text}");
    }
}
