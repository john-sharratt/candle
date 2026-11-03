//! `view: selection`: the dialogue turns and summaries the latest projection
//! of this timeline kept, and why each (`GET
//! /v1/substrate/zend/timeline/{tl}/selection`) — held in the memory of the
//! member that projected it, so it is the latest there, and gone after its
//! restart.

use host::substrate_views::TimelineDetail;
use serde_json::Value;

use super::head;
use crate::mcp::ops_tools::OpsCall;

/// The selection `s` as an answer.
pub fn render(t: &TimelineDetail, s: &Value) -> String {
    let selected: Vec<String> = s
        .get("selected")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|n| {
            let origin = match n.get("origin") {
                Some(Value::String(o)) => o.clone(),
                Some(other) => other.to_string(),
                None => "?".into(),
            };
            format!(
                "node {} — {origin}{}",
                n.get("node_id").and_then(Value::as_u64).unwrap_or(0),
                n.get("effective_score")
                    .and_then(Value::as_f64)
                    .map_or(String::new(), |s| format!(", scored {s:.0}"))
            )
        })
        .collect();
    let mut out = head(t);
    out.push_str(&format!(
        "\nlatest selection: {} nodes, {} tokens of a {} budget, {} pending",
        selected.len(),
        s.get("selected_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        s.get("budget").and_then(Value::as_u64).unwrap_or(0),
        s.get("pending_count").and_then(Value::as_u64).unwrap_or(0)
    ));
    for line in selected {
        out.push('\n');
        out.push_str(&line);
    }
    out
}

pub fn read(call: &OpsCall, t: &TimelineDetail) -> Result<String, String> {
    let raw: u64 = t.timeline.parse().map_err(|_| {
        format!(
            "timeline {} has no number to look its selection up by",
            t.timeline
        )
    })?;
    let s = call.session.substrate_selection(raw).ok_or_else(|| {
        format!(
            "the member that answered holds no selection for timeline {raw}: none has been \
             projected there since it started, or its layer selects by rule"
        )
    })?;
    let v = serde_json::to_value(&s).map_err(|e| e.to_string())?;
    Ok(render(t, &v))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::tests::timeline;
    use super::*;

    #[test]
    fn a_selection_reads_as_its_nodes_and_its_budget() {
        let s = json!({"selected": [{"node_id": 4, "origin": "recency", "effective_score": null},
                                    {"node_id": 9, "origin": "scored", "effective_score": 812.4}],
                       "pending_count": 1, "selected_tokens": 3000, "budget": 8000});
        let text = render(&timeline(), &s);
        assert!(
            text.ends_with(
                "\nlatest selection: 2 nodes, 3000 tokens of a 8000 budget, 1 pending\nnode 4 — \
                 recency\nnode 9 — scored, scored 812"
            ),
            "{text}"
        );
    }
}
