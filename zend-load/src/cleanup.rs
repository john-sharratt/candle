//! A load run leaves its conversations in the cluster's substrate, which is
//! someone's working memory: every one is deleted once the run ends. A named
//! conversation goes by its id; a passthrough one is found by the run's mark
//! in its opening, which its label shows.

use serde_json::Value;

use crate::client::Target;

/// Delete `named` and every passthrough conversation whose label carries
/// `mark`, through the gateway at `base`. Returns how many went, and how many
/// refused.
pub async fn delete_run(
    target: &Target,
    base: &str,
    named: &[String],
    mark: &str,
) -> (usize, usize) {
    let mut ids: Vec<String> = named.to_vec();
    ids.extend(marked(target, base, mark).await);
    let (mut gone, mut kept) = (0, 0);
    for id in ids {
        let res = target
            .http
            .delete(format!("{base}/v1/conversations/{id}"))
            .header("host", &target.host)
            .send()
            .await;
        match res {
            Ok(r) if r.status().is_success() => gone += 1,
            _ => kept += 1,
        }
    }
    (gone, kept)
}

/// The passthrough conversations whose label carries `mark`.
async fn marked(target: &Target, base: &str, mark: &str) -> Vec<String> {
    let listing = target
        .http
        .get(format!("{base}/v1/conversations"))
        .header("host", &target.host)
        .send()
        .await;
    let Ok(listing) = listing else {
        return Vec::new();
    };
    let Ok(v) = listing.json::<Value>().await else {
        return Vec::new();
    };
    matching(&v, mark)
}

/// The ids in a `/v1/conversations` listing of passthrough conversations
/// whose label carries `mark`.
pub fn matching(listing: &Value, mark: &str) -> Vec<String> {
    listing["conversations"]
        .as_array()
        .map(|all| {
            all.iter()
                .filter(|c| {
                    let id = c["id"].as_str().unwrap_or("");
                    let label = c["label"].as_str().unwrap_or("");
                    id.starts_with("passthrough-") && label.contains(mark)
                })
                .filter_map(|c| c["id"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Only passthrough conversations carrying the run's mark are taken;
    /// someone's own conversation, or another run's, is left.
    #[test]
    fn only_the_runs_own_are_matched() {
        let listing = json!({"conversations": [
            {"id": "passthrough-1", "label": "[zend-load zl-9 client 0] You are a storyteller."},
            {"id": "passthrough-2", "label": "[zend-load zl-8 client 0] You are a storyteller."},
            {"id": "passthrough-3", "label": "my own notes on zl-9"},
            {"id": "zl-9-c1", "label": "[zend-load zl-9 client 1]"},
        ]});
        assert_eq!(
            matching(&listing, "[zend-load zl-9 "),
            vec!["passthrough-1"]
        );
    }
}
