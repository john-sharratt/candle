//! Reading a route handler's answer back: its body as the route's type, or
//! what it refused, worded as it worded it.

use axum::body::to_bytes;
use axum::response::Response;
use serde::de::DeserializeOwned;

use crate::engine::bodies::ApiError;

/// The most of a handler's answer read back.
const BODY_LIMIT: usize = 4 << 20;

/// The body `res` carries, as `T`; a refusal in the route's own words.
pub async fn read<T: DeserializeOwned>(res: Response) -> Result<T, String> {
    let status = res.status();
    let bytes = to_bytes(res.into_body(), BODY_LIMIT)
        .await
        .map_err(|e| format!("the answer could not be read: {e}"))?;
    if !status.is_success() {
        return Err(refused(status.as_u16(), &bytes));
    }
    serde_json::from_slice(&bytes)
        .map_err(|e| format!("the route answered in a shape the tool does not read ({e})"))
}

/// The words a refusal `res` carries.
pub async fn refusal(res: Response) -> String {
    let status = res.status().as_u16();
    match to_bytes(res.into_body(), BODY_LIMIT).await {
        Ok(bytes) => refused(status, &bytes),
        Err(_) => format!("refused ({status})"),
    }
}

/// A refusal's words: the route's own `{error, detail}` when it gave them, its
/// status otherwise.
pub fn refused(status: u16, body: &[u8]) -> String {
    match serde_json::from_slice::<ApiError>(body) {
        Ok(e) if e.detail.is_empty() => format!("{} ({status})", e.error),
        Ok(e) => format!("{}: {} ({status})", e.error, e.detail),
        Err(_) => {
            let text = String::from_utf8_lossy(body);
            let text = text.trim();
            if text.is_empty() {
                format!("refused ({status})")
            } else {
                format!(
                    "refused ({status}): {}",
                    text.chars().take(300).collect::<String>()
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::Json;
    use serde_json::{json, Value};

    use super::*;

    #[test]
    fn a_refusal_is_worded_as_the_route_worded_it() {
        assert_eq!(
            refused(
                404,
                br#"{"error":"npc_not_found","detail":"no such character"}"#
            ),
            "npc_not_found: no such character (404)"
        );
        assert_eq!(
            refused(409, br#"{"error":"not_in_a_world"}"#),
            "not_in_a_world (409)"
        );
        assert_eq!(refused(502, b""), "refused (502)");
        assert_eq!(refused(500, b"boom"), "refused (500): boom");
    }

    #[tokio::test]
    async fn an_answer_is_read_as_its_type_and_a_refusal_as_its_words() {
        let ok = Json(json!({"cancelled": true})).into_response();
        assert_eq!(read::<Value>(ok).await.unwrap(), json!({"cancelled": true}));
        let no = (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "npc_not_found", "detail": "no such character"})),
        )
            .into_response();
        assert_eq!(
            read::<Value>(no).await.unwrap_err(),
            "npc_not_found: no such character (404)"
        );
    }
}
