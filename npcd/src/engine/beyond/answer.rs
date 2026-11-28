//! A tool's answer as the mind reads it: its JSON as the tool wrote it — or,
//! for an answer that is one text, the text — cut at [`ANSWER_CHARS`], and a
//! tool's error as a refusal in its own words.

use serde_json::Value;

/// The most of an answer the mind reads, in characters.
pub const ANSWER_CHARS: usize = 6_000;

/// What zend's tool answered, `result`, as the mind reads it.
pub fn answer_of(result: &Value) -> Result<String, String> {
    if let Some(code) = result.get("error").and_then(Value::as_str) {
        return Err(match result.get("detail").and_then(Value::as_str) {
            Some(detail) => format!("{code}: {detail}"),
            None => code.to_string(),
        });
    }
    let text = match result.as_object().map(|o| (o.len(), o.get("text"))) {
        Some((1, Some(Value::String(t)))) => t.clone(),
        _ => result.to_string(),
    };
    Ok(cut(&text))
}

/// `text`, cut at [`ANSWER_CHARS`] on a character, saying that it was.
pub fn cut(text: &str) -> String {
    match text.char_indices().nth(ANSWER_CHARS) {
        Some((at, _)) => format!(
            "{}… (cut: the rest is past what one answer holds)",
            &text[..at]
        ),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **An answer is the tool's JSON as it wrote it, one text as itself, an
    /// error in its own words.**
    #[test]
    fn an_answer_is_the_tools_own() {
        assert_eq!(
            answer_of(&json!({"expression": "2+2", "result": {"value": 4}})),
            Ok(r#"{"expression":"2+2","result":{"value":4}}"#.into())
        );
        assert_eq!(
            answer_of(&json!({"text": "Guides: deploy, test"})),
            Ok("Guides: deploy, test".into())
        );
        assert_eq!(
            answer_of(&json!({"error": "url_blocked", "detail": "private/localhost URLs blocked"})),
            Err("url_blocked: private/localhost URLs blocked".into())
        );
        assert_eq!(
            answer_of(&json!({"error": "not_permitted"})),
            Err("not_permitted".into())
        );
    }

    /// **A long answer is cut on a character, and says so.**
    #[test]
    fn a_long_answer_is_cut_on_a_character() {
        let long = "é".repeat(ANSWER_CHARS + 10);
        let said = cut(&long);
        assert!(said.starts_with(&"é".repeat(ANSWER_CHARS)));
        assert!(said.ends_with("… (cut: the rest is past what one answer holds)"));
        assert_eq!(cut("short"), "short");
    }
}
