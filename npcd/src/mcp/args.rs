//! Reading a Battle Cities tool's arguments.

use serde_json::Value;

/// A string argument, refused when missing or empty.
pub fn string<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    optional(args, key).ok_or_else(|| format!("`{key}` is missing: it is a non-empty string"))
}

/// A string argument when given and not empty.
pub fn optional<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// A boolean argument, `default` when not given.
pub fn flag(args: &Value, key: &str, default: bool) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// A list of strings, the empty ones left out; empty when not given.
pub fn strings(args: &Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn arguments_are_trimmed_and_an_empty_one_is_absent() {
        let a = json!({"npc": " 1z ", "blank": "  ", "start": false, "steps": ["a", " ", "b"]});
        assert_eq!(string(&a, "npc").unwrap(), "1z");
        assert!(string(&a, "blank").is_err());
        assert_eq!(optional(&a, "missing"), None);
        assert!(!flag(&a, "start", true));
        assert!(flag(&a, "other", true));
        assert_eq!(strings(&a, "steps"), ["a", "b"]);
        assert!(strings(&a, "none").is_empty());
    }
}
