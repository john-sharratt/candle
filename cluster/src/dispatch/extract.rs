//! Reading a request's key (§5.2): the first source that yields one wins. A
//! header or path key is read without touching the body; a body or derived key
//! needs the body buffered — at most [`BODY_LIMIT`] bytes — and the caller is
//! told so rather than having it read here.

use http::HeaderMap;
use serde_json::Value;

use super::KeySource;

/// The most of a request body read to find its key.
pub const BODY_LIMIT: usize = 8 << 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Extracted {
    Key(String),
    /// A later source reads the body, which has not been buffered.
    NeedsBody,
    /// No source yields a key: a request that starts something new.
    None,
}

/// The key `sources` yield for a request with these path `params` and
/// `headers`, and `body` once it has been buffered.
pub fn extract(
    sources: &[KeySource],
    params: &[(String, String)],
    headers: &HeaderMap,
    body: Option<&[u8]>,
) -> Extracted {
    let mut json: Option<Option<Value>> = None;
    for source in sources {
        let found = match source {
            KeySource::Path(name) => params
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone()),
            KeySource::Header(name) => headers
                .get(name.as_str())
                .and_then(|v| v.to_str().ok())
                .filter(|v| !v.is_empty())
                .map(str::to_owned),
            KeySource::Body(_) | KeySource::Derived(_) => {
                let Some(bytes) = body else {
                    return Extracted::NeedsBody;
                };
                let value = json.get_or_insert_with(|| serde_json::from_slice(bytes).ok());
                match (source, value) {
                    (KeySource::Body(field), Some(v)) => v
                        .get(field)
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned),
                    (KeySource::Derived(d), Some(v)) => d.key(v),
                    _ => None,
                }
            }
        };
        if let Some(key) = found {
            return Extracted::Key(key);
        }
    }
    Extracted::None
}

#[cfg(test)]
mod tests {
    use http::HeaderValue;

    use super::*;
    use crate::dispatch::Derived;

    fn chat() -> Vec<KeySource> {
        vec![
            KeySource::Header("x-tokera-conversation".into()),
            KeySource::Body("conv_id".into()),
            KeySource::Derived(Derived::Passthrough),
        ]
    }

    #[test]
    fn the_first_source_with_a_key_wins() {
        let mut h = HeaderMap::new();
        h.insert("x-tokera-conversation", HeaderValue::from_static("c-1"));
        // The header wins without the body ever being needed.
        assert_eq!(
            extract(&chat(), &[], &h, None),
            Extracted::Key("c-1".into())
        );
        let none = HeaderMap::new();
        assert_eq!(extract(&chat(), &[], &none, None), Extracted::NeedsBody);
        let body = br#"{"conv_id":"c-2","messages":[]}"#;
        assert_eq!(
            extract(&chat(), &[], &none, Some(body)),
            Extracted::Key("c-2".into())
        );
        let passthrough = br#"{"model":"passthrough","messages":[{"role":"user","content":"t"}]}"#;
        let Extracted::Key(k) = extract(&chat(), &[], &none, Some(passthrough)) else {
            panic!()
        };
        assert_eq!(k.len(), 32);
        // A new conversation names none.
        let fresh = br#"{"messages":[{"role":"user","content":"hi"}]}"#;
        assert_eq!(extract(&chat(), &[], &none, Some(fresh)), Extracted::None);
        assert_eq!(
            extract(&chat(), &[], &none, Some(b"not json")),
            Extracted::None
        );
    }

    #[test]
    fn a_path_key_reads_its_parameter() {
        let params = vec![("id".to_string(), "c-9".to_string())];
        assert_eq!(
            extract(
                &[KeySource::Path("id".into())],
                &params,
                &HeaderMap::new(),
                None
            ),
            Extracted::Key("c-9".into())
        );
    }
}
