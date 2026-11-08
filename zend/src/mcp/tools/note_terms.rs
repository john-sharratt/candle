//! A memory note's terms as a `journal_note` call states them
//! (`docs/zen_memory_and_guides.md` Phase 2): `by`, `source`, `expires`,
//! `replaces`. Absent, a note states none.

use journals::line::LineId;
use journals::terms::Terms;
use journals::time::Stamp;
use serde_json::Value;

/// A day, in seconds: a date given for `expires` runs to its end.
const DAY: i64 = 86_400;

/// The terms `args` state.
pub fn terms_of(args: &Value) -> Result<Terms, String> {
    let text = |k: &str| -> Result<Option<&str>, String> {
        match args.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) if s.trim().is_empty() => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.trim())),
            Some(_) => Err(format!("`{k}` is text")),
        }
    };
    let concluded = match text("by")? {
        None | Some("speaker") => false,
        Some("assistant") => true,
        Some(other) => {
            return Err(format!(
                "`by` is `speaker` (the person said it) or `assistant` (you concluded it), not `{other}`"
            ))
        }
    };
    let expires = text("expires")?.map(expiry).transpose()?;
    let replaces = text("replaces")?
        .map(|s| LineId::parse(s.trim_start_matches('#')))
        .transpose()?;
    Ok(Terms {
        concluded,
        source: text("source")?.map(str::to_string),
        expires,
        replaces,
    })
}

/// When a note given `expires: s` stops being shown: the moment named, or
/// the end of the day a bare date names.
fn expiry(s: &str) -> Result<Stamp, String> {
    if let Ok(at) = Stamp::parse(s) {
        return Ok(at);
    }
    Stamp::parse(&format!("{s}T00:00:00Z"))
        .map(|day| day.plus(DAY))
        .map_err(|_| format!("`expires` is a date (`2026-11-30`) or a moment (`2026-11-30T09:00:00Z`), not `{s}`"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **Each term is read as the call states it**, and a call with none
    /// states none.
    #[test]
    fn each_term_is_read_as_stated() {
        assert_eq!(
            terms_of(&json!({"journal": "here", "text": "x"})).unwrap(),
            Terms::default()
        );
        let t = terms_of(&json!({
            "by": "assistant",
            "source": " zend/src/branch_ingest/keys.rs:12 ",
            "expires": "2026-11-30",
            "replaces": "#2026-10-31-0034-7c25",
        }))
        .unwrap();
        assert_eq!(
            t,
            Terms {
                concluded: true,
                source: Some("zend/src/branch_ingest/keys.rs:12".into()),
                expires: Some(Stamp::parse("2026-12-01T00:00:00Z").unwrap()),
                replaces: Some(LineId::parse("2026-10-31-0034-7c25").unwrap()),
            }
        );
        assert_eq!(
            terms_of(&json!({"expires": "2026-11-30T09:00:00Z", "by": "speaker"}))
                .unwrap()
                .expires,
            Some(Stamp::parse("2026-11-30T09:00:00Z").unwrap())
        );
    }

    /// **A term stated wrongly is refused, saying what it takes.**
    #[test]
    fn a_wrong_term_is_refused_saying_what_it_takes() {
        assert!(terms_of(&json!({"by": "claude"}))
            .unwrap_err()
            .contains("`speaker`"));
        assert!(terms_of(&json!({"expires": "soon"}))
            .unwrap_err()
            .contains("a date"));
        assert!(terms_of(&json!({"replaces": "7e3a"}))
            .unwrap_err()
            .contains("not a line id"));
        assert_eq!(
            terms_of(&json!({"source": 3})).unwrap_err(),
            "`source` is text"
        );
    }
}
