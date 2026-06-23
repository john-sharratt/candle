//! Structured log framing for `GET /ws/logs`.
//!
//! The log bus broadcasts the `tracing` fmt-layer's formatted lines
//! (`2026-06-21T11:55:00.123456Z  INFO zend::substrate: message`). The socket
//! frames each one as JSON `{ ts, level, target, msg }` so the UI renders it
//! without any client-side parsing (docs/zend_ui_redesign.md decision 6). `ts`
//! is reduced to `HH:MM:SS` to match the UI's log row.

use serde::Serialize;

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct LogLine {
    pub ts: String,
    pub level: String,
    pub target: String,
    pub msg: String,
}

const LEVELS: [&str; 5] = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"];

/// Parse one formatted log line into a [`LogLine`]. Tolerant: an unparseable
/// line yields the whole text as `msg` with an empty `target` and `INFO` level.
pub fn parse(line: &str) -> LogLine {
    let ts = hms(line.split_whitespace().next().unwrap_or(""));

    let mut level = "INFO";
    let mut after = line.trim();
    for lv in LEVELS {
        if let Some(idx) = find_token(line, lv) {
            level = lv;
            after = line[idx + lv.len()..].trim_start();
            break;
        }
    }

    let (target, msg) = match after.split_once(": ") {
        Some((t, m)) => (t.trim().to_string(), m.trim().to_string()),
        None => (String::new(), after.trim().to_string()),
    };

    LogLine {
        ts,
        level: level.to_string(),
        target,
        msg,
    }
}

/// Reduce an RFC3339 timestamp to `HH:MM:SS`; pass anything else through.
fn hms(ts: &str) -> String {
    let b = ts.as_bytes();
    if ts.len() >= 19 && b.get(10) == Some(&b'T') {
        ts[11..19].to_string()
    } else {
        ts.to_string()
    }
}

/// Find `needle` in `line` as a whitespace-bounded token (so a level name
/// inside the message text isn't mistaken for the level field).
fn find_token(line: &str, needle: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut start = 0;
    while let Some(rel) = line[start..].find(needle) {
        let i = start + rel;
        let before_ok = i == 0 || bytes[i - 1] == b' ';
        let end = i + needle.len();
        let after_ok = end >= line.len() || bytes[end] == b' ';
        if before_ok && after_ok {
            return Some(i);
        }
        start = end;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_standard_info_line() {
        let l =
            parse("2026-06-21T11:55:00.123456Z  INFO zend::substrate: recovered 5 conversations");
        assert_eq!(
            l,
            LogLine {
                ts: "11:55:00".to_string(),
                level: "INFO".to_string(),
                target: "zend::substrate".to_string(),
                msg: "recovered 5 conversations".to_string(),
            }
        );
    }

    #[test]
    fn parses_warn_with_double_colon_target() {
        let l = parse("2026-06-21T02:14:09.880Z  WARN zend::decode: ctx pressure 96%");
        assert_eq!(l.level, "WARN");
        assert_eq!(l.target, "zend::decode");
        assert_eq!(l.msg, "ctx pressure 96%");
        assert_eq!(l.ts, "02:14:09");
    }

    #[test]
    fn message_containing_a_level_word_is_not_mistaken() {
        let l = parse("2026-06-21T02:14:09.880Z  DEBUG zend::http: returned INFO payload");
        assert_eq!(l.level, "DEBUG");
        assert_eq!(l.target, "zend::http");
        assert_eq!(l.msg, "returned INFO payload");
    }

    #[test]
    fn unparseable_line_falls_back_to_msg() {
        let l = parse("a bare line with no structure");
        assert_eq!(l.level, "INFO");
        assert_eq!(l.target, "");
        assert_eq!(l.msg, "a bare line with no structure");
    }

    #[test]
    fn serializes_to_expected_json() {
        let l = parse("2026-06-21T11:55:00Z  INFO zend::http: listening on 127.0.0.1:8731");
        let json = serde_json::to_string(&l).unwrap();
        assert_eq!(
            json,
            r#"{"ts":"11:55:00","level":"INFO","target":"zend::http","msg":"listening on 127.0.0.1:8731"}"#
        );
    }
}
