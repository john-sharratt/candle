//! Each conversation ref's last use, kept where every member and both
//! aggregators read it (`docs/private_mode.md` §3.2): a record per
//! conversation, `mcp/ref/<conversation>`, holding nothing but `used_ms` — no
//! account, no flag, never the ref itself.
//!
//! The aggregators tell a use at most every [`REF_TOUCH_EVERY_MS`] and ask
//! only when a ref is older than they can vouch for; the tend ceases every
//! record idle past [`REF_IDLE_MS`].

use mcp_server::refs::{REF_IDLE_MS, REF_TOUCH_EVERY_MS};
use serde_json::{json, Value};

use super::records::Records;

const PREFIX: &str = "mcp/ref/";

fn key(conversation: &str) -> String {
    format!("{PREFIX}{conversation}")
}

fn used_of(v: &Value) -> Option<u64> {
    v.get("used_ms").and_then(Value::as_u64)
}

/// When `conversation`'s ref was last told used, if it ever was.
pub fn used(records: &dyn Records, conversation: &str) -> Option<u64> {
    used_of(&records.get(&key(conversation))?)
}

/// Tell `conversation`'s ref used at `used_ms`: written only when the record
/// is [`REF_TOUCH_EVERY_MS`] behind it, or absent. When it was last used, as
/// the record now says.
pub fn touch(records: &dyn Records, conversation: &str, used_ms: u64) -> anyhow::Result<u64> {
    let held = used(records, conversation);
    match held {
        Some(at) if used_ms.saturating_sub(at) < REF_TOUCH_EVERY_MS => Ok(at.max(used_ms)),
        _ => {
            records.put(&key(conversation), json!({"used_ms": used_ms}))?;
            Ok(used_ms)
        }
    }
}

/// Cease every record idle past [`REF_IDLE_MS`] at `now_ms`, and one that
/// does not read as a use; how many were ceased.
pub fn sweep(records: &dyn Records, now_ms: u64) -> usize {
    let mut ceased = 0;
    for (k, v) in records.under(PREFIX) {
        let stale = used_of(&v).is_none_or(|at| now_ms.saturating_sub(at) >= REF_IDLE_MS);
        if stale && records.cease(&k).is_ok() {
            ceased += 1;
        }
    }
    ceased
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;

    #[test]
    fn a_use_is_written_when_the_record_is_far_enough_behind() {
        let r = MemRecords::default();
        assert_eq!(used(&r, "c1"), None);
        assert_eq!(touch(&r, "c1", 1_000).unwrap(), 1_000);
        assert_eq!(r.get("mcp/ref/c1"), Some(json!({"used_ms": 1_000})));
        assert_eq!(
            touch(&r, "c1", 1_000 + REF_TOUCH_EVERY_MS - 1).unwrap(),
            1_000 + REF_TOUCH_EVERY_MS - 1,
            "the newer use is answered"
        );
        assert_eq!(used(&r, "c1"), Some(1_000), "and not written");
        touch(&r, "c1", 1_000 + REF_TOUCH_EVERY_MS).unwrap();
        assert_eq!(used(&r, "c1"), Some(1_000 + REF_TOUCH_EVERY_MS));
        assert_eq!(
            touch(&r, "c1", 5).unwrap(),
            1_000 + REF_TOUCH_EVERY_MS,
            "an older use moves nothing"
        );
        assert_eq!(used(&r, "c1"), Some(1_000 + REF_TOUCH_EVERY_MS));
    }

    #[test]
    fn a_sweep_ceases_the_idle_and_keeps_the_live() {
        let r = MemRecords::default();
        touch(&r, "old", 0).unwrap();
        touch(&r, "new", REF_IDLE_MS).unwrap();
        r.put("mcp/ref/odd", json!("x")).unwrap();
        assert_eq!(sweep(&r, REF_IDLE_MS), 2);
        assert_eq!(used(&r, "old"), None);
        assert_eq!(used(&r, "new"), Some(REF_IDLE_MS));
        assert_eq!(r.get("mcp/ref/odd"), None);
    }
}
