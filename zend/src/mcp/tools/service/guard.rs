//! What a worker's deploy may not take down (`docs/build_deploy_tools.md` §4).
//! keep refuses anything but a build of origin/main and restores the previous
//! binary when one does not serve, so a bad build cannot keep a member down;
//! what is left here is what keep cannot see:
//!
//! - **Where the worker runs.** A worker runs inside `motherd`, and a trial
//!   stops the service it replaces: a worker trying `motherd` on its own
//!   machine would end itself mid-deploy.
//! - **What it takes down.** A trial of `motherd` stops every app on the
//!   machine, so one leaving an app with no ready member elsewhere is the
//!   user's decision, not a worker's. A status that cannot be read refuses too.

use axum::http::Method;
use host::cluster::up::send_up;
use serde_json::Value;

use crate::mcp::ops::{self, json_of};
use crate::mcp::ops_tools::OpsCall;
use crate::worker_hop;

use super::KEEP_READ;

/// The ready tenants of the member on `host` in `status` — `/v1/status` as the
/// gateway gathers it — that no member on another machine has ready.
pub fn only_here(status: &Value, host: &str) -> Vec<String> {
    let rows = status.get("members").and_then(Value::as_array);
    let ready = |m: &Value| -> Vec<String> {
        m.pointer("/body/tenants")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .filter(|(_, t)| t.get("state").and_then(Value::as_str) == Some("ready"))
            .map(|(name, _)| name.clone())
            .collect()
    };
    let on = |m: &&Value| {
        m.get("host")
            .and_then(Value::as_str)
            .is_some_and(|h| h.eq_ignore_ascii_case(host))
    };
    let elsewhere: Vec<String> = rows
        .into_iter()
        .flatten()
        .filter(|m| !on(m))
        .flat_map(ready)
        .collect();
    let mut only: Vec<String> = rows
        .into_iter()
        .flatten()
        .filter(on)
        .flat_map(ready)
        .filter(|app| !elsewhere.contains(app))
        .collect();
    only.sort();
    only.dedup();
    only
}

/// Why a worker's trial of `service` on `host` may not run from member `own`,
/// with `/v1/status` as `status`.
pub fn in_the_way(status: &Value, own: u64, host: &str, service: &str) -> Option<String> {
    if service != "motherd" {
        return None;
    }
    let own = format!("{own:016x}");
    let here = status
        .get("members")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|m| m.get("node").and_then(Value::as_str) == Some(own.as_str()))
        .and_then(|m| m.get("host"))
        .and_then(Value::as_str);
    let Some(here) = here else {
        return Some(
            "this member is not in the cluster's status, so what a trial would stop is unknown"
                .to_string(),
        );
    };
    if here.eq_ignore_ascii_case(host.trim()) {
        return Some(format!(
            "this task runs inside motherd on {here}, and a trial stops it: try it on another \
             machine, from a task running elsewhere"
        ));
    }
    let only = only_here(status, host.trim());
    (!only.is_empty()).then(|| {
        format!(
            "a trial on {host} takes {} out of service, with no ready member elsewhere: that is \
             the user's decision, so ask",
            only.join(", ")
        )
    })
}

/// Refuse a worker's deploy of `service` to `host` that [`in_the_way`] names.
pub async fn check(call: &OpsCall, host: &str, service: &str) -> Result<(), String> {
    if service != "motherd" {
        return Ok(());
    }
    let own = worker_hop::own_node().ok_or("this member is in no cluster")?;
    let member = ops::member(&call.session)?;
    let status = send_up(
        &member,
        Method::GET,
        "/v1/status",
        &[],
        Vec::new(),
        Some(KEEP_READ),
    )
    .await
    .map_err(|e| format!("the cluster's status could not be read: {e:#}"))?;
    let status = json_of(&status).map_err(|e| format!("/v1/status: {e}"))?;
    in_the_way(&status, own, host, service).map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const HERE: u64 = 0xa;

    fn member(host: &str, node: u64, tenants: Value) -> Value {
        json!({"host": host, "node": format!("{node:016x}"), "status": 200,
               "body": {"tenants": tenants}})
    }

    fn cluster() -> Value {
        json!({"members": [
            member("blackworld", HERE, json!({"zend": {"state": "ready"}})),
            member("whiteworld", 0xb, json!({
                "zend": {"state": "ready"}, "npcd": {"state": "ready"}
            })),
            member("alienworld", 0xc, json!({"zend": {"state": "loading"}})),
        ]})
    }

    /// **A task never stops the motherd it runs in**, in any spelling of the
    /// machine's name.
    #[test]
    fn a_trial_of_its_own_machine_is_refused() {
        let why = in_the_way(&cluster(), HERE, " BlackWorld ", "motherd").unwrap();
        assert!(
            why.starts_with("this task runs inside motherd on blackworld"),
            "{why}"
        );
    }

    /// **An app with no ready member elsewhere stops the trial**, and is
    /// named; one that is only loading elsewhere does not count as ready.
    #[test]
    fn a_trial_that_takes_down_an_only_member_is_refused() {
        let why = in_the_way(&cluster(), HERE, "whiteworld", "motherd").unwrap();
        assert_eq!(
            why,
            "a trial on whiteworld takes npcd out of service, with no ready member elsewhere: \
             that is the user's decision, so ask"
        );
        assert_eq!(only_here(&cluster(), "alienworld"), Vec::<String>::new());
        let status = json!({"members": [
            member("blackworld", HERE, json!({"zend": {"state": "ready"}})),
            member("whiteworld", 0xb, json!({"zend": {"state": "ready"}})),
        ]});
        assert_eq!(in_the_way(&status, HERE, "whiteworld", "motherd"), None);
    }

    /// A member missing from the status refuses rather than guessing; the
    /// gateway and the connector hold no worker and no app.
    #[test]
    fn only_motherd_is_in_the_way_and_an_unknown_refuses() {
        assert!(in_the_way(&cluster(), 0xfff, "alienworld", "motherd")
            .unwrap()
            .contains("unknown"));
        assert_eq!(in_the_way(&cluster(), HERE, "blackworld", "web"), None);
        assert_eq!(in_the_way(&cluster(), HERE, "blackworld", "mcp"), None);
    }
}
