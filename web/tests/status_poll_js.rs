//! `zend/web/status-poll.js`, run as written: coming back to the console
//! must not treat a hidden tab, an aborted fetch, or a silent gateway as
//! zend reloading.

use boa_engine::builtins::promise::PromiseState;
use boa_engine::module::Module;
use boa_engine::{Context, JsString, JsValue, Script, Source};
use serde_json::{json, Value};

const POLL: &str = include_str!("../../zend/web/status-poll.js");
const PAGE: &str = include_str!("../../zend/web/index.html");
const LIVE: &str = include_str!("../../zend/web/zend-api.live.js");

fn call(args: Value) -> Value {
    call_export("decideStatus", &[args])
}

fn call_export(name: &str, args: &[Value]) -> Value {
    let mut ctx = Context::default();
    let module = Module::parse(Source::from_bytes(POLL), None, &mut ctx).unwrap();
    let evaluated = module.load_link_evaluate(&mut ctx);
    ctx.run_jobs().unwrap();
    assert!(
        matches!(evaluated.state(), PromiseState::Fulfilled(_)),
        "the module did not evaluate"
    );
    let export = module
        .namespace(&mut ctx)
        .get(JsString::from(name), &mut ctx)
        .unwrap();
    let f = export.as_callable().expect(name);
    let args: Vec<JsValue> = args
        .iter()
        .map(|a| JsValue::from_json(a, &mut ctx).unwrap())
        .collect();
    let out = f.call(&JsValue::undefined(), &args, &mut ctx).unwrap();
    out.to_json(&mut ctx).unwrap().unwrap_or(Value::Null)
}

/// `(unreachable, delay ms)`. A JS number has no integer kind, so the delay
/// is read as the number it is.
fn boot_retry(failures: u32) -> (bool, f64) {
    let out = call_export("bootRetry", &[json!(failures)]);
    (
        out["unreachable"].as_bool().expect("unreachable"),
        out["delay"].as_f64().expect("delay"),
    )
}

fn reason(name: Value, message: Value) -> Value {
    call_export("unreachableReason", &[name, message])
}

fn poll(extra: Value) -> Value {
    let mut body = json!({
        "hidden": false,
        "aborted": false,
        "misses": 0,
        "zendState": "ready",
        "startedAt": 1_700_000_000_000u64,
        "lastStartedAt": 1_700_000_000_000u64,
    });
    if let (Value::Object(dst), Value::Object(src)) = (&mut body, extra) {
        for (k, v) in src {
            dst.insert(k, v);
        }
    }
    call(body)
}

/// A hidden page and an aborted fetch leave the chat alone, and do not
/// count toward the reconnect streak.
#[test]
fn a_hidden_or_aborted_poll_is_ignored() {
    assert_eq!(
        poll(json!({ "hidden": true, "zendState": "" })),
        json!("ignore")
    );
    assert_eq!(
        poll(json!({ "aborted": true, "zendState": "" })),
        json!("ignore")
    );
}

/// One unanswered poll keeps the chat. The next one, still visible, is
/// when the console may say it is reconnecting.
#[test]
fn one_miss_keeps_the_chat_and_the_next_says_reconnecting() {
    assert_eq!(
        poll(json!({ "zendState": "", "startedAt": 0 })),
        json!("miss")
    );
    assert_eq!(
        poll(json!({ "zendState": "", "startedAt": 0, "misses": 1 })),
        json!("reconnect")
    );
    // The loading line a gateway invents when no member answered.
    assert_eq!(
        poll(json!({ "zendState": "loading", "startedAt": 0 })),
        json!("miss")
    );
}

/// A daemon that answered and is loading is a real reload. A new process
/// start is a restart. The same process, still ready, stays.
#[test]
fn a_real_answer_reloads_restarts_or_stays() {
    assert_eq!(
        poll(json!({ "zendState": "loading", "startedAt": 1_700_000_000_000u64 })),
        json!("reload")
    );
    assert_eq!(
        poll(json!({ "startedAt": 1_700_000_000_100u64 })),
        json!("restart")
    );
    assert_eq!(poll(json!({})), json!("stay"));
    // First sight of a ready daemon, and the mock's start of 0, are not
    // restarts.
    assert_eq!(poll(json!({ "lastStartedAt": null })), json!("stay"));
    assert_eq!(
        poll(json!({ "startedAt": 0, "lastStartedAt": 0 })),
        json!("stay")
    );
}

/// A gateway that is down never answers the boot poll. The first two misses
/// keep the splash; the third says the server cannot be reached. The retry
/// doubles from a second and stops at ten, however long the tab sits there.
#[test]
fn an_unanswered_boot_poll_says_so_and_backs_off() {
    assert_eq!(boot_retry(1), (false, 1000.0));
    assert_eq!(boot_retry(2), (false, 2000.0));
    assert_eq!(boot_retry(3), (true, 4000.0));
    assert_eq!(boot_retry(4), (true, 8000.0));
    assert_eq!(boot_retry(5), (true, 10000.0));
    assert_eq!(boot_retry(10_000), (true, 10000.0));
    // A count that never went up is still one attempt, not zero.
    assert_eq!(boot_retry(0), (false, 1000.0));
}

/// The card names what failed in words a reader can use: the browser's own
/// network-error text is replaced, a gateway status is named, and the
/// adapter's own messages (a timeout, no daemon behind the gateway) pass
/// through.
#[test]
fn the_unreachable_reason_is_readable() {
    assert_eq!(
        reason(json!("TypeError"), json!("Failed to fetch")),
        json!("The server could not be reached.")
    );
    assert_eq!(
        reason(json!("Error"), json!("GET /v1/status -> 502")),
        json!("The gateway answered 502.")
    );
    assert_eq!(
        reason(json!("Error"), json!("GET /v1/status -> 503")),
        json!("Zend is restarting — back in a moment."),
        "every member still loading"
    );
    assert_eq!(
        reason(
            json!("Error"),
            json!("/v1/status did not answer within 8s.")
        ),
        json!("/v1/status did not answer within 8s.")
    );
    assert_eq!(
        reason(Value::Null, Value::Null),
        json!("The server did not answer.")
    );
}

/// The console module still parses. A syntax slip there is a blank page.
#[test]
fn the_console_module_parses() {
    let tag = "<script type=\"module\">";
    let open = PAGE.find(tag).expect("an inline module") + tag.len();
    let close = PAGE[open..].find("</script>").expect("a closed module") + open;
    let mut ctx = Context::default();
    if let Err(e) = Module::parse(Source::from_bytes(&PAGE[open..close]), None, &mut ctx) {
        panic!("the console module does not parse: {e}");
    }
}

/// The live adapter still parses. It is a classic script loaded before the
/// console module, which reads `window.ZendAPI` from it on its first line.
#[test]
fn the_live_adapter_parses() {
    let mut ctx = Context::default();
    if let Err(e) = Script::parse(Source::from_bytes(LIVE), None, &mut ctx) {
        panic!("zend-api.live.js does not parse: {e}");
    }
}
