//! `zend/web/status-poll.js`, run as written: coming back to the console
//! must not treat a hidden tab, an aborted fetch, or a silent gateway as
//! zend reloading.

use boa_engine::builtins::promise::PromiseState;
use boa_engine::module::Module;
use boa_engine::{Context, JsString, JsValue, Source};
use serde_json::{json, Value};

const POLL: &str = include_str!("../../zend/web/status-poll.js");
const PAGE: &str = include_str!("../../zend/web/index.html");

fn call(args: Value) -> Value {
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
        .get(JsString::from("decideStatus"), &mut ctx)
        .unwrap();
    let f = export.as_callable().expect("decideStatus");
    let arg = JsValue::from_json(&args, &mut ctx).unwrap();
    let out = f.call(&JsValue::undefined(), &[arg], &mut ctx).unwrap();
    out.to_json(&mut ctx).unwrap().unwrap_or(Value::Null)
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
