//! `content/common/lib/aggregate.js`, run as written: the consoles read
//! `/v1/status` and the instruments through it, so what it returns for a
//! process's answer and for a clustered gateway's envelope
//! (`docs/replicated_substrate.md` §5.5) is asserted here, value for value.

use std::slice::from_ref;

use boa_engine::builtins::promise::PromiseState;
use boa_engine::module::Module;
use boa_engine::{Context, JsString, JsValue, Source};
use serde_json::{json, Value};

const AGGREGATE: &str = include_str!("../content/common/lib/aggregate.js");

/// Call the module's export `name` with `args`, and read its answer back.
fn call(name: &str, args: &[Value]) -> Value {
    let mut ctx = Context::default();
    let module = Module::parse(Source::from_bytes(AGGREGATE), None, &mut ctx).unwrap();
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
    let f = export.as_callable().expect("a function export");
    let args: Vec<JsValue> = args
        .iter()
        .map(|a| JsValue::from_json(a, &mut ctx).unwrap())
        .collect();
    let out = f.call(&JsValue::undefined(), &args, &mut ctx).unwrap();
    out.to_json(&mut ctx).unwrap().unwrap_or(Value::Null)
}

fn status(state: &str, build: &str) -> Value {
    json!({ "state": state, "build": build,
            "tenants": { "zend": { "state": state, "loading": null } } })
}

fn member(node: &str, host: &str, body: Value) -> Value {
    json!({ "node": node, "host": host, "build": 3234, "status": 200, "body": body })
}

fn late(node: &str) -> Value {
    json!({ "node": node, "host": "far", "build": 3234, "status": 504, "error": "deadline" })
}

/// **One process's status is the status.**
#[test]
fn a_process_status_passes_through() {
    let s = status("ready", "b1");
    assert_eq!(call("memberStatus", &[s.clone(), json!("zend")]), s);
    assert_eq!(call("isEnvelope", &[s]), json!(false));
}

/// **Through a clustered gateway the ready member speaks for the app**, ahead
/// of a loading one listed first and a member past its deadline. What a
/// console held its loading screen on was the envelope itself: no `tenants`.
#[test]
fn the_ready_member_speaks_for_the_app() {
    let envelope = json!({
        "members": [
            late("aa"),
            member("bb", ".6", status("loading", "b1")),
            member("cc", ".5", status("ready", "b2")),
        ],
        "group": { "floor": null, "model_lock": "m", "members": 3 },
    });
    assert_eq!(call("isEnvelope", from_ref(&envelope)), json!(true));
    assert_eq!(
        call("memberStatus", &[envelope, json!("zend")]),
        status("ready", "b2")
    );
}

/// With no member ready, the first that answered shows its load; with none
/// answering, the console reads as connecting.
#[test]
fn no_ready_member_shows_the_first_answer_or_connecting() {
    let loading = json!({ "members": [late("aa"), member("bb", ".5", status("loading", "b1"))] });
    assert_eq!(
        call("memberStatus", &[loading, json!("zend")]),
        status("loading", "b1")
    );
    let silent = json!({ "members": [late("aa")] });
    assert_eq!(
        call("memberStatus", &[silent, json!("npcd")]),
        json!({
            "state": "loading",
            "started_at_ms": 0,
            "tenants": { "npcd": { "state": "loading", "loading": {
                "label": "Connecting", "detail": "Waiting for npcd to start",
                "progress": 0, "completed": [] } } },
        })
    );
}

/// An overview with no state per tenant comes from the first member holding
/// the tenant's line; none holding it reads as no line.
#[test]
fn an_overview_comes_from_a_member_holding_the_line() {
    let without = json!({ "tenants": { "zend": null, "npcd": { "bytes": 1 } } });
    let with = json!({ "tenants": { "zend": { "bytes": 7 } } });
    let envelope = json!({ "members": [
        member("aa", ".6", without.clone()),
        member("bb", ".5", with.clone()),
    ] });
    assert_eq!(call("memberWith", &[envelope, json!("zend")]), with);
    let none = json!({ "members": [member("aa", ".6", without.clone())] });
    assert_eq!(
        call("memberWith", &[none, json!("zend")]),
        json!({ "tenants": {} })
    );
    assert_eq!(
        call("memberWith", &[without.clone(), json!("zend")]),
        without
    );
}

/// A per-member page picks one member by node, the first when the one named
/// is gone, and lists the members that answered; one process's answer has no
/// members to pick from.
#[test]
fn a_page_picks_a_member_by_node() {
    let envelope = json!({ "members": [
        member("aa", ".6", json!({ "waves": [1] })),
        late("bb"),
        member("cc", ".5", json!({ "waves": [2] })),
    ] });
    assert_eq!(
        call("pick", &[envelope.clone(), json!("cc")]),
        json!({
            "members": [{ "node": "aa", "host": ".6" }, { "node": "cc", "host": ".5" }],
            "node": "cc",
            "body": { "waves": [2] },
        })
    );
    assert_eq!(
        call("pick", &[envelope, json!("gone")]),
        json!({
            "members": [{ "node": "aa", "host": ".6" }, { "node": "cc", "host": ".5" }],
            "node": "aa",
            "body": { "waves": [1] },
        })
    );
    assert_eq!(
        call("pick", &[json!({ "waves": [3] }), Value::Null]),
        json!({ "members": [], "node": null, "body": { "waves": [3] } })
    );
    assert_eq!(
        call("pick", &[json!({ "members": [late("bb")] }), Value::Null]),
        json!({ "members": [], "node": null, "body": null })
    );
}

/// **The ingest backlog is the cluster's.** The member that speaks for the
/// app can have none of its own while another reads on — through a gateway
/// with two members the indexing bar vanished whenever the first to answer
/// was the idle one. Summed across members, naming the busiest one's item.
#[test]
fn the_ingest_backlog_is_every_members() {
    let line = |backlog: Value| {
        json!({ "state": "ready", "build": "b",
                "tenants": { "zend": { "state": "ready", "loading": null, "ingest_backlog": backlog } } })
    };
    let envelope = json!({ "members": [
        member("ww", ".6", line(Value::Null)),
        member("bw", ".5", line(json!({ "processed": 5, "total": 233, "last_item": "candle/a.rs" }))),
        member("xx", ".7", line(json!({ "processed": 10, "total": 12, "last_item": "candle/b.rs" }))),
    ]});
    let got = call("memberStatus", &[envelope, json!("zend")]);
    assert_eq!(got["build"], "b");
    assert_eq!(
        got["tenants"]["zend"]["ingest_backlog"],
        json!({ "processed": 15, "total": 245, "last_item": "candle/a.rs" })
    );
    let idle = json!({ "members": [member("ww", ".6", line(Value::Null)), member("bw", ".5", line(Value::Null))] });
    assert_eq!(
        call("memberStatus", &[idle, json!("zend")])["tenants"]["zend"]["ingest_backlog"],
        Value::Null
    );
}
