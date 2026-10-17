//! zend's performance page's script parses: `zend/web/perf.html` carries every
//! chart, the time model and the verdict inline, as one module, and a syntax
//! slip there is a blank page found only by opening it.

use boa_engine::{Context, Module, Source};

const PAGE: &str = include_str!("../../zend/web/perf.html");

#[test]
fn the_performance_pages_module_parses() {
    let tag = "<script type=\"module\">";
    let open = PAGE.find(tag).expect("an inline module") + tag.len();
    let close = PAGE[open..].find("</script>").expect("a closed module") + open;
    let mut ctx = Context::default();
    if let Err(e) = Module::parse(Source::from_bytes(&PAGE[open..close]), None, &mut ctx) {
        panic!("the performance page's module does not parse: {e}");
    }
}

/// The cluster view's fusion (`fuseRows`, `fuseMix`, `sumHost`), lifted out of
/// the page as written and run: what every panel draws for the cluster as a
/// whole is one engine's worth made of every member's.
fn fused(expr: &str) -> serde_json::Value {
    let from = PAGE
        .find("/* ── the cluster as a whole")
        .expect("the fusion block");
    let to = PAGE[from..]
        .find("async function refresh()")
        .expect("its end")
        + from;
    let script = format!("{}\nJSON.stringify({expr})", &PAGE[from..to]);
    let mut ctx = Context::default();
    let out = ctx
        .eval(Source::from_bytes(script.as_bytes()))
        .unwrap_or_else(|e| panic!("the fusion block does not run: {e}"));
    let text = out.to_string(&mut ctx).unwrap().to_std_string_escaped();
    serde_json::from_str(&text).unwrap()
}

/// **Two members' waves in one slot are one wave of the cluster's**: tokens,
/// backlog and every VRAM band summed; the wave's wall-clock the longer, so
/// tokens over it is the cluster's rate. A slot only one member ran in is
/// that member's.
#[test]
fn the_cluster_view_sums_members_and_keeps_the_longest_wave() {
    let w = fused(
        "fuseRows([\
           [{ at: 10000, age: 1, tok: 300, ws: 2000, backlog: 5, vram: { kv: 100, free: 50 } },\
            { at: 14000, age: 0, tok: 10, ws: 1000, backlog: 1, vram: { kv: 1, free: 1 } }],\
           [{ at: 10400, age: 1, tok: 700, ws: 1500, backlog: 7, vram: { kv: 200, free: 25 } }]\
         ])",
    );
    assert_eq!(w[0]["tok"], 1000.0);
    assert_eq!(w[0]["ws"], 2000.0, "the longer wave, not their sum");
    assert_eq!(w[0]["backlog"], 12.0);
    assert_eq!(w[0]["vram"]["kv"], 300.0);
    assert_eq!(w[0]["vram"]["free"], 75.0);
    assert_eq!(w[1]["tok"], 10.0, "a slot one member ran in is its");
}

/// The arena mix's formats are every member's, each member's counts moved
/// onto them; host memory is added up, and a reading one member lacks stays
/// absent rather than reading low.
#[test]
fn the_cluster_view_merges_formats_and_adds_host_memory() {
    let mix = fused(
        "fuseMix([\
           { formats: ['c4', 'c6'], rows: [{ at: 2000, tok: [10, 20] }] },\
           { formats: ['c6', 'c8'], rows: [{ at: 2100, tok: [5, 7] }] }\
         ])",
    );
    assert_eq!(mix["formats"], serde_json::json!(["c4", "c6", "c8"]));
    assert_eq!(mix["rows"][0]["tok"], serde_json::json!([10, 25, 7]));
    let host = fused(
        "sumHost([{ total_mib: 64000, free_mib: 20000, swap_mib: 4 }, { total_mib: 32000, free_mib: null, swap_mib: 2 }])",
    );
    assert_eq!(host["total_mib"], 96000);
    assert_eq!(host["free_mib"], serde_json::Value::Null);
    assert_eq!(host["swap_mib"], 6);
}
