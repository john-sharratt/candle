//! The shared performance and substrate pages' scripts parse.
//! `common/lib/perf.js` carries every chart, the time model and the verdict,
//! and `common/lib/substrate.js` the whole inspector; zend's and npcd's consoles
//! both mount them, and a syntax slip there is a blank page on both, found only
//! by opening it. What the performance page draws for the cluster as a whole is
//! added up by the gateway (`cluster::dispatch::fuse`) and tested there.

use boa_engine::{Context, Module, Source};

const PERF: &str = include_str!("../content/common/lib/perf.js");
const SUBSTRATE: &str = include_str!("../content/common/lib/substrate.js");
/// zend's two pages, each an inline module that mounts one of the above.
const ZEND_PERF: &str = include_str!("../../zend/web/perf.html");
const ZEND_SUBSTRATE: &str = include_str!("../../zend/web/substrate.html");

fn parses(what: &str, source: &str) {
    let mut ctx = Context::default();
    if let Err(e) = Module::parse(Source::from_bytes(source), None, &mut ctx) {
        panic!("{what} does not parse: {e}");
    }
}

/// The inline module of an HTML page.
fn inline_module(page: &str) -> &str {
    let tag = "<script type=\"module\">";
    let open = page.find(tag).expect("an inline module") + tag.len();
    let close = page[open..].find("</script>").expect("a closed module") + open;
    &page[open..close]
}

#[test]
fn the_shared_pages_modules_parse() {
    parses("common/lib/perf.js", PERF);
    parses("common/lib/substrate.js", SUBSTRATE);
}

#[test]
fn zends_pages_modules_parse() {
    parses("zend/web/perf.html's module", inline_module(ZEND_PERF));
    parses(
        "zend/web/substrate.html's module",
        inline_module(ZEND_SUBSTRATE),
    );
}
