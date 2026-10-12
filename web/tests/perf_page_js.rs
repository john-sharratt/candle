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
