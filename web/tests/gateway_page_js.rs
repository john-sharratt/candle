//! The cluster page's script parses: `src/cluster/gateway.html` carries its
//! whole operator UI inline, and a syntax slip there is a blank page found only
//! by opening it.

use boa_engine::{Context, Script, Source};

const PAGE: &str = include_str!("../src/cluster/gateway.html");

#[test]
fn the_cluster_pages_script_parses() {
    let open = PAGE.rfind("<script>").expect("an inline script") + "<script>".len();
    let close = PAGE[open..].find("</script>").expect("a closed script") + open;
    let mut ctx = Context::default();
    if let Err(e) = Script::parse(Source::from_bytes(&PAGE[open..close]), None, &mut ctx) {
        panic!("the cluster page's script does not parse: {e}");
    }
}
