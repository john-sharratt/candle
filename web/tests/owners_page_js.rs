//! npcd's Owners page parses: a syntax slip in a console page is a blank page
//! found only by opening it, and this one is the only way to hand a character
//! to another account.

use boa_engine::{Context, Module, Source};

const PAGE: &str = include_str!("../content/npcd/pages/owners.js");

#[test]
fn the_owners_pages_module_parses() {
    let mut ctx = Context::default();
    if let Err(e) = Module::parse(Source::from_bytes(PAGE), None, &mut ctx) {
        panic!("the owners page's module does not parse: {e}");
    }
}
