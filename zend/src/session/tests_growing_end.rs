//! Where a journal's growing end begins (`docs/placed_section_layout.md`
//! §3.7): what a request may change as it goes.

use super::*;

fn section(name: &str) -> Section {
    Section {
        name: name.to_string(),
        text: String::new(),
    }
}

/// **The growing end is the current request's heading, or the close where
/// there are no steps**; a journal with neither may change nowhere ahead of
/// its end.
#[test]
fn the_growing_end_is_the_current_heading_or_the_close() {
    let with_steps = [
        section(HISTORY_SECTION),
        section(&entry_section(1)),
        section(&entry_section(2)),
        section(CURRENT_SECTION),
        section("journal/step/1"),
        section(CLOSE_SECTION),
    ];
    assert_eq!(growing_end(&with_steps), 3);
    let closed = [section(&entry_section(1)), section(CLOSE_SECTION)];
    assert_eq!(growing_end(&closed), 1);
    let bare = [section("memory/topic")];
    assert_eq!(growing_end(&bare), 1);
}
