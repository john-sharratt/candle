//! A character's experience as journal lines (`docs/npc_conversations.md`
//! §5): what it heard, said, wrote, promised, did, and what changed for it,
//! written into `groups/<group>/bots/<bot>/` of its world's group.
//!
//! A conversation knows what happened since it last spoke with its party
//! only from these lines (§6.1), and the main loop knows what its
//! conversations did from them (§5.4).
//!
//! | Module | |
//! |---|---|
//! | [`buffer`] | the unwritten lines, per bot, and their caps |
//! | [`concern`] | which journals hear of a change in the world |
//! | [`days`] | the world's day rolling over |
//! | [`line`] | how each kind of line reads |
//! | [`store`] | the checkout, the per-world groups, the flush |

pub mod buffer;
pub mod concern;
pub mod days;
pub mod line;
pub mod store;

/// Whether an act of a character's own is worth a `did` line: one whose
/// outcome changes the world's state — a device, the house, a part, an item
/// handed over, a mission's work — never walking, waiting, speech or the
/// phone, which are written as what was said, or not at all.
pub fn notable(tool: &str) -> bool {
    const ACTS: &[&str] = &[
        "give",
        "operate",
        "house_mode",
        "post_notice",
        "record_verdict",
        "make",
        "claim",
        "release",
        "collect_mission",
        "report_done",
        "report_stuck",
        "file_write",
        "file_edit",
        "compose",
        "onboard_part",
        "onboard_thing",
        "link_part",
        "offboard_part",
        "discover_door",
        "undiscover_door",
        "describe_room",
        "pair_device",
        "wake_device",
    ];
    const DEVICES: &[&str] = &["sonos_", "receiver_", "cast_", "wiz_"];
    let reads = tool.ends_with("_status") || tool.ends_with("_playing");
    ACTS.contains(&tool) || (DEVICES.iter().any(|d| tool.starts_with(d)) && !reads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_changes_the_world_is_notable_and_nothing_else_is() {
        for yes in [
            "give",
            "house_mode",
            "sonos_pause",
            "wiz_on",
            "report_done",
            "file_write",
            "discover_door",
            "describe_room",
        ] {
            assert!(notable(yes), "{yes}");
        }
        for no in ["move_to", "wait", "tell", "message", "reply", "reflect", "sonos_status"] {
            assert!(!notable(no), "{no}");
        }
    }
}
