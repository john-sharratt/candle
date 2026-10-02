//! Part id → world-API namespace, in one place.
//!
//! Every situated route lives under a namespace segment: a chronicle terminal's
//! routes at `http://local/chronicle/<id>`, an easel's at
//! `http://local/portrait/<id>`. The namespace is not the part's own id — it is
//! the *world surface* the part belongs to, so several parts that all write the
//! same store share one (`accession-desk`, `catalogue`, `appraisal-bench` and
//! `mending-bench` are all `record`), and the surface is named for what it does
//! rather than for whichever fixture happens to reach it.
//!
//! The table is the executable form of the effector design's Appendix C.0 route
//! map. Keeping it here — one function, one match — means the router, and every
//! handler that later mounts under a namespace, read the same mapping, and a
//! part that grows a route needs one line rather than a scattered agreement.

/// The `<ns>` path segment a part's routes mount under.
///
/// A part with no place in the table falls back to its own id, so a world that
/// adds a part gets a working (if unshared) namespace for free rather than a
/// panic or a blank segment — the same default the map's own ids follow.
pub fn namespace_of(part_id: &str) -> &str {
    match part_id {
        // The reading/working stations, each its own authoring surface.
        "chronicle-terminal" | "concordance-table" | "archive" | "timeline-wall" => "chronicle",
        "character-terminal" | "relations-table" => "character",
        "story-desk" | "gap-ledger" | "filed-stories" => "story",
        "easel" | "plate-rack" | "hung-faces" | "likeness-table" => "portrait",
        "survey-desk" | "place-index" | "road-table" => "place",
        "map-table" | "gallery-rail" => "map",
        // The record surface: intake, catalogue, and the benches that mend it.
        "accession-desk" | "catalogue" | "appraisal-bench" | "mending-bench" => "record",
        // The command table: where a character takes up a mission and reports it.
        // Its own routable namespace, **distinct from the plan-bridge `/orders`**
        // (the projected agency store, which is excluded from station routing) —
        // so the mission acts (`collect_mission`/`report_done`/`report_stuck`)
        // reach the device as `http://local/command/<id>/<verb>` (effector design
        // Step 6; the mission half of the command migration).
        "order-table" => "command",
        // The muster board's order acts.
        "muster-board" => "orders",
        "creators-chair" => "creator",
        "dispatch-board" => "dispatch",
        "enquiry-desk" => "enquiry",
        "plant-panel" => "plant",
        "structure-board" => "structure",
        "reading-table" => "gather",
        "watch-desk" => "cast",
        "roster" => "roster",
        "house-palette" => "standard",
        "stores" | "fabricator" => "stores",
        "bridge-console" => "tower",
        "seat" => "room",
        // Anything unmapped addresses itself.
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reading_stations_map_to_their_authoring_surfaces() {
        assert_eq!(namespace_of("chronicle-terminal"), "chronicle");
        assert_eq!(namespace_of("character-terminal"), "character");
        assert_eq!(namespace_of("story-desk"), "story");
        assert_eq!(namespace_of("easel"), "portrait");
        assert_eq!(namespace_of("survey-desk"), "place");
        assert_eq!(namespace_of("map-table"), "map");
    }

    #[test]
    fn the_parts_that_write_the_record_share_one_namespace() {
        for part in [
            "accession-desk",
            "catalogue",
            "appraisal-bench",
            "mending-bench",
        ] {
            assert_eq!(namespace_of(part), "record", "{part} is a record surface");
        }
    }

    #[test]
    fn a_part_with_no_mapping_addresses_itself() {
        assert_eq!(namespace_of("some-new-part"), "some-new-part");
    }

    /// **The command table has its own routable namespace**, distinct from the
    /// plan-bridge `orders` (which is excluded from station routing) — so the
    /// mission acts reach the device rather than sharing the projected store's
    /// prefix.
    #[test]
    fn the_command_table_routes_under_command_not_orders() {
        assert_eq!(namespace_of("order-table"), "command");
        assert_eq!(namespace_of("muster-board"), "orders");
    }
}
