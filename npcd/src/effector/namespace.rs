//! Which namespace a placed part is served under.
//!
//! An instance's url is `http://local/<namespace>/<part>~<ordinal>`. Several
//! parts of one office share a namespace — the chronicle's terminal, table,
//! archive and wall are all `chronicle` — so a character reads the office as one
//! place and the instance id still names the single machine.

/// The namespace a part's instances are served under. A part the table does not
/// name is served under its own id.
pub fn namespace_of(part_id: &str) -> &str {
    match part_id {
        "chronicle-terminal" | "concordance-table" | "archive" | "timeline-wall" => "chronicle",
        "character-terminal" | "relations-table" => "character",
        "story-desk" | "gap-ledger" | "filed-stories" => "story",
        "easel" | "plate-rack" | "hung-faces" | "likeness-table" => "portrait",
        "survey-desk" | "place-index" | "road-table" => "place",
        "map-table" | "gallery-rail" => "map",
        "accession-desk" | "catalogue" | "appraisal-bench" | "mending-bench" => "record",
        "enquiry-desk" => "enquiry",
        "order-table" => "command",
        "creators-chair" => "creator",
        "dispatch-board" => "dispatch",
        "stores" | "fabricator" => "stores",
        "plant-panel" => "plant",
        "structure-board" => "structure",
        "reading-table" => "gather",
        "watch-desk" => "cast",
        "roster" => "roster",
        "house-palette" => "standard",
        "seat" => "room",
        "muster-board" => "orders",
        "bridge-console" => "tower",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_office_shares_one_namespace() {
        for part in ["chronicle-terminal", "concordance-table", "archive", "timeline-wall"] {
            assert_eq!(namespace_of(part), "chronicle");
        }
        assert_eq!(namespace_of("stores"), "stores");
        assert_eq!(namespace_of("fabricator"), "stores");
    }

    #[test]
    fn a_part_is_served_under_the_name_its_office_gives_it() {
        assert_eq!(namespace_of("order-table"), "command");
        assert_eq!(namespace_of("muster-board"), "orders");
        assert_eq!(namespace_of("seat"), "room");
        assert_eq!(namespace_of("bridge-console"), "tower");
    }

    #[test]
    fn a_part_the_table_does_not_name_is_its_own_namespace() {
        assert_eq!(namespace_of("blast-door"), "blast-door");
        assert_eq!(namespace_of("wall-turret"), "wall-turret");
    }
}
