//! Tests reshaping a world's map while it runs: adding rooms, relocating
//! stranded bodies and refusing impossible edits.
//!
//! A reshape that adds a walkable room makes it reachable at once. A body
//! standing in a drowned room is moved to the area's entry and logged both ways.
//! An impossible edit changes nothing, and a reshape within a level leaves the
//! lift as it was.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_bodies::tests::{at, vault};

    /// A plain room node for a reshape test — somewhere to stand that opens off
    /// one existing room.
    pub(in super::super) fn room(id: &str, off: &[&str]) -> Node {
        Node {
            id: id.into(),
            kind: NodeKind::Social,
            name: id.into(),
            plural: false,
            stand: None,
            off: off.iter().map(|s| s.to_string()).collect(),
            character: None,
            parts: vec![],
            ground: vec![],
            habit: None,
            sees: vec![],
            exits: vec![],
            visible: vec![],
        }
    }

    /// **A reshape adds a room, and it is walkable at once.** The new node opens
    /// off the green room; after the reshape the green room opens back onto it —
    /// the door woven both ways, exactly as a loaded map's is — and a body can
    /// route to it.
    #[test]
    fn a_reshape_adds_a_walkable_room() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("green-room")).unwrap();
        let moved = w
            .reshape(&MapEdit::AddNode {
                area: "vault-casting".into(),
                node: Box::new(room("annex", &["green-room"])),
            })
            .expect("a room off the green room is a valid reshape");
        assert!(moved.is_empty(), "adding a room displaces nobody");
        let green = w
            .map()
            .get("vault-casting")
            .unwrap()
            .node("green-room")
            .unwrap();
        assert!(
            green.exits.contains(&"annex".to_string()),
            "the door was not woven back: {:?}",
            green.exits
        );
        // And it is genuinely reachable from where the body stands.
        let route = w.route("m1", &at("annex")).expect("the annex is walkable");
        assert_eq!(route.last().unwrap(), &at("annex"));
    }

    /// **A body standing where a room is drowned is relocated, not stranded.**
    /// It is moved to the area's way in, and the move is logged both ways so the
    /// vanished room and the new one each perceive it.
    #[test]
    fn a_reshape_relocates_a_body_off_a_drowned_room() {
        let mut w = vault();
        w.reshape(&MapEdit::AddNode {
            area: "vault-casting".into(),
            node: Box::new(room("annex", &["green-room"])),
        })
        .unwrap();
        w.enter("m1", "Maker-01", at("annex")).unwrap();
        assert_eq!(w.actor("m1").unwrap().at, at("annex"));

        let moved = w
            .reshape(&MapEdit::RemoveNode(at("annex")))
            .expect("drowning an empty leaf room is valid");
        assert_eq!(moved, vec!["m1".to_string()], "the body was not relocated");
        // vault-casting's way in is its core, where the stranded body lands.
        assert_eq!(w.actor("m1").unwrap().at, at("core"));
        assert!(w
            .map()
            .get("vault-casting")
            .unwrap()
            .node("annex")
            .is_none());
    }

    /// **An impossible reshape refuses and changes nothing.** A room opening off
    /// a door to nowhere cannot validate, so the running world keeps the map it
    /// had — the annex never appears.
    #[test]
    fn an_impossible_reshape_leaves_the_world_untouched() {
        let mut w = vault();
        let before = w.map().get("vault-casting").unwrap().nodes.len();
        let err = w
            .reshape(&MapEdit::AddNode {
                area: "vault-casting".into(),
                node: Box::new(room("annex", &["nowhere"])),
            })
            .expect_err("a door to nowhere cannot be reshaped in");
        assert!(err.contains("not a node here"), "{err}");
        assert_eq!(
            w.map().get("vault-casting").unwrap().nodes.len(),
            before,
            "a refused reshape must not change the map"
        );
    }

    /// **Adding a room to a level leaves the lift exactly as it was.** The shaft
    /// is a function of the levels and their cores, and neither changed — so the
    /// car is not disturbed and nobody mid-ride is set down.
    #[test]
    fn a_reshape_within_a_level_does_not_touch_the_lift() {
        let mut w = vault();
        let shaft_before = w.shaft().to_vec();
        w.reshape(&MapEdit::AddNode {
            area: "vault-casting".into(),
            node: Box::new(room("annex", &["green-room"])),
        })
        .unwrap();
        assert_eq!(
            w.shaft(),
            shaft_before.as_slice(),
            "the shaft must be unchanged"
        );
        assert!(w.lift().is_some(), "the lift is still there");
    }
}
