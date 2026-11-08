//! What the house adds to a body's situation (`docs/zen_moon_mind.md`
//! §13–§15, `docs/house_modes.md`): the bridges it can work where it stands,
//! the house's modes where it may set them, and what its carried mission lets
//! it build there. One computation, so the runtime's grammar and
//! anything checking it read the same sets.

use npc_map::schema::Where;

use crate::bridge;
use crate::bridge::finding::Finding;
use crate::engine::build::{edit, grants_here, open_findings};
use crate::engine::mission_gen::standing::HOUSE_LINK;
use crate::engine::tools::Within;
use crate::world::Hosted;

impl Within {
    /// This situation with the house's sets filled in for `body` — left as it
    /// is outside the mirror of a real place, where the house offers nothing.
    pub fn in_the_house(self, hosted: &Hosted, body: &str) -> Self {
        let place = hosted.standpoint_of(body);
        let Some(at) = Where::parse(&place) else {
            return self;
        };
        if !hosted.read(|w| w.map().is_mirrored(&at.area)) {
            return self;
        }
        let parts = bridge::parts_here(hosted, body);
        let built = hosted.sim(|s| s.built.edits().to_vec());
        let (unlinked, doors) = hosted.read(|w| {
            let map = w.map();
            let unlinked = parts
                .iter()
                .filter(|p| p.id != HOUSE_LINK && edit::linkable(map, &p.id))
                .map(|p| p.name.clone())
                .collect();
            // A room somebody is in is never taken away from under them.
            let doors = edit::doors(map, &at, &built)
                .into_iter()
                .filter(|(_, room)| w.actors_at(room).is_empty())
                .map(|(name, _)| name)
                .collect();
            (unlinked, doors)
        });
        // The house's mode is Zen's to set, from the room its devices are in.
        let house_modes = hosted.sim(|s| {
            match s.missions.is_unordered(body) {
                true => s.house.modes_at(&place),
                false => Vec::new(),
            }
        });
        Within {
            links: bridge::links_of(&parts),
            sonos: bridge::sonos_of(&parts),
            receiver: bridge::receiver_of(&parts),
            cast: bridge::cast_of(&parts),
            pairable: bridge::pairable_of(&parts),
            pairing: bridge::pairing_of(&parts),
            house_modes,
            grants: grants_here(hosted, body),
            findings: open_findings(hosted, &place)
                .iter()
                .map(Finding::label)
                .collect(),
            unlinked,
            onboarded: parts
                .iter()
                .filter(|p| edit::onboarded(&p.id))
                .map(|p| p.name.clone())
                .collect(),
            doors,
            ..self
        }
    }
}
