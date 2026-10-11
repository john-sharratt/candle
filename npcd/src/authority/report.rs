//! What this member reports of its worlds in `COORD`
//! (`docs/replicated_substrate.md` §12.1): each world's role, epoch, since
//! when and the handover it waits on — with, as a world's authority, every
//! lease it lends — and the NPC leases it holds.

use cluster::coord::{AppCoord, NpcLeaseLine, WorldLine};

use super::npc_lease::{hex, leases_of, NpcLease};
use super::role::Role;
use crate::engine::runtime::Runtime;

fn line(world: &str, npc: u64, lease: &NpcLease) -> NpcLeaseLine {
    NpcLeaseLine {
        world: world.to_string(),
        npc,
        holder: lease.holder.clone(),
        expires_ms: lease.expires_ms,
        next: lease.next.clone(),
    }
}

/// The app's part of this member's `COORD`.
pub fn app_coord(rt: &Runtime) -> AppCoord {
    let me = rt.standing.node().map(hex);
    let mut coord = AppCoord::default();
    for world in rt.hosted.ids() {
        let role = rt.standing.role(&world);
        let leases = leases_of(&rt.standing.written(&world));
        coord.worlds.push(WorldLine {
            world: world.clone(),
            role: role.name().to_string(),
            epoch: match role {
                Role::Authority { epoch } => Some(epoch),
                _ => None,
            },
            since_ms: rt.standing.since(&world),
            waiting: rt
                .standing
                .waiting(&world)
                .map(|(by, through)| format!("{} through {through}", hex(by))),
            lent: match role {
                Role::Authority { .. } => leases
                    .iter()
                    .map(|(npc, lease)| line(&world, *npc, lease))
                    .collect(),
                _ => Vec::new(),
            },
        });
        coord.npc_leases.extend(
            leases
                .iter()
                .filter(|(_, l)| Some(&l.holder) == me.as_ref())
                .map(|(npc, lease)| line(&world, *npc, lease)),
        );
    }
    coord
}
