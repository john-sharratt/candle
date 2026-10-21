//! The status page's World leases, NPC leases and Disconnected worlds blocks
//! (§12.2), drawn from what the members report of their worlds in `COORD`
//! (§12.1) and what this gateway knows of the app's membership.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use super::status::MemberRow;
use crate::consensus::rendezvous::{owner, Weighted};
use crate::coord::NpcLeaseLine;

/// One world of an app.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorldView {
    pub app: String,
    pub world: String,
    /// The member that says it is the world's authority.
    pub authority: Option<String>,
    pub epoch: Option<u64>,
    /// The authority's term left on its own clock.
    pub term_left_ms: Option<u64>,
    /// The member rendezvous names to ask for the lease (§9.5).
    pub proposer: Option<String>,
    /// The proposer is not the authority: a handover is due, or the world
    /// has none.
    pub proposer_flagged: bool,
    /// Why a world has no authority, when it has none.
    pub no_holder: Option<String>,
    /// A handover in progress: who takes the world, once what it waits on is
    /// applied.
    pub handover: Option<String>,
    /// Every character's lease, as the authority lends it.
    pub lent: Vec<NpcLeaseLine>,
    /// Characters whose holder is not among the members now: not lent again
    /// until it returns (§9.3).
    pub away: Vec<u64>,
}

/// A member running a world in disconnected mode (§9.4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DisconnectedRow {
    pub app: String,
    pub node: String,
    pub host: String,
    pub world: String,
    pub since_ms: u64,
    /// The characters it runs there.
    pub characters: Vec<u64>,
}

/// The rendezvous key of a world's lease proposer.
pub fn proposer_key(world: &str) -> &[u8] {
    world.as_bytes()
}

/// Every world any member of an app reports, from `members`.
pub fn worlds(members: &[MemberRow]) -> Vec<WorldView> {
    let mut apps: BTreeMap<&str, Vec<&MemberRow>> = BTreeMap::new();
    for m in members {
        apps.entry(m.app.as_str()).or_default().push(m);
    }
    let mut out = Vec::new();
    for (app, rows) in apps {
        let ready: Vec<Weighted> = rows
            .iter()
            .filter(|r| r.state == "ready")
            .filter_map(|r| {
                Some(Weighted {
                    node: u64::from_str_radix(&r.node, 16).ok()?,
                    weight: r.weight as f64,
                })
            })
            .collect();
        let present: BTreeSet<&str> = rows.iter().map(|r| r.node.as_str()).collect();
        let quorum = rows
            .iter()
            .filter_map(|r| r.coord.as_ref())
            .map(|c| c.paxos.quorum)
            .max()
            .unwrap_or(1);
        let names: BTreeSet<&str> = rows
            .iter()
            .filter_map(|r| r.coord.as_ref())
            .flat_map(|c| c.worlds.iter().map(|w| w.world.as_str()))
            .collect();
        for world in names {
            let lines: Vec<(&MemberRow, _)> = rows
                .iter()
                .filter_map(|r| {
                    let c = r.coord.as_ref()?;
                    Some((*r, c.worlds.iter().find(|w| w.world == world)?))
                })
                .collect();
            let authority = lines.iter().find(|(_, w)| w.role == "authority");
            let proposer = owner(proposer_key(world), &ready).map(|n| format!("{n:016x}"));
            let lease_key = format!("world/{world}");
            let term_left_ms = authority.and_then(|(r, _)| {
                r.coord
                    .as_ref()?
                    .paxos
                    .keys
                    .iter()
                    .find(|k| k.key == lease_key)?
                    .held_ms
            });
            let lent = authority.map(|(_, w)| w.lent.clone()).unwrap_or_default();
            let away = lent
                .iter()
                .filter(|l| !l.holder.is_empty() && !present.contains(l.holder.as_str()))
                .map(|l| l.npc)
                .collect();
            let handover = lines.iter().find_map(|(r, w)| {
                w.waiting
                    .as_ref()
                    .map(|on| format!("{} takes it once {on} is applied", r.node))
            });
            let no_holder = match authority {
                Some(_) => None,
                None if ready.len() < quorum => Some(format!(
                    "no quorum reachable: {} ready of the {quorum} it needs",
                    ready.len()
                )),
                None => Some("a term still running out".to_string()),
            };
            out.push(WorldView {
                app: app.to_string(),
                world: world.to_string(),
                authority: authority.map(|(r, _)| r.node.clone()),
                epoch: authority.and_then(|(_, w)| w.epoch),
                term_left_ms,
                proposer_flagged: proposer.as_deref() != authority.map(|(r, _)| r.node.as_str()),
                proposer,
                no_holder,
                handover,
                lent,
                away,
            });
        }
    }
    out
}

/// Every member running a world disconnected.
pub fn disconnected(members: &[MemberRow]) -> Vec<DisconnectedRow> {
    members
        .iter()
        .filter_map(|r| Some((r, r.coord.as_ref()?)))
        .flat_map(|(r, c)| {
            c.worlds
                .iter()
                .filter(|w| w.role == "disconnected")
                .map(move |w| DisconnectedRow {
                    app: r.app.clone(),
                    node: r.node.clone(),
                    host: r.host.clone(),
                    world: w.world.clone(),
                    since_ms: w.since_ms,
                    characters: c
                        .npc_leases
                        .iter()
                        .filter(|l| l.world == w.world)
                        .map(|l| l.npc)
                        .collect(),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::{CoordState, KeyLine, PaxosView, WorldLine};
    use crate::frame::BuildId;
    use crate::gateway::status::Queues;

    fn row(
        node: u64,
        state: &'static str,
        worlds: Vec<WorldLine>,
        held: Vec<NpcLeaseLine>,
    ) -> MemberRow {
        MemberRow {
            app: "npcd".into(),
            node: format!("{node:016x}"),
            host: format!("m{node}"),
            build: BuildId {
                count: 1,
                hash: "h".into(),
                local: String::new(),
            },
            local_build: false,
            weight: 1,
            power: None,
            idle: None,
            slots_free: None,
            prefill_backlog: None,
            prefill_tok_s: None,
            decode_step_next_us: None,
            decode_step_wide_us: None,
            prefill_tok_s_wide: None,
            prompt_tokens: None,
            reply_tokens: None,
            tokens_per_step_milli: None,
            in_flight: 0,
            state,
            session: 1,
            connected_ms: 0,
            clock_offset_ms: 0,
            coord: Some(CoordState {
                paxos: PaxosView {
                    max_members: 3,
                    quorum: 2,
                    keys: vec![KeyLine {
                        key: "world/w1".into(),
                        held_ms: Some(20_000),
                        ..KeyLine::default()
                    }],
                },
                worlds,
                npc_leases: held,
                ..CoordState::default()
            }),
            last_close: None,
            high_water: 0,
            lagging: false,
            syncing: false,
            queues: Queues::default(),
        }
    }

    fn line(role: &str, lent: Vec<NpcLeaseLine>) -> WorldLine {
        WorldLine {
            world: "w1".into(),
            role: role.into(),
            epoch: (role == "authority").then_some(5),
            since_ms: 100,
            waiting: None,
            lent,
        }
    }

    fn lease(npc: u64, holder: u64) -> NpcLeaseLine {
        NpcLeaseLine {
            world: "w1".into(),
            npc,
            holder: format!("{holder:016x}"),
            expires_ms: 9,
            next: None,
        }
    }

    /// The authority and its term, the proposer rendezvous names, the leases
    /// it lends, and the characters of a member that is away.
    #[test]
    fn a_world_shows_its_authority_proposer_and_leases() {
        let rows = vec![
            row(
                1,
                "ready",
                vec![line("authority", vec![lease(7, 1), lease(8, 9)])],
                vec![],
            ),
            row(2, "ready", vec![line("replica", vec![])], vec![]),
        ];
        let views = worlds(&rows);
        assert_eq!(views.len(), 1);
        let w = &views[0];
        assert_eq!(w.authority.as_deref(), Some("0000000000000001"));
        assert_eq!((w.epoch, w.term_left_ms), (Some(5), Some(20_000)));
        let ready = [
            Weighted {
                node: 1,
                weight: 1.0,
            },
            Weighted {
                node: 2,
                weight: 1.0,
            },
        ];
        let named = owner(b"w1", &ready).unwrap();
        assert_eq!(w.proposer, Some(format!("{named:016x}")));
        assert_eq!(w.proposer_flagged, named != 1);
        assert_eq!(w.lent.len(), 2);
        assert_eq!(w.away, vec![8], "member 9 is not here");
        assert_eq!(w.no_holder, None);
    }

    /// With no authority, the page says why.
    #[test]
    fn a_world_with_no_authority_says_why() {
        let one = vec![row(2, "ready", vec![line("replica", vec![])], vec![])];
        assert_eq!(
            worlds(&one)[0].no_holder.as_deref(),
            Some("no quorum reachable: 1 ready of the 2 it needs")
        );
        let two = vec![
            row(2, "ready", vec![line("replica", vec![])], vec![]),
            row(3, "ready", vec![line("replica", vec![])], vec![]),
        ];
        assert_eq!(
            worlds(&two)[0].no_holder.as_deref(),
            Some("a term still running out")
        );
    }

    #[test]
    fn a_disconnected_world_lists_its_characters() {
        let rows = vec![row(
            3,
            "ready",
            vec![line("disconnected", vec![])],
            vec![lease(4, 3), lease(6, 3)],
        )];
        assert_eq!(
            disconnected(&rows),
            vec![DisconnectedRow {
                app: "npcd".into(),
                node: "0000000000000003".into(),
                host: "m3".into(),
                world: "w1".into(),
                since_ms: 100,
                characters: vec![4, 6],
            }]
        );
    }
}
