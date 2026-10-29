//! NPC leases (`docs/replicated_substrate.md` §9.3): each character runs on one
//! machine, the holder of its lease, which the world's authority lends as a
//! state record, `world/<wid>/npclease/<nid> = { holder, expires_ms, next }`.
//!
//! - It lends by weighted rendezvous over the npcd members (§9.5), from the
//!   start, so losing any one member loses only its share of the cast.
//! - It renews a lease while the holder is in the group.
//! - **A lease is held until it expires *and* its holder comes back.** A
//!   holder that drops out keeps its characters: they are not lent again until
//!   it rejoins, so a character never runs in two places and its memory never
//!   gains two pasts.
//! - When rendezvous names a new holder for a character whose holder is still
//!   here, the lease names it as `next`; the holder hands the character back
//!   at its next turn boundary, and the authority then lends it to `next`,
//!   naming what the old holder wrote — its turns, its journal — for the new
//!   one to have applied before the character wakes there.

use std::collections::{BTreeMap, BTreeSet};

use cluster::consensus::rendezvous::{owner, Weighted};
use cluster::frame::Plan;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::world_state::Records;

/// How long a lent lease runs.
pub const NPC_TTL_MS: u64 = 60_000;

/// The record names of a world's NPC leases begin so.
pub const LEASES: &str = "npclease/";

/// One character's lease, as its record holds it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NpcLease {
    /// The member that runs the character, its node in hex; empty when an
    /// admin freed the lease.
    pub holder: String,
    pub expires_ms: u64,
    /// The member rendezvous names now, waiting for the holder to hand the
    /// character back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    /// The writes of the member that handed the character back, which the
    /// holder applies before it runs the character.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<After>,
}

/// A member's own records, through an `oseq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct After {
    pub origin: u64,
    pub through: u64,
}

/// Every character's lease in a world's `records`.
pub fn leases_of(records: &Records) -> BTreeMap<u64, NpcLease> {
    records
        .iter()
        .filter_map(|(name, v)| {
            let nid = name.strip_prefix(LEASES)?.parse::<u64>().ok()?;
            Some((nid, serde_json::from_value(v.clone()).ok()?))
        })
        .collect()
}

/// `records` with their leases replaced by `leases`.
pub fn with_leases(records: &mut Records, leases: &BTreeMap<u64, NpcLease>) {
    records.retain(|name, _| !name.starts_with(LEASES));
    for (nid, lease) in leases {
        let value: Value = serde_json::to_value(lease).expect("a lease always encodes");
        records.insert(lease_entity(*nid), value);
    }
}

/// The characters whose lease names `node` as holder.
pub fn held_by(leases: &BTreeMap<u64, NpcLease>, node: u64) -> BTreeSet<u64> {
    let me = hex(node);
    leases
        .iter()
        .filter(|(_, l)| l.holder == me)
        .map(|(nid, _)| *nid)
        .collect()
}

pub fn hex(node: u64) -> String {
    format!("{node:016x}")
}

pub fn node_of(hex: &str) -> Option<u64> {
    u64::from_str_radix(hex, 16).ok()
}

/// The record name of `nid`'s lease under its world.
pub fn lease_entity(nid: u64) -> String {
    format!("{LEASES}{nid}")
}

/// The member rendezvous lends `nid` to among `members`.
pub fn lent_to(nid: u64, members: &[Weighted]) -> Option<u64> {
    owner(nid.to_string().as_bytes(), members)
}

/// `ready` weighted as the gateway's plan lends characters
/// (`docs/cooperative_admission.md` §4.4): each member by the weight the plan
/// names for it, and nothing for one it does not name — a member the plan has
/// not yet seen is not yet fit to host. With no plan, as they stand: the power
/// each was admitted at, as worlds are weighed.
pub fn by_plan(ready: Vec<Weighted>, plan: Option<&Plan>) -> Vec<Weighted> {
    let Some(plan) = plan else {
        return ready;
    };
    ready
        .into_iter()
        .map(|m| Weighted {
            node: m.node,
            weight: f64::from(plan.weight_of(m.node).unwrap_or(0)),
        })
        .collect()
}

/// A lease `lend` changed: a character lent for the first time, moved, or
/// asked to move at its next turn boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Moved {
    pub nid: u64,
    /// Its holder before, empty for one never lent.
    pub from: String,
    pub holder: String,
    /// The member it is to move to once its holder hands it back.
    pub next: Option<String>,
}

/// Every lease in `lent` whose holder or next holder differs from `current`.
pub fn moves(current: &BTreeMap<u64, NpcLease>, lent: &BTreeMap<u64, NpcLease>) -> Vec<Moved> {
    lent.iter()
        .filter_map(|(nid, l)| {
            let before = current.get(nid);
            let same = before.is_some_and(|b| b.holder == l.holder && b.next == l.next);
            (!same).then(|| Moved {
                nid: *nid,
                from: before.map(|b| b.holder.clone()).unwrap_or_default(),
                holder: l.holder.clone(),
                next: l.next.clone(),
            })
        })
        .collect()
}

/// The leases the authority holds the cast to now: for each character of the
/// world in `cast`, its lease from `current` renewed, moved or lent; a
/// character no longer in the cast has none. `joined` are the members in the
/// group now.
pub fn lend(
    cast: &[u64],
    current: &BTreeMap<u64, NpcLease>,
    members: &[Weighted],
    joined: &BTreeSet<u64>,
    now_ms: u64,
) -> BTreeMap<u64, NpcLease> {
    let mut out = BTreeMap::new();
    for &nid in cast {
        let Some(want) = lent_to(nid, members) else {
            if let Some(l) = current.get(&nid) {
                out.insert(nid, l.clone());
            }
            continue;
        };
        let lease = match current.get(&nid) {
            // Never lent, or freed by an admin: lend it.
            None => fresh(want, now_ms),
            Some(l) if l.holder.is_empty() => fresh(want, now_ms),
            Some(l) => {
                let holder = node_of(&l.holder);
                match holder {
                    // Away: held as it stands, not renewed, not lent again.
                    Some(h) if !joined.contains(&h) => l.clone(),
                    // Here, and where rendezvous wants it: renewed while it
                    // has less than half its term left, and a move asked
                    // before rendezvous changed back is called off.
                    Some(h) if h == want => {
                        if l.expires_ms.saturating_sub(now_ms) < NPC_TTL_MS / 2 || l.next.is_some()
                        {
                            NpcLease {
                                after: l.after,
                                ..fresh(want, now_ms)
                            }
                        } else {
                            l.clone()
                        }
                    }
                    // Here, and rendezvous wants it elsewhere: renewed, with
                    // the next holder named, until the holder hands it back.
                    Some(_) => NpcLease {
                        holder: l.holder.clone(),
                        expires_ms: if l.expires_ms.saturating_sub(now_ms) < NPC_TTL_MS / 2 {
                            now_ms + NPC_TTL_MS
                        } else {
                            l.expires_ms
                        },
                        next: Some(hex(want)),
                        after: l.after,
                    },
                    None => fresh(want, now_ms),
                }
            }
        };
        out.insert(nid, lease);
    }
    out
}

fn fresh(holder: u64, now_ms: u64) -> NpcLease {
    NpcLease {
        holder: hex(holder),
        expires_ms: now_ms + NPC_TTL_MS,
        next: None,
        after: None,
    }
}

/// The holder `from` handed `nid` back, its own writes reaching `through`: the
/// lease goes to its `next`, who waits for them. `None` when `from` is not its
/// holder or no move was asked.
pub fn hand_back(current: &NpcLease, from: u64, through: u64, now_ms: u64) -> Option<NpcLease> {
    if node_of(&current.holder) != Some(from) {
        return None;
    }
    let next = node_of(current.next.as_deref()?)?;
    Some(NpcLease {
        after: Some(After {
            origin: from,
            through,
        }),
        ..fresh(next, now_ms)
    })
}

/// An admin's release: the lease names nobody, and the authority lends it
/// again on its next beat.
pub fn released() -> NpcLease {
    NpcLease {
        holder: String::new(),
        expires_ms: 0,
        next: None,
        after: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn members(nodes: &[u64]) -> Vec<Weighted> {
        nodes
            .iter()
            .map(|&node| Weighted { node, weight: 1.0 })
            .collect()
    }

    fn joined(nodes: &[u64]) -> BTreeSet<u64> {
        nodes.iter().copied().collect()
    }

    const NOW: u64 = 1_000_000;

    /// A cast is lent across the members by rendezvous, every character to
    /// exactly one, from the start.
    #[test]
    fn a_cast_is_spread_by_rendezvous() {
        let cast: Vec<u64> = (1..=40).collect();
        let ms = members(&[1, 2, 3]);
        let leases = lend(&cast, &BTreeMap::new(), &ms, &joined(&[1, 2, 3]), NOW);
        assert_eq!(leases.len(), 40);
        let holders: BTreeSet<&str> = leases.values().map(|l| l.holder.as_str()).collect();
        assert_eq!(holders.len(), 3, "every member holds some");
        for (nid, l) in &leases {
            assert_eq!(node_of(&l.holder), lent_to(*nid, &ms));
            assert_eq!(l.expires_ms, NOW + NPC_TTL_MS);
        }
    }

    /// Renewed only past half its term; a lease of a member that dropped out
    /// is held as it stands — not renewed, not lent elsewhere.
    #[test]
    fn a_lease_is_renewed_while_its_holder_is_here_and_kept_while_away() {
        let ms = members(&[1, 2]);
        let nid = (1..).find(|n| lent_to(*n, &ms) == Some(1)).unwrap();
        let first = lend(&[nid], &BTreeMap::new(), &ms, &joined(&[1, 2]), NOW);
        let early = lend(&[nid], &first, &ms, &joined(&[1, 2]), NOW + 10_000);
        assert_eq!(early, first, "plenty of term left");
        let late = lend(&[nid], &first, &ms, &joined(&[1, 2]), NOW + 40_000);
        assert_eq!(late[&nid].expires_ms, NOW + 40_000 + NPC_TTL_MS);

        // Member 1 drops out; rendezvous over the rest would name 2, but the
        // lease waits for 1.
        let away = lend(&[nid], &first, &members(&[2]), &joined(&[2]), NOW + 90_000);
        assert_eq!(away, first);
    }

    /// Rendezvous moving a character marks the lease with its next holder;
    /// the holder handing it back moves it; anyone else cannot.
    #[test]
    fn a_move_waits_for_the_holder_to_hand_back() {
        let before = members(&[1]);
        let after = members(&[1, 2]);
        let nid = (1..).find(|n| lent_to(*n, &after) == Some(2)).unwrap();
        let held = lend(&[nid], &BTreeMap::new(), &before, &joined(&[1]), NOW);
        assert_eq!(held[&nid].holder, hex(1));
        let moving = lend(&[nid], &held, &after, &joined(&[1, 2]), NOW + 1_000);
        assert_eq!(moving[&nid].holder, hex(1), "still running where it is");
        assert_eq!(moving[&nid].next.as_deref(), Some(hex(2).as_str()));
        assert_eq!(hand_back(&moving[&nid], 2, 9, NOW), None, "not its holder");
        let moved = hand_back(&moving[&nid], 1, 40, NOW + 2_000).unwrap();
        assert_eq!(moved.holder, hex(2));
        assert_eq!(moved.next, None);
        assert_eq!(
            moved.after,
            Some(After {
                origin: 1,
                through: 40
            }),
            "the new holder waits for what the old one wrote"
        );
        // Renewed by its new holder, it still names what to wait for.
        let mut current = BTreeMap::new();
        current.insert(nid, moved.clone());
        let renewed = lend(&[nid], &current, &after, &joined(&[1, 2]), NOW + 50_000);
        assert_eq!(renewed[&nid].after, moved.after);
    }

    /// **Characters are lent by the plan's weights** (§4.4 of
    /// `docs/cooperative_admission.md`): a member the plan gives nothing, or
    /// does not name, is lent nothing, so its share moves to the others; with
    /// no plan, by the admitted powers.
    #[test]
    fn characters_are_lent_by_the_plan() {
        let ready = vec![
            Weighted {
                node: 1,
                weight: 300.0,
            },
            Weighted {
                node: 2,
                weight: 140.0,
            },
            Weighted {
                node: 3,
                weight: 50.0,
            },
        ];
        assert_eq!(by_plan(ready.clone(), None), ready, "no plan yet");
        let plan = Plan {
            epoch: 4,
            weights: vec![(1, 0), (2, 140)],
        };
        let lent = by_plan(ready, Some(&plan));
        assert_eq!(
            lent.iter().map(|w| (w.node, w.weight)).collect::<Vec<_>>(),
            vec![(1, 0.0), (2, 140.0), (3, 0.0)]
        );
        let cast: Vec<u64> = (1..=40).collect();
        let leases = lend(&cast, &BTreeMap::new(), &lent, &joined(&[1, 2, 3]), NOW);
        assert!(
            leases.values().all(|l| node_of(&l.holder) == Some(2)),
            "only the member the plan lends by holds any"
        );
    }

    /// **What `lend` changed is listed**: a character lent for the first time,
    /// one asked to move at its next boundary, and nothing for a lease renewed
    /// in place.
    #[test]
    fn what_lend_changed_is_listed() {
        let mut current = BTreeMap::new();
        current.insert(7, fresh(1, NOW));
        current.insert(8, fresh(1, NOW));
        let mut lent = current.clone();
        lent.insert(
            8,
            NpcLease {
                next: Some(hex(2)),
                ..fresh(1, NOW + 40_000)
            },
        );
        lent.insert(9, fresh(2, NOW));
        lent.insert(7, fresh(1, NOW + 40_000));
        assert_eq!(
            moves(&current, &lent),
            vec![
                Moved {
                    nid: 8,
                    from: hex(1),
                    holder: hex(1),
                    next: Some(hex(2)),
                },
                Moved {
                    nid: 9,
                    from: String::new(),
                    holder: hex(2),
                    next: None,
                },
            ]
        );
    }

    /// An admin's release is lent again at once; a character gone from the
    /// cast has no lease.
    #[test]
    fn a_released_lease_is_lent_again_and_a_gone_character_has_none() {
        let ms = members(&[1, 2]);
        let mut current = BTreeMap::new();
        current.insert(7, released());
        current.insert(8, released());
        let leases = lend(&[7], &current, &ms, &joined(&[1, 2]), NOW);
        assert_eq!(node_of(&leases[&7].holder), lent_to(7, &ms));
        assert!(!leases.contains_key(&8));
    }

    #[test]
    fn a_lease_record_is_its_holder_term_and_next() {
        let l = NpcLease {
            holder: hex(1),
            expires_ms: 5,
            next: None,
            after: None,
        };
        assert_eq!(
            serde_json::to_value(&l).unwrap(),
            serde_json::json!({"holder": "0000000000000001", "expires_ms": 5})
        );
        assert_eq!(lease_entity(42), "npclease/42");
    }

    /// A world's records carry its leases; reading them back, replacing them
    /// and asking which a member holds.
    #[test]
    fn leases_live_in_the_world_s_records() {
        let mut records = Records::new();
        records.insert("sim/meta".into(), serde_json::json!({}));
        let mut leases = BTreeMap::new();
        leases.insert(7, fresh(1, NOW));
        leases.insert(8, fresh(2, NOW));
        with_leases(&mut records, &leases);
        assert_eq!(leases_of(&records), leases);
        assert_eq!(held_by(&leases, 1), BTreeSet::from([7]));
        leases.remove(&8);
        with_leases(&mut records, &leases);
        assert!(!records.contains_key("npclease/8"));
        assert!(
            records.contains_key("sim/meta"),
            "the world's own records stay"
        );
    }
}
