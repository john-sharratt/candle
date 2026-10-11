//! The substrate's replicated state, rendered canonically.
//!
//! Two substrates that have applied the same records must hold the same state,
//! whatever order the records arrived in (`docs/replicated_substrate.md` §6.9).
//! This renders everything that promise covers — every keyed value, the
//! identity of the record behind it, every set, every held record — as one
//! string in a fixed order, so the promise is checked by comparing two strings.
//!
//! Where a record *lives* is left out. Each member gives the records it holds
//! its own locations, so two equal substrates differ in every offset; what they
//! must agree on is which record holds each value, and that is its identity.
//!
//! So is everything a timeline tombstone has killed. A tombstoned timeline that
//! is not distilled is dead at read — no projection, listing or rewrite ever
//! sees its records again — and a member may hold them or may have refused
//! them on arrival (§6.5), so what remains of it is its tombstone.

use std::collections::BTreeSet;
use std::fmt::Write;

use super::Substrate;
use crate::persistence::content_hash::hash_bytes;
use crate::persistence::identity::RecordIdentity;
use crate::persistence::streams::StreamDecl;
use crate::projection::TimelineId;

/// `origin/oseq@gen.epoch.hlc`.
fn id(i: &RecordIdentity) -> String {
    format!("{:x}/{}@{}.{}.{}", i.origin, i.oseq, i.gen, i.epoch, i.hlc)
}

impl Substrate {
    /// The canonical rendering — see the module docs.
    pub fn replicated_state(&self) -> String {
        let mut out = String::new();
        self.render_streams(&mut out);
        self.render_timelines(&mut out);
        self.render_recurrent(&mut out);
        self.render_objects(&mut out);
        self.render_markers(&mut out);
        out
    }

    /// Whether `timeline` is dead: tombstoned and not distilled.
    fn is_dead(&self, timeline: u64) -> bool {
        TimelineId::from_raw(timeline).is_some_and(|tl| {
            self.tombstoned_timelines.contains(&tl) && !self.distilled_timelines.contains_key(&tl)
        })
    }

    fn render_streams(&self, out: &mut String) {
        let mut streams: Vec<_> = self
            .streams
            .iter_physical()
            .filter(|(_, s)| match &s.decl {
                Some(StreamDecl::Turn(t)) => !self.is_dead(t.timeline_id),
                _ => true,
            })
            .collect();
        streams.sort_unstable_by_key(|(key, _)| *key);
        for (key, s) in streams {
            let ids = &s.ids;
            let _ = writeln!(
                out,
                "stream {:016x} writer {:x}",
                key.stream_id.0, key.writer
            );
            if let Some(decl) = &s.decl {
                let _ = writeln!(out, "  decl {} {:?}", id(&ids.decl), decl);
            }
            for (idx, loc) in &s.chunks {
                let cid = ids.chunks.get(idx).copied().unwrap_or_default();
                let _ = writeln!(
                    out,
                    "  chunk {idx} {} len={} tokens={} format={}",
                    id(&cid),
                    loc.payload_len,
                    loc.token_count,
                    loc.format
                );
            }
            if let Some(loc) = &s.tokens {
                let _ = writeln!(out, "  tokens {} len={}", id(&ids.tokens), loc.payload_len);
            }
            if let Some(p) = &s.projection_events {
                let _ = writeln!(
                    out,
                    "  events {} {:?}",
                    id(&ids.projection_events),
                    hash_bytes(p)
                );
            }
            if let Some(p) = &s.wide_q_sigs {
                let _ = writeln!(out, "  sigs {} {:?}", id(&ids.wide_q_sigs), hash_bytes(p));
            }
            if let Some(p) = &s.index_page {
                let _ = writeln!(
                    out,
                    "  index_page {} {:?}",
                    id(&ids.index_page),
                    hash_bytes(p.bytes())
                );
            }
            if let Some(through) = s.committed_through {
                let _ = writeln!(out, "  committed {through} {}", id(&ids.commit));
            }
        }
        // Each timeline's turn order: the position every declared turn holds,
        // and whose stream holds it.
        let mut timelines: Vec<u64> = self
            .streams
            .timelines()
            .filter(|tl| !self.is_dead(*tl))
            .collect();
        timelines.sort_unstable();
        for tl in timelines {
            let _ = writeln!(out, "turns {tl:016x}");
            for (pos, key) in self.streams.turns_of(tl) {
                let _ = writeln!(
                    out,
                    "  {pos} {:016x} writer {:x}",
                    key.stream_id.0, key.writer
                );
            }
        }
    }

    fn render_timelines(&self, out: &mut String) {
        let mut tls: Vec<_> = self
            .timelines
            .keys()
            .copied()
            .filter(|t| !self.is_dead(t.raw()))
            .collect();
        tls.sort_unstable_by_key(|t| t.raw());
        for tl in tls {
            let e = &self.timelines[&tl];
            let ids = &e.ids;
            let _ = writeln!(
                out,
                "timeline {:016x} {:?} {:?} order={}",
                tl.raw(),
                e.layer,
                e.group,
                e.order
            );
            if let Some(c) = &e.conv_id {
                let _ = writeln!(out, "  conv_id {} {c}", id(&ids.conv_id));
            }
            if let Some(named) = &ids.named {
                let _ = writeln!(out, "  named {} {}", id(named), ids.named_conv_id);
            }
            if let Some(l) = &e.label {
                let _ = writeln!(out, "  label {} {l}", id(&ids.label));
            }
            for (k, v) in &e.custom {
                let kid = ids.custom.get(k).copied().unwrap_or_default();
                let _ = writeln!(out, "  custom {} {k}={v}", id(&kid));
            }
            let _ = writeln!(
                out,
                "  state {} archived={} active={} branches={:?}",
                id(&ids.state),
                e.archived,
                e.active,
                e.branches
            );
            if let Some(d) = &e.debug_id {
                let _ = writeln!(out, "  debug_id {} {d}", id(&ids.debug_id));
            }
            for (idx, meta) in &e.tree_meta {
                let mid = ids.tree_meta.get(idx).copied().unwrap_or_default();
                let _ = writeln!(out, "  tree {} {} {:?}", idx.0, id(&mid), meta);
            }
            let couplings: BTreeSet<u32> = e.couplings.iter().copied().collect();
            for from in couplings {
                let cid = ids.couplings.get(&from).copied().unwrap_or_default();
                let _ = writeln!(out, "  coupling {from} {}", id(&cid));
            }
        }
        for (tl, rt, payload, rid) in self.pending.records() {
            if self.is_dead(tl) {
                continue;
            }
            let _ = writeln!(
                out,
                "pending {tl:016x} {rt:?} {} {:?}",
                id(&rid),
                hash_bytes(&payload)
            );
        }
    }

    fn render_recurrent(&self, out: &mut String) {
        let mut sids: Vec<_> = self.snapshot_ids.keys().copied().collect();
        sids.sort_unstable_by_key(|s| s.0);
        for sid in sids {
            let sid_id = self.snapshot_ids[&sid];
            match self.recurrent_snapshots.get(&sid) {
                Some(loc) => {
                    let _ = writeln!(
                        out,
                        "snapshot {:016x} {} len={}",
                        sid.0,
                        id(&sid_id),
                        loc.payload_len
                    );
                }
                None => {
                    let _ = writeln!(out, "snapshot {:016x} {} retracted", sid.0, id(&sid_id));
                }
            }
        }
        let mut sids: Vec<_> = self.branch_checkpoints.keys().copied().collect();
        sids.sort_unstable_by_key(|s| s.0);
        for sid in sids {
            let _ = writeln!(
                out,
                "branch {:016x} {} len={}",
                sid.0,
                id(&self.branch_checkpoint_id(sid)),
                self.branch_checkpoints[&sid].payload_len
            );
        }
    }

    fn render_objects(&self, out: &mut String) {
        let mut keys: Vec<u64> = self.custom_objects.keys().copied().collect();
        keys.sort_unstable();
        for key in keys {
            let obj = &self.custom_objects[&key];
            let _ = writeln!(
                out,
                "object {key:016x} {} {:?}",
                id(&self.custom_object_id(key)),
                hash_bytes(&obj.encode())
            );
        }
    }

    fn render_markers(&self, out: &mut String) {
        let mut tombstoned: Vec<u64> = self.tombstoned_timelines.iter().map(|t| t.raw()).collect();
        tombstoned.sort_unstable();
        let mut turns: Vec<(u64, u32, u64)> = self
            .tombstoned_turns
            .iter()
            .map(|m| (m.timeline.raw(), m.turn_index, m.writer))
            .collect();
        turns.sort_unstable();
        let sections: Vec<(u64, u64, u64)> = self
            .ended_sections
            .generations()
            .map(|(s, g)| (s.0, g.writer, g.oseq))
            .collect();
        let unstamped: Vec<u64> = self.ended_sections.unstamped().map(|s| s.0).collect();
        let mut distilled: Vec<(u64, String)> = self
            .distilled_timelines
            .iter()
            .map(|(t, m)| (t.raw(), format!("{m:?}")))
            .collect();
        distilled.sort_unstable();
        let _ = writeln!(out, "tombstoned {tombstoned:?}");
        let _ = writeln!(out, "tombstoned_turns {turns:?}");
        let _ = writeln!(out, "ended_sections {sections:?} unstamped {unstamped:?}");
        let _ = writeln!(out, "distilled {distilled:?}");
    }
}
