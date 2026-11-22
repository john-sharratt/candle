//! Assembles the gallery of past turns that a section collection votes with, and
//! preloads it onto the GPU arena.
//!
//! Scope is a tag partition: a tagged collection admits only turns carrying its
//! tags, and an untagged one admits only dialogue, so calibration corpus and live
//! conversation never bleed into each other. A tombstoned turn is a dead
//! placeholder and must not vote. Galleries are cached against the substrate's
//! gallery epoch, so an assembly reruns only when a stream, blob or tombstone
//! changed. The preload loads what every conversation reads before the first
//! projection, so that projection uploads nothing.

use super::*;

/// A collection's belief gallery: each member's stored wide-Q window, the slot
/// it votes for, and its stream id (the window's arena residency key).
pub type BeliefGallery = (Vec<SigWindow>, Vec<usize>, Vec<StreamId>);

/// What [`Conversation::preload_galleries`] loaded.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GalleryPreload {
    /// Turns walked across the scanned groups and the collections.
    pub turns: usize,
    /// Turns resident on the arena, and their signature tokens.
    pub resident: usize,
    pub tokens: usize,
    /// The card is too small to preload on ([`GalleryArena::preloads`]):
    /// nothing was walked, and the scans upload what they read.
    pub skipped: bool,
}

/// Which gallery an assembly is for: the collection's name, its scope tags, and
/// its members in slot order. The members are part of the key because the
/// slots are: builders for different tool modes or identities carry different
/// members under the same collection name, and a gallery assembled for one
/// names slots that do not exist, or mean another member, in the other.
pub(super) type GalleryKey = (String, Vec<String>, Vec<String>);

impl Conversation {
    /// Assemble the belief gallery for a section `collection`: every in-scope
    /// turn's wide-Q window mapped to the belief slot it stands for.
    ///
    /// Scope is a **tag partition**, so a tagged corpus and untagged live
    /// conversation never bleed into each other:
    /// - a **tagged** policy (`tags` non-empty, e.g. `["tool"]`) admits only
    ///   turns carrying one of those tags — its fixed calibration corpus;
    /// - an **untagged** policy (`tags` empty) admits only *untagged* turns —
    ///   live conversation, the self-reinforcing case where a past turn that
    ///   scored high gets pulled back in (tagged calibration turns are excluded
    ///   so they don't leak into general memory).
    ///
    /// Label: a turn's tag that names a slot wins (so tagged calibration turns
    /// are labelled directly — no cold-start bootstrap); otherwise the turn is
    /// labelled by the section its own last projection selected in `collection`
    /// (self-reinforcing). Turns that resolve to no slot are skipped. Returns
    /// `(windows, slot_per_window, stream_per_window)` — the windows and slots
    /// feed [`crate::provenance::score_slots`]; the stream ids key the windows'
    /// residency in the GPU gallery arena.
    ///
    /// Reused while [`Substrate::gallery_epoch`] is unchanged — see
    /// `gallery_cache`. The epoch is read from the same version the assembly
    /// reads, so a cached gallery is always the one that version would
    /// assemble.
    pub fn belief_gallery(
        &self,
        collection: &str,
        tags: &[String],
        members: &[String],
    ) -> BeliefGallery {
        let slot_of = |name: &str| members.iter().position(|m| m == name);
        let sub = self.inner.read();
        let epoch = sub.gallery_epoch();
        let key: GalleryKey = (collection.to_string(), tags.to_vec(), members.to_vec());
        if let Some(held) = self.gallery_cache.pin().get(&key) {
            if held.0 == epoch {
                return held.1.clone();
            }
        }
        let mut windows: Vec<SigWindow> = Vec::new();
        let mut slots: Vec<usize> = Vec::new();
        let mut sids: Vec<StreamId> = Vec::new();
        // A tagged collection admits only turns carrying its tags, and an
        // untagged one only dialogue — the turns carrying none — and the
        // substrate indexes both, so neither walks the turns it would refuse.
        let streams: Box<dyn Iterator<Item = TurnStream<'_>>> = if tags.is_empty() {
            Box::new(sub.dialogue_turn_streams())
        } else {
            Box::new(sub.tagged_turn_streams(tags))
        };
        for turn in streams {
            let (sid, d) = (turn.stream_id, turn.decl);
            // A tombstoned turn (or timeline) must not keep voting in
            // retrieval: its walker-replayed sig blob still exists, but the
            // turn itself is a dead placeholder whose KV can never
            // materialise — selecting it would point projection at content
            // that cannot be elevated.
            if let Some(tl) = TimelineId::from_raw(d.timeline_id) {
                if sub.is_tombstoned(tl) || sub.is_turn_tombstoned(tl, turn.index.0) {
                    continue;
                }
            }
            // A working-set mark is not scope: a dialogue turn carrying only
            // marks is still an untagged turn.
            let in_scope = if tags.is_empty() {
                is_dialogue(&d.tags)
            } else {
                d.tags.iter().any(|t| tags.contains(t))
            };
            if !in_scope {
                continue;
            }
            // The stored window, not decoded: the scan decodes only the turns
            // it has to upload or score on the host. `None` covers absent,
            // empty and unreadable windows.
            let Some(window) = sub.wide_sig_window(sid) else {
                continue;
            };
            let slot = d.tags.iter().find_map(|t| slot_of(t)).or_else(|| {
                sub.last_selected_in(sid, collection)
                    .and_then(|name| slot_of(&name))
            });
            let Some(slot) = slot else {
                continue;
            };
            windows.push(window);
            slots.push(slot);
            sids.push(sid);
        }
        let gallery = (windows, slots, sids);
        // Two assemblies of one key race only to store equal galleries of one
        // epoch, or the later epoch's over the earlier — never the reverse,
        // which would serve a stale gallery as fresh: an older epoch does not
        // replace a newer one.
        let fresh = Arc::new((epoch, gallery.clone()));
        let _ = self.gallery_cache.pin().compute(key, |held| match held {
            Some((_, held)) if held.0 > epoch => papaya::Operation::Abort(()),
            _ => papaya::Operation::Insert(Arc::clone(&fresh)),
        });
        gallery
    }

    /// Load everything a projection's belief scan reads before it first runs:
    /// every scanned turn group's signature windows gathered (and their events
    /// digested), every section collection's gallery assembled, and all of them
    /// resident on `arena` under the fingerprints the scan itself takes — so
    /// the scan finds them all and uploads nothing. Each window is decoded only
    /// for its upload; nothing decoded outlives it.
    ///
    /// Without it the first projection after a start paid for all of that
    /// inline, ahead of its first token: measured at ~515 ms decoding and
    /// ~435 ms uploading a 1,000-file `code_reading` gallery, and ~115 ms
    /// assembling the tool catalog's. Every scope sees the same turns resident,
    /// so a group is preloaded whole rather than for one conversation's scope.
    ///
    /// **Only what every conversation reads.** A layer gathered per
    /// conversation (`gather_scope: conversation` — zend's dialogue) is read
    /// only by its own conversation, and only while it is loaded: its scans
    /// upload its own turns, and the scheduler gives them up when it unloads
    /// (`docs/gallery_residency.md`). Preloading every such conversation the
    /// substrate holds put 60,285 dialogue turns on BlackWorld's card for the
    /// handful of chats open.
    ///
    /// A failed upload is logged and skipped — the scan uploads what it does
    /// not find, so a preload that stops short costs that projection the
    /// difference and nothing else.
    ///
    /// **Not on a card under 64 GiB** (`gallery_arena::eviction::preloads_on`):
    /// there the turns it would load hold ground the expert zone streams
    /// without, and the first projections upload what they read instead.
    pub fn preload_galleries(
        &self,
        schema: &Schema,
        arena: Option<&GalleryArena>,
    ) -> GalleryPreload {
        if arena.is_some_and(|a| !a.preloads()) {
            return GalleryPreload {
                skipped: true,
                ..GalleryPreload::default()
            };
        }
        // Each turn with the fold groups its scans weigh — a file's content by
        // its group's `layer_weights`, its calibrations by the
        // `calibration_layer_weights`, a collection's gallery by the
        // collection's — the union where several read it, so the preload puts
        // on the card exactly what the scans will read
        // (`docs/gallery_residency.md` §4).
        // Held under the union of what its readers keep of it, too
        // (`docs/gallery_dedup.md` §3).
        let n_groups = arena.map_or(0, GalleryArena::n_groups);
        let mut order: Vec<StreamId> = Vec::new();
        let mut wanted: HashMap<StreamId, (SigWindow, GroupSet, Keep)> = HashMap::new();
        let mut want = |sid: StreamId, window: SigWindow, groups: GroupSet, keep: Keep| {
            wanted
                .entry(sid)
                .and_modify(|(_, g, k)| {
                    *g = g.union(groups);
                    *k = k.union(keep);
                })
                .or_insert_with(|| {
                    order.push(sid);
                    (window, groups, keep)
                });
        };
        let mut preload = GalleryPreload::default();
        {
            let sub = self.inner.read();
            let shared = schema
                .layers
                .iter()
                .filter(|l| l.gather_scope == GatherScope::Shared);
            for group in shared.flat_map(|l| l.groups.iter()) {
                if !group.is_scanned() {
                    continue;
                }
                let content = GroupSet::weighed(&group.policy.layer_weights, n_groups);
                let calibration =
                    GroupSet::weighed(&group.policy.calibration_layer_weights, n_groups);
                let timelines: Vec<TimelineId> = sub.timelines_for_group(group.id).collect();
                let (files, turns) = assemble_file_scans(&sub, timelines);
                preload.turns += turns;
                let scan = &group.policy.scan;
                for f in files {
                    let mut by_arc = vec![(GroupSet::default(), Keep::All); f.n_arcs()];
                    let mut read = |k: usize, groups: GroupSet, keep: Keep| {
                        let (g, kept) = by_arc[k];
                        // An arc no window reads is never preloaded; its keep
                        // is the first reader's, not a union with `All`.
                        let kept = if g.is_empty() { keep } else { kept.union(keep) };
                        by_arc[k] = (g.union(groups), kept);
                    };
                    for &(k, ..) in &f.windows {
                        read(k, content, scan.gallery_keep);
                    }
                    for &(k, ..) in &f.cal_windows {
                        read(k, calibration, scan.calibration_gallery_keep);
                    }
                    for (k, (groups, keep)) in by_arc.into_iter().enumerate() {
                        want(f.sid(k), f.arc(k).clone(), groups, keep);
                    }
                }
            }
        }
        for item in &schema.system_prompt.items {
            let SystemPromptItem::Collection(coll) = item else {
                continue;
            };
            let members: Vec<String> = coll.sections.iter().map(|s| s.name.clone()).collect();
            let (gallery, _, sids) = self.belief_gallery(&coll.name, &coll.policy.tags, &members);
            preload.turns += gallery.len();
            let groups = GroupSet::weighed(&coll.policy.layer_weights, n_groups);
            for (sid, window) in sids.into_iter().zip(gallery) {
                want(sid, window, groups, coll.policy.scan.gallery_keep);
            }
        }
        let Some(arena) = arena else {
            return preload;
        };
        let windows: Vec<(StreamId, SigWindow, GroupSet, Keep)> = order
            .into_iter()
            .filter_map(|sid| {
                let (window, groups, keep) = wanted.remove(&sid)?;
                Some((sid, window, groups, keep))
            })
            .filter(|(_, _, groups, _)| !groups.is_empty())
            .collect();
        let ensured =
            arena.ensure_all_resident(windows.iter().map(|(sid, window, groups, keep)| {
                (*sid, window.into(), window.fingerprint(), *groups, *keep)
            }));
        for ((sid, window, ..), done) in windows.iter().zip(ensured) {
            match done {
                Ok(()) => {
                    preload.resident += 1;
                    preload.tokens += window.len();
                }
                Err(e) => tracing::warn!(
                    target: "candle_conversation::provenance",
                    stream = sid.0,
                    "gallery preload: upload refused ({e}) — the first scan uploads this turn"
                ),
            }
        }
        preload
    }
}
