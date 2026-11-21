//! Reads and writes a conversation's label, conv id and custom metadata as redo-log
//! records.
//!
//! Each record sets only its own field and keeps the rest, so a rename never
//! restates or rolls back a tag written meanwhile. A read-then-write of one key runs
//! as one substrate edit. A try that loses to a later identity starts again under a
//! higher one, up to a fixed limit, after which it stores nothing and says so.
//! Utility ingests use the custom bag as a content-addressed cache tag.

use super::*;

/// How many times a metadata edit starts again before it gives up: far past
/// any run of members writing one field at once.
pub(super) const EDIT_TRIES: usize = 64;

impl Conversation {
    /// The sidebar label for `timeline`, or `None` if no label has been
    /// recorded — either the conversation hasn't had its first user turn
    /// yet, or the recovered redo log carries no label for it.
    pub fn label_of(&self, timeline: TimelineId) -> Option<String> {
        self.read().label_of(timeline).map(|s| s.to_string())
    }

    /// The client-supplied `conv_id` string for `timeline`, or `None` if
    /// no submit has been recorded yet. Recovered from the redo log on
    /// substrate reload — drives the daemon's sidebar id field.
    pub fn conv_id_of(&self, timeline: TimelineId) -> Option<String> {
        self.read().conv_id_of(timeline).map(|s| s.to_string())
    }

    /// Persist a sidebar label for `timeline`, as a `Label` record that sets
    /// the label field alone: every other field keeps whatever record set it
    /// (§6.5), so a rename never restates — and never rolls back — a tag
    /// another write set meanwhile.
    pub fn set_conversation_label(&self, timeline: TimelineId, label: &str) -> candle::Result<()> {
        if label.is_empty() {
            return Ok(());
        }
        let payload = manifest::encode_label_payload(timeline.raw(), "", label, &BTreeMap::new());
        let identity = self.minter.mint(0);
        self.update(|s| s.set_label(timeline, label, identity));
        self.persistence.enqueue(WriteJob::ConvMeta {
            record: RecordType::Label,
            payload,
            identity,
        });
        Ok(())
    }

    /// Merge a single `(key, value)` into `timeline`'s `custom` metadata
    /// bag and persist the full ConvMeta. Used as a content-addressed
    /// cache tag by utility ingests (code_read / repo_map). Reads the
    /// sibling fields (conv_id, label, existing custom) so the Label
    /// record stays complete.
    pub fn set_conversation_metadata(
        &self,
        timeline: TimelineId,
        key: &str,
        value: &str,
    ) -> candle::Result<()> {
        if key.is_empty() {
            return Ok(());
        }
        let mut one = std::collections::BTreeMap::new();
        one.insert(key.to_string(), value.to_string());
        self.set_conversation_metadata_many(timeline, &one)
    }

    /// Merge several `(key, value)` pairs into `timeline`'s `custom`
    /// metadata in a single persisted Label record (cheaper than one
    /// `set_conversation_metadata` per key when tagging a conversation
    /// with several fields at once). The record sets these keys alone; every
    /// other field keeps the record that set it (§6.5).
    pub fn set_conversation_metadata_many(
        &self,
        timeline: TimelineId,
        kv: &std::collections::BTreeMap<String, String>,
    ) -> candle::Result<()> {
        if kv.is_empty() {
            return Ok(());
        }
        // Encode on this thread, append on the owner's. The record is small,
        // so it carries the same crash-loss window as every other enqueued
        // metadata record: the in-RAM merge below is what this session reads,
        // and a record that never lands leaves the tag absent on the next load
        // rather than wrong.
        let payload = manifest::encode_label_payload(timeline.raw(), "", "", kv);
        let identity = self.minter.mint(0);
        self.update(|s| s.merge_custom(timeline, kv, identity));
        self.persistence.enqueue(WriteJob::ConvMeta {
            record: RecordType::Label,
            payload,
            identity,
        });
        Ok(())
    }

    /// Rewrite `key` of `timeline`'s `custom` metadata from the value it holds,
    /// as one step: `next` is handed the substrate and the current value and
    /// returns the one to store, or `None` to leave it as it is — a decision
    /// it makes from anything else the substrate holds is part of the same
    /// step. Returns what was stored.
    ///
    /// A plain read then [`Self::set_conversation_metadata`] loses an update
    /// made between them — two uploads each reading the list and writing it
    /// back with one file added keep one file. Here the read and the write are
    /// one substrate edit, and the write's identity is minted before the edit,
    /// so an edit that read a value set by a later identity than its own would
    /// lose to it field by field: it starts again under a fresh identity. An
    /// identity that writes nothing is marked never-coming, so no peer waits
    /// on it. `next` may run more than once and must not act on anything.
    ///
    /// **Every new start gets further.** The identity it lost to is taken into
    /// the minter, so the next one outranks it. A field this member wrote cut
    /// off holds the provisional generation, which no identity outranks once
    /// the member has rejoined, until the rejoin's reload: it is settled under
    /// the generation minted now. Trying again alone never returned, and said
    /// nothing. Past [`EDIT_TRIES`] the edit gives up, loudly, and stores nothing.
    ///
    /// **A try that cannot store publishes nothing**, so it is looked at on the
    /// latest version first: every published version sends every other writer
    /// round again, and one edit trying without pause starved them all.
    pub fn update_conversation_metadata(
        &self,
        timeline: TimelineId,
        key: &str,
        mut next: impl FnMut(&Substrate, Option<&str>) -> Option<String>,
    ) -> Option<String> {
        enum Edit {
            Stored(String),
            Unchanged,
            Overtaken(Option<RecordIdentity>),
        }
        for _ in 0..EDIT_TRIES {
            let identity = self.minter.mint(0);
            let latest = self.read();
            let edit = if latest.custom_field_takes(timeline, key, identity) {
                drop(latest);
                self.update(|s| {
                    if !s.custom_field_takes(timeline, key, identity) {
                        return Edit::Overtaken(s.custom_field_id(timeline, key));
                    }
                    let current = s.custom_of(timeline).and_then(|m| m.get(key).cloned());
                    match next(s, current.as_deref()) {
                        Some(value) => {
                            let kv = BTreeMap::from([(key.to_string(), value.clone())]);
                            s.merge_custom(timeline, &kv, identity);
                            Edit::Stored(value)
                        }
                        None => Edit::Unchanged,
                    }
                })
            } else {
                Edit::Overtaken(latest.custom_field_id(timeline, key))
            };
            match edit {
                Edit::Stored(value) => {
                    let kv = BTreeMap::from([(key.to_string(), value.clone())]);
                    self.persistence.enqueue(WriteJob::ConvMeta {
                        record: RecordType::Label,
                        payload: manifest::encode_label_payload(timeline.raw(), "", "", &kv),
                        identity,
                    });
                    return Some(value);
                }
                Edit::Unchanged => {
                    self.persistence.enqueue(WriteJob::Abandoned { identity });
                    return None;
                }
                Edit::Overtaken(held) => {
                    self.persistence.enqueue(WriteJob::Abandoned { identity });
                    let Some(held) = held else { continue };
                    self.minter.observe(&held);
                    if held.gen == GEN_DETACHED && identity.gen != GEN_DETACHED {
                        self.update(|s| s.settle_custom_field(timeline, key, identity.gen));
                    }
                }
            }
        }
        tracing::error!(
            timeline = timeline.raw(),
            key,
            tries = EDIT_TRIES,
            "a metadata edit could not outrank the value it edits: nothing was stored"
        );
        None
    }

    /// `timeline`'s `custom` metadata bag, or `None` if unregistered.
    pub fn conversation_metadata(
        &self,
        timeline: TimelineId,
    ) -> Option<std::collections::BTreeMap<String, String>> {
        self.read()
            .custom_of(timeline)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }

    /// One value of `timeline`'s `custom` metadata — `None` when it is
    /// unregistered or holds no `key` — without copying the rest of the bag,
    /// which can hold a whole journal.
    pub fn conversation_metadata_value(&self, timeline: TimelineId, key: &str) -> Option<String> {
        self.read().custom_of(timeline)?.get(key).cloned()
    }

    /// Whether `timeline`'s `custom` metadata holds `key` — false when it is
    /// unregistered.
    pub fn has_conversation_metadata_key(&self, timeline: TimelineId, key: &str) -> bool {
        self.read()
            .custom_of(timeline)
            .is_some_and(|m| m.contains_key(key))
    }

    /// `f` over `timeline`'s `custom` metadata, read in place under the
    /// snapshot — `None` when it is unregistered. For a reader of several keys.
    pub fn conversation_metadata_with<T>(
        &self,
        timeline: TimelineId,
        f: impl FnOnce(&POrdMap<String, String>) -> T,
    ) -> Option<T> {
        self.read().custom_of(timeline).map(f)
    }

    /// The live timelines whose `custom[key]` is each of `values`, in one
    /// pass of the snapshot. A value no live timeline carries is absent.
    pub fn find_timelines_by_metadata_values(
        &self,
        key: &str,
        values: &BTreeSet<String>,
    ) -> BTreeMap<String, Vec<TimelineId>> {
        self.read().timelines_with_metadata_values(key, values)
    }

    /// Whether a live timeline whose K/V is intact carries `key == value`,
    /// stopping at the first.
    pub fn has_live_intact_metadata(&self, key: &str, value: &str) -> bool {
        self.read().has_live_intact_metadata(key, value)
    }

    /// Every live timeline whose `custom` metadata contains `key == value`.
    /// The reload-time cache lookup utility ingests use to skip units
    /// already present in the substrate. Tombstoned timelines are excluded.
    pub fn find_timelines_by_metadata(&self, key: &str, value: &str) -> Vec<TimelineId> {
        self.read().timelines_with_metadata(key, value)
    }

    /// [`Self::find_timelines_by_metadata`] plus tombstoned timelines that are
    /// DISTILLED — see
    /// [`crate::substrate::Substrate::timelines_with_metadata_including_distilled`].
    /// For the calibration resume filter; ordinary tombstones stay excluded.
    pub fn find_timelines_by_metadata_including_distilled(
        &self,
        key: &str,
        value: &str,
    ) -> Vec<TimelineId> {
        self.read()
            .timelines_with_metadata_including_distilled(key, value)
    }

    /// Distinct `custom[key]` values across live timelines — a one-pass
    /// snapshot for O(1) membership probing (e.g. the resume cache).
    pub fn metadata_values_for_key(&self, key: &str) -> std::collections::HashSet<String> {
        self.read().metadata_values_for_key(key)
    }

    /// Live timelines carrying `key`, paired with that key's value.
    /// Drives ingest reconciliation (tombstone units whose source is gone).
    pub fn timelines_with_metadata_key(&self, key: &str) -> Vec<(TimelineId, String)> {
        self.read().timelines_with_metadata_key(key)
    }

    /// Persist the client-supplied `conv_id` for `timeline`. Idempotent: the
    /// typical caller is the daemon's chat handler, invoking this on every
    /// submit so the conv_id reaches the redo log immediately (well before the
    /// titler completes), and a conv_id the substrate already holds writes
    /// nothing. The record sets the conv_id alone; the label and tags keep the
    /// records that set them (§6.5).
    pub fn set_conversation_conv_id(
        &self,
        timeline: TimelineId,
        conv_id: &str,
    ) -> candle::Result<()> {
        if conv_id.is_empty() || self.conv_id_of(timeline).as_deref() == Some(conv_id) {
            return Ok(());
        }
        let payload = manifest::encode_label_payload(timeline.raw(), conv_id, "", &BTreeMap::new());
        let identity = self.minter.mint(0);
        self.update(|s| s.set_conv_id(timeline, conv_id, identity));
        self.persistence.enqueue(WriteJob::ConvMeta {
            record: RecordType::Label,
            payload,
            identity,
        });
        Ok(())
    }

    /// Every conversation the workspace substrate knows about —
    /// `(timeline, conv_id, label, archived, order)` tuples drawn from the
    /// in-RAM `Substrate::timelines` map (`order` = creation-order rank).
    /// Drives `GET /v1/conversations` directly; no sidecar involved.
    pub fn known_conversations(&self) -> Vec<(TimelineId, String, String, bool, u64)> {
        self.read().known_conversations()
    }

    /// Live conversations whose `conv_id` starts with `prefix` — see
    /// [`crate::substrate::Substrate::conversations_with_conv_id_prefix`].
    pub fn conversations_with_conv_id_prefix(&self, prefix: &str) -> Vec<(TimelineId, String)> {
        self.read().conversations_with_conv_id_prefix(prefix)
    }

    /// Every live timeline, named or not — see
    /// [`crate::substrate::Substrate::live_timeline_ids`].
    pub fn live_timeline_ids(&self) -> Vec<TimelineId> {
        self.read().live_timeline_ids()
    }
}
