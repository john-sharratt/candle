//! A tenant's substrate as the cluster's replicator drives it
//! (`docs/replicated_substrate.md` §6–§7): the `cluster` crate's [`Store`]
//! over a [`Conversation`].
//!
//! Consensus decisions and `max_members` are custom objects under
//! `cluster/`, replicated like any keyed value (§5.9): a decision's bytes as a
//! JSON array, `max_members` as a number — a set under `max_members/set`, and
//! each member's latest raise under `max_members/raise/<node>`.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use applied_set::AppliedSet;
use candle_conversation::persistence::deleted::{DeletedEntry, DeletedKey, DeletedTimeline};
use candle_conversation::persistence::identity::VersionStamp;
use candle_conversation::persistence::record::{decode_header, RecordType};
use candle_conversation::persistence::remote::RemoteOutcome;
use candle_conversation::projection::replication::CLUSTER_PREFIX;
use candle_conversation::projection::Conversation;
use cluster::consensus::votes::{FileJournal, VoteStore, DEFAULT_SIZE};
use cluster::consensus::{Change, MaxMembers};
use cluster::frame::{Gone, RecordHeader};
use cluster::replicate::{Applied, Disk, Listing, Replication, Store, Tap, Tombstones};
use serde_json::{json, Value};

use super::disk::volume_of;

/// The file a member's acceptor keeps its votes in, beside its log (§5.9).
pub const VOTES: &str = "VOTES";

const GENERATION: &str = "gen/";
const MAX_MEMBERS_SET: &str = "max_members/set";
const MAX_MEMBERS_RAISE: &str = "max_members/raise/";

/// See the module documentation.
pub struct ConversationStore {
    conversation: Conversation,
    kv_layers: usize,
    reload: Box<dyn Fn() + Send + Sync>,
    /// Run after every batch applied and every reload: what the app checks
    /// against what the substrate now holds (`Self::on_applied`).
    applied_hook: OnceLock<Box<dyn Fn() + Send + Sync>>,
}

impl ConversationStore {
    /// `conversation`'s substrate, on a model with `kv_layers` K/V layers;
    /// `reload` rebuilds its in-memory state from the log and returns once
    /// done.
    pub fn new(
        conversation: Conversation,
        kv_layers: usize,
        reload: impl Fn() + Send + Sync + 'static,
    ) -> Arc<ConversationStore> {
        Arc::new(ConversationStore {
            conversation,
            kv_layers,
            reload: Box::new(reload),
            applied_hook: OnceLock::new(),
        })
    }

    /// Run `hook` after every batch of records from other members is applied
    /// and after every reload — the moments what this member holds can grow.
    /// The first hook set is the one kept.
    pub fn on_applied(&self, hook: impl Fn() + Send + Sync + 'static) {
        let _ = self.applied_hook.set(Box::new(hook));
    }

    fn applied_changed(&self) {
        if let Some(hook) = self.applied_hook.get() {
            hook();
        }
    }

    /// Replication of this substrate, with the acceptor's votes in its vote
    /// journal.
    pub fn replication(self: &Arc<Self>) -> anyhow::Result<Replication> {
        let votes = FileJournal::open(&self.votes_path(), DEFAULT_SIZE)?;
        Ok(Replication::open(
            self.clone(),
            Box::new(votes) as Box<dyn VoteStore>,
        )?)
    }

    fn votes_path(&self) -> PathBuf {
        self.conversation.substrate_dir().join(VOTES)
    }

    /// Every object under `cluster/`, by its key below the prefix.
    fn objects(&self) -> Vec<(String, Value, VersionStamp)> {
        self.conversation
            .cluster_objects()
            .into_iter()
            .map(|(k, v, s)| (k[CLUSTER_PREFIX.len()..].to_string(), v, s))
            .collect()
    }

    fn put(&self, key: &str, blob: Value) {
        if let Err(e) = self
            .conversation
            .put_cluster_object(&format!("{CLUSTER_PREFIX}{key}"), blob)
        {
            tracing::error!(key, "a cluster record was not written: {e}");
        }
    }
}

/// A deleted-set entry as a `HEADERS` gone entry carries it: a kind byte —
/// `0` a timeline's, `1` a keyed value's — then its words, little-endian:
/// `timeline`, `origin`, `oseq`; or `key`, `gen`, `epoch`, `hlc`, `origin`,
/// `oseq`.
pub fn encode_entry(e: &DeletedEntry) -> Vec<u8> {
    let (kind, words): (u8, Vec<u64>) = match e {
        DeletedEntry::Timeline(t) => (0, vec![t.timeline, t.origin, t.oseq]),
        DeletedEntry::Key(k) => (1, vec![k.key, k.gen, k.epoch, k.hlc, k.origin, k.oseq]),
    };
    let mut out = vec![kind];
    out.extend(words.iter().flat_map(|w| w.to_le_bytes()));
    out
}

pub fn decode_entry(b: &[u8]) -> Option<DeletedEntry> {
    let (&kind, rest) = b.split_first()?;
    let word = |i: usize| u64::from_le_bytes(rest[i * 8..i * 8 + 8].try_into().unwrap());
    match (kind, rest.len()) {
        (0, 24) => Some(DeletedEntry::Timeline(DeletedTimeline {
            timeline: word(0),
            origin: word(1),
            oseq: word(2),
        })),
        (1, 48) => Some(DeletedEntry::Key(DeletedKey {
            key: word(0),
            gen: word(1),
            epoch: word(2),
            hlc: word(3),
            origin: word(4),
            oseq: word(5),
        })),
        _ => None,
    }
}

impl Store for ConversationStore {
    fn node(&self) -> u64 {
        self.conversation.node()
    }

    fn own_through(&self) -> u64 {
        self.conversation.applied_set().through(self.node())
    }

    fn set_tap(&self, tap: Tap) {
        self.conversation.set_replication_tap(tap);
    }

    fn read(&self, origin: u64, oseq: u64) -> Option<Vec<u8>> {
        self.conversation.read_event(origin, oseq)
    }

    fn read_many(&self, origin: u64, oseqs: &[u64]) -> Vec<Option<Vec<u8>>> {
        self.conversation.read_events(origin, oseqs)
    }

    fn applied(&self) -> AppliedSet {
        self.conversation.applied_set()
    }

    fn apply(&self, records: &[(u64, u64, Vec<u8>)]) -> Applied {
        // A record whose header names another identity than it travelled under
        // is refused: the applied set is kept by header, and the requester
        // counts what it asked for by the frame's.
        let honest: Vec<bool> = records
            .iter()
            .map(|(origin, oseq, bytes)| {
                decode_header(bytes).is_ok_and(|(h, _)| {
                    let id = h.identity();
                    (id.origin, id.oseq) == (*origin, *oseq)
                })
            })
            .collect();
        let batch: Vec<&[u8]> = records
            .iter()
            .zip(&honest)
            .filter(|(_, ok)| **ok)
            .map(|((_, _, b), _)| b.as_slice())
            .collect();
        let result = self.conversation.apply_remote(&batch, self.kv_layers);
        let (mut outcomes, reload) = match result {
            Ok(b) => (b.outcomes.into_iter(), b.reload),
            Err(e) => {
                tracing::error!("applying records from other members failed: {e}");
                return Applied {
                    taken: vec![false; records.len()],
                    reload: false,
                };
            }
        };
        // A refused record is fetched again, so one refused for good is fetched
        // for ever — a catch-up that receives and never holds. Say which, and
        // why: the refusal is the only place the reason exists.
        let mut refused = Vec::new();
        let taken = honest
            .iter()
            .zip(records)
            .map(|(ok, (origin, oseq, bytes))| {
                if !*ok {
                    refused.push(format!(
                        "{origin:x}/{oseq} ({} B): its header names another identity",
                        bytes.len()
                    ));
                    return false;
                }
                match outcomes.next() {
                    Some(RemoteOutcome::Refused(why)) => {
                        refused.push(format!("{origin:x}/{oseq} ({} B): {why:?}", bytes.len()));
                        false
                    }
                    Some(_) => true,
                    None => false,
                }
            })
            .collect();
        if !refused.is_empty() {
            tracing::warn!(
                refused = refused.len(),
                of = records.len(),
                first = %refused.iter().take(4).cloned().collect::<Vec<_>>().join("; "),
                "records from other members refused — each is fetched again",
            );
        }
        self.applied_changed();
        Applied { taken, reload }
    }

    fn list(&self, origin: u64, from: u64, to: u64) -> Listing {
        let held = self.conversation.list_held(origin, from, to);
        Listing {
            records: held
                .records
                .into_iter()
                .map(|r| RecordHeader {
                    oseq: r.oseq,
                    record_type: r.record_type,
                    stream_id: r.stream_id,
                    chunk_index: r.chunk_index,
                    len: r.len,
                    digest: r.digest,
                })
                .collect(),
            gone: held
                .gone
                .into_iter()
                .map(|(oseq, e)| Gone {
                    oseq,
                    deleted_entry: e.as_ref().map(encode_entry),
                })
                .collect(),
        }
    }

    fn gone(&self, origin: u64, gone: &[Gone]) {
        let gone: Vec<(u64, Option<DeletedEntry>)> = gone
            .iter()
            .map(|g| (g.oseq, g.deleted_entry.as_deref().and_then(decode_entry)))
            .collect();
        if let Err(e) = self.conversation.record_gone(origin, &gone) {
            tracing::error!(
                origin,
                "what a catch-up learned is gone was not recorded: {e}"
            );
        }
    }

    fn travels(&self, record_type: u8) -> bool {
        !RecordType::from_tag(record_type).is_local_only()
    }

    fn ticket(&self) -> u64 {
        self.conversation.write_ticket()
    }

    fn wait_written(&self, ticket: u64) {
        self.conversation.wait_written(ticket);
    }

    fn detach(&self) {
        self.conversation.detach();
    }

    fn detached(&self) -> bool {
        self.conversation.holds_detached()
    }

    fn rejoin(&self, gen: u64) {
        if let Err(e) = self.conversation.rejoin_at(gen) {
            tracing::error!(gen, "the rejoin's generation was not recorded: {e}");
        }
    }

    fn generation(&self) -> u64 {
        let decided = self
            .objects()
            .iter()
            .filter_map(|(k, _, _)| k.strip_prefix(GENERATION)?.parse::<u64>().ok())
            .max()
            .unwrap_or(0);
        self.conversation.generation().max(decided)
    }

    fn record_decision(&self, key: &[u8], value: &[u8]) {
        self.put(&String::from_utf8_lossy(key), json!(value));
    }

    fn decision(&self, key: &[u8]) -> Option<Vec<u8>> {
        let key = String::from_utf8_lossy(key);
        self.objects()
            .into_iter()
            .find(|(k, _, _)| *k == key)
            .and_then(|(_, v, _)| serde_json::from_value(v).ok())
    }

    fn max_members(&self) -> u32 {
        let mut m = MaxMembers::<VersionStamp>::new();
        for (key, value, stamp) in self.objects() {
            let Some(n) = value.as_u64() else { continue };
            if key == MAX_MEMBERS_SET {
                m.apply(stamp, Change::Set(n as u32));
            } else if key.starts_with(MAX_MEMBERS_RAISE) {
                m.apply(stamp, Change::Raise(n as u32));
            }
        }
        m.value()
    }

    fn raise_max_members(&self, seen: u32) {
        self.put(
            &format!("{MAX_MEMBERS_RAISE}{:016x}", self.node()),
            json!(seen),
        );
    }

    fn set_max_members(&self, n: u32) {
        self.put(MAX_MEMBERS_SET, json!(n));
    }

    fn reload(&self) {
        (self.reload)();
        self.applied_changed();
    }

    fn disk(&self) -> Disk {
        let dir = self.conversation.substrate_dir();
        let (free, total) = volume_of(&dir).unwrap_or_else(|| {
            tracing::error!(
                "the substrate's volume is on no measured mount dir={}",
                dir.display()
            );
            (0, 0)
        });
        Disk {
            free,
            total,
            write_failed: self.conversation.out_of_space(),
        }
    }

    fn test_write(&self) -> bool {
        match self.conversation.test_write() {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!("the test write failed: {e}");
                false
            }
        }
    }

    fn drop_dead_segment(&self) -> bool {
        self.conversation.drop_dead_segment().unwrap_or_else(|e| {
            tracing::error!("dropping a dead segment failed: {e}");
            false
        })
    }

    fn deferred(&self) -> Vec<(String, u64, usize)> {
        self.conversation
            .deferred_records()
            .into_iter()
            .map(|((record_type, origin), n)| (record_type, origin, n))
            .collect()
    }

    fn census(&self) -> Vec<u8> {
        self.conversation.held_record_tags()
    }

    fn decisions(&self, prefix: &str) -> Vec<(String, Vec<u8>)> {
        self.objects()
            .into_iter()
            .filter(|(k, _, _)| k.starts_with(prefix))
            .filter_map(|(k, v, _)| Some((k, serde_json::from_value(v).ok()?)))
            .collect()
    }

    fn tombstones(&self) -> Tombstones {
        let t = self.conversation.tombstone_counts();
        Tombstones {
            held: t.held,
            swept: t.swept,
            deleted: t.deleted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timelines_entry_is_a_kind_byte_and_24_bytes() {
        let e = DeletedEntry::Timeline(DeletedTimeline {
            timeline: 0x0102,
            origin: 0xA,
            oseq: 7,
        });
        let mut want = vec![0, 0x02, 0x01, 0, 0, 0, 0, 0, 0];
        want.extend([0x0A, 0, 0, 0, 0, 0, 0, 0]);
        want.extend([7, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(encode_entry(&e), want);
        assert_eq!(decode_entry(&want), Some(e));
        assert_eq!(decode_entry(&want[..24]), None);
    }

    #[test]
    fn a_keyed_entry_is_a_kind_byte_and_48_bytes() {
        let e = DeletedEntry::Key(DeletedKey {
            key: 0x55,
            gen: 1,
            epoch: 2,
            hlc: 3,
            origin: 0xB,
            oseq: 9,
        });
        let mut want = vec![1];
        for w in [0x55u8, 1, 2, 3, 0x0B, 9] {
            want.extend([w, 0, 0, 0, 0, 0, 0, 0]);
        }
        assert_eq!(encode_entry(&e), want);
        assert_eq!(decode_entry(&want), Some(e));
        assert_eq!(decode_entry(&[2, 0]), None, "an unknown kind");
        assert_eq!(decode_entry(&[]), None);
    }
}
