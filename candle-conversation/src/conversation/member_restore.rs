//! Restoring the knowledge base's members at boot, from their records alone
//! (`docs/provenance_system_collection.md` §12 D15).
//!
//! A record names each member by its content address and token count. A
//! member's address is its tokens and what it is sealed behind — the prompt
//! before the collection, the hole of its range, and the members before it in
//! its file — so the address the current prompt gives each member is worked
//! out from the record without the file's text: the record is restored only
//! when it matches, member by member. Each member's tokens are read from its
//! own stream and checked against the address before the member is installed.
//!
//! Members are restored rank by rank across every file: every file's first
//! member, then every second, because a member's address names the members
//! before it, which must already be installed. Each rank goes to the scheduler
//! in batches, not a request per member. A file restored whole is named and
//! placeable at once, and the ingest's pass leaves it alone; a file any member
//! of which does not restore is released whole, and the pass seals it as it
//! seals a new one.

use super::prefix_state::chain_of;
use super::{MembersPrefix, Sequence};
use crate::error::ConversationError;
use crate::files::files_owner;
use crate::persistence::content_hash::{hash_tokens, section_stream_id, shared_section_prefix};
use crate::persistence::resume::decode_token_ids;
use crate::persistence::streams::{ContentAddress, StreamId};
use crate::projection::SectionId;
use crate::scheduler::{RestoreItem, SchedulerRequest};
use crate::token_buffer::TokenBuffer;

/// The members a scheduler request restores at most: a rank of the whole
/// knowledge base is ten thousand members, which one request would hold the
/// wave thread for.
const RESTORE_BATCH: usize = 256;

/// One member as its file's record names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedMember {
    /// The name it registers under (after `shared/files/`).
    pub name: String,
    pub address: ContentAddress,
    pub tokens: usize,
}

/// A file's members as its record names them, in line order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedChain {
    pub members: Vec<RecordedMember>,
}

/// One chain's walk: the members it has installed, and whether it is still
/// restoring.
struct Walk {
    installed: Vec<SectionId>,
    through: usize,
    alive: bool,
}

impl Sequence {
    /// Restore every chain of `collection` in `chains` whose members are all
    /// persisted, whole and at the addresses the current prompt gives them,
    /// and name each one restored whole. Returns, in order, whether each chain
    /// was.
    pub fn restore_member_chains(
        &self,
        collection: &str,
        chains: &[RecordedChain],
    ) -> crate::Result<Vec<bool>> {
        let prefix = self.members_prefix(collection)?;
        let mut walks: Vec<Walk> = chains
            .iter()
            .map(|_| Walk {
                installed: Vec::new(),
                through: 0,
                alive: true,
            })
            .collect();
        let depth = chains.iter().map(|c| c.members.len()).max().unwrap_or(0);
        for rank in 0..depth {
            let mut wanted: Vec<(usize, RestoreItem)> = Vec::new();
            for (i, chain) in chains.iter().enumerate() {
                let walk = &mut walks[i];
                let Some(member) = chain.members.get(rank).filter(|_| walk.alive) else {
                    continue;
                };
                let Some((stream_id, tokens)) = self.restorable(&prefix, walk, member) else {
                    walk.alive = false;
                    continue;
                };
                let section_id = self.substrate.allocate_owned_section().ok_or_else(|| {
                    ConversationError::Other("owned section ids are exhausted".into())
                })?;
                wanted.push((
                    i,
                    RestoreItem {
                        section_id,
                        stream_id,
                        tokens,
                    },
                ));
            }
            while !wanted.is_empty() {
                let take = wanted.len().min(RESTORE_BATCH);
                let (owners, items): (Vec<(usize, SectionId)>, Vec<RestoreItem>) = wanted
                    .drain(..take)
                    .map(|(i, item)| ((i, item.section_id), item))
                    .unzip();
                let (tx, rx) = flume::bounded(1);
                self.scheduler_tx
                    .send(SchedulerRequest::RestoreSections {
                        conversation: self.substrate.clone(),
                        items,
                        response_tx: tx,
                    })
                    .map_err(|_| ConversationError::SchedulerGone)?;
                let results = rx.recv().map_err(|_| ConversationError::SchedulerGone)?;
                for ((i, section), result) in owners.into_iter().zip(results) {
                    let walk = &mut walks[i];
                    match result {
                        Ok(_) => {
                            self.substrate.record_section_loads(1, 0);
                            walk.installed.push(section);
                            walk.through += chains[i].members[rank].tokens;
                        }
                        Err(e) => {
                            tracing::warn!(
                                member = %chains[i].members[rank].name,
                                "member restore failed at boot ({e}) — its file is sealed by \
                                 the ingest's pass"
                            );
                            walk.alive = false;
                        }
                    }
                }
            }
        }
        let mut whole = Vec::with_capacity(chains.len());
        for (chain, walk) in chains.iter().zip(walks) {
            let restored = walk.alive && walk.installed.len() == chain.members.len();
            for (member, &section) in chain.members.iter().zip(&walk.installed) {
                self.substrate
                    .register_owned_section(files_owner(), &member.name, section);
                if !restored {
                    // A member's stream is shared: releasing its entry ends
                    // nothing on disk.
                    self.substrate
                        .release_owned_section(files_owner(), section)
                        .map_err(ConversationError::Model)?;
                }
            }
            whole.push(restored);
        }
        Ok(whole)
    }

    /// The stream and tokens of `member`, the next of `walk`'s chain, when it
    /// is persisted at the address the current prompt gives it and its stream
    /// holds the tokens that address names.
    fn restorable(
        &self,
        prefix: &MembersPrefix,
        walk: &Walk,
        member: &RecordedMember,
    ) -> Option<(StreamId, TokenBuffer)> {
        let through = walk.through + member.tokens;
        let chain = chain_of(&self.substrate, &prefix.addressed(through, &walk.installed)).0;
        if shared_section_prefix(chain) != member.address.prefix_hash {
            return None;
        }
        let stream = section_stream_id(member.address);
        if !self.substrate.section_stream_is_persisted(stream) {
            return None;
        }
        let tokens = self.stream_tokens(stream)?;
        (tokens.len() == member.tokens && hash_tokens(&tokens) == member.address.section_hash)
            .then_some((stream, tokens))
    }

    /// The tokens `stream`'s `Tokens` record holds, read from the log.
    fn stream_tokens(&self, stream: StreamId) -> Option<TokenBuffer> {
        let (id, loc, payloads) = {
            let view = self.substrate.read();
            let s = view.stream_of(stream)?;
            (s.ids.tokens, s.tokens?, view.payloads().clone())
        };
        let bytes = payloads.read(id, loc).ok()?;
        decode_token_ids(&bytes)
            .ok()
            .map(|ids| TokenBuffer::from(&ids[..]))
    }
}
