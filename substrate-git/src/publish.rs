//! Writing a repository's new objects, then the refs that name them
//! (`docs/substrate_git.md` §6.1).
//!
//! Objects first: every object a ref reaches that the log's refs do not is
//! packed and written before the ref is, so no member ever holds a ref whose
//! commit it cannot read — and a ref that replicates ahead of its packs waits
//! for them in [`crate::Store::sync`].

use anyhow::Context;
use zend_vfs::{object_count, pack_checksum, Oid, RefName, Repo, Rev};

use crate::records::{Log, PackEntry, RefEntry, Records};
use crate::MAX_PACK;

/// What [`publish`] wrote.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Published {
    /// Packs written.
    pub packs: usize,
    /// Refs written, with where they point.
    pub refs: Vec<(RefName, Option<Oid>)>,
    /// Refs left as they were because the new value does not descend from the
    /// log's, with the log's value.
    pub refused: Vec<(RefName, Oid)>,
}

/// Write `refs` of repository `repo` to `records`, taking their objects from
/// `source`. A ref the log already holds is replaced only by a descendant,
/// unless `forced`; a `None` deletes, and is itself a forced move.
pub fn publish(
    records: &dyn Records,
    repo: &str,
    source: &Repo,
    refs: &[(RefName, Option<Oid>)],
    forced: bool,
) -> anyhow::Result<Published> {
    let log = Log::new(records, repo)?;
    let held = log.refs()?;
    let mut published = Published::default();

    let mut moving = Vec::new();
    for (name, new) in refs {
        let old = match held.get(name.as_str()).and_then(|e| e.oid.as_deref()) {
            Some(old) => Some(Oid::parse(old)?),
            None => None,
        };
        if old == *new {
            continue;
        }
        if let (Some(old), Some(new)) = (&old, new) {
            if !forced && !descends(source, old, new)? {
                published.refused.push((name.clone(), old.clone()));
                continue;
            }
        }
        if new.is_none() && !forced {
            if let Some(old) = old {
                published.refused.push((name.clone(), old));
            }
            continue;
        }
        moving.push((name.clone(), new.clone()));
    }
    if moving.is_empty() {
        return Ok(published);
    }

    let tips: Vec<Oid> = moving.iter().filter_map(|(_, oid)| oid.clone()).collect();
    let mut known = Vec::new();
    let mut needs: Vec<String> = Vec::new();
    for entry in held.values() {
        let Some(oid) = entry.oid.as_deref() else {
            continue;
        };
        known.push(Oid::parse(oid)?);
    }
    // Each pack is written and let go before the next is read: a first
    // publish of a whole history is never in memory at once.
    published.packs = source
        .pack_objects(&tips, &known, MAX_PACK, |pack| {
            let entry = PackEntry {
                checksum: pack_checksum(&pack).context("a pack with no trailer")?,
                objects: object_count(&pack),
            };
            needs.push(log.put_pack(pack, &entry)?);
            Ok::<(), anyhow::Error>(())
        })
        .with_context(|| format!("packing {repo}"))?;

    // Packing took a while, and the log is last writer wins: a ref another
    // member moved meanwhile is replaced only by a descendant of where it
    // stands now, else left for the next tick to publish again.
    let now = log.refs()?;
    for (name, oid) in moving {
        let before = held.get(name.as_str()).and_then(|e| e.oid.as_deref());
        let current = now.get(name.as_str()).and_then(|e| e.oid.as_deref());
        if current != before && !forced {
            match (current, &oid) {
                (Some(current), Some(new)) => {
                    let current = Oid::parse(current)?;
                    if !descends(source, &current, new)? {
                        published.refused.push((name, current));
                        continue;
                    }
                }
                (Some(current), None) => {
                    published.refused.push((name, Oid::parse(current)?));
                    continue;
                }
                // Deleted meanwhile: deleting is a forced move, and wins.
                (None, _) => continue,
            }
        }
        log.put_ref(
            &name,
            &RefEntry {
                oid: oid.as_ref().map(|o| o.to_string()),
                needs: needs.clone(),
                forced,
            },
        )?;
        published.refs.push((name, oid));
    }
    Ok(published)
}

/// Whether `new` descends from `old` in `source`; not when `source` lacks
/// `old`, since then it cannot say.
fn descends(source: &Repo, old: &Oid, new: &Oid) -> anyhow::Result<bool> {
    if !source.has_object(old)? {
        return Ok(false);
    }
    Ok(source.is_ancestor(&Rev::Oid(old.clone()), &Rev::Oid(new.clone()))?)
}
