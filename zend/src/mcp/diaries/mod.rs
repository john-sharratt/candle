//! Diaries as zend serves them (`docs/private_mode.md` §5.4): unlocked per
//! private conversation by a passphrase, read and written through the memory's
//! store as sealed files, and locked again on `diary_close` or when the
//! conversation's ref expires. The member that holds the journal checkout
//! serves them; what is unlocked lives in its memory alone.
//!
//! | Module | |
//! |---|---|
//! | [`unlocked`] | what a conversation holds while its diaries are open |
//! | [`backoff`] | the wait after opens that unlocked nothing, and the cap on stretching at once |
//!
//! **Locks.** A passphrase is stretched with no lock held. A change takes the
//! write turn — so two conversations changing one diary keep both changes —
//! and, inside it, the store; what is unlocked is held only long enough to
//! copy a key out or note a change, never across a commit. Every key copied
//! out is wiped when dropped.
//!
//! **A diary moves with its passphrase.** Each change first checks that the
//! file still opens under this conversation's passphrase: a diary another
//! conversation moved to a new one is closed here, never sealed back under
//! the old.

pub mod backoff;
pub mod unlocked;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, MutexGuard};

use journals::diaries::book::{Author, Diary};
use journals::diaries::crypto::{self, seal, unwrap, Dek, KdfParams, Kek};
use journals::diaries::layout::{new_salt, DiaryId, SALT, SALT_LEN};
use journals::store::diff::Changes;
use journals::time::Stamp;
use mcp_server::refs::REF_IDLE_MS;
use zeroize::Zeroizing;

use self::backoff::{Backoff, Stretching};
use self::unlocked::{Opened, Unlocked};
use super::memory::Memory;

/// What a diary call answers when its conversation has opened no passphrase.
pub const NOT_OPEN: &str = "no diary passphrase is open in this conversation: diary_open first";

/// What a change answers for a diary another conversation moved to a new
/// passphrase.
pub const MOVED: &str = "that diary is held by another passphrase now: diary_open with it";

/// The shortest passphrase taken.
pub const MIN_PASSPHRASE: usize = 8;

/// The longest topic.
pub const MAX_TOPIC: usize = 200;

/// The most a diary holds, as its plaintext: its file is then 64 MiB, within
/// what the repository's host takes in one file.
pub const MAX_DIARY: usize = 32 * 1024 * 1024 - 4096;

/// One open diary as `diary_list` tells it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listed {
    pub topic: String,
    pub entries: usize,
    pub first: Option<String>,
    pub last: Option<String>,
}

/// Every private conversation's open diaries, on this member.
pub struct Diaries {
    params: KdfParams,
    unlocked: Mutex<HashMap<String, Unlocked>>,
    /// Held across a diary's read, change and write.
    writing: Mutex<()>,
    backoff: Backoff,
    stretching: Stretching,
}

impl Default for Diaries {
    fn default() -> Self {
        Diaries::with(KdfParams::PRODUCTION)
    }
}

fn held<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn check_passphrase(passphrase: &str) -> Result<(), String> {
    match passphrase.chars().count() >= MIN_PASSPHRASE {
        true => Ok(()),
        false => Err(format!(
            "a diary passphrase is at least {MIN_PASSPHRASE} characters"
        )),
    }
}

/// The diary sealed in `file`, under `dek`.
fn opened(dek: &Dek, id: &DiaryId, file: &[u8]) -> Result<Diary, String> {
    let plain = crypto::open(dek, id, file).map_err(|_| "a diary's file did not open".to_string())?;
    Diary::from_bytes(&plain).map_err(|_| "a diary's file holds no diary".to_string())
}

/// Every diary `kek` unwraps among `files`, by id: its key and its topic.
fn unwrapped(kek: &Kek, files: &[(DiaryId, Vec<u8>)]) -> BTreeMap<DiaryId, Opened> {
    let mut open = BTreeMap::new();
    for (id, file) in files {
        let Some(dek) = unwrap(kek, id, file) else {
            continue;
        };
        if let Ok(diary) = opened(&dek, id, file) {
            open.insert(
                *id,
                Opened {
                    dek,
                    topic: diary.topic,
                },
            );
        }
    }
    open
}

/// The kek a confirmed passphrase wraps with, and the salt to write beside
/// the change when the repository has none yet.
struct Confirmed {
    kek: Kek,
    salt: Option<[u8; SALT_LEN]>,
}

impl Diaries {
    /// Diaries whose passphrases are stretched under `params`.
    pub fn with(params: KdfParams) -> Self {
        Diaries {
            params,
            unlocked: Mutex::new(HashMap::new()),
            writing: Mutex::new(()),
            backoff: Backoff::default(),
            stretching: Stretching::default(),
        }
    }

    /// `passphrase` stretched with `salt`, within the cap on stretches at
    /// once.
    fn derive(&self, passphrase: &str, salt: &[u8]) -> Result<Kek, String> {
        let _slot = self.stretching.take()?;
        Kek::derive(passphrase, salt, self.params)
            .map_err(|_| "the passphrase could not be stretched".to_string())
    }

    /// Unlock every diary `passphrase` holds for `conversation`, in place of
    /// any passphrase it held before; how many opened. Each open is counted
    /// against `owner` before the passphrase is tried, and opens that unlock
    /// nothing wait longer each time ([`backoff`]).
    pub fn open(
        &self,
        memory: &Memory,
        owner: &str,
        conversation: &str,
        passphrase: &str,
        now_ms: u64,
    ) -> Result<usize, String> {
        check_passphrase(passphrase)?;
        let attempt = self.backoff.attempt(owner, now_ms).map_err(|wait| {
            format!(
                "too many opens found nothing; try again in {} s",
                wait.div_ceil(1000)
            )
        })?;
        let (files, salt) = memory.diaries()?;
        let pending = salt.is_none().then(new_salt);
        let salt = salt.unwrap_or_else(|| pending.expect("made when none is there").to_vec());
        let kek = self.derive(passphrase, &salt)?;
        let open = unwrapped(&kek, &files);
        let n = open.len();
        attempt.found(n > 0);
        held(&self.unlocked).insert(
            conversation.to_string(),
            Unlocked {
                kek,
                pending_salt: pending,
                open,
                used_ms: now_ms,
            },
        );
        Ok(n)
    }

    /// That `passphrase` is the one `conversation` opened, and the kek to wrap
    /// with: re-derived against the repository's salt when another
    /// conversation wrote one since this one opened. Stretches with no lock
    /// held.
    fn confirm(&self, memory: &Memory, conversation: &str, passphrase: &str) -> Result<Confirmed, String> {
        let (held_kek, pending) = {
            let unlocked = held(&self.unlocked);
            let u = unlocked.get(conversation).ok_or(NOT_OPEN)?;
            (u.kek.clone(), u.pending_salt)
        };
        let (_, repo_salt) = memory.diaries()?;
        let opened_with = match (pending, &repo_salt) {
            (Some(p), _) => p.to_vec(),
            (None, Some(s)) => s.clone(),
            (None, None) => return Err("no diary salt is kept, and none was opened with".into()),
        };
        let given = self.derive(passphrase, &opened_with)?;
        if !held_kek.same(&given) {
            return Err("that is not the passphrase this conversation opened".into());
        }
        match (pending, repo_salt) {
            (Some(p), None) => Ok(Confirmed {
                kek: given,
                salt: Some(p),
            }),
            (Some(_), Some(s)) => Ok(Confirmed {
                kek: self.derive(passphrase, &s)?,
                salt: None,
            }),
            (None, _) => Ok(Confirmed {
                kek: given,
                salt: None,
            }),
        }
    }

    /// Take into `conversation`'s open diaries every one its passphrase
    /// unwraps that it has not opened yet — another conversation's new ones.
    fn rescan(&self, memory: &Memory, conversation: &str) -> Result<(), String> {
        let kek = held(&self.unlocked)
            .get(conversation)
            .ok_or(NOT_OPEN)?
            .kek
            .clone();
        let (files, _) = memory.diaries()?;
        let found = unwrapped(&kek, &files);
        let mut unlocked = held(&self.unlocked);
        let u = unlocked.get_mut(conversation).ok_or(NOT_OPEN)?;
        for (id, o) in found {
            u.open.entry(id).or_insert(o);
        }
        Ok(())
    }

    /// A diary on `topic` under `conversation`'s open passphrase, which
    /// `passphrase` must repeat, created at `now_ms`. No two of a
    /// passphrase's diaries share a topic.
    pub fn create(
        &self,
        memory: &Memory,
        conversation: &str,
        topic: &str,
        passphrase: &str,
        now_ms: u64,
    ) -> Result<(), String> {
        let topic = topic.trim();
        if topic.is_empty() || topic.chars().count() > MAX_TOPIC {
            return Err(format!("a topic is 1 to {MAX_TOPIC} characters"));
        }
        let confirmed = self.confirm(memory, conversation, passphrase)?;
        let _turn = held(&self.writing);
        {
            let mut unlocked = held(&self.unlocked);
            let u = unlocked.get_mut(conversation).ok_or(NOT_OPEN)?;
            // Stretched again against the repository's salt, it holds that
            // one from now on; still to write its own, it waits for this.
            u.kek = confirmed.kek.clone();
            u.pending_salt = confirmed.salt;
        }
        self.rescan(memory, conversation)?;
        if held(&self.unlocked)
            .get(conversation)
            .ok_or(NOT_OPEN)?
            .find(topic)
            .is_some()
        {
            return Err("a diary on that topic is already kept under this passphrase".into());
        }
        let now = Stamp::from_secs((now_ms / 1000) as i64);
        let id = DiaryId::random();
        let dek = Dek::random();
        let plain = Zeroizing::new(Diary::new(topic, now).to_bytes());
        let file = seal(&confirmed.kek, &dek, &id, &plain);
        let mut changes = Changes::from([(id.path(), Some(file))]);
        if let Some(salt) = confirmed.salt {
            changes.insert(SALT.to_string(), Some(salt.to_vec()));
        }
        memory.write_diaries(&changes).map_err(String::from)?;
        let mut unlocked = held(&self.unlocked);
        let u = unlocked.get_mut(conversation).ok_or(NOT_OPEN)?;
        u.pending_salt = None;
        u.open.insert(
            id,
            Opened {
                dek,
                topic: topic.to_string(),
            },
        );
        u.used_ms = u.used_ms.max(now_ms);
        Ok(())
    }

    /// The keys of `conversation`'s open diary on `topic` — copied out, and
    /// wiped when dropped — with the conversation marked used at `now_ms`.
    fn keys(&self, conversation: &str, topic: &str, now_ms: u64) -> Result<(DiaryId, Kek, Dek), String> {
        let mut unlocked = held(&self.unlocked);
        let u = unlocked.get_mut(conversation).ok_or(NOT_OPEN)?;
        u.used_ms = u.used_ms.max(now_ms);
        let (id, o) = u.find(topic.trim()).ok_or_else(|| no_topic(topic))?;
        Ok((id, u.kek.clone(), o.dek.clone()))
    }

    /// Close `conversation`'s diary `id`: it is no longer this passphrase's.
    fn forget(&self, conversation: &str, id: &DiaryId) {
        if let Some(u) = held(&self.unlocked).get_mut(conversation) {
            u.open.remove(id);
        }
    }

    /// Diary `id`'s file as it stands, when it still opens under `kek`: one
    /// moved to another passphrase is closed here.
    fn current(
        &self,
        memory: &Memory,
        conversation: &str,
        id: &DiaryId,
        kek: &Kek,
    ) -> Result<Vec<u8>, String> {
        let file = memory.diary_file(id)?.ok_or("the diary's file is gone")?;
        if unwrap(kek, id, &file).is_none() {
            self.forget(conversation, id);
            return Err(MOVED.into());
        }
        Ok(file)
    }

    /// The open diaries: each topic, how many entries, and the first's and
    /// last's moments — those another conversation made under this
    /// passphrase among them.
    pub fn list(&self, memory: &Memory, conversation: &str, now_ms: u64) -> Result<Vec<Listed>, String> {
        self.rescan(memory, conversation)?;
        let open: Vec<(DiaryId, Kek, Dek)> = {
            let mut unlocked = held(&self.unlocked);
            let u = unlocked.get_mut(conversation).ok_or(NOT_OPEN)?;
            u.used_ms = u.used_ms.max(now_ms);
            u.open
                .iter()
                .map(|(id, o)| (*id, u.kek.clone(), o.dek.clone()))
                .collect()
        };
        let mut listed = Vec::new();
        for (id, kek, dek) in open {
            let Ok(file) = self.current(memory, conversation, &id, &kek) else {
                continue;
            };
            let d = opened(&dek, &id, &file)?;
            let span = d.span().map(|(a, b)| (a.to_string(), b.to_string()));
            listed.push(Listed {
                topic: d.topic.clone(),
                entries: d.entries.len(),
                first: span.as_ref().map(|s| s.0.clone()),
                last: span.map(|s| s.1),
            });
        }
        listed.sort_by(|a, b| a.topic.cmp(&b.topic));
        Ok(listed)
    }

    /// The open diary on `topic`, whole.
    pub fn read(&self, memory: &Memory, conversation: &str, topic: &str, now_ms: u64) -> Result<Diary, String> {
        let (id, kek, dek) = self.keys(conversation, topic, now_ms)?;
        let file = self.current(memory, conversation, &id, &kek)?;
        opened(&dek, &id, &file)
    }

    /// Change the open diary on `topic` with `change`, sealed again under a
    /// fresh nonce and written as one commit.
    fn change<T>(
        &self,
        memory: &Memory,
        conversation: &str,
        topic: &str,
        now_ms: u64,
        change: impl FnOnce(&mut Diary) -> Result<T, String>,
    ) -> Result<T, String> {
        let (id, kek, dek) = self.keys(conversation, topic, now_ms)?;
        let _turn = held(&self.writing);
        let file = self.current(memory, conversation, &id, &kek)?;
        let mut diary = opened(&dek, &id, &file)?;
        let out = change(&mut diary)?;
        let plain = Zeroizing::new(diary.to_bytes());
        if plain.len() > MAX_DIARY {
            return Err(format!(
                "a diary holds at most {} MiB: compact it, or begin another",
                MAX_DIARY / (1024 * 1024) + 1
            ));
        }
        let sealed = seal(&kek, &dek, &id, &plain);
        memory
            .write_diaries(&Changes::from([(id.path(), Some(sealed))]))
            .map_err(String::from)?;
        Ok(out)
    }

    /// An entry in the open diary on `topic`, by `author`, about the moment
    /// `ts`; its id.
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &self,
        memory: &Memory,
        conversation: &str,
        topic: &str,
        ts: Stamp,
        author: Author,
        text: &str,
        now_ms: u64,
    ) -> Result<String, String> {
        if text.trim().is_empty() {
            return Err("an entry holds some text".into());
        }
        self.change(memory, conversation, topic, now_ms, |d| Ok(d.write(ts, author, text)))
    }

    /// Delete the entries `ids` from the open diary on `topic`; how many.
    pub fn delete(
        &self,
        memory: &Memory,
        conversation: &str,
        topic: &str,
        ids: &[String],
        now_ms: u64,
    ) -> Result<usize, String> {
        if ids.is_empty() {
            return Err("name the entries to delete by their ids".into());
        }
        self.change(memory, conversation, topic, now_ms, |d| {
            d.delete(ids)
                .map_err(|_| "not every id names an entry of this diary; none was deleted".to_string())
        })
    }

    /// Hold the open diary on `topic` under `new_passphrase` instead of the
    /// open one, which `passphrase` must repeat: sealed again under a new key
    /// of its own, wrapped under the new passphrase, so neither the old
    /// passphrase nor the old key opens any version written from now on. It
    /// is then that passphrase's, and closed here.
    pub fn rotate(
        &self,
        memory: &Memory,
        conversation: &str,
        topic: &str,
        passphrase: &str,
        new_passphrase: &str,
        now_ms: u64,
    ) -> Result<(), String> {
        check_passphrase(new_passphrase)?;
        let (id, kek, dek) = self.keys(conversation, topic, now_ms)?;
        let confirmed = self.confirm(memory, conversation, passphrase)?;
        if !kek.same(&confirmed.kek) {
            return Err("that is not the passphrase this conversation opened".into());
        }
        let (_, salt) = memory.diaries()?;
        let salt = salt.ok_or("no diary salt is kept")?;
        let new = self.derive(new_passphrase, &salt)?;
        let _turn = held(&self.writing);
        let file = self.current(memory, conversation, &id, &kek)?;
        let plain = Zeroizing::new(opened(&dek, &id, &file)?.to_bytes());
        let moved = seal(&new, &Dek::random(), &id, &plain);
        memory
            .write_diaries(&Changes::from([(id.path(), Some(moved))]))
            .map_err(String::from)?;
        self.forget(conversation, &id);
        Ok(())
    }

    /// Lock `conversation`'s diaries; whether any were open.
    pub fn close(&self, conversation: &str) -> bool {
        held(&self.unlocked).remove(conversation).is_some()
    }

    /// Lock every conversation's diaries idle past [`REF_IDLE_MS`] at
    /// `now_ms` — by its own last diary call and by `used`, its ref's last
    /// use as the record says; how many were locked.
    pub fn sweep(&self, now_ms: u64, used: impl Fn(&str) -> Option<u64>) -> usize {
        let mut unlocked = held(&self.unlocked);
        let before = unlocked.len();
        unlocked.retain(|conversation, u| {
            let last = used(conversation).unwrap_or(0).max(u.used_ms);
            now_ms.saturating_sub(last) < REF_IDLE_MS
        });
        before - unlocked.len()
    }

    /// Whether `conversation` holds any passphrase open.
    pub fn is_open(&self, conversation: &str) -> bool {
        held(&self.unlocked).contains_key(conversation)
    }
}

fn no_topic(topic: &str) -> String {
    format!(
        "no open diary is on `{}`: diary_list names the open ones",
        topic.trim()
    )
}

#[cfg(test)]
mod tests;
