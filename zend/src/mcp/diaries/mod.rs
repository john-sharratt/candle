//! Diaries as zend serves them (`docs/private_mode.md` §5.4): unlocked per
//! private conversation by a passphrase, read and written through the memory's
//! store as sealed files, and locked again on `diary_close` or when the
//! conversation's ref expires. The member that holds the journal checkout
//! serves them; what is unlocked lives in its memory alone.
//!
//! | Module | |
//! |---|---|
//! | [`unlocked`] | what a conversation holds while its diaries are open |
//! | [`backoff`] | the wait after opens that unlocked nothing |
//!
//! Locks are taken in one order — the write turn, then what is unlocked,
//! then the store — so no two calls wait on each other.

pub mod backoff;
pub mod unlocked;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, MutexGuard};

use journals::diaries::book::{Author, Diary};
use journals::diaries::crypto::{self, rewrap, seal, unwrap, Dek, KdfParams, Kek};
use journals::diaries::layout::{new_salt, DiaryId, SALT, SALT_LEN};
use journals::store::diff::Changes;
use journals::time::Stamp;
use mcp_server::refs::REF_IDLE_MS;
use zeroize::Zeroizing;

use self::backoff::Backoff;
use self::unlocked::{Opened, Unlocked};
use super::memory::Memory;

/// What a diary call answers when its conversation has opened no passphrase.
pub const NOT_OPEN: &str = "no diary passphrase is open in this conversation: diary_open first";

/// The shortest passphrase taken.
pub const MIN_PASSPHRASE: usize = 8;

/// The longest topic.
pub const MAX_TOPIC: usize = 200;

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
    /// Held across a diary's read, change and write, so two calls changing
    /// one diary keep both changes.
    writing: Mutex<()>,
    backoff: Backoff,
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
        }
    }

    fn derive(&self, passphrase: &str, salt: &[u8]) -> Result<Kek, String> {
        Kek::derive(passphrase, salt, self.params)
            .map_err(|_| "the passphrase could not be stretched".to_string())
    }

    /// Unlock every diary `passphrase` holds for `conversation`, in place of
    /// any passphrase it held before; how many opened. `owner`'s opens that
    /// unlock nothing wait longer each time ([`backoff`]).
    pub fn open(
        &self,
        memory: &Memory,
        owner: &str,
        conversation: &str,
        passphrase: &str,
        now_ms: u64,
    ) -> Result<usize, String> {
        if let Err(wait) = self.backoff.check(owner, now_ms) {
            return Err(format!(
                "too many opens found nothing; try again in {} s",
                wait.div_ceil(1000)
            ));
        }
        check_passphrase(passphrase)?;
        let (files, salt) = memory.diaries()?;
        let pending = salt.is_none().then(new_salt);
        let salt = salt.unwrap_or_else(|| pending.expect("made when none is there").to_vec());
        let kek = self.derive(passphrase, &salt)?;
        let mut open = BTreeMap::new();
        for (id, file) in &files {
            let Some(dek) = unwrap(&kek, id, file) else {
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
        let n = open.len();
        match n {
            0 => self.backoff.failed(owner, now_ms),
            _ => self.backoff.cleared(owner),
        }
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
    /// conversation wrote one since this one opened.
    fn confirm(&self, memory: &Memory, conversation: &str, passphrase: &str) -> Result<Confirmed, String> {
        let pending = held(&self.unlocked)
            .get(conversation)
            .ok_or(NOT_OPEN)?
            .pending_salt;
        let (_, repo_salt) = memory.diaries()?;
        let opened_with = match (pending, &repo_salt) {
            (Some(p), _) => p.to_vec(),
            (None, Some(s)) => s.clone(),
            (None, None) => return Err("no diary salt is kept, and none was opened with".into()),
        };
        let given = self.derive(passphrase, &opened_with)?;
        let same = held(&self.unlocked)
            .get(conversation)
            .is_some_and(|u| u.kek.same(&given));
        if !same {
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

    /// A diary on `topic` under `conversation`'s open passphrase, which
    /// `passphrase` must repeat, created at `now_ms`.
    pub fn create(
        &self,
        memory: &Memory,
        conversation: &str,
        topic: &str,
        passphrase: &str,
        now_ms: u64,
    ) -> Result<(), String> {
        let now = Stamp::from_secs((now_ms / 1000) as i64);
        let topic = topic.trim();
        if topic.is_empty() || topic.chars().count() > MAX_TOPIC {
            return Err(format!("a topic is 1 to {MAX_TOPIC} characters"));
        }
        let _turn = held(&self.writing);
        if held(&self.unlocked)
            .get(conversation)
            .ok_or(NOT_OPEN)?
            .find(topic)
            .is_some()
        {
            return Err("a diary on that topic is already open".into());
        }
        let confirmed = self.confirm(memory, conversation, passphrase)?;
        let id = DiaryId::random();
        let dek = Dek::random();
        let file = seal(&confirmed.kek, &dek, &id, &Diary::new(topic, now).to_bytes());
        let mut changes = Changes::from([(id.path(), Some(file))]);
        if let Some(salt) = confirmed.salt {
            changes.insert(SALT.to_string(), Some(salt.to_vec()));
        }
        memory.write_diaries(&changes).map_err(String::from)?;
        let mut unlocked = held(&self.unlocked);
        let u = unlocked.get_mut(conversation).ok_or(NOT_OPEN)?;
        u.kek = confirmed.kek;
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

    /// Run `f` on `conversation`'s open diaries, marked used at `now_ms`.
    fn in_open<T>(
        &self,
        conversation: &str,
        now_ms: u64,
        f: impl FnOnce(&mut Unlocked) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut unlocked = held(&self.unlocked);
        let u = unlocked.get_mut(conversation).ok_or(NOT_OPEN)?;
        u.used_ms = u.used_ms.max(now_ms);
        f(u)
    }

    /// The open diaries: each topic, how many entries, and the first's and
    /// last's moments.
    pub fn list(&self, memory: &Memory, conversation: &str, now_ms: u64) -> Result<Vec<Listed>, String> {
        self.in_open(conversation, now_ms, |u| {
            let mut listed = Vec::new();
            for (id, o) in &u.open {
                let file = memory.diary_file(id)?.ok_or("an open diary's file is gone")?;
                let d = opened(&o.dek, id, &file)?;
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
        })
    }

    /// The open diary on `topic`, whole.
    pub fn read(&self, memory: &Memory, conversation: &str, topic: &str, now_ms: u64) -> Result<Diary, String> {
        self.in_open(conversation, now_ms, |u| {
            let (id, o) = u.find(topic.trim()).ok_or_else(|| no_topic(topic))?;
            let file = memory.diary_file(&id)?.ok_or("the diary's file is gone")?;
            opened(&o.dek, &id, &file)
        })
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
        let _turn = held(&self.writing);
        self.in_open(conversation, now_ms, |u| {
            let (id, o) = u.find(topic.trim()).ok_or_else(|| no_topic(topic))?;
            let file = memory.diary_file(&id)?.ok_or("the diary's file is gone")?;
            let mut diary = opened(&o.dek, &id, &file)?;
            let out = change(&mut diary)?;
            let plain = Zeroizing::new(diary.to_bytes());
            let sealed = seal(&u.kek, &o.dek, &id, &plain);
            memory
                .write_diaries(&Changes::from([(id.path(), Some(sealed))]))
                .map_err(String::from)?;
            Ok(out)
        })
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
    /// open one, which `passphrase` must repeat. It is then that passphrase's,
    /// and closed here: `diary_open` with it opens it.
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
        let _turn = held(&self.writing);
        let confirmed = self.confirm(memory, conversation, passphrase)?;
        let (_, salt) = memory.diaries()?;
        let salt = salt.ok_or("no diary salt is kept")?;
        let new = self.derive(new_passphrase, &salt)?;
        self.in_open(conversation, now_ms, |u| {
            let (id, o) = u.find(topic.trim()).ok_or_else(|| no_topic(topic))?;
            if !u.kek.same(&confirmed.kek) {
                return Err("that is not the passphrase this conversation opened".into());
            }
            let file = memory.diary_file(&id)?.ok_or("the diary's file is gone")?;
            let moved = rewrap(&new, &o.dek, &id, &file);
            memory
                .write_diaries(&Changes::from([(id.path(), Some(moved))]))
                .map_err(String::from)?;
            u.open.remove(&id);
            Ok(())
        })
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
