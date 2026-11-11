//! What a diary holds once opened (`docs/private_mode.md` §5.3): its topic,
//! when it was created, and its entries in time order — each with a random,
//! stable id, the moment it is about, who wrote it, and its text. The topic is
//! inside the sealed file, never beside it.

use rand::rngs::OsRng;
use rand::TryRngCore;
use serde::{Deserialize, Serialize};

use crate::time::Stamp;

/// Who wrote an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Author {
    User,
    Assistant,
}

impl Author {
    pub fn parse(s: &str) -> Option<Author> {
        match s {
            "user" => Some(Author::User),
            "assistant" => Some(Author::Assistant),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Author::User => "user",
            Author::Assistant => "assistant",
        }
    }
}

/// One entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub id: String,
    /// The moment it is about, RFC 3339.
    pub ts: String,
    pub author: Author,
    pub text: String,
}

/// A diary, opened.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diary {
    pub topic: String,
    pub created: String,
    pub entries: Vec<Entry>,
}

/// Why a diary's change was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BookError {
    /// No entry has this id.
    NoEntry(String),
    /// What was opened is not a diary.
    NotADiary(String),
}

fn entry_id() -> String {
    let mut b = [0u8; 8];
    OsRng
        .try_fill_bytes(&mut b)
        .expect("the OS must provide entropy");
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// An entry's place in time: its moment, then its id, so equal moments keep
/// one order however they were written.
fn order(e: &Entry) -> (i64, &str) {
    let at = Stamp::parse(&e.ts).map_or(i64::MIN, Stamp::secs);
    (at, e.id.as_str())
}

impl Diary {
    /// A diary on `topic`, created at `now`, with nothing in it.
    pub fn new(topic: &str, now: Stamp) -> Diary {
        Diary {
            topic: topic.to_string(),
            created: now.to_string(),
            entries: Vec::new(),
        }
    }

    /// An entry by `author` about the moment `ts`, put in its place: after
    /// every entry before it in time, before every one after. Its id.
    pub fn write(&mut self, ts: Stamp, author: Author, text: &str) -> String {
        let entry = Entry {
            id: entry_id(),
            ts: ts.to_string(),
            author,
            text: text.to_string(),
        };
        let id = entry.id.clone();
        let at = self.entries.partition_point(|e| order(e) <= order(&entry));
        self.entries.insert(at, entry);
        id
    }

    /// Delete the entries `ids` names, every one of which must be here; none
    /// is deleted when one is not.
    pub fn delete(&mut self, ids: &[String]) -> Result<usize, BookError> {
        if let Some(missing) = ids
            .iter()
            .find(|id| !self.entries.iter().any(|e| &e.id == *id))
        {
            return Err(BookError::NoEntry(missing.clone()));
        }
        let before = self.entries.len();
        self.entries.retain(|e| !ids.contains(&e.id));
        Ok(before - self.entries.len())
    }

    /// The first entry's moment and the last's, when there are any.
    pub fn span(&self) -> Option<(&str, &str)> {
        Some((&self.entries.first()?.ts, &self.entries.last()?.ts))
    }

    /// The diary as the bytes its file seals.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("a diary is JSON")
    }

    /// The diary `bytes` hold.
    pub fn from_bytes(bytes: &[u8]) -> Result<Diary, BookError> {
        serde_json::from_slice(bytes).map_err(|e| BookError::NotADiary(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(s: &str) -> Stamp {
        Stamp::parse(s).unwrap()
    }

    /// **An entry goes where its moment is**: one written later about an
    /// earlier moment reads before those after it.
    #[test]
    fn an_entry_goes_in_its_place_in_time() {
        let mut d = Diary::new("synthetic topic", at("2026-11-11T09:00:00Z"));
        d.write(at("2026-11-10T10:00:00Z"), Author::User, "second");
        d.write(at("2026-11-12T10:00:00Z"), Author::User, "fourth");
        d.write(at("2026-11-09T10:00:00Z"), Author::Assistant, "first");
        d.write(at("2026-11-11T10:00:00Z"), Author::User, "third");
        let texts: Vec<&str> = d.entries.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(texts, ["first", "second", "third", "fourth"]);
        assert_eq!(
            d.span(),
            Some(("2026-11-09T10:00:00Z", "2026-11-12T10:00:00Z"))
        );
        assert_eq!(d.entries[0].id.len(), 16);
    }

    #[test]
    fn deleting_takes_every_entry_named_or_none() {
        let mut d = Diary::new("t", at("2026-11-11T09:00:00Z"));
        let a = d.write(at("2026-11-11T10:00:00Z"), Author::User, "a");
        let b = d.write(at("2026-11-11T11:00:00Z"), Author::User, "b");
        assert_eq!(
            d.delete(&[a.clone(), "nope".into()]),
            Err(BookError::NoEntry("nope".into()))
        );
        assert_eq!(d.entries.len(), 2, "none deleted");
        assert_eq!(d.delete(&[a, b]), Ok(2));
        assert_eq!(d.span(), None);
    }

    #[test]
    fn a_diary_reads_back_as_written() {
        let mut d = Diary::new("synthetic topic", at("2026-11-11T09:00:00Z"));
        d.write(
            at("2026-11-11T10:00:00Z"),
            Author::Assistant,
            "line\nwith a break",
        );
        let bytes = d.to_bytes();
        assert_eq!(Diary::from_bytes(&bytes), Ok(d));
        assert!(Diary::from_bytes(b"{}").is_err());
        assert_eq!(Author::parse("user"), Some(Author::User));
        assert_eq!(Author::parse("me"), None);
    }
}
