//! Running-hash states for incremental MAC computation.

use std::collections::HashMap;
use std::sync::RwLock;

use chrono::Utc;
use digest::Digest;

enum HashInner {
    Sha256(sha2::Sha256),
    Sha512(sha2::Sha512),
    Sha1(sha1::Sha1),
    Md5(md5::Md5),
    Sha3_256(sha3::Sha3_256),
    Sha3_512(sha3::Sha3_512),
    Blake3(blake3::Hasher),
}

impl HashInner {
    fn update(&mut self, data: &[u8]) {
        match self {
            HashInner::Sha256(h) => h.update(data),
            HashInner::Sha512(h) => h.update(data),
            HashInner::Sha1(h) => h.update(data),
            HashInner::Md5(h) => h.update(data),
            HashInner::Sha3_256(h) => h.update(data),
            HashInner::Sha3_512(h) => h.update(data),
            HashInner::Blake3(h) => {
                h.update(data);
            }
        }
    }

    fn finalize(&self) -> Vec<u8> {
        match self {
            HashInner::Sha256(h) => h.clone().finalize().to_vec(),
            HashInner::Sha512(h) => h.clone().finalize().to_vec(),
            HashInner::Sha1(h) => h.clone().finalize().to_vec(),
            HashInner::Md5(h) => h.clone().finalize().to_vec(),
            HashInner::Sha3_256(h) => h.clone().finalize().to_vec(),
            HashInner::Sha3_512(h) => h.clone().finalize().to_vec(),
            HashInner::Blake3(h) => h.finalize().as_bytes().to_vec(),
        }
    }

    fn algo_name(&self) -> &'static str {
        match self {
            HashInner::Sha256(_) => "sha256",
            HashInner::Sha512(_) => "sha512",
            HashInner::Sha1(_) => "sha1",
            HashInner::Md5(_) => "md5",
            HashInner::Sha3_256(_) => "sha3_256",
            HashInner::Sha3_512(_) => "sha3_512",
            HashInner::Blake3(_) => "blake3",
        }
    }
}

pub struct HashStateEntry {
    pub id: String,
    pub algo: String,
    inner: HashInner,
    pub total_bytes: u64,
    pub created_at: String,
}

impl HashStateEntry {
    fn new(id: String, algo: &str) -> Result<Self, String> {
        // Canonicalize casing/separators so "MD5", "SHA-256", "SHA3-256" match.
        let norm: String = algo
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .map(|c| c.to_ascii_lowercase())
            .collect();
        let inner = match norm.as_str() {
            "sha256" => HashInner::Sha256(sha2::Sha256::new()),
            "sha512" => HashInner::Sha512(sha2::Sha512::new()),
            "sha1" => HashInner::Sha1(sha1::Sha1::new()),
            "md5" => HashInner::Md5(md5::Md5::new()),
            "sha3256" => HashInner::Sha3_256(sha3::Sha3_256::new()),
            "sha3512" => HashInner::Sha3_512(sha3::Sha3_512::new()),
            "blake3" => HashInner::Blake3(blake3::Hasher::new()),
            _ => return Err(format!("unknown algorithm: {algo}")),
        };
        Ok(Self {
            id,
            algo: algo.to_string(),
            inner,
            total_bytes: 0,
            created_at: Utc::now().to_rfc3339(),
        })
    }

    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
        self.total_bytes += data.len() as u64;
    }

    pub fn finalize(&self) -> Vec<u8> {
        self.inner.finalize()
    }
}

#[derive(Default)]
pub struct HashStateStore {
    inner: RwLock<HashMap<String, HashStateEntry>>,
}

impl HashStateStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(&self, id: &str, algo: &str) -> Result<(), String> {
        let mut guard = self.inner.write().unwrap();
        if guard.contains_key(id) {
            return Err(format!("hash state id {id:?} already exists"));
        }
        let entry = HashStateEntry::new(id.to_string(), algo)?;
        guard.insert(id.to_string(), entry);
        Ok(())
    }

    pub fn update(&self, id: &str, data: &[u8]) -> Option<u64> {
        let mut guard = self.inner.write().unwrap();
        let entry = guard.get_mut(id)?;
        entry.update(data);
        Some(entry.total_bytes)
    }

    pub fn finalize(&self, id: &str) -> Option<(Vec<u8>, String)> {
        let guard = self.inner.read().unwrap();
        let entry = guard.get(id)?;
        let digest = entry.finalize();
        let algo = entry.inner.algo_name().to_string();
        Some((digest, algo))
    }

    pub fn delete(&self, id: &str) -> bool {
        self.inner.write().unwrap().remove(id).is_some()
    }

    pub fn list_ids(&self) -> Vec<String> {
        let guard = self.inner.read().unwrap();
        let mut ids: Vec<String> = guard.keys().cloned().collect();
        ids.sort();
        ids
    }

    pub fn get_created_at(&self, id: &str) -> Option<String> {
        self.inner
            .read()
            .unwrap()
            .get(id)
            .map(|e| e.created_at.clone())
    }
}
