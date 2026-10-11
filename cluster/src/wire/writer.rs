//! Appends a frame body.

use super::varint::write_varint;

/// A growing frame body.
#[derive(Debug, Default)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            buf: Vec::with_capacity(n),
        }
    }

    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.buf.push(v);
        self
    }

    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn bool(&mut self, v: bool) -> &mut Self {
        self.u8(v as u8)
    }

    /// Bytes of a fixed width, unprefixed.
    pub fn fixed(&mut self, v: &[u8]) -> &mut Self {
        self.buf.extend_from_slice(v);
        self
    }

    /// A `u32` length, then the bytes.
    pub fn blob(&mut self, v: &[u8]) -> &mut Self {
        self.u32(v.len() as u32);
        self.fixed(v)
    }

    /// A string as a blob of its UTF-8.
    pub fn str(&mut self, v: &str) -> &mut Self {
        self.blob(v.as_bytes())
    }

    /// A `u32` count, then each item written by `f`.
    pub fn list<T>(&mut self, items: &[T], mut f: impl FnMut(&mut Self, &T)) -> &mut Self {
        self.u32(items.len() as u32);
        for item in items {
            f(self, item);
        }
        self
    }

    /// `0`, or `1` and the value written by `f`.
    pub fn option<T>(&mut self, v: Option<&T>, f: impl FnOnce(&mut Self, &T)) -> &mut Self {
        match v {
            None => self.u8(0),
            Some(v) => {
                self.u8(1);
                f(self, v);
                self
            }
        }
    }

    pub fn varint(&mut self, v: u64) -> &mut Self {
        write_varint(&mut self.buf, v);
        self
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}
