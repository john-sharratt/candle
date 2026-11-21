//! Little-endian byte writer and reader for the binary payloads.
//!
//! The writer appends fixed-width integers, floats and length-prefixed blobs and
//! strings, and the reader takes them back in the same order. A reader fails when
//! the buffer is too short rather than returning partial values. A cold load can
//! ask the reader for a byte range and keep that range in place, instead of
//! copying it.

use super::*;

// Little-endian byte primitives — the encoding used by `ChunkPayload`
// and any other still-binary structured payload (kept public for
// re-use; metadata payloads are now JSON via serde).

/// Append-only little-endian byte writer.
#[derive(Default)]
pub struct ByteWriter {
    pub(super) buf: Vec<u8>,
}

impl ByteWriter {
    pub fn new() -> ByteWriter {
        ByteWriter { buf: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn put_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn put_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_raw(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    pub fn put_blob(&mut self, data: &[u8]) {
        self.put_u32(data.len() as u32);
        self.buf.extend_from_slice(data);
    }

    pub fn put_str(&mut self, s: &str) {
        self.put_blob(s.as_bytes());
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Cursor-based little-endian byte reader.
pub struct ByteReader<'a> {
    pub(super) buf: &'a [u8],
    pub(super) pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(buf: &'a [u8]) -> ByteReader<'a> {
        ByteReader { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn is_done(&self) -> bool {
        self.pos == self.buf.len()
    }

    /// Current cursor position within the underlying buffer. Used by the
    /// cold-load batched-read path to capture the `kv_bytes` slice range
    /// without copying it out.
    pub fn position(&self) -> usize {
        self.pos
    }

    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(PersistenceError::Truncated {
                need: n,
                have: self.remaining(),
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn get_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn get_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn get_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn get_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn get_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn get_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn get_raw(&mut self, n: usize) -> Result<&'a [u8]> {
        self.take(n)
    }

    pub fn get_blob(&mut self) -> Result<&'a [u8]> {
        let n = self.get_u32()? as usize;
        self.take(n)
    }

    pub fn get_str(&mut self) -> Result<String> {
        let bytes = self.get_blob()?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| PersistenceError::Corrupt(format!("invalid utf-8: {e}")))
    }
}
