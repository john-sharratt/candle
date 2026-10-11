//! A little-endian cursor over a binary record payload — what the fixed-layout
//! housekeeping records (`Deleted`, `Replica`) decode through. Every read
//! names the payload it is reading, so a truncated one is reported as that.

use super::{PersistenceError, Result};

/// See the module documentation.
pub struct LeReader<'a> {
    buf: &'a [u8],
    at: usize,
    what: &'static str,
}

impl<'a> LeReader<'a> {
    /// A cursor at the start of `buf`, a `what` payload.
    pub fn new(buf: &'a [u8], what: &'static str) -> LeReader<'a> {
        LeReader { buf, at: 0, what }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let bytes = self.bytes(N)?;
        Ok(bytes.try_into().expect("a slice of N bytes"))
    }

    /// The next `n` bytes.
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let bytes = self
            .buf
            .get(self.at..self.at + n)
            .ok_or_else(|| PersistenceError::Corrupt(format!("{} payload truncated", self.what)))?;
        self.at += n;
        Ok(bytes)
    }

    pub fn u16(&mut self) -> Result<u16> {
        self.take::<2>().map(u16::from_le_bytes)
    }

    pub fn u32(&mut self) -> Result<u32> {
        self.take::<4>().map(u32::from_le_bytes)
    }

    pub fn u64(&mut self) -> Result<u64> {
        self.take::<8>().map(u64::from_le_bytes)
    }

    /// Refuse what is left over: a payload is exactly its fields.
    pub fn finish(self) -> Result<()> {
        if self.at == self.buf.len() {
            Ok(())
        } else {
            Err(PersistenceError::Corrupt(format!(
                "{} payload has {} trailing bytes",
                self.what,
                self.buf.len() - self.at
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_fields_in_order_and_refuses_leftovers() {
        let buf = [1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 9];
        let mut r = LeReader::new(&buf, "test");
        assert_eq!(r.u16().unwrap(), 1);
        assert_eq!(r.u32().unwrap(), 2);
        assert_eq!(r.u64().unwrap(), 3);
        assert!(r.finish().is_err(), "one byte left");
        let mut r = LeReader::new(&buf[..3], "test");
        r.u16().unwrap();
        assert!(r.u16().is_err(), "truncated");
    }
}
