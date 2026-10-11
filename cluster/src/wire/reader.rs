//! Reads a frame body, refusing anything short, long or out of range.

use super::varint::read_varint;
use super::WireError;

/// A cursor over one frame body. `what` names the frame in every error.
#[derive(Debug)]
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    what: &'static str,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8], what: &'static str) -> Self {
        Self { buf, pos: 0, what }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        let left = self.buf.len() - self.pos;
        if n > left {
            return Err(WireError::Truncated {
                what: self.what,
                need: n - left,
            });
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    pub fn u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, WireError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32, WireError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64, WireError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// `0` or `1`; any other byte is refused.
    pub fn bool(&mut self) -> Result<bool, WireError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(WireError::Invalid {
                what: self.what,
                value: v as u64,
            }),
        }
    }

    pub fn fixed<const N: usize>(&mut self) -> Result<[u8; N], WireError> {
        Ok(self.take(N)?.try_into().unwrap())
    }

    pub fn blob(&mut self) -> Result<&'a [u8], WireError> {
        let n = self.u32()? as usize;
        self.take(n)
    }

    pub fn blob_vec(&mut self) -> Result<Vec<u8>, WireError> {
        Ok(self.blob()?.to_vec())
    }

    pub fn str(&mut self) -> Result<String, WireError> {
        let what = self.what;
        String::from_utf8(self.blob_vec()?).map_err(|_| WireError::NotUtf8 { what })
    }

    /// A `u32` count, then that many items read by `f`. A count no remaining
    /// byte could satisfy is refused before anything is allocated for it.
    pub fn list<T>(
        &mut self,
        mut f: impl FnMut(&mut Self) -> Result<T, WireError>,
    ) -> Result<Vec<T>, WireError> {
        let n = self.u32()? as usize;
        let left = self.remaining();
        if n > left {
            return Err(WireError::Truncated {
                what: self.what,
                need: n - left,
            });
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(f(self)?);
        }
        Ok(out)
    }

    pub fn option<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, WireError>,
    ) -> Result<Option<T>, WireError> {
        if self.bool()? {
            Ok(Some(f(self)?))
        } else {
            Ok(None)
        }
    }

    pub fn varint(&mut self) -> Result<u64, WireError> {
        let (v, n) = read_varint(&self.buf[self.pos..], self.what)?;
        self.pos += n;
        Ok(v)
    }

    /// Every byte left, consuming them.
    pub fn rest(&mut self) -> &'a [u8] {
        let out = &self.buf[self.pos..];
        self.pos = self.buf.len();
        out
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Refuse a body with bytes left over.
    pub fn finish(self) -> Result<(), WireError> {
        match self.remaining() {
            0 => Ok(()),
            extra => Err(WireError::Trailing {
                what: self.what,
                extra,
            }),
        }
    }

    pub fn what(&self) -> &'static str {
        self.what
    }
}

#[cfg(test)]
mod tests {
    use super::super::Writer;
    use super::*;

    #[test]
    fn primitives_read_back_what_was_written() {
        let mut w = Writer::new();
        w.u8(7)
            .u16(0x0102)
            .u32(0x0304_0506)
            .u64(9)
            .bool(true)
            .str("hé")
            .list(&[1u64, 2], |w, v| {
                w.u64(*v);
            })
            .option(Some(&5u32), |w, v| {
                w.u32(*v);
            })
            .varint(300);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes, "t");
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.u16().unwrap(), 0x0102);
        assert_eq!(r.u32().unwrap(), 0x0304_0506);
        assert_eq!(r.u64().unwrap(), 9);
        assert!(r.bool().unwrap());
        assert_eq!(r.str().unwrap(), "hé");
        assert_eq!(r.list(|r| r.u64()).unwrap(), vec![1, 2]);
        assert_eq!(r.option(|r| r.u32()).unwrap(), Some(5));
        assert_eq!(r.varint().unwrap(), 300);
        r.finish().unwrap();
    }

    #[test]
    fn short_long_and_out_of_range_are_refused() {
        assert_eq!(
            Reader::new(&[1, 2, 3], "t").u32(),
            Err(WireError::Truncated { what: "t", need: 1 })
        );
        let mut r = Reader::new(&[0, 9], "t");
        r.u8().unwrap();
        assert_eq!(
            r.finish(),
            Err(WireError::Trailing {
                what: "t",
                extra: 1
            })
        );
        assert_eq!(
            Reader::new(&[2], "t").bool(),
            Err(WireError::Invalid {
                what: "t",
                value: 2
            })
        );
        assert_eq!(
            Reader::new(&[1, 0, 0, 0, 0xff], "t").str(),
            Err(WireError::NotUtf8 { what: "t" })
        );
        // A count larger than the bytes left is refused before allocating.
        assert_eq!(
            Reader::new(&[0xff, 0xff, 0xff, 0xff], "t").list(|r| r.u8()),
            Err(WireError::Truncated {
                what: "t",
                need: 0xffff_ffff
            })
        );
    }
}
