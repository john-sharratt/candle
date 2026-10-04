//! A record payload as the slices it is written from.
//!
//! A recurrent snapshot is ~65 MB of state the conversation already holds as
//! per-layer blobs. Concatenating it into one buffer before writing cost a full
//! copy into fresh, page-faulting memory — and the record framing and the
//! group-commit buffer each copied it again. Laid out here instead as small
//! encoded fields interleaved with borrowed blobs, the payload is checksummed
//! across the slices and written slice by slice, so the state bytes go from the
//! conversation's buffers to the file without an intermediate copy.

/// Little-endian fields and borrowed blobs, in wire order. Their concatenation
/// is the payload's encoded form.
#[derive(Default)]
pub struct PayloadParts<'a> {
    fields: Vec<u8>,
    spans: Vec<Span<'a>>,
}

enum Span<'a> {
    /// `fields[start..end]`.
    Fields(usize, usize),
    Blob(&'a [u8]),
}

impl<'a> PayloadParts<'a> {
    pub fn u8(&mut self, v: u8) {
        self.field(&[v]);
    }

    pub fn u32(&mut self, v: u32) {
        self.field(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.field(&v.to_le_bytes());
    }

    pub fn raw(&mut self, bytes: &[u8]) {
        self.field(bytes);
    }

    /// A `u32` length followed by the bytes, which are borrowed, not copied.
    pub fn blob(&mut self, bytes: &'a [u8]) {
        self.u32(bytes.len() as u32);
        if !bytes.is_empty() {
            self.spans.push(Span::Blob(bytes));
        }
    }

    /// Append to the open run of fields, or open one after a blob.
    fn field(&mut self, bytes: &[u8]) {
        let start = self.fields.len();
        self.fields.extend_from_slice(bytes);
        let end = self.fields.len();
        match self.spans.last_mut() {
            Some(Span::Fields(_, open_end)) if *open_end == start => *open_end = end,
            _ => self.spans.push(Span::Fields(start, end)),
        }
    }

    /// The payload's slices, in order.
    pub fn slices(&self) -> Vec<&[u8]> {
        self.spans
            .iter()
            .map(|s| match *s {
                Span::Fields(start, end) => &self.fields[start..end],
                Span::Blob(b) => b,
            })
            .collect()
    }

    /// The encoded payload's length in bytes.
    pub fn byte_len(&self) -> usize {
        self.spans
            .iter()
            .map(|s| match *s {
                Span::Fields(start, end) => end - start,
                Span::Blob(b) => b.len(),
            })
            .sum()
    }

    /// The encoded payload as one buffer, sized once.
    pub fn concat(&self) -> Vec<u8> {
        let slices = self.slices();
        let mut out = Vec::with_capacity(self.byte_len());
        for s in slices {
            out.extend_from_slice(s);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_between_blobs_run_together_and_blobs_are_length_prefixed() {
        let state = [9u8, 8, 7];
        let mut p = PayloadParts::default();
        p.u32(1);
        p.u8(2);
        p.blob(&state);
        p.u64(3);
        p.blob(&[]);
        let slices = p.slices();
        // Version and tag and the blob's length are one run; the blob is its own
        // slice; the trailing u64 and the empty blob's length are the last run.
        assert_eq!(slices.len(), 3);
        assert_eq!(slices[0], &[1, 0, 0, 0, 2, 3, 0, 0, 0]);
        assert_eq!(slices[1], &[9, 8, 7]);
        assert_eq!(slices[2], &[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            p.concat(),
            vec![1, 0, 0, 0, 2, 3, 0, 0, 0, 9, 8, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(p.byte_len(), 24);
    }

    #[test]
    fn the_blob_is_borrowed_not_copied() {
        let state = vec![5u8; 64];
        let mut p = PayloadParts::default();
        p.blob(&state);
        assert!(std::ptr::eq(p.slices()[1].as_ptr(), state.as_ptr()));
    }
}
