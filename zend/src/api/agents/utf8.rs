//! A command's output as text, decoded as it arrives.
//!
//! A log is read in byte chunks, and a chunk can end inside a character: the
//! bytes of a character cut in two are held back until the rest arrives,
//! rather than each half being shown as a replacement mark. Bytes that are not
//! UTF-8 at all — a command's output need not be — are shown as `U+FFFD`.

/// The decoder for one log.
#[derive(Debug, Default)]
pub struct Utf8Stream {
    held: Vec<u8>,
}

/// How many trailing bytes of `b` begin a character that has not ended.
fn unfinished(b: &[u8]) -> usize {
    for back in 1..=b.len().min(3) {
        let byte = b[b.len() - back];
        if byte & 0xC0 == 0x80 {
            continue;
        }
        let needs = match byte {
            0xF0.. => 4,
            0xE0.. => 3,
            0xC0.. => 2,
            _ => 1,
        };
        return if needs > back { back } else { 0 };
    }
    0
}

impl Utf8Stream {
    /// The text `bytes` complete, holding back a character they cut short.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.held.extend_from_slice(bytes);
        let keep = unfinished(&self.held);
        let tail = self.held.split_off(self.held.len() - keep);
        let text = String::from_utf8_lossy(&self.held).into_owned();
        self.held = tail;
        text
    }

    /// Whatever is still held, once the log has ended.
    pub fn finish(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.held).into_owned();
        self.held.clear();
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A character cut between two chunks is held until it is whole**, and
    /// the text either side passes straight through.
    #[test]
    fn a_character_cut_in_two_is_held_until_whole() {
        let mut s = Utf8Stream::default();
        // "ok → done": the arrow is E2 86 92.
        assert_eq!(s.push(b"ok \xE2\x86"), "ok ");
        assert_eq!(s.push(b"\x92 done\n"), "\u{2192} done\n");
        // A four-byte character cut after its first byte.
        assert_eq!(s.push(b"\xF0"), "");
        assert_eq!(s.push(b"\x9F\x92"), "");
        assert_eq!(s.push(b"\xA1!"), "\u{1F4A1}!");
        assert_eq!(s.finish(), "");
    }

    /// Bytes that are not UTF-8 come out as replacement marks, and a cut
    /// character still held when the log ends is one too.
    #[test]
    fn bytes_that_are_not_utf8_are_marked() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(b"a\xFFb\x80c"), "a\u{FFFD}b\u{FFFD}c");
        assert_eq!(s.push(b"x\xE2\x86"), "x");
        assert_eq!(s.finish(), "\u{FFFD}");
        assert_eq!(s.push(b"\n"), "\n");
    }
}
