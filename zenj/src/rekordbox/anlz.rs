//! rekordbox's analysis files (`ANLZ0000.DAT` and `.EXT`): big-endian
//! tagged sections after a `PMAI` header. The ones a mix needs:
//!
//! | Tag | File | What |
//! |---|---|---|
//! | `PPTH` | DAT | the track's path, UTF-16BE |
//! | `PQTZ` | DAT | the beat grid: each beat's number in its bar, tempo ×100, time in ms |
//! | `PCOB` | DAT | the cue points, memory and hot (`PCPT` entries) |
//! | `PSSI` | EXT | the phrases, masked since rekordbox 6 |
//!
//! Layouts after Deep Symmetry's `crate-digger` analysis.

use crate::music::{HotCue, Phrase, PhraseKind};

/// One beat of the grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beat {
    /// 1–4: its place in the bar.
    pub number: u16,
    pub bpm: f32,
    pub ms: u32,
}

/// One cue point, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cue {
    /// 0 for a memory cue, else the pad 1–8.
    pub hot: u8,
    pub ms: u32,
}

/// One phrase as rekordbox writes it: from `beat` (1-based on the grid).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawPhrase {
    pub beat: u16,
    pub kind: u16,
}

/// The phrases' section.
#[derive(Debug, Clone, PartialEq)]
pub struct Phrases {
    /// 1 high, 2 mid, 3 low.
    pub mood: u16,
    pub end_beat: u16,
    pub entries: Vec<RawPhrase>,
}

/// What one analysis file held.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Anlz {
    pub path: Option<String>,
    pub beats: Vec<Beat>,
    pub cues: Vec<Cue>,
    pub phrases: Option<Phrases>,
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

/// `PSSI`'s mask, from offset 18 of the tag, each byte offset by the
/// entry count.
const MASK: [u8; 19] = [
    0xCB, 0xE1, 0xEE, 0xFA, 0xE5, 0xEE, 0xAD, 0xEE, 0xE9, 0xD2, 0xE9, 0xEB, 0xE1, 0xE9, 0xF3, 0xE8,
    0xE9, 0xF4, 0xE1,
];

/// `tag` with the phrase mask applied (it is its own inverse).
pub fn unmask(tag: &[u8]) -> Vec<u8> {
    let mut t = tag.to_vec();
    let Some(n) = u16_at(tag, 16) else {
        return t;
    };
    for (i, b) in t.iter_mut().enumerate().skip(18) {
        *b ^= MASK[(i - 18) % MASK.len()].wrapping_add(n as u8);
    }
    t
}

fn phrases_in(tag: &[u8]) -> Option<Phrases> {
    let header = u32_at(tag, 4)? as usize;
    let entry = u32_at(tag, 12)? as usize;
    let n = u16_at(tag, 16)? as usize;
    let mood = u16_at(tag, 18)?;
    if !(1..=3).contains(&mood) || entry < 6 {
        return None;
    }
    let end_beat = u16_at(tag, 26)?;
    let mut entries = Vec::with_capacity(n);
    for i in 0..n {
        let at = header + i * entry;
        entries.push(RawPhrase {
            beat: u16_at(tag, at + 2)?,
            kind: u16_at(tag, at + 4)?,
        });
    }
    entries
        .windows(2)
        .all(|w| w[0].beat < w[1].beat)
        .then_some(Phrases {
            mood,
            end_beat,
            entries,
        })
}

/// The phrases of a `PSSI` tag, masked or not: whichever reading has a mood
/// and beats that rise.
fn phrases(tag: &[u8]) -> Option<Phrases> {
    phrases_in(tag).or_else(|| phrases_in(&unmask(tag)))
}

fn beats(tag: &[u8]) -> Option<Vec<Beat>> {
    let header = u32_at(tag, 4)? as usize;
    let n = u32_at(tag, 20)? as usize;
    (0..n)
        .map(|i| {
            let at = header + i * 8;
            Some(Beat {
                number: u16_at(tag, at)?,
                bpm: u16_at(tag, at + 2)? as f32 / 100.0,
                ms: u32_at(tag, at + 4)?,
            })
        })
        .collect()
}

fn path(tag: &[u8]) -> Option<String> {
    let len = u32_at(tag, 12)? as usize;
    let raw = tag.get(16..16 + len)?;
    let units: Vec<u16> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_be_bytes(*c))
        .take_while(|&u| u != 0)
        .collect();
    Some(String::from_utf16_lossy(&units))
}

fn cues(tag: &[u8]) -> Option<Vec<Cue>> {
    let header = u32_at(tag, 4)? as usize;
    let n = u16_at(tag, 18)? as usize;
    let mut out = Vec::with_capacity(n);
    let mut at = header;
    for _ in 0..n {
        let entry = tag.get(at..)?;
        if entry.get(..4)? != b"PCPT" {
            return None;
        }
        let len = u32_at(entry, 8)? as usize;
        out.push(Cue {
            hot: u32_at(entry, 12)? as u8,
            ms: u32_at(entry, 32)?,
        });
        at += len.max(1);
    }
    Some(out)
}

/// Every section of `file` this reads; one that does not parse is left out.
pub fn parse(file: &[u8]) -> Result<Anlz, String> {
    if file.get(..4) != Some(b"PMAI") {
        return Err("not an analysis file: no PMAI header".into());
    }
    let mut at = u32_at(file, 4).ok_or("a short header")? as usize;
    let mut out = Anlz::default();
    while at + 12 <= file.len() {
        let fourcc = &file[at..at + 4];
        let len = u32_at(file, at + 8).ok_or("a short tag")? as usize;
        if len < 12 {
            return Err(format!("a tag {} bytes long at {at}", len));
        }
        let tag = file.get(at..at + len).ok_or("a tag past the file's end")?;
        match fourcc {
            b"PPTH" => out.path = path(tag),
            b"PQTZ" => out.beats = beats(tag).unwrap_or_default(),
            b"PCOB" => out.cues.extend(cues(tag).unwrap_or_default()),
            b"PSSI" => out.phrases = phrases(tag),
            _ => {}
        }
        at += len;
    }
    Ok(out)
}

/// A phrase kind in mood `mood`, folded into what a mix needs.
pub fn kind(mood: u16, kind: u16) -> Option<PhraseKind> {
    use PhraseKind::*;
    Some(match (mood, kind) {
        (1, 1) => Intro,
        (1, 2) => Up,
        (1, 3) => Down,
        (1, 5) => Chorus,
        (1, 6) => Outro,
        (2 | 3, 1) => Intro,
        (2 | 3, 2..=7) => Up,
        (2 | 3, 8) => Bridge,
        (2 | 3, 9) => Chorus,
        (2 | 3, 10) => Outro,
        _ => return None,
    })
}

impl Phrases {
    /// The phrases as the profile counts them: 0-based beats, each to the
    /// next's start, the last to the end.
    pub fn folded(&self) -> Vec<Phrase> {
        let mut out = Vec::new();
        for (i, e) in self.entries.iter().enumerate() {
            let end = self.entries.get(i + 1).map_or(self.end_beat, |n| n.beat);
            let Some(kind) = kind(self.mood, e.kind) else {
                continue;
            };
            let (start, end) = (
                e.beat.saturating_sub(1) as u32,
                end.saturating_sub(1) as u32,
            );
            if end > start {
                out.push(Phrase { kind, start, end });
            }
        }
        out
    }
}

impl Anlz {
    /// The hot cues as beats on a grid whose beat 0 is at `first_ms` with
    /// `ms_per_beat`.
    pub fn hot_cues(&self, first_ms: f64, ms_per_beat: f64) -> Vec<HotCue> {
        self.cues
            .iter()
            .filter(|c| c.hot > 0)
            .map(|c| HotCue {
                pad: c.hot,
                beat: (c.ms as f64 - first_ms) / ms_per_beat,
            })
            .collect()
    }

    /// The first downbeat's time and the grid's dominant tempo.
    pub fn grid(&self) -> Option<(f64, f32)> {
        let first = self.beats.iter().find(|b| b.number == 1)?;
        let mut tempos: Vec<u32> = self.beats.iter().map(|b| (b.bpm * 100.0) as u32).collect();
        tempos.sort_unstable();
        let median = tempos[tempos.len() / 2] as f32 / 100.0;
        Some((first.ms as f64, median))
    }
}

#[cfg(test)]
pub mod build {
    //! Analysis files as bytes, for tests.

    fn be16(v: u16) -> [u8; 2] {
        v.to_be_bytes()
    }

    fn be32(v: u32) -> [u8; 4] {
        v.to_be_bytes()
    }

    pub fn tag(fourcc: &[u8; 4], header: &[u8], body: &[u8]) -> Vec<u8> {
        let len_header = 12 + header.len();
        let mut t = Vec::new();
        t.extend_from_slice(fourcc);
        t.extend_from_slice(&be32(len_header as u32));
        t.extend_from_slice(&be32((len_header + body.len()) as u32));
        t.extend_from_slice(header);
        t.extend_from_slice(body);
        t
    }

    pub fn file(tags: &[Vec<u8>]) -> Vec<u8> {
        let body: Vec<u8> = tags.concat();
        let mut f = b"PMAI".to_vec();
        f.extend_from_slice(&be32(28));
        f.extend_from_slice(&be32(28 + body.len() as u32));
        f.extend_from_slice(&[0u8; 16]);
        f.extend(body);
        f
    }

    pub fn grid(beats: &[(u16, u16, u32)]) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend_from_slice(&be32(0));
        header.extend_from_slice(&be32(0x80000));
        header.extend_from_slice(&be32(beats.len() as u32));
        let mut body = Vec::new();
        for &(n, tempo, ms) in beats {
            body.extend_from_slice(&be16(n));
            body.extend_from_slice(&be16(tempo));
            body.extend_from_slice(&be32(ms));
        }
        tag(b"PQTZ", &header, &body)
    }

    pub fn path(p: &str) -> Vec<u8> {
        let mut units: Vec<u8> = p.encode_utf16().flat_map(|u| u.to_be_bytes()).collect();
        units.extend_from_slice(&[0, 0]);
        tag(b"PPTH", &be32(units.len() as u32), &units)
    }

    pub fn cues(hot: &[(u32, u32)]) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend_from_slice(&be32(1));
        header.extend_from_slice(&be16(0));
        header.extend_from_slice(&be16(hot.len() as u16));
        header.extend_from_slice(&be32(u32::MAX));
        let mut body = Vec::new();
        for &(pad, ms) in hot {
            let mut e = Vec::new();
            e.extend_from_slice(&be32(pad));
            e.extend_from_slice(&be32(4));
            e.extend_from_slice(&be32(0x10000));
            e.extend_from_slice(&be16(0xFFFF));
            e.extend_from_slice(&be16(0xFFFF));
            e.extend_from_slice(&[1, 0]);
            e.extend_from_slice(&be16(1000));
            e.extend_from_slice(&be32(ms));
            e.extend_from_slice(&be32(u32::MAX));
            e.resize(0x38 - 12, 0);
            body.extend(tag(b"PCPT", &e[..16], &e[16..]));
        }
        tag(b"PCOB", &header, &body)
    }

    /// A `PSSI` tag, masked as rekordbox 6+ writes it when `masked`.
    pub fn phrases(mood: u16, end_beat: u16, entries: &[(u16, u16)], masked: bool) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend_from_slice(&be32(24));
        header.extend_from_slice(&be16(entries.len() as u16));
        header.extend_from_slice(&be16(mood));
        header.extend_from_slice(&[0u8; 6]);
        header.extend_from_slice(&be16(end_beat));
        header.extend_from_slice(&be16(0));
        header.extend_from_slice(&[0, 0]);
        let mut body = Vec::new();
        for (i, &(beat, kind)) in entries.iter().enumerate() {
            let mut e = Vec::new();
            e.extend_from_slice(&be16(i as u16 + 1));
            e.extend_from_slice(&be16(beat));
            e.extend_from_slice(&be16(kind));
            e.resize(24, 0);
            body.extend(e);
        }
        let t = tag(b"PSSI", &header, &body);
        if masked {
            super::unmask(&t)
        } else {
            t
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A DAT file reads to its path, grid and hot cues**, from bytes laid
    /// out as rekordbox writes them.
    #[test]
    fn a_dat_file_reads_its_path_grid_and_cues() {
        let f = build::file(&[
            build::path("/Music/TIDAL/123.flac"),
            build::grid(&[
                (1, 12800, 120),
                (2, 12800, 589),
                (3, 12800, 1058),
                (4, 12850, 1527),
            ]),
            build::cues(&[(1, 120), (0, 600), (3, 15120)]),
            build::tag(b"PWAV", &[0, 0, 0, 0], &[9, 9]),
        ]);
        let a = parse(&f).unwrap();
        assert_eq!(a.path.as_deref(), Some("/Music/TIDAL/123.flac"));
        assert_eq!(a.beats.len(), 4);
        assert_eq!(
            a.beats[3],
            Beat {
                number: 4,
                bpm: 128.5,
                ms: 1527
            }
        );
        assert_eq!(a.grid(), Some((120.0, 128.0)));
        assert_eq!(
            a.cues,
            vec![
                Cue { hot: 1, ms: 120 },
                Cue { hot: 0, ms: 600 },
                Cue { hot: 3, ms: 15120 }
            ]
        );
        let hot = a.hot_cues(120.0, 468.75);
        assert_eq!(hot.len(), 2, "memory cues are not pads");
        assert_eq!((hot[1].pad, hot[1].beat), (3, 32.0));
    }

    /// **Phrases read masked or plain**, and fold to 0-based beats, each to
    /// the next's start — an unknown kind left out.
    #[test]
    fn phrases_read_masked_or_plain() {
        let entries = [(1, 1), (33, 2), (65, 9), (129, 4), (161, 9), (225, 10)];
        for masked in [false, true] {
            let f = build::file(&[build::phrases(3, 257, &entries, masked)]);
            let p = parse(&f).unwrap().phrases.unwrap();
            assert_eq!(
                (p.mood, p.end_beat, p.entries.len()),
                (3, 257, 6),
                "masked {masked}"
            );
            use PhraseKind::*;
            assert_eq!(
                p.folded()
                    .iter()
                    .map(|p| (p.kind, p.start, p.end))
                    .collect::<Vec<_>>(),
                vec![
                    (Intro, 0, 32),
                    (Up, 32, 64),
                    (Chorus, 64, 128),
                    (Up, 128, 160),
                    (Chorus, 160, 224),
                    (Outro, 224, 256),
                ]
            );
        }
        let masked = build::phrases(1, 100, &[(1, 1)], true);
        assert_ne!(
            masked,
            build::phrases(1, 100, &[(1, 1)], false),
            "the mask changes bytes"
        );
        assert_eq!(kind(1, 4), None, "high mood has no kind 4");
    }

    #[test]
    fn a_file_that_is_not_analysis_is_refused() {
        assert_eq!(
            parse(b"RIFF....").unwrap_err(),
            "not an analysis file: no PMAI header"
        );
        let mut f = build::file(&[build::path("x")]);
        let at = 28 + 8;
        f[at..at + 4].copy_from_slice(&9999u32.to_be_bytes());
        assert_eq!(parse(&f).unwrap_err(), "a tag past the file's end");
    }
}
