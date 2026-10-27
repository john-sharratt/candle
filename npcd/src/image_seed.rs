//! The seed a picture keeps in its own file.
//!
//! A draw's seed is what an edit of it should start from: a prompt edit with
//! the seed that drew its picture starts from the same noise, so at a low
//! `hold` the layout stays put rather than being drawn anew. The seed is kept
//! in the file itself — a JPEG comment segment, a PNG `tEXt` chunk — because
//! the file is already the record: an image is named by its hash, so the seed
//! needs no store entry, no replication and no expiry of its own, and it
//! travels with a downloaded picture that is uploaded again —
//! an upload is re-encoded, so it carries the seed across itself
//! (`api::put_upload`). Something outside Zen that re-encodes the file drops
//! it, and an edit then draws a fresh seed.
//!
//! A seed names the noise for one geometry: an edit that keeps the picture's
//! size and layout — a colour adjustment, a cutout — passes it on; a crop, a
//! resize or a turn does not, and the next prompt edit chooses its own.
//!
//! Every viewer ignores both: a JPEG `COM` segment and a PNG `tEXt` chunk are
//! the formats' own places for a note.

use candle_conversation::guest::EXACT_IN_JSON;
use crc32fast::Hasher;

/// What the note says before the number, in both formats.
const KEY: &str = "zen-seed";

const JPEG_SOI: [u8; 2] = [0xFF, 0xD8];
const JPEG_COM: u8 = 0xFE;
const JPEG_SOS: u8 = 0xDA;
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// `file` with `seed` kept in it, replacing a seed it already kept. A file in
/// neither format, or one too damaged to walk, is returned as it is.
pub fn stamp(file: &[u8], seed: u64) -> Vec<u8> {
    if file.starts_with(&JPEG_SOI) {
        stamp_jpeg(file, seed)
    } else if file.starts_with(&PNG_SIGNATURE) {
        stamp_png(file, seed)
    } else {
        file.to_vec()
    }
}

/// The seed `file` keeps, if it keeps one.
pub fn read(file: &[u8]) -> Option<u64> {
    if file.starts_with(&JPEG_SOI) {
        jpeg_segments(file)?
            .into_iter()
            .filter(|s| s.marker == JPEG_COM)
            .find_map(|s| seed_of(s.body(file), b':'))
    } else if file.starts_with(&PNG_SIGNATURE) {
        png_chunks(file)?
            .into_iter()
            .filter(|c| &c.kind == b"tEXt")
            .find_map(|c| seed_of(c.data(file), 0))
    } else {
        None
    }
}

/// `to` with `from`'s seed kept in it, when `from` keeps one: what an edit
/// that draws nothing new — a crop, a resize, a cutout — passes on.
pub fn carry(from: &[u8], to: Vec<u8>) -> Vec<u8> {
    match read(from) {
        Some(seed) => stamp(&to, seed),
        None => to,
    }
}

/// The number a note of `KEY`, `sep`, digits holds — only one a JSON number
/// carries exactly, since the seed is reported and may be passed back.
fn seed_of(note: &[u8], sep: u8) -> Option<u64> {
    let rest = note.strip_prefix(KEY.as_bytes())?.strip_prefix(&[sep])?;
    let seed: u64 = std::str::from_utf8(rest).ok()?.parse().ok()?;
    (seed <= EXACT_IN_JSON).then_some(seed)
}

/// A JPEG application segment — JFIF's APP0, Exif's APP1 — which readers
/// expect straight after SOI.
fn is_app(marker: u8) -> bool {
    (0xE0..=0xEF).contains(&marker)
}

/// One JPEG marker segment before the scan: its marker and where its body
/// lies (after the two length bytes).
struct Segment {
    marker: u8,
    start: usize,
    end: usize,
}

impl Segment {
    fn body<'a>(&self, file: &'a [u8]) -> &'a [u8] {
        &file[self.start..self.end]
    }
}

/// The marker segments between a JPEG's SOI and its first scan; `None` for a
/// file that does not walk.
fn jpeg_segments(file: &[u8]) -> Option<Vec<Segment>> {
    let mut out = Vec::new();
    let mut at = 2;
    loop {
        if *file.get(at)? != 0xFF {
            return None;
        }
        let marker = *file.get(at + 1)?;
        if marker == JPEG_SOS {
            return Some(out);
        }
        let len = u16::from_be_bytes([*file.get(at + 2)?, *file.get(at + 3)?]) as usize;
        if len < 2 || at + 2 + len > file.len() {
            return None;
        }
        out.push(Segment {
            marker,
            start: at + 4,
            end: at + 2 + len,
        });
        at += 2 + len;
    }
}

fn stamp_jpeg(file: &[u8], seed: u64) -> Vec<u8> {
    let Some(segments) = jpeg_segments(file) else {
        return file.to_vec();
    };
    // After the application segments that lead the file, where readers look
    // for JFIF and Exif.
    let at = segments
        .iter()
        .take_while(|s| is_app(s.marker))
        .last()
        .map_or(JPEG_SOI.len(), |s| s.end);
    let note = format!("{KEY}:{seed}");
    let len = (note.len() + 2) as u16;
    let mut out = Vec::with_capacity(file.len() + note.len() + 4);
    out.extend_from_slice(&file[..at]);
    out.extend_from_slice(&[0xFF, JPEG_COM]);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(note.as_bytes());
    // Everything after as it was, less a seed kept before.
    let mut copied = at;
    for s in segments.iter().filter(|s| s.start - 4 >= at) {
        if s.marker == JPEG_COM && seed_of(s.body(file), b':').is_some() {
            out.extend_from_slice(&file[copied..s.start - 4]);
            copied = s.end;
        }
    }
    out.extend_from_slice(&file[copied..]);
    out
}

/// One PNG chunk: its type and where its data lies.
struct Chunk {
    kind: [u8; 4],
    /// The chunk's first byte (its length field).
    start: usize,
    data: (usize, usize),
    /// One past its CRC.
    end: usize,
}

impl Chunk {
    fn data<'a>(&self, file: &'a [u8]) -> &'a [u8] {
        &file[self.data.0..self.data.1]
    }
}

/// A PNG's chunks; `None` for a file that does not walk.
fn png_chunks(file: &[u8]) -> Option<Vec<Chunk>> {
    let mut out = Vec::new();
    let mut at = PNG_SIGNATURE.len();
    while at < file.len() {
        let len = u32::from_be_bytes(file.get(at..at + 4)?.try_into().ok()?) as usize;
        let kind: [u8; 4] = file.get(at + 4..at + 8)?.try_into().ok()?;
        let end = at + 12 + len;
        if end > file.len() {
            return None;
        }
        out.push(Chunk {
            kind,
            start: at,
            data: (at + 8, at + 8 + len),
            end,
        });
        at = end;
        if &kind == b"IEND" {
            break;
        }
    }
    Some(out)
}

fn stamp_png(file: &[u8], seed: u64) -> Vec<u8> {
    let Some(chunks) = png_chunks(file) else {
        return file.to_vec();
    };
    let Some(ihdr) = chunks.first().filter(|c| &c.kind == b"IHDR") else {
        return file.to_vec();
    };
    let mut data = Vec::with_capacity(KEY.len() + 21);
    data.extend_from_slice(KEY.as_bytes());
    data.push(0);
    data.extend_from_slice(seed.to_string().as_bytes());
    let mut crc = Hasher::new();
    crc.update(b"tEXt");
    crc.update(&data);
    let mut out = Vec::with_capacity(file.len() + data.len() + 12);
    out.extend_from_slice(&file[..ihdr.end]);
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(b"tEXt");
    out.extend_from_slice(&data);
    out.extend_from_slice(&crc.finalize().to_be_bytes());
    // Every chunk after IHDR as it was, less a seed kept before, and anything
    // a tool left past the last chunk.
    for c in &chunks[1..] {
        if &c.kind == b"tEXt" && seed_of(c.data(file), 0).is_some() {
            continue;
        }
        out.extend_from_slice(&file[c.start..c.end]);
    }
    let walked = chunks.last().map_or(PNG_SIGNATURE.len(), |c| c.end);
    out.extend_from_slice(&file[walked..]);
    out
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{load_from_memory, DynamicImage, ImageFormat, Rgb, RgbImage};

    use super::*;

    fn picture(format: ImageFormat) -> Vec<u8> {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 2, Rgb([200, 40, 9])));
        let mut out = Cursor::new(Vec::new());
        img.write_to(&mut out, format).unwrap();
        out.into_inner()
    }

    /// A JFIF APP0 segment: what most encoders put straight after SOI.
    const APP0: [u8; 18] = [
        0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00,
        0x01, 0x00, 0x00,
    ];

    /// The comment segment `zen-seed:42` is kept in.
    fn com_42() -> Vec<u8> {
        let mut com = vec![0xFF, 0xFE, 0x00, 13];
        com.extend_from_slice(b"zen-seed:42");
        com
    }

    /// **A JPEG keeps its seed in a comment after its leading application
    /// segments** — straight after SOI with none, after JFIF's APP0 with one —
    /// byte for byte, and still decodes to the same pixels.
    #[test]
    fn a_jpeg_keeps_its_seed_in_a_comment() {
        let bare = picture(ImageFormat::Jpeg);
        let body = if bare[2..4] == [0xFF, 0xE0] {
            let len = u16::from_be_bytes([bare[4], bare[5]]) as usize;
            bare[4 + len..].to_vec()
        } else {
            bare[2..].to_vec()
        };
        let plain: Vec<u8> = [&JPEG_SOI[..], &body].concat();
        let jfif: Vec<u8> = [&JPEG_SOI[..], &APP0, &body].concat();

        let want: Vec<u8> = [&JPEG_SOI[..], &com_42(), &body].concat();
        assert_eq!(stamp(&plain, 42), want);
        let want: Vec<u8> = [&JPEG_SOI[..], &APP0, &com_42(), &body].concat();
        assert_eq!(stamp(&jfif, 42), want, "JFIF stays straight after SOI");

        assert_eq!(read(&stamp(&jfif, 42)), Some(42));
        assert_eq!(read(&jfif), None);
        let a = load_from_memory(&jfif).unwrap().to_rgb8();
        let b = load_from_memory(&stamp(&jfif, 42)).unwrap().to_rgb8();
        assert_eq!(a, b);
    }

    /// **A seed a JSON number would round is not read**: it is reported, and
    /// passed back it would name another draw.
    #[test]
    fn a_seed_past_two_to_the_fifty_three_is_not_read() {
        let png = picture(ImageFormat::Png);
        assert_eq!(read(&stamp(&png, EXACT_IN_JSON)), Some(EXACT_IN_JSON));
        assert_eq!(read(&stamp(&png, EXACT_IN_JSON + 1)), None);
    }

    /// **Bytes past a PNG's last chunk are kept**, as a tool left them.
    #[test]
    fn a_pngs_trailing_bytes_are_kept() {
        let mut png = picture(ImageFormat::Png);
        png.extend_from_slice(b"trailer");
        assert!(stamp(&png, 3).ends_with(b"IEND\xAE\x42\x60\x82trailer"));
    }

    /// **A PNG keeps its seed in a `tEXt` chunk after IHDR**, with the CRC
    /// the format asks for, and still decodes.
    #[test]
    fn a_png_keeps_its_seed_in_a_text_chunk() {
        let png = picture(ImageFormat::Png);
        let stamped = stamp(&png, 7);
        let ihdr_end = 8 + 12 + 13;
        let chunk = &stamped[ihdr_end..ihdr_end + 12 + 10];
        assert_eq!(&chunk[..4], &[0, 0, 0, 10]);
        assert_eq!(&chunk[4..8], b"tEXt");
        assert_eq!(&chunk[8..18], b"zen-seed\x007");
        let mut crc = Hasher::new();
        crc.update(b"tEXtzen-seed\x007");
        assert_eq!(&chunk[18..22], &crc.finalize().to_be_bytes());
        assert_eq!(read(&stamped), Some(7));
        assert_eq!(read(&png), None);
        assert!(load_from_memory(&stamped).is_ok());
    }

    /// **Stamping again replaces the seed**, never keeps two.
    #[test]
    fn a_second_stamp_replaces_the_first() {
        for format in [ImageFormat::Jpeg, ImageFormat::Png] {
            let file = picture(format);
            let twice = stamp(&stamp(&file, 1), 9_007_199_254_740_991);
            assert_eq!(read(&twice), Some(9_007_199_254_740_991));
            assert_eq!(twice.len(), stamp(&file, 9_007_199_254_740_991).len());
        }
    }

    /// **An edit that keeps the layout passes the seed on**; a source with
    /// none passes none.
    #[test]
    fn a_seed_is_carried_from_the_source() {
        let from = stamp(&picture(ImageFormat::Jpeg), 5);
        let to = picture(ImageFormat::Png);
        assert_eq!(read(&carry(&from, to.clone())), Some(5));
        assert_eq!(carry(&picture(ImageFormat::Jpeg), to.clone()), to);
    }

    /// Anything else is left alone.
    #[test]
    fn a_file_in_neither_format_is_untouched() {
        assert_eq!(stamp(b"GIF89a", 3), b"GIF89a");
        assert_eq!(stamp(&[0xFF, 0xD8, 0x00], 3), [0xFF, 0xD8, 0x00]);
        assert_eq!(read(b"GIF89a"), None);
    }
}
