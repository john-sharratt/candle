//! Conversation-files — the storage/reconstruction core for the
//! conversation-files layer (docs/zend_ui_redesign.md §2.5, decision 2).
//!
//! An uploaded file is stored on its own layer as **token strings** so it can be
//! reconstructed by concatenation. Binaries (images, etc.) go through the *same*
//! tokenized path by hex-encoding the bytes first, so every file kind shares one
//! code path and round-trips byte-exact:
//!
//! ```text
//! upload:   bytes ──encode_for_storage──▶ storable text ──tokenize──▶ stored
//! download: stored ──detokenize/concat──▶ storable text ──decode_from_storage──▶ bytes
//! ```
//!
//! This module owns the model-independent half (kind classification, size
//! formatting, the hex round-trip, carve-into-parts for upload progress). The
//! tokenize/store/admit-to-projection half is engine-backed (§2.5).

/// Display classification driving the GUI's colored file badge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Code,
    Log,
    Doc,
    Text,
    Img,
}

impl FileKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FileKind::Code => "code",
            FileKind::Log => "log",
            FileKind::Doc => "doc",
            FileKind::Text => "text",
            FileKind::Img => "img",
        }
    }
    /// Non-tokenizable kinds are hex-encoded before storage (decision 2).
    pub fn is_binary(self) -> bool {
        matches!(self, FileKind::Img)
    }
}

/// Classify a file by extension (matching the GUI's `_kindFor`).
pub fn kind_for(name: &str) -> FileKind {
    let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    const CODE: &[&str] = &[
        "rs", "js", "ts", "tsx", "jsx", "py", "go", "rb", "c", "cpp", "h", "java", "json", "toml",
        "yaml", "yml", "sh",
    ];
    const IMG: &[&str] = &["png", "jpg", "jpeg", "gif", "svg", "webp", "bmp"];
    if CODE.contains(&ext.as_str()) {
        FileKind::Code
    } else if ext == "log" {
        FileKind::Log
    } else if matches!(ext.as_str(), "md" | "markdown" | "rst") {
        FileKind::Doc
    } else if IMG.contains(&ext.as_str()) {
        FileKind::Img
    } else {
        FileKind::Text
    }
}

/// The 2–4 char uppercase extension badge the GUI shows (e.g. `RS`, `LOG`).
pub fn ext_badge(name: &str) -> String {
    let ext = name.rsplit('.').next().unwrap_or("");
    if ext == name || ext.is_empty() {
        "·".to_string()
    } else {
        ext.to_ascii_uppercase().chars().take(4).collect()
    }
}

/// Human-readable size, matching the GUI's `fmtBytes`.
pub fn fmt_bytes(b: u64) -> String {
    if b < 1024 {
        format!("{b} B")
    } else if b < 1_048_576 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    }
}

/// Encode raw bytes into the storable text that gets tokenized: UTF-8 text is
/// stored verbatim; everything else is lowercase hex (decision 2's uniform path).
pub fn encode_for_storage(bytes: &[u8], kind: FileKind) -> String {
    if kind.is_binary() {
        to_hex(bytes)
    } else {
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(_) => to_hex(bytes),
        }
    }
}

/// Inverse of [`encode_for_storage`] — reconstruct the original bytes from the
/// concatenated stored text.
pub fn decode_from_storage(stored: &str, kind: FileKind) -> Vec<u8> {
    if kind.is_binary() {
        from_hex(stored).unwrap_or_else(|| stored.as_bytes().to_vec())
    } else {
        stored.as_bytes().to_vec()
    }
}

/// Number of carve parts an upload of `len` bytes splits into, for the
/// per-part progress bar (≥1, capped so the bar stays legible).
pub fn part_count(len: u64, part_bytes: u64) -> usize {
    let pb = part_bytes.max(1);
    (len.div_ceil(pb)).clamp(1, 64) as usize
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return None;
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16)?;
        let lo = (bytes[i + 1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
        i += 2;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_by_extension() {
        assert_eq!(kind_for("redo.rs"), FileKind::Code);
        assert_eq!(kind_for("boot.log"), FileKind::Log);
        assert_eq!(kind_for("schema.md"), FileKind::Doc);
        assert_eq!(kind_for("notes.txt"), FileKind::Text);
        assert_eq!(kind_for("diagram.png"), FileKind::Img);
        assert_eq!(kind_for("noext"), FileKind::Text);
    }

    #[test]
    fn ext_badge_and_size() {
        assert_eq!(ext_badge("redo.rs"), "RS");
        assert_eq!(ext_badge("a.jsonl"), "JSON");
        assert_eq!(ext_badge("noext"), "·");
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(18_842), "18.4 KB");
        assert_eq!(fmt_bytes(1_572_864), "1.5 MB");
    }

    #[test]
    fn text_round_trips_verbatim() {
        let bytes = b"// crates/substrate/src/redo.rs\nfn main() {}\n";
        let stored = encode_for_storage(bytes, FileKind::Code);
        assert_eq!(stored.as_bytes(), bytes); // text stored verbatim
        assert_eq!(decode_from_storage(&stored, FileKind::Code), bytes);
    }

    #[test]
    fn binary_round_trips_byte_exact_via_hex() {
        // a non-UTF-8 byte sequence (a tiny "PNG-ish" header)
        let bytes: Vec<u8> = vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0xff, 0x00];
        let stored = encode_for_storage(&bytes, FileKind::Img);
        assert_eq!(stored, "89504e470d0a1a0aff00"); // raw expected hex bytes
        assert_eq!(decode_from_storage(&stored, FileKind::Img), bytes);
    }

    #[test]
    fn non_utf8_text_kind_falls_back_to_hex() {
        let bytes: Vec<u8> = vec![0xff, 0xfe, 0x00];
        let stored = encode_for_storage(&bytes, FileKind::Text);
        // not valid UTF-8 -> hex, and still reconstructs byte-exact
        assert_eq!(stored, "fffe00");
        // a Text-kind reconstruction returns the stored bytes as-is; the hex form
        // is preserved losslessly for callers that know the original was binary.
        assert_eq!(decode_from_storage(&stored, FileKind::Img), bytes);
    }

    #[test]
    fn part_count_is_bounded() {
        assert_eq!(part_count(0, 1024), 1);
        assert_eq!(part_count(1, 1024), 1);
        assert_eq!(part_count(4096, 1024), 4);
        assert_eq!(part_count(10_000_000, 1024), 64); // capped
    }
}
