//! How a turn says it holds an image, in its text and in its ids.
//!
//! In the ids: `<|vision_start|>`, one `<|image_pad|>` per row, `<|vision_end|>`.
//! In the text: `[image <id> <W>×<H>]`, the image's 16-hex id and the size it
//! arrived at — what a transcript, a history view or a summariser sees, and
//! what names the image's rows in the substrate. A turn's images open its user
//! half — the markers its text, one a line, and the blocks its ids, back to
//! back — and the `k`-th block is the `k`-th marker, so a path that rebuilds a
//! turn from its ids and text finds its images again. Only that head is read:
//! text that looks like a marker further on is the user's own.

use super::prepare::ID_HEX;

/// The marker a turn's text holds for an image.
pub fn marker(id: &str, width: u32, height: u32) -> String {
    format!("[image {id} {width}×{height}]")
}

/// The image id a marker line names, when `line` is one.
fn marker_id(line: &str) -> Option<String> {
    let tail = line.strip_prefix("[image ")?;
    let id = tail.get(..ID_HEX)?;
    let size = tail[ID_HEX..].strip_prefix(' ')?.strip_suffix(']')?;
    let (w, h) = size.split_once('×')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    (id.bytes().all(|b| b.is_ascii_hexdigit()) && digits(w) && digits(h)).then(|| id.to_string())
}

/// The image ids the marker lines opening `text` name, in order, and the
/// text after them — a turn's text taken apart into its images and its
/// message.
pub fn split_markers(text: &str) -> (Vec<String>, &str) {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some((line, after)) = rest.split_once('\n') {
        match marker_id(line) {
            Some(id) => out.push(id),
            None => break,
        }
        rest = after;
    }
    (out, rest)
}

/// The image ids the marker lines opening `text` name, in order.
pub fn marker_ids(text: &str) -> Vec<String> {
    split_markers(text).0
}

/// The image blocks opening `ids`, back to back: each block's first
/// `<|image_pad|>` and its count of them.
pub fn image_blocks(ids: &[u32], start: u32, pad: u32, end: u32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while ids.get(i) == Some(&start) {
        let first = i + 1;
        let mut j = first;
        while ids.get(j) == Some(&pad) {
            j += 1;
        }
        if j == first || ids.get(j) != Some(&end) {
            break;
        }
        out.push((first, j - first));
        i = j + 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_marker_names_its_image_and_size() {
        assert_eq!(
            marker("0123456789abcdef", 640, 480),
            "[image 0123456789abcdef 640×480]"
        );
    }

    /// The marker lines opening the text, in order; a marker anywhere else,
    /// or a line that only looks like one, ends them.
    #[test]
    fn marker_ids_are_the_lines_that_open_the_text() {
        let text = "[image 0123456789abcdef 640×480]\n[image fedcba9876543210 32×32]\n\
                    what is [image 1111111111111111 1×1] this?";
        assert_eq!(marker_ids(text), ["0123456789abcdef", "fedcba9876543210"]);
        assert!(marker_ids("no images here").is_empty());
        assert!(marker_ids("said [image 0123456789abcdef 1×1]\n").is_empty());
        assert!(marker_ids("[image short 1×1]\n[image 0123456789abcdef 1×1]\n").is_empty());
        assert!(marker_ids("[image 0123456789abcdef 1x1]\n").is_empty());
        // A marker with nothing after it is still not a line.
        assert!(marker_ids("[image 0123456789abcdef 1×1]").is_empty());
    }

    /// A turn's text comes apart into its images and its message, exactly as
    /// the submitter put them together.
    #[test]
    fn a_turns_text_splits_into_its_images_and_its_message() {
        let text = format!("{}\n{}\nwhat?", marker("0123456789abcdef", 2, 3), marker("fedcba9876543210", 4, 5));
        assert_eq!(
            split_markers(&text),
            (
                vec!["0123456789abcdef".to_string(), "fedcba9876543210".to_string()],
                "what?"
            )
        );
        assert_eq!(split_markers("what?"), (Vec::new(), "what?"));
        // An image alone: its message is empty.
        let alone = format!("{}\n", marker("0123456789abcdef", 2, 3));
        assert_eq!(split_markers(&alone), (vec!["0123456789abcdef".to_string()], ""));
    }

    /// The blocks opening the ids, back to back; anything after the first
    /// id that is not a block's start is the message's own.
    #[test]
    fn image_blocks_are_the_pad_runs_that_open_the_ids() {
        let (s, p, e) = (1, 2, 3);
        let ids = [s, p, p, p, e, s, p, e, 9, s, p, e];
        assert_eq!(image_blocks(&ids, s, p, e), [(1, 3), (6, 1)]);
        assert!(image_blocks(&[9, s, p, e], s, p, e).is_empty());
        assert!(image_blocks(&[s, e, 9], s, p, e).is_empty());
        assert!(image_blocks(&[s, p, p], s, p, e).is_empty());
    }
}
