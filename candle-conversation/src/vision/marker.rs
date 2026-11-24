//! How a turn says it holds an image, in its text and in its ids.
//!
//! In the ids: `<|vision_start|>`, one `<|image_pad|>` per row, `<|vision_end|>`.
//! In the text: `[image <id> <W>×<H>]`, the image's 16-hex id and the size it
//! arrived at — what a transcript, a history view or a summariser sees, and
//! what names the image's rows in the substrate. A turn's `k`-th image block in
//! its ids is the `k`-th marker in its text, so a path that rebuilds a turn from
//! its ids and text finds its images again.

use super::prepare::ID_HEX;

/// The marker a turn's text holds for an image.
pub fn marker(id: &str, width: u32, height: u32) -> String {
    format!("[image {id} {width}×{height}]")
}

/// Every image id the markers in `text` name, in order.
pub fn marker_ids(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find("[image ") {
        let tail = &rest[at + "[image ".len()..];
        let id: String = tail.chars().take(ID_HEX).collect();
        let well_formed = id.len() == ID_HEX
            && id.chars().all(|c| c.is_ascii_hexdigit())
            && tail[ID_HEX..].starts_with(' ')
            && tail[ID_HEX..].contains(']');
        if well_formed {
            out.push(id);
        }
        rest = &rest[at + 1..];
    }
    out
}

/// The image blocks in `ids`: each block's first `<|image_pad|>` and its
/// count of them.
pub fn image_blocks(ids: &[u32], start: u32, pad: u32, end: u32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == start {
            let first = i + 1;
            let mut j = first;
            while j < ids.len() && ids[j] == pad {
                j += 1;
            }
            if j < ids.len() && ids[j] == end && j > first {
                out.push((first, j - first));
                i = j + 1;
                continue;
            }
        }
        i += 1;
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

    /// Markers in order; text that only looks like one is not one.
    #[test]
    fn marker_ids_are_read_in_order() {
        let text = "[image 0123456789abcdef 640×480]\n[image short 1×1] and \
                    [image fedcba9876543210 32×32] what is this?";
        assert_eq!(
            marker_ids(text),
            ["0123456789abcdef", "fedcba9876543210"]
        );
        assert!(marker_ids("no images here").is_empty());
    }

    /// Blocks are found by their markers' ids; a start with no pads or no end
    /// is not a block.
    #[test]
    fn image_blocks_are_their_pad_runs() {
        let (s, p, e) = (1, 2, 3);
        let ids = [9, s, p, p, p, e, 9, s, e, s, p, e, s, p];
        assert_eq!(image_blocks(&ids, s, p, e), [(2, 3), (10, 1)]);
    }
}
