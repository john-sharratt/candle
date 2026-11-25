//! Cutting a file into chunks whose seams tokenize as the whole file does.
//!
//! A chunk is sealed on its own, so its tokens are the tokens of its text
//! alone. Placed one after another, the chunks must hold exactly the tokens
//! the file's text would have held: a seam where the tokenizer would have
//! merged across the cut — a blank line opening the next chunk, whose `\n`
//! joins the one ending this chunk — is a sequence the model never saw. So
//! each cut is checked, and a cut whose seam tokenizes differently moves
//! forward a line at a time until it does not.

use super::chunk::Chunk;
use super::key::LineSpan;

/// `text` cut into chunks of whole lines, up to `max_lines` each, every cut a
/// seam `tokens` tokenizes alike: the tokens of the chunk and the text after
/// it, joined, equal the chunk's tokens followed by the rest's. A cut that
/// fails moves forward one line at a time, past `max_lines`, until one holds;
/// when none does, the chunk runs to the end of the file. An empty `text` has
/// no chunks.
///
/// Each line keeps its `\n`; a last line without one is given one (see
/// [`Chunk`]). The text after a cut is checked as far as the next chunk would
/// run, `max_lines` lines.
pub fn cut_lines(text: &str, max_lines: usize, tokens: impl Fn(&str) -> Vec<u32>) -> Vec<Chunk> {
    assert!(max_lines > 0, "a chunk holds at least one line");
    let lines: Vec<String> = text
        .split_inclusive('\n')
        .map(|l| {
            if l.ends_with('\n') {
                l.to_string()
            } else {
                format!("{l}\n")
            }
        })
        .collect();
    let mut out = Vec::new();
    let mut first = 0;
    while first < lines.len() {
        let mut end = (first + max_lines).min(lines.len());
        while end < lines.len() {
            let after = &lines[end..(end + max_lines).min(lines.len())];
            if seam_holds(&lines[first..end].concat(), &after.concat(), &tokens) {
                break;
            }
            end += 1;
        }
        out.push(Chunk {
            lines: LineSpan {
                first: first as u32 + 1,
                last: end as u32,
            },
            text: lines[first..end].concat(),
        });
        first = end;
    }
    out
}

/// Whether `a` then `b` tokenize as `a` and `b` apart.
fn seam_holds(a: &str, b: &str, tokens: &impl Fn(&str) -> Vec<u32>) -> bool {
    let mut apart = tokens(a);
    apart.extend(tokens(b));
    tokens(&format!("{a}{b}")) == apart
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy vocabulary that, as Qwen's does, merges a blank line's `\n\n`
    /// into one token: `\n\n` is 1, `\n` is 2, any other character its code
    /// point.
    fn toy(text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\n' {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                    out.push(1);
                } else {
                    out.push(2);
                }
            } else {
                out.push(c as u32);
            }
        }
        out
    }

    fn chunk(first: u32, last: u32, text: &str) -> Chunk {
        Chunk {
            lines: LineSpan { first, last },
            text: text.to_string(),
        }
    }

    /// **A cut that would open the next chunk on a blank line moves forward
    /// until its seam tokenizes alike**: after line 2 the next chunk would
    /// start `\n`, merging with line 2's own `\n`; after line 3 it starts
    /// `c`, and the cut holds.
    #[test]
    fn a_chunk_cut_moves_to_a_seam_that_tokenizes_alike() {
        let cut = cut_lines("a\nb\n\nc\nd\n", 2, toy);
        assert_eq!(cut, vec![chunk(1, 3, "a\nb\n\n"), chunk(4, 5, "c\nd\n")]);
        assert_eq!(toy("a\nb\n\n"), vec![97, 2, 98, 1]);
    }

    /// With no seam that holds, the chunk keeps the rest of the file.
    #[test]
    fn a_cut_with_no_seam_that_holds_keeps_the_rest_of_the_file() {
        let cut = cut_lines("a\n\n", 1, toy);
        assert_eq!(cut, vec![chunk(1, 2, "a\n\n")]);
    }

    /// Whole lines up to the limit, each keeping its `\n`, the last given one.
    #[test]
    fn a_file_is_cut_into_whole_lines_up_to_the_limit() {
        let cut = cut_lines("a\nb\nc\nd\ne", 2, toy);
        assert_eq!(
            cut,
            vec![
                chunk(1, 2, "a\nb\n"),
                chunk(3, 4, "c\nd\n"),
                chunk(5, 5, "e\n")
            ]
        );
        assert_eq!(cut_lines("", 2, toy), Vec::<Chunk>::new());
    }
}
