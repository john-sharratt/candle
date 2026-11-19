//! The `<tool_response>` blocks of a tool round's results, as a stored turn's
//! user half holds them.

const OPEN: &str = "<tool_response>";
const CLOSE: &str = "</tool_response>";

/// Each `<tool_response>` block's body in `user`, in order, without the
/// `[n/total name]` label a multi-call round puts on its first line.
pub fn response_bodies(user: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = user;
    while let Some(open) = rest.find(OPEN) {
        let after = &rest[open + OPEN.len()..];
        let Some(end) = after.find(CLOSE) else {
            break;
        };
        out.push(strip_label(&after[..end]));
        rest = &after[end + CLOSE.len()..];
    }
    out
}

/// `body` without a leading `[n/total name]\n` label.
fn strip_label(body: &str) -> &str {
    let Some((first, rest)) = body.split_once('\n') else {
        return body;
    };
    let is_label = first
        .strip_prefix('[')
        .and_then(|l| l.strip_suffix(']'))
        .and_then(|l| l.split_once(' '))
        .and_then(|(count, _name)| count.split_once('/'))
        .is_some_and(|(n, total)| {
            !n.is_empty()
                && !total.is_empty()
                && n.bytes().all(|b| b.is_ascii_digit())
                && total.bytes().all(|b| b.is_ascii_digit())
        });
    if is_label {
        rest
    } else {
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Each block's body comes back in order, its round label stripped**;
    /// text between blocks is not a body.
    #[test]
    fn bodies_come_back_in_order_without_their_labels() {
        let user = "<tool_response>[1/2 file_read]\nfirst</tool_response>\nbetween\n\
                    <tool_response>[2/2 file_list]\nsecond</tool_response>\n";
        assert_eq!(response_bodies(user), vec!["first", "second"]);
    }

    /// A one-call round carries no label, and its first line stays.
    #[test]
    fn an_unlabelled_body_is_kept_whole() {
        let user = "<tool_response>[not a label]\nbody</tool_response>";
        assert_eq!(response_bodies(user), vec!["[not a label]\nbody"]);
        assert_eq!(
            response_bodies("<tool_response>one line</tool_response>"),
            vec!["one line"]
        );
    }

    /// An unclosed block ends the read; no block, no bodies.
    #[test]
    fn an_unclosed_block_ends_the_read() {
        assert_eq!(
            response_bodies("<tool_response>a</tool_response><tool_response>b"),
            vec!["a"]
        );
        assert!(response_bodies("no blocks here").is_empty());
    }
}
