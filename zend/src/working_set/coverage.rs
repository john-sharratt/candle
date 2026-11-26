//! The tool responses a turn's user half carries: each `<tool_response>`
//! block's body, in order, without the label a multi-call round puts on it.

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

    /// A multi-call round's labels are stripped, and the bodies come back in
    /// order; text outside the blocks is no body.
    #[test]
    fn labelled_responses_come_back_in_order_without_their_labels() {
        let user = "before<tool_response>[1/2 file_read]\nfirst</tool_response>\n\
                    <tool_response>[2/2 file_read]\n{\"error\":\"x\"}</tool_response>\n\
                    <tool_response>plain</tool_response>";
        assert_eq!(
            response_bodies(user),
            vec!["first", "{\"error\":\"x\"}", "plain"]
        );
        assert_eq!(
            response_bodies("<tool_response>[a b]\nkept</tool_response>"),
            vec!["[a b]\nkept"],
            "a first line that is no count label stays"
        );
    }
}
