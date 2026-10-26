//! Reading a backend's event stream (`docs/mcp_app.md` §3): `event:` and
//! `data:` lines, an event ending at a blank line. Bytes arrive in chunks cut
//! anywhere, so lines are assembled across them.

use std::mem::take;

/// One event read off a stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    /// Its `event:` name; `message` when it named none.
    pub name: String,
    /// Its `data:` lines, joined by newlines.
    pub data: String,
}

/// Events assembled from chunks of a stream.
#[derive(Debug, Default)]
pub struct Reader {
    pending: Vec<u8>,
    name: Option<String>,
    data: Vec<String>,
}

impl Reader {
    /// Take in `chunk`; every event it completes, in order.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<Event> {
        self.pending.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(at) = self.pending.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=at).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if !self.data.is_empty() || self.name.is_some() {
                    out.push(Event {
                        name: self.name.take().unwrap_or_else(|| "message".into()),
                        data: take(&mut self.data).join("\n"),
                    });
                }
            } else if line.starts_with(':') {
                // A comment: a keep-alive.
            } else if let Some(name) = field(line, "event") {
                self.name = Some(name.to_string());
            } else if let Some(data) = field(line, "data") {
                self.data.push(data.to_string());
            }
        }
        out
    }
}

/// The value of `line` when it is the field `name`.
fn field<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let rest = line.strip_prefix(name)?.strip_prefix(':')?;
    Some(rest.strip_prefix(' ').unwrap_or(rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_are_read_across_chunks_cut_anywhere() {
        let stream = b"event: progress\ndata: {\"message\":\"a\"}\n\n: keep-alive\n\nevent: result\r\ndata: {\"text\":\"ok\"}\r\n\r\n";
        for cut in 0..stream.len() {
            let mut r = Reader::default();
            let mut events = r.feed(&stream[..cut]);
            events.extend(r.feed(&stream[cut..]));
            assert_eq!(
                events,
                vec![
                    Event {
                        name: "progress".into(),
                        data: r#"{"message":"a"}"#.into()
                    },
                    Event {
                        name: "result".into(),
                        data: r#"{"text":"ok"}"#.into()
                    },
                ],
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn data_lines_join_and_an_unnamed_event_is_a_message() {
        let mut r = Reader::default();
        assert_eq!(
            r.feed(b"data: one\ndata:two\n\n"),
            vec![Event {
                name: "message".into(),
                data: "one\ntwo".into()
            }]
        );
    }
}
