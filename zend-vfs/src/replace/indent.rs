//! Carrying `new` over to the file's indentation, from how `old`'s lines
//! matched the file's with their indentation ignored.

use super::ReplaceError;

/// Why a match's indentation cannot be carried over.
const UNEVEN: &str = "`old_text` matches the file only with its indentation ignored, and lines it \
                      quotes at one indentation stand at different indentations in the file, so \
                      where `new_text`'s lines belong cannot be worked out. Copy the lines to \
                      change exactly as the file indents them";

/// A line's leading whitespace.
pub(super) fn indent(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

/// How the indentation `old` was written at maps onto the file's, in one
/// matched window.
///
/// Per depth rather than one shift for the whole edit, because a model does
/// not misquote evenly: settling a merge, one quoted the conflict markers at
/// the margin, as the file has them, and the code between them two spaces too
/// deep. The first line's shift — none — then left the settled code too deep.
#[derive(Debug)]
pub(super) struct Depths {
    /// For each depth an `old` line is at, the depth of the file line it
    /// matched.
    map: Vec<(String, String)>,
    /// One level of indentation as `old` writes it and as the file does, when
    /// both can be read and they differ in kind — tabs on one side, spaces on
    /// the other.
    units: Option<(String, String)>,
}

impl Depths {
    /// The depths `old`'s lines map to in `window`, the file lines they
    /// matched. Blank lines say nothing. One quoted depth standing at two
    /// file depths cannot be carried over, and is refused.
    pub(super) fn of(old: &[&str], window: &[&str]) -> Result<Self, ReplaceError> {
        let mut map: Vec<(String, String)> = Vec::new();
        for (quoted, line) in old.iter().zip(window) {
            if quoted.trim().is_empty() {
                continue;
            }
            let (from, to) = (indent(quoted), indent(line));
            match map.iter().find(|(known, _)| known == from) {
                None => map.push((from.to_string(), to.to_string())),
                Some((_, mapped)) if mapped == to => {}
                Some(_) => return Err(ReplaceError::Unmatched(UNEVEN.to_string())),
            }
        }
        let quoted = unit(map.iter().map(|(from, _)| from.as_str()));
        let file = unit(map.iter().map(|(_, to)| to.as_str()));
        let units = quoted
            .zip(file)
            .filter(|(quoted, file)| quoted.contains('\t') != file.contains('\t'));
        Ok(Self { map, units })
    }

    /// `line` moved from the indentation `old` was written at to the file's:
    /// the deepest quoted depth that begins the line's own indentation is
    /// replaced by the file's for it, and whatever the line has beyond that
    /// stays — in the file's kind of indentation where `old` wrote the other.
    /// A blank line stays empty, and a line no depth begins keeps its own.
    pub(super) fn reindent(&self, line: &str) -> String {
        if line.trim().is_empty() {
            return String::new();
        }
        let own = indent(line);
        let deepest = self
            .map
            .iter()
            .filter(|(from, _)| own.starts_with(from.as_str()))
            .max_by_key(|(from, _)| from.len());
        match deepest {
            Some((from, to)) => {
                let extra = self.in_file_units(&own[from.len()..]);
                format!("{to}{extra}{}", &line[own.len()..])
            }
            None => line.to_string(),
        }
    }

    /// `extra` indentation, whole levels of `old`'s unit, as that many of the
    /// file's; anything else unchanged.
    fn in_file_units(&self, extra: &str) -> String {
        match &self.units {
            Some((quoted, file))
                if !extra.is_empty()
                    && extra.len().is_multiple_of(quoted.len())
                    && extra == quoted.repeat(extra.len() / quoted.len()) =>
            {
                file.repeat(extra.len() / quoted.len())
            }
            _ => extra.to_string(),
        }
    }
}

/// One level of indentation among `depths`: the whitespace the smallest step
/// between two of them adds. `None` with fewer than two depths, or where no
/// deeper one extends a shallower.
fn unit<'a>(depths: impl Iterator<Item = &'a str>) -> Option<String> {
    let mut sorted: Vec<&str> = depths.collect();
    sorted.sort_by_key(|d| d.len());
    sorted.dedup();
    sorted
        .windows(2)
        .filter_map(|pair| pair[1].strip_prefix(pair[0]))
        .filter(|step| !step.is_empty())
        .min_by_key(|step| step.len())
        .map(str::to_string)
}
