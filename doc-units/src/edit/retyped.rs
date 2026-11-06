//! A `text` that retypes what stands around `old`. With `old`, only those
//! words are replaced, so a `text` written as the whole region — the line
//! above them, or the start of their own line, and then the new lines — writes
//! what was already there a second time. A coder replacing `mod tests {` with
//! `#[cfg(test)]` / `mod tests {` / its new test left two `#[cfg(test)]`
//! headers, twice in one task.

/// The fewest characters a retyped start or end of `old`'s own line must hold
/// to count: a closing `)` or `;` after `old` is as often the new text's own.
const LINE_PART: usize = 3;

/// Why replacing the one `old` in `unit` with `new` writes the unit's own text
/// around it again, if it does: `new`, written over several lines, opens with
/// the lines just above `old`'s line or closes with those just below, or
/// repeats the part of `old`'s line before or after it.
pub(super) fn retyped(unit: &str, old: &str, new: &str) -> Option<String> {
    if !new.contains('\n') {
        return None;
    }
    let at = unit.find(old)?;
    let start = unit[..at].rfind('\n').map_or(0, |i| i + 1);
    let old_end = at + old.trim_end_matches(['\r', '\n']).len();
    let end = unit[old_end..]
        .find('\n')
        .map_or(unit.len(), |i| old_end + i);
    let above: Vec<&str> = unit[..start].lines().collect();
    let below: Vec<&str> = unit[end..]
        .strip_prefix('\n')
        .unwrap_or("")
        .lines()
        .collect();
    let lines: Vec<&str> = new.lines().collect();
    let same = |a: &[&str], b: &[&str]| {
        a.iter().zip(b).all(|(x, y)| x.trim() == y.trim()) && a.iter().any(|l| !l.trim().is_empty())
    };
    let most = lines.len() - 1;
    if let Some(k) = (1..=most.min(above.len()))
        .rev()
        .find(|&k| same(&lines[..k], &above[above.len() - k..]))
    {
        return Some(twice(k, "above", lines[0]));
    }
    if let Some(k) = (1..=most.min(below.len()))
        .rev()
        .find(|&k| same(&lines[lines.len() - k..], &below[..k]))
    {
        return Some(twice(k, "below", lines[lines.len() - k]));
    }
    let before = unit[start..at].trim();
    if before.len() >= LINE_PART && lines[0].trim_start().starts_with(before) {
        return Some(line_twice("begins", before));
    }
    let after = unit[old_end..end].trim();
    let last = lines[lines.len() - 1];
    if after.len() >= LINE_PART && last.trim_end().ends_with(after) {
        return Some(line_twice("ends", after));
    }
    None
}

fn twice(k: usize, side: &str, first: &str) -> String {
    let lines = match k {
        1 => "a line".to_string(),
        k => format!("{k} lines"),
    };
    format!(
        "with `old`, `text` replaces only those words, but it retypes {lines} the part already \
         holds just {side} them, starting `{}`, which would then stand twice. Give `text` as \
         only what replaces `old`, or replace whole lines with `from` and `to`",
        first.trim()
    )
}

fn line_twice(how: &str, part: &str) -> String {
    format!(
        "with `old`, `text` replaces only those words, but it {how} with `{part}`, which their \
         line already holds around them and keeps, so the line would hold it twice. Give `text` \
         as only what replaces `old`, or replace whole lines with `from` and `to`"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const PART: &str = "use super::*;\n\n#[cfg(test)]\nmod tests {\n    fn a() {}\n}\n";

    /// **Retyping the line above `old` is refused**, as the coder did: the
    /// header `old` sits under, written again over `old`'s own line.
    #[test]
    fn the_line_above_retyped_is_refused() {
        let got = retyped(
            PART,
            "mod tests {",
            "#[cfg(test)]\nmod tests {\n    fn b() {}",
        );
        assert_eq!(
            got.as_deref(),
            Some(
                "with `old`, `text` replaces only those words, but it retypes a line the part \
                 already holds just above them, starting `#[cfg(test)]`, which would then stand \
                 twice. Give `text` as only what replaces `old`, or replace whole lines with \
                 `from` and `to`"
            )
        );
    }

    /// **Retyping the lines below is refused too**, and counts them.
    #[test]
    fn the_lines_below_retyped_are_refused() {
        let got = retyped(
            PART,
            "#[cfg(test)]",
            "#[cfg(all(test, unix))]\nmod tests {\n    fn a() {}",
        );
        assert_eq!(
            got.as_deref(),
            Some(
                "with `old`, `text` replaces only those words, but it retypes 2 lines the part \
                 already holds just below them, starting `mod tests {`, which would then stand \
                 twice. Give `text` as only what replaces `old`, or replace whole lines with \
                 `from` and `to`"
            )
        );
    }

    /// **Retyping the start of `old`'s own line is refused**: the words
    /// before `old` stay, so the line would open with them twice.
    #[test]
    fn the_start_of_its_line_retyped_is_refused() {
        let unit = "fn f() {\n    while self.len > max {\n        stop();\n    }\n}\n";
        assert_eq!(
            retyped(
                unit,
                "self.len > max",
                "while self.len >= max {\n        warn();"
            )
            .as_deref(),
            Some(
                "with `old`, `text` replaces only those words, but it begins with `while`, which \
                 their line already holds around them and keeps, so the line would hold it \
                 twice. Give `text` as only what replaces `old`, or replace whole lines with \
                 `from` and `to`"
            )
        );
        let unit = "    let total = count(items, limit);\n";
        assert_eq!(
            retyped(
                unit,
                "items",
                "items\n        .iter()\n        .count(), limit);"
            )
            .as_deref(),
            Some(
                "with `old`, `text` replaces only those words, but it ends with `, limit);`, \
                 which their line already holds around them and keeps, so the line would hold \
                 it twice. Give `text` as only what replaces `old`, or replace whole lines with \
                 `from` and `to`"
            )
        );
    }

    /// **New lines that only replace `old` are taken**: one line, new lines
    /// unlike their neighbours, and a short `)` the new text closes itself.
    #[test]
    fn a_text_that_only_replaces_old_is_taken() {
        assert_eq!(retyped(PART, "fn a() {}", "fn a() {}\n    fn b() {}"), None);
        assert_eq!(retyped(PART, "mod tests {", "mod checks {"), None);
        assert_eq!(retyped("f(a)\n", "a", "g(\n    1)"), None);
    }
}
