//! Heading ids, stable and unique within a document.
//!
//! A paper's headings are its addresses: `#9-4-throughput` has to survive a
//! re-render, or every link anyone ever shared into the document breaks. So the
//! slug is a pure function of the heading text, with a counter suffix as the
//! only tie-break — and the counter is the reason [`Slugger`] is stateful
//! rather than a free function.

use std::collections::HashMap;

#[derive(Default)]
pub struct Slugger {
    seen: HashMap<String, u32>,
}

impl Slugger {
    /// A url-safe id for this heading text, unique within the document.
    ///
    /// Lowercased; runs of anything that is not alphanumeric collapse to a
    /// single `-`. Non-ASCII letters are kept — a heading in another script
    /// should still get an id rather than collapsing to `section-4`.
    pub fn slug(&mut self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut dash = false;
        for c in text.chars() {
            if c.is_alphanumeric() {
                out.extend(c.to_lowercase());
                dash = false;
            } else if !out.is_empty() && !dash {
                out.push('-');
                dash = true;
            }
        }
        while out.ends_with('-') {
            out.pop();
        }
        if out.is_empty() {
            out.push_str("section");
        }

        // The suffix search re-checks the map rather than trusting the counter,
        // because a document can contain both a repeated "Results" and a
        // literal "Results 1" — and handing the same id to two headings sends
        // one of the two links to the wrong place.
        if !self.seen.contains_key(&out) {
            self.seen.insert(out.clone(), 0);
            return out;
        }
        let mut n = self.seen[&out];
        loop {
            n += 1;
            let candidate = format!("{out}-{n}");
            if !self.seen.contains_key(&candidate) {
                self.seen.insert(out, n);
                self.seen.insert(candidate.clone(), 0);
                return candidate;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punctuation_and_case_collapse() {
        let mut s = Slugger::default();
        assert_eq!(
            s.slug("§9.4 — Throughput (measured)"),
            "9-4-throughput-measured"
        );
        assert_eq!(s.slug("The O(1) Error Theorem"), "the-o-1-error-theorem");
    }

    #[test]
    fn repeats_get_a_counter_and_the_first_keeps_the_bare_slug() {
        // The first occurrence must stay unsuffixed: it is the one already
        // linked to from elsewhere.
        let mut s = Slugger::default();
        assert_eq!(s.slug("Results"), "results");
        assert_eq!(s.slug("Results"), "results-1");
        assert_eq!(s.slug("Results"), "results-2");
    }

    #[test]
    fn a_heading_of_pure_punctuation_still_gets_an_id() {
        let mut s = Slugger::default();
        assert_eq!(s.slug("——"), "section");
        assert_eq!(s.slug("***"), "section-1");
    }

    #[test]
    fn a_generated_suffix_never_collides_with_a_real_heading() {
        // "Results" twice plus a literal "Results 1" is the case a plain
        // counter gets wrong, by handing the same id to two headings.
        let mut s = Slugger::default();
        assert_eq!(s.slug("Results"), "results");
        assert_eq!(s.slug("Results 1"), "results-1");
        assert_eq!(s.slug("Results"), "results-2");
        assert_eq!(s.slug("Results"), "results-3");
    }

    #[test]
    fn non_ascii_letters_survive() {
        let mut s = Slugger::default();
        assert_eq!(s.slug("Über Alles"), "über-alles");
    }
}
