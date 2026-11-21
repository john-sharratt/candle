//! Reads the fields of a call as trimmed text, with length and loop checks.
//!
//! A field is refused when it falls outside its word range or repeats in a loop, since a loop is the decode failing and not a choice the model made. A field that loops is kept up to where the loop begins when what came before it is substantive, because a reading that had already said what it found should not be discarded for its tail. A long field that opens with its substance is kept up to its last whole sentence rather than refused.

use super::*;

/// The arguments of one call, read as trimmed text.
pub(in super::super) struct Fields<'a>(pub(in super::super) &'a Map<String, Value>);

impl Fields<'_> {
    /// The field as trimmed text — a list's entries a line each, its empty
    /// entries dropped and one written twice kept once: a list that goes round
    /// repeats its entries whole.
    pub(in super::super) fn get(&self, k: &str) -> String {
        match self.0.get(k) {
            Some(Value::String(s)) => s.trim().to_string(),
            Some(Value::Array(entries)) => {
                let mut kept: Vec<&str> = Vec::new();
                for e in entries.iter().filter_map(Value::as_str).map(str::trim) {
                    if !e.is_empty() && !kept.contains(&e) {
                        kept.push(e);
                    }
                }
                kept.join("\n")
            }
            _ => String::new(),
        }
    }

    /// The field, refused when it is outside `min..=max` words or goes round
    /// in a loop.
    ///
    /// **A loop is the decode failing, not the model choosing.** A `happens`
    /// that began well ended "She is Zen, and nothing matters. She is Zen, and
    /// everything matters. She is Zen…" for a hundred words; the cap and the
    /// repetition check both refuse it, and the retry is asked again.
    pub(in super::super) fn words(
        &self,
        k: &str,
        min: usize,
        max: usize,
        what: &str,
    ) -> Result<String, String> {
        let v = self.get(k);
        let n = v.split_whitespace().count();
        if n < min {
            return Err(format!("`{k}` is {what}; it needs at least {min} words."));
        }
        if n > max {
            return Err(format!(
                "`{k}` is {n} words; it is {what} and must be under {max}. Say it once, plainly."
            ));
        }
        // **A field that ends in a loop is kept up to the loop.** A reading
        // whose `checked` said everything it had to and then could not stop —
        // "I am done. I will output. I am done. I am ready to output…" — was
        // refused three times over its tail, and the reviewer got no reading
        // at all. What came before the loop is the field.
        if loops(&v) {
            let kept = before_the_loop(&v);
            if kept.split_whitespace().count() >= min && !loops(&kept) {
                return Ok(kept);
            }
            return Err(format!(
                "`{k}` repeats itself. Say {what} once, plainly, and stop."
            ));
        }
        Ok(v)
    }
}

impl Fields<'_> {
    /// [`Self::words`] for a field whose opening is its substance: one over
    /// `max` is kept up to the last sentence that ends within it, rather than
    /// refused.
    ///
    /// **A long reading is a reading.** The table's `checked`, its findings
    /// stated, ran to eleven hundred words of going over them again, and was
    /// refused, read again and refused — the reviewer given nothing for a
    /// reading that had said what it found in its first paragraphs.
    pub(in super::super) fn words_kept(
        &self,
        k: &str,
        min: usize,
        max: usize,
        what: &str,
    ) -> Result<String, String> {
        let v = self.get(k);
        if v.split_whitespace().count() <= max {
            return self.words(k, min, max, what);
        }
        let kept = within_words(&v, max);
        match kept.split_whitespace().count() >= min && !loops(&kept) {
            true => Ok(kept),
            false => self.words(k, min, max, what),
        }
    }
}

/// A proposal's fields, by name.
pub(super) fn fields<const N: usize>(pairs: [(&str, String); N]) -> BTreeMap<String, String> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}
