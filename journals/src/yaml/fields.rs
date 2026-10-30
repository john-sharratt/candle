//! Taking a mapping's fields one by one, so that a missing, wrong or unknown
//! field is a [`Problem`] naming the file and line.

use std::collections::BTreeSet;

use serde_yaml::{Mapping, Sequence, Value};

use crate::error::Problem;
use crate::yaml::doc::key_line;

pub(crate) struct Fields<'a> {
    map: &'a Mapping,
    taken: BTreeSet<String>,
    file: &'a str,
    /// The entry's own line, when its key's cannot be found.
    line: Option<usize>,
    /// The file's text, for finding a key's line.
    text: Option<&'a str>,
}

impl<'a> Fields<'a> {
    /// The fields of `v`, which must be a mapping with string keys. `line` is
    /// the entry's line when it has one; `text` the file's, to find a key in.
    pub(crate) fn of(
        v: &'a Value,
        file: &'a str,
        line: Option<usize>,
        text: Option<&'a str>,
    ) -> Result<Fields<'a>, Problem> {
        let map = v
            .as_mapping()
            .ok_or_else(|| Problem::new(file, line, "expected a mapping of fields"))?;
        if let Some(k) = map.keys().find(|k| !k.is_string()) {
            return Err(Problem::new(
                file,
                line,
                format!("a field's name must be text, not {}", describe(k)),
            ));
        }
        Ok(Fields {
            map,
            taken: BTreeSet::new(),
            file,
            line,
            text,
        })
    }

    /// A problem at `key`'s line, or the entry's.
    pub(crate) fn problem(&self, key: &str, what: impl Into<String>) -> Problem {
        let line = self.text.and_then(|t| key_line(t, key)).or(self.line);
        Problem::new(self.file, line, what)
    }

    fn get(&mut self, key: &str) -> Option<&'a Value> {
        self.taken.insert(key.to_string());
        self.map.get(key).filter(|v| !v.is_null())
    }

    /// Text. A number is refused rather than read as digits it may not have
    /// been written with — `1e3` would come back `1000` — so a hand edit
    /// that left a name unquoted is told to quote it.
    pub(crate) fn opt_str(&mut self, key: &str) -> Result<Option<String>, Problem> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(v) => Err(self.not_text(key, v)),
        }
    }

    fn not_text(&self, key: &str, v: &Value) -> Problem {
        let hint = if matches!(v, Value::Number(_) | Value::Bool(_)) {
            " — quote it"
        } else {
            ""
        };
        self.problem(
            key,
            format!("`{key}` must be text, not {}{hint}", describe(v)),
        )
    }

    /// Text, or a whole number read as its digits: a year written bare by
    /// hand, `period: 2026`, is the year it says.
    pub(crate) fn opt_text_or_whole(&mut self, key: &str) -> Result<Option<String>, Problem> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(Value::Number(n)) if n.is_u64() => Ok(Some(n.to_string())),
            Some(v) => Err(self.not_text(key, v)),
        }
    }

    /// A count: a whole number.
    pub(crate) fn count(&mut self, key: &str) -> Result<usize, Problem> {
        match self.get(key) {
            Some(Value::Number(n)) => n
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| self.problem(key, format!("`{key}` is not a count"))),
            None => Err(self.problem(key, format!("`{key}` is missing"))),
            Some(v) => Err(self.problem(key, format!("`{key}` is not a count: {}", describe(v)))),
        }
    }

    pub(crate) fn str(&mut self, key: &str) -> Result<String, Problem> {
        self.opt_str(key)?
            .ok_or_else(|| self.problem(key, format!("`{key}` is missing")))
    }

    /// Text parsed by `parse`, whose error says what is wrong with it.
    pub(crate) fn parsed<T>(
        &mut self,
        key: &str,
        parse: impl Fn(&str) -> Result<T, String>,
    ) -> Result<T, Problem> {
        let s = self.str(key)?;
        parse(&s).map_err(|why| self.problem(key, format!("`{key}`: {why}")))
    }

    pub(crate) fn opt_parsed<T>(
        &mut self,
        key: &str,
        parse: impl Fn(&str) -> Result<T, String>,
    ) -> Result<Option<T>, Problem> {
        match self.opt_str(key)? {
            None => Ok(None),
            Some(s) => parse(&s)
                .map(Some)
                .map_err(|why| self.problem(key, format!("`{key}`: {why}"))),
        }
    }

    pub(crate) fn opt_bool(&mut self, key: &str) -> Result<Option<bool>, Problem> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(v) => Err(self.problem(
                key,
                format!("`{key}` must be true or false, not {}", describe(v)),
            )),
        }
    }

    pub(crate) fn opt_seq(&mut self, key: &str) -> Result<Option<&'a Sequence>, Problem> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Sequence(s)) => Ok(Some(s)),
            Some(v) => {
                Err(self.problem(key, format!("`{key}` must be a list, not {}", describe(v))))
            }
        }
    }

    /// A list of text.
    pub(crate) fn opt_strs(&mut self, key: &str) -> Result<Option<Vec<String>>, Problem> {
        let Some(seq) = self.opt_seq(key)? else {
            return Ok(None);
        };
        seq.iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                v => Err(self.problem(key, format!("`{key}` must list text, not {}", describe(v)))),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    pub(crate) fn opt_map(&mut self, key: &str) -> Result<Option<&'a Value>, Problem> {
        match self.get(key) {
            None => Ok(None),
            Some(v @ Value::Mapping(_)) => Ok(Some(v)),
            Some(v) => Err(self.problem(
                key,
                format!("`{key}` must be a mapping, not {}", describe(v)),
            )),
        }
    }

    /// Every field was taken: anything left is unknown, and an error.
    pub(crate) fn done(self) -> Result<(), Problem> {
        let unknown = self
            .map
            .keys()
            .filter_map(Value::as_str)
            .find(|k| !self.taken.contains(*k));
        match unknown {
            Some(k) => Err(self.problem(k, format!("`{k}` is not a field here"))),
            None => Ok(()),
        }
    }
}

/// What kind of value `v` is, for an error.
pub(crate) fn describe(v: &Value) -> String {
    match v {
        Value::Null => "nothing".into(),
        Value::Bool(b) => format!("`{b}`"),
        Value::Number(n) => format!("the number {n}"),
        Value::String(s) => format!("the text `{s}`"),
        Value::Sequence(_) => "a list".into(),
        Value::Mapping(_) => "a mapping".into(),
        Value::Tagged(t) => format!("a value tagged `{}`", t.tag),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Value {
        serde_yaml::from_str(s).unwrap()
    }

    #[test]
    fn fields_are_taken_by_name_and_type() {
        let doc = v("{a: x, n: 2026, c: 12, b: true, l: [p, \"3\"], m: {k: v}, z: null}");
        let mut f = Fields::of(&doc, "f", Some(4), None).unwrap();
        assert_eq!(f.str("a").unwrap(), "x");
        assert_eq!(f.opt_text_or_whole("n").unwrap().as_deref(), Some("2026"));
        assert_eq!(f.count("c").unwrap(), 12);
        assert_eq!(f.opt_bool("b").unwrap(), Some(true));
        assert_eq!(f.opt_strs("l").unwrap(), Some(vec!["p".into(), "3".into()]));
        assert!(f.opt_map("m").unwrap().is_some());
        assert_eq!(f.opt_str("z").unwrap(), None);
        assert_eq!(f.opt_str("absent").unwrap(), None);
        f.done().unwrap();
    }

    #[test]
    fn a_number_where_text_belongs_is_refused_not_rewritten() {
        let doc = v("{name: 1e3, key: 1234567890123456, n: -3, f: 1.5, l: [1]}");
        let mut f = Fields::of(&doc, "f", Some(2), None).unwrap();
        let err = f.str("name").unwrap_err().to_string();
        assert!(
            err.starts_with("f:2: `name` must be text, not the number ")
                && err.ends_with(" — quote it"),
            "{err}"
        );
        assert!(f.str("key").is_err());
        assert!(f.opt_text_or_whole("n").is_err(), "a negative is no year");
        assert!(f.opt_text_or_whole("f").is_err());
        assert!(f.count("f").is_err());
        assert!(f.opt_strs("l").is_err());
    }

    #[test]
    fn each_fault_is_named() {
        let doc = v("{a: [1], b: x, c: {d: 1}}");
        let mut f = Fields::of(&doc, "f", Some(9), None).unwrap();
        assert_eq!(
            f.str("a").unwrap_err().to_string(),
            "f:9: `a` must be text, not a list"
        );
        assert_eq!(
            f.opt_bool("b").unwrap_err().to_string(),
            "f:9: `b` must be true or false, not the text `x`"
        );
        assert_eq!(
            f.str("missing").unwrap_err().to_string(),
            "f:9: `missing` is missing"
        );
        assert_eq!(
            f.done().unwrap_err().to_string(),
            "f:9: `c` is not a field here"
        );
    }

    #[test]
    fn a_multi_line_file_names_the_key_line() {
        let text = "name: a\nfolded: maybe\n";
        let doc = v(text);
        let mut f = Fields::of(&doc, "journal.yaml", None, Some(text)).unwrap();
        assert_eq!(
            f.opt_bool("folded").unwrap_err().to_string(),
            "journal.yaml:2: `folded` must be true or false, not the text `maybe`"
        );
    }

    #[test]
    fn non_mappings_and_non_text_keys_are_refused() {
        assert!(Fields::of(&v("[1, 2]"), "f", None, None).is_err());
        assert!(Fields::of(&v("{1: a}"), "f", None, None).is_err());
        let parsed = v("{t: !custom x}");
        let mut f = Fields::of(&parsed, "f", None, None).unwrap();
        assert!(f.str("t").unwrap_err().what.contains("tagged"));
    }
}
