//! The figure CSVs `analyse` writes, read back: a header and its rows, each
//! field looked up by its column's name. A field may be double-quoted, as a
//! contrast's name is.

use std::fs;
use std::mem;
use std::path::Path;

use anyhow::{anyhow, Context};

/// A CSV's header and rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

/// One row, its fields by column name.
pub struct Record<'a> {
    table: &'a Table,
    row: &'a [String],
}

/// The fields of one CSV line, quotes taken off.
pub fn fields(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    for c in line.chars() {
        match c {
            '"' => quoted = !quoted,
            ',' if !quoted => out.push(mem::take(&mut field)),
            c => field.push(c),
        }
    }
    out.push(field);
    out
}

impl Table {
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let mut lines = text.lines().filter(|l| !l.is_empty());
        let header = fields(lines.next().ok_or_else(|| anyhow!("no header"))?);
        let rows: Vec<Vec<String>> = lines.map(fields).collect();
        if let Some(bad) = rows.iter().position(|r| r.len() != header.len()) {
            return Err(anyhow!(
                "row {} has {} fields against the header's {}",
                bad + 1,
                rows[bad].len(),
                header.len()
            ));
        }
        Ok(Self { header, rows })
    }

    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing {}", path.display()))
    }

    fn column(&self, name: &str) -> anyhow::Result<usize> {
        self.header
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| anyhow!("no column {name}"))
    }

    pub fn records(&self) -> impl Iterator<Item = Record<'_>> {
        self.rows.iter().map(|row| Record { table: self, row })
    }
}

impl Record<'_> {
    pub fn get(&self, name: &str) -> anyhow::Result<&str> {
        Ok(&self.row[self.table.column(name)?])
    }

    pub fn num(&self, name: &str) -> anyhow::Result<f64> {
        let s = self.get(name)?;
        s.parse()
            .with_context(|| format!("column {name}: {s:?} is not a number"))
    }

    /// A `;`-separated list of numbers.
    pub fn nums(&self, name: &str) -> anyhow::Result<Vec<f64>> {
        self.get(name)?
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse()
                    .with_context(|| format!("column {name}: {s:?} is not a number"))
            })
            .collect()
    }

    pub fn flag(&self, name: &str) -> anyhow::Result<bool> {
        match self.get(name)? {
            "true" => Ok(true),
            "false" => Ok(false),
            s => Err(anyhow!("column {name}: {s:?} is not true or false")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_fields_keep_their_commas_and_lose_their_quotes() {
        assert_eq!(fields("\"a, b\",C1,0.5"), vec!["a, b", "C1", "0.5"]);
        assert_eq!(fields("x,,y"), vec!["x", "", "y"]);
    }

    #[test]
    fn a_record_reads_its_fields_by_column_name() {
        let t = Table::parse("arm,mean,run_means,placed\nC1,0.25,0.5;0.25,true\n").unwrap();
        let r = t.records().next().unwrap();
        assert_eq!(r.get("arm").unwrap(), "C1");
        assert_eq!(r.num("mean").unwrap(), 0.25);
        assert_eq!(r.nums("run_means").unwrap(), vec![0.5, 0.25]);
        assert!(r.flag("placed").unwrap());
        assert!(r.num("arm").is_err());
        assert!(r.get("nope").is_err());
        assert!(Table::parse("a,b\n1\n").is_err());
    }
}
