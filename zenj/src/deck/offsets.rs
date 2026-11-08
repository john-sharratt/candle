//! Where rekordbox keeps its live deck state (§4.2), as pointer chains from
//! its module's base — the `rkbx_link` offsets format, so offsets found for a
//! new release with that project's method drop straight in.
//!
//! ```text
//! 7.2.19                      a block: the version first
//! 05F19B48 20 0 280 124       the master deck's index
//!
//! 05E09790 0 2C0 1A0          per deck: its BPM,
//! 05E09790 0 2C0 130          its sample position,
//! 05F53DA8 8 3F0 0            its track's description,
//! 05F53DA8 8 3F0 0            its track's analysis file,
//! #uuid 05F53DA8 8 10 0       and optionally its track's uuid
//! ```
//!
//! A chain is hex: every value but the last is added and dereferenced, the
//! last is added. Blocks are separated by two or more blank lines; `#` starts
//! a comment, except `#uuid`.

use std::collections::BTreeMap;

/// A pointer chain from the module's base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain {
    pub hops: Vec<usize>,
    pub last: usize,
}

impl Chain {
    pub fn parse(line: &str) -> Result<Chain, String> {
        let values = line
            .split_whitespace()
            .map(|v| usize::from_str_radix(v, 16).map_err(|_| format!("`{v}` is not hex in `{line}`")))
            .collect::<Result<Vec<usize>, String>>()?;
        let (last, hops) = values
            .split_last()
            .ok_or_else(|| format!("`{line}` is no chain"))?;
        Ok(Chain {
            hops: hops.to_vec(),
            last: *last,
        })
    }
}

/// One deck's chains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeckChains {
    pub bpm: Chain,
    pub samples: Chain,
    pub info: Chain,
    pub anlz: Chain,
    pub uuid: Option<Chain>,
}

/// One release's chains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Offsets {
    pub version: String,
    pub master: Chain,
    pub decks: Vec<DeckChains>,
}

fn block(lines: &[String]) -> Result<Offsets, String> {
    let mut rows = lines.iter().peekable();
    let version = rows.next().ok_or("an empty block")?.clone();
    let master = Chain::parse(rows.next().ok_or_else(|| format!("{version}: no master chain"))?)?;
    let mut decks = Vec::new();
    while rows.peek().is_some() {
        let mut next = |what: &str| {
            rows.next()
                .ok_or_else(|| format!("{version}: deck {} has no {what} chain", decks.len() + 1))
                .and_then(|l| Chain::parse(l))
        };
        let bpm = next("BPM")?;
        let samples = next("position")?;
        let info = next("track")?;
        let anlz = next("analysis")?;
        let uuid = match rows.peek().and_then(|r| r.strip_prefix("uuid ")) {
            Some(chain) => {
                let c = Chain::parse(chain)?;
                rows.next();
                Some(c)
            }
            None => None,
        };
        decks.push(DeckChains {
            bpm,
            samples,
            info,
            anlz,
            uuid,
        });
    }
    Ok(Offsets {
        version,
        master,
        decks,
    })
}

/// Every release's chains in `text`, by version.
pub fn parse(text: &str) -> Result<BTreeMap<String, Offsets>, String> {
    let mut out = BTreeMap::new();
    let mut lines: Vec<String> = Vec::new();
    let mut blanks = 0;
    let mut flush = |lines: &mut Vec<String>| -> Result<(), String> {
        if !lines.is_empty() {
            let o = block(lines)?;
            out.insert(o.version.clone(), o);
            lines.clear();
        }
        Ok(())
    };
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            blanks += 1;
            if blanks >= 2 {
                flush(&mut lines)?;
            }
            continue;
        }
        blanks = 0;
        if let Some(uuid) = line.strip_prefix("#uuid ") {
            lines.push(format!("uuid {uuid}"));
        } else if !line.starts_with('#') {
            lines.push(line.to_string());
        }
    }
    flush(&mut lines)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIPPED: &str = include_str!("../../offsets/rekordbox.txt");

    /// **The shipped offsets read**: 7.2.19's master chain and its decks
    /// exactly, each with its uuid; 7.2.2's with none; the last block read
    /// though the file ends without two blank lines.
    #[test]
    fn the_shipped_offsets_read() {
        let all = parse(SHIPPED).unwrap();
        let v = &all["7.2.19"];
        assert_eq!(
            v.master,
            Chain {
                hops: vec![0x05F19B48, 0x20, 0x0, 0x280],
                last: 0x124
            }
        );
        assert_eq!(v.decks.len(), 4);
        assert_eq!(
            v.decks[1].samples,
            Chain {
                hops: vec![0x05E09790, 0x8, 0x2C0],
                last: 0x130
            }
        );
        assert_eq!(
            v.decks[0].uuid,
            Some(Chain {
                hops: vec![0x05F53DA8, 0x8, 0x10],
                last: 0
            })
        );
        assert_eq!(all["7.2.2"].decks.len(), 4);
        assert!(all["7.2.2"].decks.iter().all(|d| d.uuid.is_none()));
    }

    #[test]
    fn a_broken_block_is_refused() {
        assert_eq!(
            parse("7.0\nZZ 1\n").unwrap_err(),
            "`ZZ` is not hex in `ZZ 1`"
        );
        assert_eq!(
            parse("7.0\n1 2\n3 4\n5 6\n").unwrap_err(),
            "7.0: deck 1 has no track chain"
        );
    }
}
