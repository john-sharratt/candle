//! Seeded random adversaries over whole calls.
//!
//! At a branch the adversary takes any token the mask allows, so it walks the
//! grammar's every shape — empty arrays, elements with any subset of fields, the
//! 64-element bound. In a free span it writes whatever is most likely to break
//! JSON: closers of both kinds, quotes, backslashes, raw control characters,
//! half literals, merged delimiter tokens, and EOS. Every call must still parse,
//! name a tool from the catalog, and carry only the fields its schema declares,
//! in the shapes it declares — for thousands of runs, on a fixed seed so a
//! failure reproduces.

use serde_json::Value;

use super::{catalog, parse, tree, Rng, EOS};
use crate::stencil::mask::AllowedSet;
use crate::stencil::session::Observe;
use crate::stencil::sim::{simulate, Oracle};
use crate::stencil::tool_call::MAX_ARRAY_ELEMENTS;
use crate::stencil::vocab::{TestVocab, TokenId};

/// Multi-byte tokens a model really emits around JSON structure — including
/// the other languages' spellings the repairs read.
const MERGED: &[&str] = &[
    "\"]",
    "\"}",
    "\",",
    "\", \"",
    "]}",
    "}]",
    "}}",
    "]}}}",
    "}, {",
    "[{",
    "[]",
    "{}",
    "20}",
    "3420]",
    "0,",
    " null",
    " true",
    "\\\"",
    "\\n",
    "a\nb",
    " True",
    " None",
    " 'x'",
    "é",
    "{'",
    "':",
    "', '",
    " nil",
    " NaN",
    " undefined",
    "{a:",
    " inf",
    ".5",
    ",]",
    ", }",
    "\\'",
    "\\x4",
    " FALSE",
];

/// Single bytes chosen to break JSON.
const HOSTILE: &[u8] = b"\"\\[]{},: \n\t\x01-+.eE0123456789truefalsnxq/'";

fn vocab() -> TestVocab {
    MERGED
        .iter()
        .enumerate()
        .fold(TestVocab::new(), |v, (i, s)| {
            v.with_special(s, 300 + i as TokenId)
        })
}

/// A mask-respecting adversary. At a branch it prefers to close (`]`, `}`)
/// often enough that arrays end well before their bound on most runs, and
/// sometimes runs one to the bound. In free text it mixes hostile bytes,
/// ordinary letters, merged tokens and EOS.
fn adversary(seed: u64) -> Oracle {
    let mut rng = Rng(seed);
    let run_to_bound = rng.chance(5);
    Oracle::Policy(Box::new(
        move |allowed: Option<&AllowedSet>| match allowed {
            Some(set) => {
                let is_closer = |t: TokenId| t == b']' as TokenId || t == b'}' as TokenId;
                let toks = set.tokens();
                let others: Vec<TokenId> =
                    toks.iter().copied().filter(|&t| !is_closer(t)).collect();
                match toks.iter().copied().find(|&t| is_closer(t)) {
                    // Never close while anything else is legal: every array runs
                    // to the bound, where the grammar closes it.
                    Some(_) if run_to_bound && !others.is_empty() => {
                        others[rng.below(others.len())]
                    }
                    Some(c) if !run_to_bound && rng.chance(40) => c,
                    _ => toks[rng.below(toks.len())],
                }
            }
            None => match rng.below(100) {
                0..=2 => EOS,
                3..=22 => 300 + rng.below(MERGED.len()) as TokenId,
                23..=62 => HOSTILE[rng.below(HOSTILE.len())] as TokenId,
                _ => b"abcdxyz_/."[rng.below(10)] as TokenId,
            },
        },
    ))
}

/// Whether `args` has exactly the shape `tool`'s schema declares: no field the
/// schema lacks, every required field, strings where strings are declared,
/// guided arrays and objects as arrays and objects. A free value (a number, or
/// `pack`'s untyped `values`) may be any JSON — the grammar holds structure,
/// not scalar type.
fn conforms(tool: &str, args: &Value) -> Result<(), String> {
    let obj = args.as_object().ok_or("arguments are not an object")?;
    let allowed_keys = |keys: &[&str]| -> Result<(), String> {
        match obj.keys().find(|k| !keys.contains(&k.as_str())) {
            Some(k) => Err(format!("undeclared field {k:?}")),
            None => Ok(()),
        }
    };
    match tool {
        "read_files" => {
            allowed_keys(&["files"])?;
            let files = obj["files"].as_array().ok_or("files is not an array")?;
            for f in files {
                let f = f.as_object().ok_or("an element is not an object")?;
                if let Some(k) = f
                    .keys()
                    .find(|k| !["path", "start_line", "end_line"].contains(&k.as_str()))
                {
                    return Err(format!("undeclared element field {k:?}"));
                }
                f.get("path")
                    .and_then(Value::as_str)
                    .ok_or("an element's path is missing or not a string")?;
            }
        }
        "run_commands" => {
            allowed_keys(&["commands"])?;
            let cmds = obj["commands"]
                .as_array()
                .ok_or("commands is not an array")?;
            if cmds.iter().any(|c| !c.is_string()) {
                return Err("a command is not a string".into());
            }
        }
        "pack" => {
            allowed_keys(&["values", "label"])?;
            if !obj.contains_key("values") {
                return Err("values is missing".into());
            }
            if obj.get("label").is_some_and(|l| !l.is_string()) {
                return Err("label is not a string".into());
            }
        }
        other => return Err(format!("{other:?} is not in the catalog")),
    }
    Ok(())
}

/// What the adversaries made the stencil do, across all runs — so a fuzz that
/// passes is known to have reached the paths it exists to test, rather than
/// having wandered down the easy ones.
#[derive(Debug, Default)]
struct Coverage {
    escaped: usize,
    completed: usize,
    delimiters_dropped: usize,
    eos_intercepted: usize,
    merged_closes: usize,
    forced: usize,
    multi_element: usize,
    at_bound: usize,
}

impl Coverage {
    fn record(&mut self, observes: &[Observe], call: &Value) {
        for o in observes {
            match o {
                Observe::Repaired { closed: false } => self.escaped += 1,
                Observe::Repaired { closed: true } => self.completed += 1,
                Observe::DelimiterDropped => self.delimiters_dropped += 1,
                Observe::TokenClosedDrop => self.eos_intercepted += 1,
                Observe::SpanClosed { leftover } if *leftover > 0 => self.merged_closes += 1,
                Observe::SpanForcedClosed => self.forced += 1,
                _ => {}
            }
        }
        let args = &call["arguments"];
        for list in [&args["files"], &args["commands"]] {
            match list.as_array().map(Vec::len) {
                Some(n) if n == MAX_ARRAY_ELEMENTS => self.at_bound += 1,
                Some(n) if n > 1 => self.multi_element += 1,
                _ => {}
            }
        }
    }
}

#[test]
fn random_adversaries_always_produce_a_call_that_fits_the_schema() {
    let v = vocab();
    let tree = tree(&v);
    let names: Vec<String> = catalog().into_iter().map(|t| t.name).collect();
    let mut seen = std::collections::HashSet::new();
    let mut coverage = Coverage::default();
    for seed in 0..3000u64 {
        let run = simulate(tree.clone(), &v, adversary(seed), 200_000)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let text = run.text(&v);
        assert!(
            !run.observes.contains(&Observe::Bailed),
            "seed {seed}: bailed: {text:?}"
        );
        let call = parse(&text);
        let name = call["name"].as_str().unwrap_or_default();
        assert!(
            names.iter().any(|n| n == name),
            "seed {seed}: name {name:?}"
        );
        if let Err(why) = conforms(name, &call["arguments"]) {
            panic!("seed {seed}: {why}: {text:?}");
        }
        seen.insert(name.to_string());
        coverage.record(&run.observes, &call);
    }
    assert_eq!(
        seen.len(),
        names.len(),
        "every tool was exercised: {seen:?}"
    );
    let c = &coverage;
    for (path, hits) in [
        ("a character escaped", c.escaped),
        ("a malformed value completed", c.completed),
        ("a delimiter dropped", c.delimiters_dropped),
        ("an EOS intercepted", c.eos_intercepted),
        ("a close merged with what follows", c.merged_closes),
        ("an array of several elements", c.multi_element),
        ("an array at its bound", c.at_bound),
    ] {
        assert!(hits > 0, "no run reached {path}: {coverage:?}");
    }
}
