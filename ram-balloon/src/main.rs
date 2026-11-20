//! `ram-balloon`: hold active and idle host RAM on a schedule, and log once a
//! second how this process and a watched one page (`docs/elastic_warm_tier.md`
//! §8).
//!
//! ```text
//! ram-balloon --watch motherd --log run.csv \
//!     --phase idle=20G,secs=300 \
//!     --phase active=16G,ramp=20,secs=180 \
//!     --phase active=0,secs=600
//! ```
//!
//! Active RAM is touched every sweep, as a build's working set is; idle RAM is
//! written once and left, as a minimized app's is. The schedule runs once; the
//! balloon then frees everything and exits.

mod hog;
mod plan;
mod report;

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use hog::Hog;
use plan::{parse_plan, second_of, target_at};
use report::{Sampler, HEADER};

struct Args {
    phases: Vec<String>,
    watch: Option<String>,
    log: Option<String>,
}

fn args() -> Result<Args> {
    let mut a = Args {
        phases: Vec::new(),
        watch: None,
        log: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--phase" => a.phases.push(value()?),
            "--watch" => a.watch = Some(value()?),
            "--log" => a.log = Some(value()?),
            other => {
                bail!("unknown flag {other:?}: --phase SPEC (repeated), --watch NAME, --log PATH")
            }
        }
    }
    if a.phases.is_empty() {
        bail!("give at least one --phase, e.g. --phase active=8G,idle=4G,secs=120,ramp=10");
    }
    Ok(a)
}

fn main() -> Result<()> {
    let a = args()?;
    let plan = parse_plan(&a.phases)?;
    let mut log = match &a.log {
        Some(path) => Some(std::fs::File::create(path).with_context(|| format!("log {path}"))?),
        None => None,
    };
    let mut emit = |line: &str| -> Result<()> {
        println!("{line}");
        if let Some(f) = log.as_mut() {
            writeln!(f, "{line}")?;
            f.flush()?;
        }
        Ok(())
    };
    emit(HEADER)?;
    let mut hog = Hog::new();
    let mut sampler = Sampler::default();
    let start = Instant::now();
    let mut from = (0u64, 0u64);
    for (i, phase) in plan.iter().enumerate() {
        let began = Instant::now();
        while let Some(s) = second_of(phase, began.elapsed()) {
            let (active, idle) = target_at(from, phase, s);
            let held = hog.set(active, idle);
            let line = sampler.line(
                start.elapsed().as_secs(),
                i,
                held,
                hog.sweeps(),
                a.watch.as_deref(),
            );
            emit(&line)?;
            let next = began + Duration::from_secs(s + 1);
            std::thread::sleep(next.saturating_duration_since(Instant::now()));
        }
        from = (phase.active, phase.idle);
    }
    hog.set(0, 0);
    Ok(())
}
