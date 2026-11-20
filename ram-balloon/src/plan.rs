//! The schedule: phases of active and idle RAM, each reached by a ramp and held
//! for its length.
//!
//! A phase is written `active=16G,idle=8G,secs=120,ramp=30`. A field left out
//! keeps the previous phase's value — `secs` and `ramp` excepted, which default
//! to 60 and 0 — so a schedule reads as what changes.

use anyhow::{bail, Context, Result};

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

/// One phase: the RAM to hold, how long to hold it, and how long to ramp to it
/// from the phase before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phase {
    /// Bytes kept resident by touching every page each sweep — a build's working
    /// set.
    pub active: u64,
    /// Bytes written once and left — a minimized app's.
    pub idle: u64,
    /// Seconds the phase lasts, the ramp included.
    pub secs: u64,
    /// Seconds over which the targets move from the previous phase's.
    pub ramp: u64,
}

/// A size: a number with `G`, `M` or no suffix (bytes).
pub fn parse_size(s: &str) -> Result<u64> {
    let s = s.trim();
    let (num, mult) = match s.as_bytes().last() {
        Some(b'G' | b'g') => (&s[..s.len() - 1], GIB),
        Some(b'M' | b'm') => (&s[..s.len() - 1], MIB),
        _ => (s, 1),
    };
    let n: f64 = num.parse().with_context(|| format!("size {s:?}"))?;
    if !n.is_finite() || n < 0.0 {
        bail!("size {s:?} is not a non-negative number");
    }
    Ok((n * mult as f64) as u64)
}

/// Parse a schedule of phase specs, each following the one before it.
pub fn parse_plan(specs: &[String]) -> Result<Vec<Phase>> {
    let mut out = Vec::with_capacity(specs.len());
    let mut prev = Phase {
        active: 0,
        idle: 0,
        secs: 60,
        ramp: 0,
    };
    for spec in specs {
        let mut p = Phase {
            secs: 60,
            ramp: 0,
            ..prev
        };
        for field in spec.split(',').map(str::trim).filter(|f| !f.is_empty()) {
            let (k, v) = field
                .split_once('=')
                .with_context(|| format!("phase field {field:?} is not key=value"))?;
            match k.trim() {
                "active" => p.active = parse_size(v)?,
                "idle" => p.idle = parse_size(v)?,
                "secs" => p.secs = v.trim().parse().with_context(|| format!("secs {v:?}"))?,
                "ramp" => p.ramp = v.trim().parse().with_context(|| format!("ramp {v:?}"))?,
                other => bail!("phase field {other:?} is not active, idle, secs or ramp"),
            }
        }
        if p.ramp > p.secs {
            bail!("phase {spec:?}: its ramp is longer than the phase");
        }
        out.push(p);
        prev = p;
    }
    Ok(out)
}

/// The targets `t` seconds into `phase`, ramping from `from`.
pub fn target_at(from: (u64, u64), phase: &Phase, t: u64) -> (u64, u64) {
    if phase.ramp == 0 || t >= phase.ramp {
        return (phase.active, phase.idle);
    }
    let lerp = |a: u64, b: u64| {
        let f = t as f64 / phase.ramp as f64;
        (a as f64 + (b as f64 - a as f64) * f) as u64
    };
    (lerp(from.0, phase.active), lerp(from.1, phase.idle))
}

/// The second of `phase` that `elapsed` into it falls in, or `None` once the
/// phase is over. Read from the clock, not counted: a tick that runs long —
/// allocating gigabytes, or a machine under the very pressure being made —
/// must not stretch the phase.
pub fn second_of(phase: &Phase, elapsed: std::time::Duration) -> Option<u64> {
    let s = elapsed.as_secs();
    (s < phase.secs).then_some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// **A phase ends on the clock**, however few ticks ran inside it.
    #[test]
    fn a_phase_ends_on_the_clock() {
        let p = Phase {
            active: GIB,
            idle: 0,
            secs: 120,
            ramp: 15,
        };
        assert_eq!(second_of(&p, Duration::from_millis(0)), Some(0));
        assert_eq!(second_of(&p, Duration::from_millis(7_900)), Some(7));
        assert_eq!(second_of(&p, Duration::from_millis(119_999)), Some(119));
        assert_eq!(second_of(&p, Duration::from_secs(120)), None);
        assert_eq!(second_of(&p, Duration::from_secs(290)), None);
    }

    #[test]
    fn sizes_take_a_suffix() {
        assert_eq!(parse_size("16G").unwrap(), 16 * GIB);
        assert_eq!(parse_size("1.5g").unwrap(), GIB + GIB / 2);
        assert_eq!(parse_size("512M").unwrap(), 512 * MIB);
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert!(parse_size("-1G").is_err());
        assert!(parse_size("lots").is_err());
    }

    /// **A field left out keeps the previous phase's value**, `secs` and
    /// `ramp` excepted.
    #[test]
    fn a_phase_inherits_what_it_leaves_out() {
        let plan = parse_plan(&[
            "idle=20G,secs=120".into(),
            "active=16G,ramp=30,secs=180".into(),
            "active=0".into(),
        ])
        .unwrap();
        assert_eq!(
            plan,
            vec![
                Phase {
                    active: 0,
                    idle: 20 * GIB,
                    secs: 120,
                    ramp: 0
                },
                Phase {
                    active: 16 * GIB,
                    idle: 20 * GIB,
                    secs: 180,
                    ramp: 30
                },
                Phase {
                    active: 0,
                    idle: 20 * GIB,
                    secs: 60,
                    ramp: 0
                },
            ]
        );
        assert!(parse_plan(&["ramp=90,secs=60".into()]).is_err());
        assert!(parse_plan(&["hot=1G".into()]).is_err());
    }

    /// **The ramp moves both targets linearly**, and the phase's own targets
    /// hold once it is done.
    #[test]
    fn the_ramp_is_linear_then_holds() {
        let p = Phase {
            active: 16 * GIB,
            idle: 0,
            secs: 60,
            ramp: 4,
        };
        let from = (0, 8 * GIB);
        assert_eq!(target_at(from, &p, 0), (0, 8 * GIB));
        assert_eq!(target_at(from, &p, 1), (4 * GIB, 6 * GIB));
        assert_eq!(target_at(from, &p, 2), (8 * GIB, 4 * GIB));
        assert_eq!(target_at(from, &p, 4), (16 * GIB, 0));
        assert_eq!(target_at(from, &p, 59), (16 * GIB, 0));
    }
}
