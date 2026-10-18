//! A forced rollout's plan (§3.11): what `rollout { app, build, from, targets,
//! parallel }` would do, worked out before anything moves — the direction and
//! the floors, each target and where its artifact comes from, the members
//! skipped and stranded, and the builds a downgrade rejects — or why it is
//! refused outright. The admin confirms it by typing the build's `count-hash`,
//! and the plan, as a [`Rollout`], is what the source member proposes.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use serde::Serialize;

use crate::consensus::seats;
use crate::coord::UpgradeView;
use crate::frame::BuildId;
use crate::upgrade::compat::{check, Needs};
use crate::upgrade::decisions::{build_name, parse_build, ArtifactLine, Rollout};

/// What the admin asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ask {
    pub app: String,
    /// `count-hash`.
    pub build: String,
    pub from: Option<u64>,
    pub targets: Option<Vec<u64>>,
    pub parallel: u32,
    pub by: String,
}

/// One member as the plan weighs it.
#[derive(Debug, Clone)]
pub struct Member<'a> {
    pub node: u64,
    pub host: &'a str,
    pub build: &'a BuildId,
    pub ready: bool,
    pub upgrade: Option<&'a UpgradeView>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TargetLine {
    pub node: String,
    pub host: String,
    pub kind: String,
    pub platform: String,
    pub current: String,
    /// `kept`: its own copy, nothing transferred; `fetch`: from a holder.
    pub source: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Aside {
    pub node: String,
    pub host: String,
    pub why: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Plan {
    /// `upgrade`, `downgrade` or `reinstall`.
    pub direction: String,
    pub floor_before: Option<String>,
    pub floor_after: String,
    pub from: String,
    pub parallel: u32,
    pub targets: Vec<TargetLine>,
    pub skipped: Vec<Aside>,
    pub stranded: Vec<Aside>,
    /// For a downgrade: the builds it rejects.
    pub rejects: Vec<String>,
    /// What the source proposes once the admin confirms.
    #[serde(skip)]
    pub decision: Rollout,
}

fn hex(node: u64) -> String {
    format!("{node:016x}")
}

/// The plan for `ask` over `members`, against the app's `floor`, `quorum`,
/// and the builds the app runs or was certifying (`seen`); or why not.
///
/// `tiebreak` is whether a gateway votes as the tie-break (§5.9). Its seat is
/// counted exactly as consensus counts it ([`seats`]): it fills the empty seat
/// of a group of two, so such a group still decides — and a rollout can
/// restart one member at a time — with one member away. Not counting it made
/// every forced rollout of a two-member group impossible, an emergency
/// downgrade among them.
pub fn plan(
    ask: &Ask,
    members: &[Member<'_>],
    floor: Option<&BuildId>,
    quorum: usize,
    tiebreak: bool,
    seen: &[BuildId],
    at_ms: u64,
) -> Result<Plan, String> {
    let (count, hash) = parse_build(&ask.build)
        .ok_or_else(|| format!("{:?} is not a build: `count-hash`", ask.build))?;
    let build = BuildId {
        count,
        hash,
        local: String::new(),
    };
    let name = build_name(&build);
    let ready = members.iter().filter(|m| m.ready).count();
    let voting = |members: usize| seats(quorum, tiebreak, members, usize::from(tiebreak));
    if voting(ready) < quorum {
        return Err(format!(
            "no quorum: {ready} members ready, {quorum} needed to decide the rollout"
        ));
    }
    // The members a step may take down at once, while those left — and the
    // tie-break's seat — still make the quorum.
    let parallel = ask.parallel.max(1);
    let spare = (0..=ready)
        .rev()
        .find(|down| voting(ready - down) >= quorum)
        .unwrap_or(0);
    if parallel as usize > spare {
        return Err(format!(
            "parallel {parallel} is more than the {ready} ready members can spare and keep \
             the quorum of {quorum}: consensus must stay available while members restart"
        ));
    }
    // The source: the member named, or the first that holds the build.
    let holds = |m: &Member<'_>| {
        m.upgrade
            .is_some_and(|v| v.holds.iter().any(|h| h.build == name))
    };
    let source = match ask.from {
        Some(n) => {
            let m = members
                .iter()
                .find(|m| m.node == n)
                .ok_or_else(|| format!("member {} is not connected", hex(n)))?;
            if !holds(m) {
                return Err(format!("{} does not hold {name}", m.host));
            }
            m
        }
        None => members
            .iter()
            .find(|m| holds(m))
            .ok_or_else(|| format!("no member holds {name}"))?,
    };
    let source_view = source.upgrade.expect("a holder reported its holdings");
    if source.build.count == count && source.build.hash == build.hash && source.build.is_local() {
        return Err(format!(
            "{name} on {} is a local build: push it, and roll out the build it becomes",
            source.host
        ));
    }
    let direction = match floor.map(|f| f.count.cmp(&count)) {
        Some(Ordering::Greater) => "downgrade",
        Some(Ordering::Equal) => "reinstall",
        _ => "upgrade",
    };
    let named: Option<BTreeSet<u64>> = ask.targets.as_ref().map(|t| t.iter().copied().collect());
    let (mut targets, mut skipped, mut stranded) = (Vec::new(), Vec::new(), Vec::new());
    let mut artifacts: Vec<ArtifactLine> = Vec::new();
    for m in members {
        let aside = |why: &str| Aside {
            node: hex(m.node),
            host: m.host.to_string(),
            why: why.to_string(),
        };
        let is_named = named.as_ref().is_some_and(|n| n.contains(&m.node));
        if named.as_ref().is_some_and(|n| !n.contains(&m.node)) {
            continue;
        }
        if m.build.count == count && m.build.hash == build.hash && !m.build.is_local() {
            skipped.push(aside("already on the build"));
            continue;
        }
        if m.build.is_local() && !is_named {
            skipped.push(aside("a local build, installed over only when named"));
            continue;
        }
        let Some(view) = m.upgrade else {
            stranded.push(aside("it has not reported what it can open"));
            continue;
        };
        if view.refused.contains(&name) && !is_named {
            skipped.push(aside(
                "an install of the build failed here: named, it is retried",
            ));
            continue;
        }
        let needs = Needs {
            log_format: view.log_format,
            census: view.census.clone(),
            model: view.model.clone(),
        };
        let own = view
            .holds
            .iter()
            .find(|h| h.build == name && h.source == "kept" && h.kind == view.kind);
        let theirs = source_view
            .holds
            .iter()
            .find(|h| h.build == name && h.kind == view.kind && h.platform == view.platform);
        let (line, how, bytes) = match (own, theirs) {
            (Some(k), _) => (k, "kept", 0),
            (None, Some(a)) => (a, "fetch", a.size),
            (None, None) => {
                stranded.push(aside(&format!(
                    "no {} artifact for {} in {name}",
                    view.kind, view.platform
                )));
                continue;
            }
        };
        if let Err(why) = check(&line.opens.clone().unwrap_or_default(), &needs) {
            stranded.push(aside(&why));
            continue;
        }
        if !artifacts.iter().any(|a| a.sha256 == line.sha256) {
            artifacts.push(ArtifactLine {
                kind: line.kind.clone(),
                platform: line.platform.clone(),
                size: line.size,
                sha256: line.sha256.clone(),
            });
        }
        targets.push(TargetLine {
            node: hex(m.node),
            host: m.host.to_string(),
            kind: view.kind.clone(),
            platform: view.platform.clone(),
            current: build_name(m.build),
            source: how.into(),
            bytes,
        });
    }
    let mut rejects: Vec<String> = if direction == "downgrade" {
        seen.iter()
            .filter(|b| b.count > count && !b.is_local())
            .map(build_name)
            .collect()
    } else {
        Vec::new()
    };
    rejects.sort();
    rejects.dedup();
    let decision = Rollout {
        build: name.clone(),
        from: hex(source.node),
        targets: targets.iter().map(|t| t.node.clone()).collect(),
        parallel,
        artifacts,
        rejects: rejects.clone(),
        by: ask.by.clone(),
        at_ms,
    };
    Ok(Plan {
        direction: direction.into(),
        floor_before: floor.map(build_name),
        floor_after: name,
        from: hex(source.node),
        parallel,
        targets,
        skipped,
        stranded,
        rejects,
        decision,
    })
}

#[cfg(test)]
mod tests;
