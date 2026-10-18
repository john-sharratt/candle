//! A forced rollout's plan (§14.1 forced rollout).

use super::*;
use crate::coord::HoldLine;
use crate::version::info::Opens;

fn b(count: u32) -> BuildId {
    BuildId {
        count,
        hash: format!("h{count}"),
        local: String::new(),
    }
}

fn opens(tags: &[u8]) -> Opens {
    Opens {
        log_format: Some(3),
        record_tags: tags.to_vec(),
        model: Some("M".into()),
    }
}

fn hold(build: &str, kind: &str, platform: &str, source: &str, tags: &[u8]) -> HoldLine {
    HoldLine {
        build: build.into(),
        kind: kind.into(),
        platform: platform.into(),
        size: 100,
        sha256: format!("{build}-{kind}-{platform}"),
        source: source.into(),
        opens: Some(opens(tags)),
    }
}

fn view(platform: &str, census: &[u8], holds: Vec<HoldLine>) -> UpgradeView {
    UpgradeView {
        kind: "zend".into(),
        platform: platform.into(),
        log_format: Some(3),
        census: census.to_vec(),
        model: Some("M".into()),
        holds,
        ..UpgradeView::default()
    }
}

fn ask(build: &str) -> Ask {
    Ask {
        app: "zend".into(),
        build: build.into(),
        from: None,
        targets: None,
        parallel: 1,
        by: "ops@tokera.com".into(),
    }
}

/// A downgrade to a build the source keeps: the target that keeps it too
/// fetches nothing, another fetches its bytes, one on another platform with no
/// artifact is stranded, one already on the build and a local build are
/// skipped, and the builds above it are rejected.
#[test]
fn a_downgrade_is_planned_member_by_member() {
    let (b10, b12, b13) = (b(10), b(12), b(13));
    let local = BuildId {
        local: "abcd".into(),
        ..b(12)
    };
    let source = view(
        "linux-x86_64",
        &[1],
        vec![hold("10-h10", "zend", "linux-x86_64", "kept", &[1, 2])],
    );
    let keeper = view(
        "linux-x86_64",
        &[1],
        vec![hold("10-h10", "zend", "linux-x86_64", "kept", &[1, 2])],
    );
    let fetcher = view("linux-x86_64", &[1], vec![]);
    let other_platform = view("windows-x86_64", &[1], vec![]);
    let members = [
        Member {
            node: 1,
            host: "m1",
            build: &b12,
            ready: true,
            upgrade: Some(&source),
        },
        Member {
            node: 2,
            host: "m2",
            build: &b12,
            ready: true,
            upgrade: Some(&keeper),
        },
        Member {
            node: 3,
            host: "m3",
            build: &b13,
            ready: true,
            upgrade: Some(&fetcher),
        },
        Member {
            node: 4,
            host: "m4",
            build: &b12,
            ready: true,
            upgrade: Some(&other_platform),
        },
        Member {
            node: 5,
            host: "m5",
            build: &b10,
            ready: true,
            upgrade: Some(&fetcher),
        },
        Member {
            node: 6,
            host: "m6",
            build: &local,
            ready: true,
            upgrade: Some(&fetcher),
        },
    ];
    let p = plan(
        &ask("10-h10"),
        &members,
        Some(&b(12)),
        3,
        false,
        &[b12.clone(), b13.clone(), b10.clone()],
        7,
    )
    .unwrap();
    assert_eq!(p.direction, "downgrade");
    assert_eq!(
        (p.floor_before.as_deref(), p.floor_after.as_str()),
        (Some("12-h12"), "10-h10")
    );
    assert_eq!(p.from, "0000000000000001");
    let targets: Vec<(&str, &str, u64)> = p
        .targets
        .iter()
        .map(|t| (t.host.as_str(), t.source.as_str(), t.bytes))
        .collect();
    assert_eq!(
        targets,
        [("m1", "kept", 0), ("m2", "kept", 0), ("m3", "fetch", 100)]
    );
    let stranded: Vec<(&str, &str)> = p
        .stranded
        .iter()
        .map(|s| (s.host.as_str(), s.why.as_str()))
        .collect();
    assert_eq!(
        stranded,
        [("m4", "no zend artifact for windows-x86_64 in 10-h10")]
    );
    let skipped: Vec<&str> = p.skipped.iter().map(|s| s.host.as_str()).collect();
    assert_eq!(skipped, ["m5", "m6"]);
    assert_eq!(p.rejects, ["12-h12", "13-h13"]);
    assert_eq!(
        p.decision.targets,
        ["0000000000000001", "0000000000000002", "0000000000000003"]
    );
    assert_eq!(p.decision.by, "ops@tokera.com");
}

/// A downgrade across a record type the older build lacks strands every
/// member holding one.
#[test]
fn a_downgrade_across_a_new_record_type_is_stranded() {
    let b12 = b(12);
    let source = view(
        "linux-x86_64",
        &[1, 31],
        vec![hold("10-h10", "zend", "linux-x86_64", "kept", &[1, 2])],
    );
    let other = view("linux-x86_64", &[1, 31], vec![]);
    let members = [
        Member {
            node: 1,
            host: "m1",
            build: &b12,
            ready: true,
            upgrade: Some(&source),
        },
        Member {
            node: 2,
            host: "m2",
            build: &b12,
            ready: true,
            upgrade: Some(&other),
        },
    ];
    let p = plan(&ask("10-h10"), &members, Some(&b(12)), 1, false, &[], 0).unwrap();
    assert!(p.targets.is_empty());
    assert_eq!(p.stranded.len(), 2);
    assert!(p.stranded[0]
        .why
        .starts_with("the substrate holds record types 31"));
}

/// The refusals: no quorum, `parallel` beyond ready members less the quorum,
/// a source not holding the build or not connected, a local build, a name
/// that is not a build.
#[test]
fn a_rollout_is_refused_outright_when_it_cannot_run() {
    let b12 = b(12);
    let local = BuildId {
        local: "abcd".into(),
        ..b(10)
    };
    let holder = view(
        "linux-x86_64",
        &[1],
        vec![hold("10-h10", "zend", "linux-x86_64", "running", &[1])],
    );
    let empty = view("linux-x86_64", &[1], vec![]);
    let members = [
        Member {
            node: 1,
            host: "m1",
            build: &local,
            ready: true,
            upgrade: Some(&holder),
        },
        Member {
            node: 2,
            host: "m2",
            build: &b12,
            ready: true,
            upgrade: Some(&empty),
        },
        Member {
            node: 3,
            host: "m3",
            build: &b12,
            ready: false,
            upgrade: Some(&empty),
        },
    ];
    assert_eq!(
        plan(&ask("10-h10"), &members, Some(&b(12)), 3, false, &[], 0).unwrap_err(),
        "no quorum: 2 members ready, 3 needed to decide the rollout"
    );
    let wide = Ask {
        parallel: 2,
        ..ask("10-h10")
    };
    assert!(plan(&wide, &members, Some(&b(12)), 1, false, &[], 0)
        .unwrap_err()
        .starts_with(
            "parallel 2 is more than the 2 ready members can spare and keep the quorum of 1"
        ));
    let from2 = Ask {
        from: Some(2),
        ..ask("10-h10")
    };
    assert_eq!(
        plan(&from2, &members, Some(&b(12)), 1, false, &[], 0).unwrap_err(),
        "m2 does not hold 10-h10"
    );
    let from9 = Ask {
        from: Some(9),
        ..ask("10-h10")
    };
    assert_eq!(
        plan(&from9, &members, Some(&b(12)), 1, false, &[], 0).unwrap_err(),
        "member 0000000000000009 is not connected"
    );
    assert!(
        plan(&ask("10-h10"), &members, Some(&b(12)), 1, false, &[], 0)
            .unwrap_err()
            .contains("is a local build")
    );
    assert!(plan(&ask("ten"), &members, None, 1, false, &[], 0)
        .unwrap_err()
        .contains("is not a build"));
}

/// **A group of two with a gateway tie-break can roll out, one member at a
/// time** (§5.9). The gateway's vote fills the seat of the member restarting,
/// as consensus counts it, so the plan gets past its availability checks —
/// here to the source's own refusal, a local build — where without the
/// tie-break it is refused for having no member to spare. Two at once is still
/// refused, and so is any step once only one member is ready.
#[test]
fn a_two_member_group_rolls_one_at_a_time_on_the_tiebreak() {
    let b12 = b(12);
    let local = BuildId {
        local: "abcd".into(),
        ..b(10)
    };
    let holder = view(
        "linux-x86_64",
        &[1],
        vec![hold("10-h10", "zend", "linux-x86_64", "running", &[1])],
    );
    let empty = view("linux-x86_64", &[1], vec![]);
    let pair = |second_ready: bool| {
        [
            Member {
                node: 1,
                host: "m1",
                build: &local,
                ready: true,
                upgrade: Some(&holder),
            },
            Member {
                node: 2,
                host: "m2",
                build: &b12,
                ready: second_ready,
                upgrade: Some(&empty),
            },
        ]
    };
    let both = pair(true);
    assert!(plan(&ask("10-h10"), &both, Some(&b(12)), 2, false, &[], 0)
        .unwrap_err()
        .contains("can spare"));
    assert!(plan(&ask("10-h10"), &both, Some(&b(12)), 2, true, &[], 0)
        .unwrap_err()
        .contains("is a local build"));
    let wide = Ask {
        parallel: 2,
        ..ask("10-h10")
    };
    assert!(plan(&wide, &both, Some(&b(12)), 2, true, &[], 0)
        .unwrap_err()
        .contains("can spare"));
    let one = pair(false);
    assert!(plan(&ask("10-h10"), &one, Some(&b(12)), 2, true, &[], 0)
        .unwrap_err()
        .contains("can spare"));
}
