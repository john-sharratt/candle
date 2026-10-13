//! The Paxos steps a gateway carries, as coordination-log lines (§12.3): a
//! round prepared, promised, refused with a `NACK`, proposed, accepted, and a
//! lease released. Each line's subject is the key, so the log filters a key's
//! whole history.

use crate::frame::{Ballot, Frame};

fn ballot(b: Ballot) -> String {
    format!("{}.{:016x}", b.round, b.node)
}

/// The key and line for `frame`, sent by `from`; `None` for a frame that is no
/// Paxos step.
pub fn paxos_step(from: u64, frame: &Frame) -> Option<(String, String)> {
    let key = |k: &[u8]| String::from_utf8_lossy(k).into_owned();
    Some(match frame {
        Frame::Prepare(p) => (
            key(&p.key),
            format!("prepared at {} by {from:016x}", ballot(p.ballot)),
        ),
        Frame::Promise(p) => (
            key(&p.key),
            format!(
                "{:016x} promised {}{}",
                p.from,
                ballot(p.ballot),
                p.accepted.as_ref().map_or(String::new(), |a| format!(
                    ", having accepted {}",
                    ballot(a.ballot)
                ))
            ),
        ),
        Frame::Nack(n) => (
            key(&n.key),
            format!(
                "{:016x} refused {:016x}: it promised {}",
                n.from,
                n.reply_to,
                ballot(n.promised)
            ),
        ),
        Frame::Propose(p) => (
            key(&p.key),
            format!(
                "proposed at {} by {from:016x}, {} bytes",
                ballot(p.ballot),
                p.value.len()
            ),
        ),
        Frame::Accepted(a) => (
            key(&a.key),
            format!("{:016x} accepted {}", a.from, ballot(a.ballot)),
        ),
        Frame::Release(r) => (
            key(&r.key),
            format!(
                "released by {from:016x} through oseq {}, its lease of {}",
                r.through,
                ballot(r.ballot)
            ),
        ),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Accepted, AcceptedValue, Nack, Prepare, Promise, Propose, Release};

    #[test]
    fn each_step_reads_as_one_line_about_its_key() {
        let b = Ballot { round: 3, node: 1 };
        let k = b"gen/4".to_vec();
        let lines: Vec<(String, String)> = [
            Frame::Prepare(Prepare {
                key: k.clone(),
                ballot: b,
            }),
            Frame::Promise(Promise {
                key: k.clone(),
                ballot: b,
                accepted: Some(AcceptedValue {
                    ballot: Ballot { round: 2, node: 2 },
                    value: vec![1],
                }),
                from: 2,
                reply_to: 1,
                tiebreak: false,
            }),
            Frame::Nack(Nack {
                key: k.clone(),
                promised: Ballot { round: 5, node: 3 },
                from: 3,
                reply_to: 1,
                tiebreak: false,
            }),
            Frame::Propose(Propose {
                key: k.clone(),
                ballot: b,
                value: vec![0; 8],
            }),
            Frame::Accepted(Accepted {
                key: k.clone(),
                ballot: b,
                from: 2,
                reply_to: 1,
                tiebreak: false,
            }),
            Frame::Release(Release {
                key: k,
                ballot: b,
                through: 77,
            }),
        ]
        .iter()
        .filter_map(|f| paxos_step(1, f))
        .collect();
        let texts: Vec<&str> = lines.iter().map(|(_, t)| t.as_str()).collect();
        assert!(lines.iter().all(|(key, _)| key == "gen/4"));
        assert_eq!(
            texts,
            [
                "prepared at 3.0000000000000001 by 0000000000000001",
                "0000000000000002 promised 3.0000000000000001, having accepted 2.0000000000000002",
                "0000000000000003 refused 0000000000000001: it promised 5.0000000000000003",
                "proposed at 3.0000000000000001 by 0000000000000001, 8 bytes",
                "0000000000000002 accepted 3.0000000000000001",
                "released by 0000000000000001 through oseq 77, its lease of 3.0000000000000001",
            ]
        );
        assert_eq!(paxos_step(1, &Frame::Ping([0; 8])), None);
    }
}
