//! Protocol v2's `ls-refs`, asking only for branches: the request body, the
//! capability check on the server's advertisement, and the answer read back
//! as branch tips.
//!
//! Over HTTP the request is `POST <url>/git-upload-pack` with the header
//! `Git-Protocol: version=2`, and the advertisement is
//! `GET <url>/info/refs?service=git-upload-pack` with the same header.

use super::pkt_line::{self, Packet, DELIM, FLUSH};
use super::ProbeError;
use crate::types::{BranchName, ObjectFormat, Oid};

/// The prefix every branch ref carries, and the only one asked for.
const HEADS: &str = "refs/heads/";

/// One branch as origin holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchTip {
    pub name: BranchName,
    pub tip: Oid,
}

/// The body of an `ls-refs` request for every branch — nothing peeled, no
/// symbolic refs, no tags. A SHA-256 repository says so, since SHA-1 is what
/// a server assumes.
pub fn request(format: ObjectFormat) -> Vec<u8> {
    let mut body = pkt_line::encode(b"command=ls-refs\n");
    if format == ObjectFormat::Sha256 {
        body.extend(pkt_line::encode(b"object-format=sha256\n"));
    }
    body.extend_from_slice(DELIM);
    body.extend(pkt_line::encode(format!("ref-prefix {HEADS}\n").as_bytes()));
    body.extend_from_slice(FLUSH);
    body
}

/// Whether a `GET info/refs` advertisement says the server speaks protocol
/// v2 and answers `ls-refs`. Over smart HTTP the advertisement may open with
/// a `# service=git-upload-pack` packet and a flush, which are skipped.
pub fn advertises_ls_refs(advertisement: &[u8]) -> Result<bool, ProbeError> {
    let mut version_2 = false;
    let mut ls_refs = false;
    for packet in pkt_line::decode(advertisement)? {
        let Packet::Data(line) = packet else {
            continue;
        };
        let line = String::from_utf8_lossy(line);
        let line = line.trim_end_matches('\n');
        if line == "version 2" {
            version_2 = true;
        } else if version_2 && (line == "ls-refs" || line.starts_with("ls-refs=")) {
            ls_refs = true;
        }
    }
    Ok(version_2 && ls_refs)
}

/// The branches in an `ls-refs` answer, in name order. An `ERR` packet is the
/// server refusing; anything that is not `<id> refs/heads/<name>` (with any
/// attributes after it) up to the flush is malformed. A ref outside
/// `refs/heads/` is skipped, since none was asked for.
pub fn parse_response(body: &[u8]) -> Result<Vec<BranchTip>, ProbeError> {
    let mut tips = Vec::new();
    let mut flushed = false;
    for packet in pkt_line::decode(body)? {
        match packet {
            Packet::Flush => {
                flushed = true;
                break;
            }
            Packet::Data(line) => {
                let line = std::str::from_utf8(line)
                    .map_err(|_| ProbeError::Malformed("a ref line is not UTF-8".into()))?;
                let line = line.trim_end_matches('\n');
                if let Some(why) = line.strip_prefix("ERR ") {
                    return Err(ProbeError::Refused(why.to_string()));
                }
                let bad = || ProbeError::Malformed(format!("{line:?} is not a ref line"));
                let mut fields = line.split(' ');
                let (Some(oid), Some(name)) = (fields.next(), fields.next()) else {
                    return Err(bad());
                };
                let tip = Oid::parse(oid).map_err(|_| bad())?;
                let Some(branch) = name.strip_prefix(HEADS) else {
                    continue;
                };
                let name = BranchName::parse(branch).map_err(|_| bad())?;
                tips.push(BranchTip { name, tip });
            }
            Packet::Delim | Packet::ResponseEnd => {
                return Err(ProbeError::Malformed(
                    "a marker other than flush in an ls-refs answer".into(),
                ))
            }
        }
    }
    if !flushed {
        return Err(ProbeError::Malformed(
            "the ls-refs answer ends without a flush".into(),
        ));
    }
    tips.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
    Ok(tips)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
    const B: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

    #[test]
    fn the_request_asks_for_branches_alone() {
        assert_eq!(
            request(ObjectFormat::Sha1),
            b"0014command=ls-refs\n0001001bref-prefix refs/heads/\n0000"
        );
        assert_eq!(
            request(ObjectFormat::Sha256),
            b"0014command=ls-refs\n0019object-format=sha256\n0001001bref-prefix refs/heads/\n0000"
                .to_vec()
        );
    }

    /// The advertisement GitHub serves, service header and all.
    #[test]
    fn a_v2_advertisement_offering_ls_refs_is_recognised() {
        let advertisement = b"001e# service=git-upload-pack\n0000000eversion 2\n0022agent=git/github-7c8d2b2f66c2\n0013ls-refs=unborn\n0020fetch=shallow wait-for-done\n0012server-option\n0017object-format=sha1\n0000";
        assert!(advertises_ls_refs(advertisement).unwrap());
    }

    /// A v0 advertisement — the ref list itself — is not v2, whatever it
    /// names.
    #[test]
    fn a_v0_advertisement_is_not_v2() {
        let line = format!("{A} refs/heads/main\0 multi_ack ls-refs\n");
        let advertisement = [
            pkt_line::encode(b"# service=git-upload-pack\n"),
            FLUSH.to_vec(),
            pkt_line::encode(line.as_bytes()),
            FLUSH.to_vec(),
        ]
        .concat();
        assert!(!advertises_ls_refs(&advertisement).unwrap());
        let no_ls_refs = b"000eversion 2\n0012server-option\n0000";
        assert!(!advertises_ls_refs(no_ls_refs).unwrap());
    }

    #[test]
    fn the_answer_reads_as_branch_tips_in_name_order() {
        let body = [
            pkt_line::encode(format!("{B} refs/heads/zen/work\n").as_bytes()),
            pkt_line::encode(format!("{A} refs/heads/main symref-target:x\n").as_bytes()),
            pkt_line::encode(format!("{A} refs/tags/v1\n").as_bytes()),
            FLUSH.to_vec(),
        ]
        .concat();
        let tips = parse_response(&body).unwrap();
        let got: Vec<(&str, &str)> = tips
            .iter()
            .map(|t| (t.name.as_str(), t.tip.as_str()))
            .collect();
        assert_eq!(got, vec![("main", A), ("zen/work", B)]);
    }

    #[test]
    fn a_repository_with_no_branches_answers_with_a_flush() {
        assert_eq!(parse_response(b"0000").unwrap(), vec![]);
    }

    #[test]
    fn an_err_packet_is_a_refusal() {
        let body = [pkt_line::encode(b"ERR access denied\n"), FLUSH.to_vec()].concat();
        assert_eq!(
            parse_response(&body),
            Err(ProbeError::Refused("access denied".into()))
        );
    }

    #[test]
    fn a_line_that_is_not_a_ref_or_a_missing_flush_is_malformed() {
        let no_flush = pkt_line::encode(format!("{A} refs/heads/main\n").as_bytes());
        let bad_oid = [pkt_line::encode(b"xyz refs/heads/main\n"), FLUSH.to_vec()].concat();
        let one_field = [
            pkt_line::encode(format!("{A}\n").as_bytes()),
            FLUSH.to_vec(),
        ]
        .concat();
        let bad_name = [
            pkt_line::encode(format!("{A} refs/heads/a..b\n").as_bytes()),
            FLUSH.to_vec(),
        ]
        .concat();
        for body in [no_flush, bad_oid, one_field, bad_name] {
            assert!(
                matches!(parse_response(&body), Err(ProbeError::Malformed(_))),
                "{:?}",
                String::from_utf8_lossy(&body)
            );
        }
    }
}
