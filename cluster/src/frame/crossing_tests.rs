//! The frames two protocols share (§3.5), pinned to raw bytes: a member on an
//! older protocol still has to read these to fetch the build that replaces it.
//! A test failing here is a change that strands every member behind it, which
//! a new frame kind avoids and a new `PROTO` does not.

use super::*;

fn id() -> BuildId {
    BuildId {
        count: 7,
        hash: "ab".into(),
        local: String::new(),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn offer() -> BuildArtifacts {
    BuildArtifacts {
        build: id(),
        artifacts: vec![Artifact {
            kind: "m".into(),
            platform: "p".into(),
            size: 1,
            sha256: [2; 32],
        }],
    }
}

#[test]
fn the_handshake_keeps_its_bytes() {
    let hello = Frame::Hello(Box::new(Hello {
        proto: 8,
        app: "z".into(),
        node: 1,
        host: "h".into(),
        card: 2,
        build: id(),
        rejected: false,
        model: None,
        power: 3,
        repos: vec![],
        now_ms: 4,
        nonce: [5; 32],
    }));
    let admit = Frame::Admit(Admit {
        session: 1,
        members: vec![],
        claims: vec![],
        floor_count: 7,
        floor_hash: "ab".into(),
        routes_wanted: false,
    });
    assert_eq!(
        hex(&hello.encode()),
        "0100000000000800010000007a01000000000000000100000068020000000000000007000000020000006162\
         0000000000000300000000000000040000000000000005050505050505050505050505050505050505050505\
         05050505050505050505"
    );
    assert_eq!(
        hex(&admit.encode()),
        "040000000000010000000000000000000000000000000700000002000000616200"
    );
}

#[test]
fn the_builds_frames_keep_their_bytes() {
    let fetch = Frame::BuildFetch(BuildFetch {
        artifact: [2; 32],
        offset: 3,
        from_node: 4,
        reply_to: 5,
    });
    let chunk = Frame::BuildChunk(BuildChunk {
        artifact: [2; 32],
        offset: 3,
        reply_to: 5,
        bytes: vec![9],
    });
    const BLOB: &str = "0100000009";
    const SHA: &str = "0202020202020202020202020202020202020202020202020202020202020202";
    const OFFER: &str = "0000000000070000000200000061620000000001000000010000006d0100000070\
                         0100000000000000";
    assert_eq!(
        hex(&Frame::BuildOffer(offer()).encode()),
        format!("1e{OFFER}{SHA}")
    );
    assert_eq!(
        hex(&Frame::BuildHave(offer()).encode()),
        format!("1f{OFFER}{SHA}")
    );
    assert_eq!(
        hex(&fetch.encode()),
        format!("200000000000{SHA}030000000000000004000000000000000500000000000000")
    );
    assert_eq!(
        hex(&chunk.encode()),
        format!("210000000000{SHA}03000000000000000500000000000000{BLOB}")
    );
    assert_eq!(
        hex(&Frame::Ping(*b"12345678").encode()),
        "0c00000000003132333435363738"
    );
    assert_eq!(
        hex(&Frame::Pong(*b"12345678").encode()),
        "0d00000000003132333435363738"
    );
}
