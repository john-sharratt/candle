//! Every frame kind encodes to exact bytes and decodes back; a truncated frame,
//! an unknown kind, a stray stream id or flag and a trailing byte are refused.
//! The expected bytes are spelled out by hand, not produced by the writer under
//! test.

use applied_set::AppliedSet;

use super::*;

/// Hand-assembled little-endian bytes.
#[derive(Default)]
struct B(Vec<u8>);

impl B {
    fn header(kind: u8, flags: u8, stream: u32) -> B {
        let mut b = B::default();
        b.0.push(kind);
        b.0.push(flags);
        b.0.extend_from_slice(&stream.to_le_bytes());
        b
    }
    fn u8(mut self, v: u8) -> B {
        self.0.push(v);
        self
    }
    fn u16(mut self, v: u16) -> B {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn u32(mut self, v: u32) -> B {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn u64(mut self, v: u64) -> B {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn raw(mut self, v: &[u8]) -> B {
        self.0.extend_from_slice(v);
        self
    }
    fn s(self, v: &str) -> B {
        self.u32(v.len() as u32).raw(v.as_bytes())
    }
    fn build(self, b: &BuildId) -> B {
        self.u32(b.count).s(&b.hash).s(&b.local)
    }
}

fn build() -> BuildId {
    BuildId {
        count: 4812,
        hash: "a1b2c3d".into(),
        local: String::new(),
    }
}

fn member() -> MemberInfo {
    MemberInfo {
        node: 0x3f,
        host: ".5".into(),
        app: "zend".into(),
        build: build(),
        weight: 7,
        state: MemberState::Ready,
        high_water: 99,
    }
}

fn member_bytes(b: B) -> B {
    b.u64(0x3f)
        .s(".5")
        .s("zend")
        .build(&build())
        .u32(7)
        .u8(1)
        .u64(99)
}

fn ballot() -> Ballot {
    Ballot { round: 3, node: 9 }
}

fn artifact() -> Artifact {
    Artifact {
        kind: "zend".into(),
        platform: "windows-x86_64".into(),
        size: 1 << 30,
        sha256: [0xab; 32],
    }
}

/// Every kind, once, with the bytes it must encode to.
fn samples() -> Vec<(Frame, Vec<u8>)> {
    let mut since = AppliedSet::default();
    since.insert_range(5, 1, 3);
    let model = ModelId {
        arch: "Qwen4Exp".into(),
        weights: [1; 32],
        tokenizer: [2; 32],
        rope: "lineage".into(),
        describe: "Flash".into(),
    };
    let header = RecordHeader {
        oseq: 4,
        record_type: 12,
        stream_id: 0x77,
        chunk_index: 2,
        len: 300,
        digest: [0xcd; 16],
    };
    vec![
        (
            Frame::Hello(Box::new(Hello {
                proto: 1,
                app: "zend".into(),
                node: 0x1122,
                host: ".5".into(),
                build: build(),
                rejected: true,
                model: Some(model.clone()),
                power: 170,
                repos: vec![RepoAd {
                    name: "candle".into(),
                    origin: "git@host:candle.git".into(),
                    push: true,
                }],
                now_ms: 1_700_000_000_000,
                nonce: [9; 32],
            })),
            B::header(1, 0, 0)
                .u16(1)
                .s("zend")
                .u64(0x1122)
                .s(".5")
                .build(&build())
                .u8(1)
                .u8(1)
                .s("Qwen4Exp")
                .raw(&[1; 32])
                .raw(&[2; 32])
                .s("lineage")
                .s("Flash")
                .u32(170)
                .u32(1)
                .s("candle")
                .s("git@host:candle.git")
                .u8(1)
                .u64(1_700_000_000_000)
                .raw(&[9; 32])
                .0,
        ),
        (
            Frame::Challenge(Challenge {
                nonce: [3; 32],
                proof: [4; 32],
            }),
            B::header(2, 0, 0).raw(&[3; 32]).raw(&[4; 32]).0,
        ),
        (
            Frame::Prove(Prove { proof: [5; 32] }),
            B::header(3, 0, 0).raw(&[5; 32]).0,
        ),
        (
            Frame::Admit(Admit {
                session: 77,
                members: vec![member()],
                claims: vec![ClaimInfo {
                    key: b"conv".to_vec(),
                    epoch: Epoch { ms: 10, node: 0x3f },
                }],
                floor_count: 4800,
                floor_hash: "ffee".into(),
                routes_wanted: true,
            }),
            member_bytes(B::header(4, 0, 0).u64(77).u32(1))
                .u32(1)
                .u32(4)
                .raw(b"conv")
                .u64(10)
                .u64(0x3f)
                .u32(4800)
                .s("ffee")
                .u8(1)
                .0,
        ),
        (
            Frame::Members(Members {
                epoch: 5,
                members: vec![member()],
            }),
            member_bytes(B::header(10, 0, 0).u64(5).u32(1)).0,
        ),
        (
            Frame::Capacity(Capacity {
                state: MemberState::Loading,
                power: 150,
                idle: true,
                slots_free: 3,
                decodes: 2,
                prefill_backlog: 4096,
                vram_free_mib: 8000,
                cores: 16,
                load_milli: 1250,
                os: "linux".into(),
                mem_free_mib: 30_000,
                jobs: 2,
                toolchains: vec!["cargo".into(), "npm".into()],
                low_priority: true,
                prefill_tok_s: 2300,
                decode_step_next_us: 41_500,
                decode_step_wide_us: 96_250,
                prefill_tok_s_wide: 640,
                prompt_tokens: 3_100,
                reply_tokens: 410,
                tokens_per_step_milli: 2_650,
            }),
            B::header(11, 0, 0)
                .u8(0)
                .u32(150)
                .u8(1)
                .u32(3)
                .u32(2)
                .u32(4096)
                .u32(8000)
                .u16(16)
                .u32(1250)
                .s("linux")
                .u32(30_000)
                .u32(2)
                .u32(2)
                .s("cargo")
                .s("npm")
                .u8(1)
                .u32(2300)
                .u32(41_500)
                .u32(96_250)
                .u32(640)
                .u32(3_100)
                .u32(410)
                .u32(2_650)
                .0,
        ),
        (
            Frame::Ping([1, 2, 3, 4, 5, 6, 7, 8]),
            B::header(12, 0, 0).raw(&[1, 2, 3, 4, 5, 6, 7, 8]).0,
        ),
        (Frame::Pong([8; 8]), B::header(13, 0, 0).raw(&[8; 8]).0),
        (
            Frame::Routes(Routes {
                table: vec![1, 2, 3],
            }),
            B::header(14, 0, 0).u32(3).raw(&[1, 2, 3]).0,
        ),
        (
            Frame::Claim(Claim {
                key: b"k".to_vec(),
                epoch: Epoch { ms: 1, node: 2 },
            }),
            B::header(15, 0, 0).u32(1).raw(b"k").u64(1).u64(2).0,
        ),
        (
            Frame::Unclaim(Unclaim {
                key: b"k".to_vec(),
                epoch: Epoch { ms: 1, node: 2 },
                through: 40,
            }),
            B::header(16, 0, 0).u32(1).raw(b"k").u64(1).u64(2).u64(40).0,
        ),
        (
            Frame::Coord(Coord { state: vec![0xee] }),
            B::header(17, 0, 0).u32(1).u8(0xee).0,
        ),
        (
            Frame::Complete(Complete {
                keys: vec![b"ab".to_vec(), b"c".to_vec()],
            }),
            B::header(18, 0, 0)
                .u32(2)
                .u32(2)
                .raw(b"ab")
                .u32(1)
                .raw(b"c")
                .0,
        ),
        (
            Frame::Prepare(Prepare {
                key: b"w".to_vec(),
                ballot: ballot(),
            }),
            B::header(20, 0, 0).u32(1).raw(b"w").u64(3).u64(9).0,
        ),
        (
            Frame::Promise(Promise {
                key: b"w".to_vec(),
                ballot: ballot(),
                accepted: Some(AcceptedValue {
                    ballot: Ballot { round: 2, node: 1 },
                    value: b"v".to_vec(),
                }),
                from: 4,
                reply_to: 6,
                tiebreak: false,
            }),
            B::header(21, 0, 0)
                .u32(1)
                .raw(b"w")
                .u64(3)
                .u64(9)
                .u8(1)
                .u64(2)
                .u64(1)
                .u32(1)
                .raw(b"v")
                .u64(4)
                .u64(6)
                .0,
        ),
        (
            Frame::Promise(Promise {
                key: b"w".to_vec(),
                ballot: ballot(),
                accepted: None,
                from: 4,
                reply_to: 6,
                tiebreak: true,
            }),
            B::header(21, 1, 0)
                .u32(1)
                .raw(b"w")
                .u64(3)
                .u64(9)
                .u8(0)
                .u64(4)
                .u64(6)
                .0,
        ),
        (
            Frame::Nack(Nack {
                key: b"w".to_vec(),
                promised: ballot(),
                from: 4,
                reply_to: 6,
                tiebreak: false,
            }),
            B::header(22, 0, 0)
                .u32(1)
                .raw(b"w")
                .u64(3)
                .u64(9)
                .u64(4)
                .u64(6)
                .0,
        ),
        (
            Frame::Propose(Propose {
                key: b"w".to_vec(),
                ballot: ballot(),
                value: b"xy".to_vec(),
            }),
            B::header(23, 0, 0)
                .u32(1)
                .raw(b"w")
                .u64(3)
                .u64(9)
                .u32(2)
                .raw(b"xy")
                .0,
        ),
        (
            Frame::Accepted(Accepted {
                key: b"w".to_vec(),
                ballot: ballot(),
                from: 4,
                reply_to: 6,
                tiebreak: false,
            }),
            B::header(24, 0, 0)
                .u32(1)
                .raw(b"w")
                .u64(3)
                .u64(9)
                .u64(4)
                .u64(6)
                .0,
        ),
        (
            Frame::Release(Release {
                key: b"w".to_vec(),
                ballot: ballot(),
                through: 8,
            }),
            B::header(25, 0, 0).u32(1).raw(b"w").u64(3).u64(9).u64(8).0,
        ),
        (
            Frame::BuildOffer(BuildArtifacts {
                build: build(),
                artifacts: vec![artifact()],
            }),
            B::header(30, 0, 0)
                .build(&build())
                .u32(1)
                .s("zend")
                .s("windows-x86_64")
                .u64(1 << 30)
                .raw(&[0xab; 32])
                .0,
        ),
        (
            Frame::BuildHave(BuildArtifacts {
                build: build(),
                artifacts: vec![],
            }),
            B::header(31, 0, 0).build(&build()).u32(0).0,
        ),
        (
            Frame::BuildFetch(BuildFetch {
                artifact: [0xab; 32],
                offset: 1 << 20,
                from_node: 4,
                reply_to: 5,
            }),
            B::header(32, 0, 0)
                .raw(&[0xab; 32])
                .u64(1 << 20)
                .u64(4)
                .u64(5)
                .0,
        ),
        (
            Frame::BuildChunk(BuildChunk {
                artifact: [0xab; 32],
                offset: 0,
                reply_to: 5,
                bytes: vec![7, 7],
            }),
            B::header(33, 0, 0)
                .raw(&[0xab; 32])
                .u64(0)
                .u64(5)
                .u32(2)
                .raw(&[7, 7])
                .0,
        ),
        (
            Frame::Verdict(Verdict {
                build: build(),
                result: VerdictResult::Fail,
                figures: vec![],
            }),
            B::header(34, 0, 0).build(&build()).u8(1).u32(0).0,
        ),
        (
            Frame::Pulse(Pulse {
                world: "w1".into(),
                kind: "move".into(),
                body: vec![1],
            }),
            B::header(40, 0, 0).s("w1").s("move").u32(1).u8(1).0,
        ),
        (
            Frame::Open(Open {
                stream: 7,
                target: "agent".into(),
            }),
            B::header(50, 0, 7).s("agent").0,
        ),
        (
            Frame::Data(Data {
                stream: 7,
                bytes: b"GET /".to_vec(),
            }),
            B::header(51, 0, 7).raw(b"GET /").0,
        ),
        (
            Frame::Window(Window {
                stream: 7,
                credit: 65536,
            }),
            B::header(52, 0, 7).u32(65536).0,
        ),
        (Frame::Fin { stream: 7 }, B::header(53, 0, 7).0),
        (Frame::Reset { stream: 9 }, B::header(54, 0, 9).0),
        (
            Frame::Event {
                catchup: false,
                event: Event {
                    origin: 0x10,
                    oseq: 3,
                    total: 2,
                    offset: 0,
                    reply_to: None,
                    bytes: vec![0xaa, 0xbb],
                },
            },
            B::header(60, 0, 0)
                .u64(0x10)
                .u64(3)
                .u32(2)
                .u32(0)
                .raw(&[0xaa, 0xbb])
                .0,
        ),
        (
            Frame::Event {
                catchup: true,
                event: Event {
                    origin: 0x10,
                    oseq: 3,
                    total: 5,
                    offset: 3,
                    reply_to: Some(0x44),
                    bytes: vec![0xaa, 0xbb],
                },
            },
            B::header(60, 3, 0)
                .u64(0x10)
                .u64(3)
                .u32(5)
                .u32(3)
                .u64(0x44)
                .raw(&[0xaa, 0xbb])
                .0,
        ),
        (
            Frame::Receipt(Receipt { through: 1000 }),
            B::header(61, 0, 0).u64(1000).0,
        ),
        (
            Frame::Ack(Ack {
                applied: vec![(1, 5), (2, 9)],
            }),
            B::header(62, 0, 0).u32(2).u64(1).u64(5).u64(2).u64(9).0,
        ),
        (
            Frame::Catchup(Catchup {
                request: 1,
                reply_to: 2,
                to: Some(3),
                since: since.clone(),
            }),
            B::header(63, 0, 0)
                .u64(1)
                .u64(2)
                .u8(1)
                .u64(3)
                .u16(1)
                .u64(5)
                .u8(3)
                .u8(0)
                .0,
        ),
        (
            Frame::Headers(Headers {
                request: 1,
                reply_to: 2,
                answerer: 9,
                origin: 5,
                from: 4,
                to: 6,
                gone: vec![Gone {
                    oseq: 5,
                    deleted_entry: Some(vec![1]),
                }],
                records: vec![header],
            }),
            B::header(64, 0, 0)
                .u64(1)
                .u64(2)
                .u64(9)
                .u64(5)
                .u64(4)
                .u64(6)
                .u32(1)
                .u64(5)
                .u8(1)
                .u32(1)
                .u8(1)
                .u32(1)
                .u64(4)
                .u8(12)
                .u64(0x77)
                .u64(2)
                .u32(300)
                .raw(&[0xcd; 16])
                .0,
        ),
        (
            Frame::Fetch(Fetch {
                request: 1,
                reply_to: 2,
                from_node: 3,
                origin: 5,
                oseqs: vec![4, 6],
            }),
            B::header(65, 0, 0)
                .u64(1)
                .u64(2)
                .u64(3)
                .u64(5)
                .u32(2)
                .u64(4)
                .u64(6)
                .0,
        ),
        (
            Frame::Done(Done {
                request: 1,
                reply_to: 2,
            }),
            B::header(66, 0, 0).u64(1).u64(2).0,
        ),
        (
            Frame::Have(Have {
                request: 1,
                reply_to: 2,
                applied: since,
            }),
            B::header(67, 0, 0)
                .u64(1)
                .u64(2)
                .u16(1)
                .u64(5)
                .u8(3)
                .u8(0)
                .0,
        ),
        (
            Frame::Lost(Lost {
                request: 1,
                reply_to: 2,
            }),
            B::header(68, 0, 0).u64(1).u64(2).0,
        ),
        (
            Frame::PeerHello(PeerHello {
                node: 0x51,
                build: build(),
                session_key: [0x11; 32],
                oauth_client: [0x22; 32],
            }),
            B::header(70, 0, 0)
                .u64(0x51)
                .build(&build())
                .raw(&[0x11; 32])
                .raw(&[0x22; 32])
                .0,
        ),
        (
            Frame::AcmeChallenge(AcmeChallenge {
                site: "tokera".into(),
                token: "tok".into(),
                answer: "tok.thumb".into(),
                expires_ms: 1_700_000_600_000,
            }),
            B::header(71, 0, 0)
                .s("tokera")
                .s("tok")
                .s("tok.thumb")
                .u64(1_700_000_600_000)
                .0,
        ),
        (
            Frame::AcmeAck {
                token: "tok".into(),
            },
            B::header(72, 0, 0).s("tok").0,
        ),
        (
            Frame::AcmeClear {
                token: "tok".into(),
            },
            B::header(73, 0, 0).s("tok").0,
        ),
        (
            Frame::AcmeCert(AcmeCert {
                site: "tokera".into(),
                chain_pem: "CHAIN".into(),
                key_pem: "KEY".into(),
                not_after: 1_800_000_000,
            }),
            B::header(74, 0, 0)
                .s("tokera")
                .s("CHAIN")
                .s("KEY")
                .u64(1_800_000_000)
                .0,
        ),
        (
            Frame::AcmeLedger {
                ledger: b"{}".to_vec(),
            },
            B::header(75, 0, 0).u32(2).raw(b"{}").0,
        ),
        (
            Frame::AcmeInventory {
                certs: vec![CertLine {
                    site: "tokera".into(),
                    not_after: 1_800_000_000,
                    digest: [0x33; 32],
                }],
            },
            B::header(76, 0, 0)
                .u32(1)
                .s("tokera")
                .u64(1_800_000_000)
                .raw(&[0x33; 32])
                .0,
        ),
        (
            Frame::AcmeOrdering {
                site: "tokera".into(),
            },
            B::header(77, 0, 0).s("tokera").0,
        ),
        (
            Frame::AcmeDone {
                site: "tokera".into(),
            },
            B::header(78, 0, 0).s("tokera").0,
        ),
        (
            Frame::AcmeAccount {
                account: b"{\"id\":1}".to_vec(),
            },
            B::header(79, 0, 0).u32(8).raw(b"{\"id\":1}").0,
        ),
    ]
}

#[test]
fn every_kind_encodes_to_exact_bytes_and_decodes_back() {
    let samples = samples();
    for kind in Kind::ALL {
        assert!(
            samples.iter().any(|(f, _)| f.kind() == *kind),
            "no sample for {}",
            name(*kind)
        );
    }
    for (frame, bytes) in &samples {
        assert_eq!(&frame.encode(), bytes, "{}", name(frame.kind()));
        assert_eq!(
            &Frame::decode(bytes).unwrap(),
            frame,
            "{}",
            name(frame.kind())
        );
    }
}

#[test]
fn every_truncation_is_refused() {
    for (frame, bytes) in samples() {
        for cut in 0..bytes.len() {
            // A DATA or EVENT body is "the rest", so a shorter one is a
            // different, valid frame unless the cut reaches its fixed part;
            // every other kind must refuse every strict prefix.
            let rest_is_body = matches!(frame.kind(), Kind::Data | Kind::Event);
            match Frame::decode(&bytes[..cut]) {
                Err(_) => {}
                Ok(f) => assert!(
                    rest_is_body && f != frame,
                    "{} cut at {cut} decoded",
                    name(frame.kind())
                ),
            }
        }
    }
}

#[test]
fn an_event_cut_inside_its_record_is_refused_by_its_total() {
    let (_, bytes) = samples()
        .into_iter()
        .find(|(f, _)| f.kind() == Kind::Event)
        .unwrap();
    // The record says it is two bytes; one byte is a fragment that ends
    // short of nothing, but a fragment can never run past the total.
    let mut long = bytes.clone();
    long.push(0xcc);
    assert!(Frame::decode(&long).is_err());
}

#[test]
fn a_trailing_byte_is_refused() {
    for (frame, mut bytes) in samples() {
        if matches!(frame.kind(), Kind::Data | Kind::Event) {
            continue;
        }
        bytes.push(0);
        assert!(Frame::decode(&bytes).is_err(), "{}", name(frame.kind()));
    }
}

#[test]
fn an_unknown_kind_is_refused() {
    let known: Vec<u8> = Kind::ALL.iter().map(|k| k.tag()).collect();
    for tag in 0..=255u8 {
        if !known.contains(&tag) {
            assert_eq!(
                Frame::decode(&[tag, 0, 0, 0, 0, 0]),
                Err(WireError::UnknownKind(tag))
            );
        }
    }
}

#[test]
fn a_stream_or_flag_where_the_kind_takes_none_is_refused() {
    // A RECEIPT with a stream id.
    assert!(Frame::decode(&B::header(61, 0, 4).u64(1).0).is_err());
    // A FIN without one.
    assert!(Frame::decode(&B::header(53, 0, 0).0).is_err());
    // A flag on a kind other than EVENT.
    assert!(Frame::decode(&B::header(61, 1, 0).u64(1).0).is_err());
    // An unknown flag on an EVENT.
    let bad = B::header(60, 4, 0).u64(1).u64(1).u32(1).u32(0).u8(0).0;
    assert!(Frame::decode(&bad).is_err());
}

#[test]
fn oversized_data_and_fragments_are_refused() {
    let data = B::header(51, 0, 1).raw(&vec![0; MAX_DATA + 1]).0;
    assert!(Frame::decode(&data).is_err());
    let fragment = B::header(60, 0, 0)
        .u64(1)
        .u64(1)
        .u32(MAX_FRAGMENT as u32 + 1)
        .u32(0)
        .raw(&vec![0; MAX_FRAGMENT + 1])
        .0;
    assert!(Frame::decode(&fragment).is_err());
    let empty_data = B::header(51, 0, 1).0;
    assert!(Frame::decode(&empty_data).is_err());
}

#[test]
fn lanes_follow_the_table() {
    let lane = |kind: Kind| {
        samples()
            .into_iter()
            .filter(|(f, _)| f.kind() == kind)
            .map(|(f, _)| f.lane())
            .collect::<Vec<_>>()
    };
    assert_eq!(lane(Kind::Hello), vec![Lane::Control]);
    assert_eq!(lane(Kind::Window), vec![Lane::Control]);
    assert_eq!(lane(Kind::Data), vec![Lane::Ordered]);
    assert_eq!(lane(Kind::Pulse), vec![Lane::Ordered]);
    assert_eq!(lane(Kind::Event), vec![Lane::Ordered, Lane::CatchUp]);
    assert_eq!(lane(Kind::Headers), vec![Lane::CatchUp]);
    assert_eq!(lane(Kind::BuildChunk), vec![Lane::CatchUp]);
    assert_eq!(lane(Kind::Fetch), vec![Lane::Control]);
    assert_eq!(lane(Kind::AcmeCert), vec![Lane::Control]);
    // A loss is news to act on: never queued behind the catch-up it reports.
    assert_eq!(lane(Kind::Lost), vec![Lane::Control]);
}
