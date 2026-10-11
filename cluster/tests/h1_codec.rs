//! H1 (§14.2): encode and decode 2 M mixed frames — byte-exact round trips at
//! no less than the frames-per-second floor. Budget 1 s.

use std::time::{Duration, Instant};

use applied_set::AppliedSet;
use cluster::frame::{
    Ack, Ballot, Capacity, Catchup, Data, Event, Frame, MemberState, Prepare, Receipt, Window,
};

const FRAMES: usize = 2_000_000;
const BUDGET: Duration = Duration::from_secs(1);
/// Frames per second, encode plus decode: a third of the first measurement on
/// the RTX PRO 5000 workstation (6.66 M/s, 2026-10-11).
const FLOOR: f64 = 2_200_000.0;

fn mix() -> Vec<Frame> {
    let mut since = AppliedSet::default();
    since.insert_range(7, 1, 1000);
    since.insert(9, 4);
    vec![
        Frame::Data(Data {
            stream: 5,
            bytes: vec![1; 1400],
        }),
        Frame::Event {
            catchup: false,
            event: Event {
                origin: 7,
                oseq: 1001,
                total: 900,
                offset: 0,
                reply_to: None,
                bytes: vec![2; 900],
            },
        },
        Frame::Window(Window {
            stream: 5,
            credit: 65536,
        }),
        Frame::Receipt(Receipt { through: 99 }),
        Frame::Ack(Ack {
            applied: vec![(7, 1000), (9, 3)],
        }),
        Frame::Capacity(Capacity {
            state: MemberState::Ready,
            power: 170,
            idle: true,
            slots_free: 4,
            decodes: 2,
            prefill_backlog: 0,
            vram_free_mib: 9000,
            cores: 16,
            load_milli: 800,
        }),
        Frame::Prepare(Prepare {
            key: b"world/w1".to_vec(),
            ballot: Ballot { round: 4, node: 2 },
        }),
        Frame::Catchup(Catchup {
            request: 3,
            reply_to: 2,
            to: None,
            since,
        }),
        Frame::Ping([5; 8]),
    ]
}

#[test]
fn h1_codec() {
    let mix = mix();
    let encoded: Vec<Vec<u8>> = mix.iter().map(Frame::encode).collect();
    let started = Instant::now();
    let mut done = 0;
    while done < FRAMES && started.elapsed() < BUDGET {
        for (frame, want) in mix.iter().zip(&encoded) {
            let bytes = frame.encode();
            assert_eq!(&bytes, want);
            assert_eq!(&Frame::decode(&bytes).unwrap(), frame);
        }
        done += mix.len();
    }
    let rate = done as f64 / started.elapsed().as_secs_f64();
    println!("H1: {done} frames round-tripped at {:.0} frames/s", rate);
    assert!(
        rate >= FLOOR,
        "{rate:.0} frames/s is under the {FLOOR:.0} floor"
    );
}
