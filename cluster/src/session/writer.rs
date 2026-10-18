//! The session's writer: drains commands into the lanes and writes the next
//! frame the lanes give, counting every frame it sends (§4.3), taking each off
//! its lane's gauge, and sending a `PING` every 15 s (§3.2).

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::time::{interval_at, Instant, MissedTickBehavior};

use super::command::Command;
use super::gauge::Gauges;
use super::lanes::{Lanes, CATCHUP_LOW_WATER};
use crate::close::Close;
use crate::delivery::FrameCounter;
use crate::frame::Frame;
use crate::pipe::{Message, PipeTx};

pub const PING_EVERY: Duration = Duration::from_secs(15);

/// A hand-off to the send pump this slow is logged.
const SLOW_HANDOFF: Duration = Duration::from_secs(5);

enum Outcome {
    Queued,
    Close(Close),
    Stop,
}

fn apply(lanes: &mut Lanes, command: Command) -> Outcome {
    match command {
        Command::Control(f) => lanes.control(f),
        Command::Ordered(f) => lanes.ordered(f),
        Command::CatchUp(f) => lanes.catchup(f),
        Command::Event(e) => lanes.event(e),
        Command::Close(close) => return Outcome::Close(close),
        Command::Stop => return Outcome::Stop,
    }
    Outcome::Queued
}

/// Act on `command`; `false` when the writer is done.
async fn take<T: PipeTx>(tx: &mut T, lanes: &mut Lanes, command: Command) -> bool {
    match apply(lanes, command) {
        Outcome::Queued => true,
        Outcome::Close(close) => {
            let _ = tx.send(Message::Close(close)).await;
            false
        }
        Outcome::Stop => false,
    }
}

pub async fn run<T: PipeTx>(
    mut tx: T,
    mut commands: UnboundedReceiver<Command>,
    gauges: Arc<Gauges>,
) {
    let mut lanes = Lanes::new();
    let mut counter = FrameCounter::default();
    // Catch-up frames wait for the socket to drain under its low-water mark,
    // and are woken by the fall itself (`pipe::drain`).
    let drain = tx.drain().clone();
    let mut ping = interval_at(Instant::now() + PING_EVERY, PING_EVERY);
    ping.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        loop {
            match commands.try_recv() {
                Ok(command) => {
                    if !take(&mut tx, &mut lanes, command).await {
                        return;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if lanes.is_empty() {
                        return;
                    }
                    break;
                }
            }
        }
        if let Some(next) = lanes.next(tx.unsent_bytes()) {
            let bytes = next.frame.encode();
            let gauged = match &next.frame {
                Frame::Event { event, .. } => event.bytes.len(),
                _ => bytes.len(),
            };
            let (kind, len) = (crate::frame::name(next.frame.kind()), bytes.len());
            let handing = std::time::Instant::now();
            if tx.send(Message::Frame(bytes)).await.is_err() {
                return;
            }
            // Every frame behind this one — a PING included — waited on it: a
            // hand-off this slow is a peer about to see "no frame for 45 s".
            if handing.elapsed() >= SLOW_HANDOFF {
                tracing::warn!(
                    frame = kind,
                    bytes = len,
                    unsent = tx.unsent_bytes(),
                    waited_ms = handing.elapsed().as_millis() as u64,
                    "a frame waited on the socket's send pump"
                );
            }
            gauges.written(next.lane, gauged);
            let number = counter.count_one();
            if let Some(done) = next.done {
                let _ = done.send(number);
            }
            continue;
        }
        tokio::select! {
            command = commands.recv() => match command {
                None => return,
                Some(command) => {
                    if !take(&mut tx, &mut lanes, command).await {
                        return;
                    }
                }
            },
            _ = ping.tick() => lanes.control(Frame::Ping(rand::random())),
            _ = drain.below(CATCHUP_LOW_WATER), if lanes.catchup_waiting() => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokio::sync::mpsc::unbounded_channel;

    use crate::frame::Data;
    use crate::pipe::memory::pair;
    use crate::pipe::PipeRx;
    use crate::session::gauge::Lane;

    /// **Catch-up goes at the reader's pace, not a timer's.** 64 MiB of
    /// catch-up frames, each the size of the low-water mark, cross a pipe that
    /// holds two of them: each waits only for the reader to take the one ahead.
    /// Woken by a 1 ms poll instead, each waited a timer tick — ~15.6 ms on
    /// Windows, over a second for these.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn catchup_frames_go_as_fast_as_the_reader_takes_them() {
        const FRAMES: usize = 64;
        let (a, mut b) = pair(2 * CATCHUP_LOW_WATER + 64);
        let (commands, rx) = unbounded_channel();
        let gauges = Arc::new(Gauges::default());
        for _ in 0..FRAMES {
            gauges.queued(Lane::CatchUp, CATCHUP_LOW_WATER);
            commands
                .send(Command::CatchUp(Frame::Data(Data {
                    stream: 1,
                    bytes: vec![0; CATCHUP_LOW_WATER],
                })))
                .unwrap();
        }
        let started = std::time::Instant::now();
        let writer = tokio::spawn(run(a.tx, rx, gauges));
        for _ in 0..FRAMES {
            b.rx.recv().await.expect("a frame");
        }
        let took = started.elapsed();
        assert!(
            took < Duration::from_millis(400),
            "{FRAMES} MiB of catch-up took {took:?}"
        );
        drop(commands);
        writer.await.unwrap();
    }
}
