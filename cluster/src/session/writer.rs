//! The session's writer: drains commands into the lanes and writes the next
//! frame the lanes give, counting every frame it sends (§4.3), taking each off
//! its lane's gauge, and sending a `PING` every 15 s (§3.2).

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::time::{interval_at, sleep, Instant, MissedTickBehavior};

use super::command::Command;
use super::gauge::Gauges;
use super::lanes::Lanes;
use crate::close::Close;
use crate::delivery::FrameCounter;
use crate::frame::Frame;
use crate::pipe::{Message, PipeTx};

pub const PING_EVERY: Duration = Duration::from_secs(15);

/// How often a writer with only catch-up frames waiting looks again at the
/// socket's unsent bytes.
const DRAIN_POLL: Duration = Duration::from_millis(1);

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
            if tx.send(Message::Frame(bytes)).await.is_err() {
                return;
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
            _ = sleep(DRAIN_POLL), if lanes.catchup_waiting() => {}
        }
    }
}
