use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

const RING: usize = 500;

/// Shared broadcast bus for log lines.
pub struct LogBus {
    tx: broadcast::Sender<String>,
    ring: Mutex<VecDeque<String>>,
}

impl LogBus {
    pub fn new() -> Arc<Self> {
        let (tx, _) = broadcast::channel(2048);
        Arc::new(Self {
            tx,
            ring: Mutex::new(VecDeque::with_capacity(RING)),
        })
    }

    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.tx.subscribe()
    }

    /// Snapshot of the most recent lines, oldest first.
    pub fn recent(&self) -> Vec<String> {
        self.ring.lock().unwrap().iter().cloned().collect()
    }

    fn push(&self, line: String) {
        {
            let mut r = self.ring.lock().unwrap();
            if r.len() >= RING {
                r.pop_front();
            }
            r.push_back(line.clone());
        }
        let _ = self.tx.send(line);
    }
}

/// Plugs into `tracing_subscriber::fmt::layer().with_writer(...)`.
#[derive(Clone)]
pub struct BusWriter(pub Arc<LogBus>);

impl Write for BusWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Ok(s) = std::str::from_utf8(buf) {
            let s = s.trim_end();
            if !s.is_empty() {
                self.0.push(s.to_owned());
            }
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for BusWriter {
    type Writer = BusWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}
