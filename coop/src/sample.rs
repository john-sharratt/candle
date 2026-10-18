//! One channel of one scheduler wave window, as its log line reports it: the
//! average forward of that channel over the window, and the window's expert
//! hit rate.

/// Which kind of forward a sample averages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channel {
    /// Prompt tokens: many per row.
    Prefill,
    /// One generated token per row.
    Decode,
}

/// The average forward of one channel over one wave window.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub channel: Channel,
    /// Forwards the window ran on this channel.
    pub fwds: u32,
    /// Rows per forward.
    pub rows: f64,
    /// Tokens per forward: the rows of a decode, the prompt tokens of a
    /// prefill.
    pub tokens: f64,
    /// Attended KV per forward, summed over its rows.
    pub kv: f64,
    /// Milliseconds per forward.
    pub ms: f64,
    /// The window's expert hit rate, 0..=1; `None` when the line carried none.
    pub hit: Option<f64>,
    /// Whether this channel had the window to itself. A decode sharing its
    /// window with prefill waits behind those forwards' expert traffic, which
    /// its own counts do not show.
    pub alone: bool,
}
