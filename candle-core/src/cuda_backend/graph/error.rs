//! Why a capture was refused.

/// A graph the driver layer refused to instantiate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// The capture recorded a different number of nodes than the caller
    /// launched. A launcher that skipped its launch, or one that ran on the
    /// default stream instead of the capture stream (where it executes at once
    /// and is not recorded), leaves the graph short — and replaying it would
    /// drop that work every wave without a word.
    NodeCount { expected: usize, recorded: usize },
    /// The capture ended without a graph: the driver handed back none for a
    /// segment that was recording.
    NoGraph,
    /// The capture recorded a node the audit does not admit — anything but a
    /// kernel, a memset, a device-to-device copy or an event record, such as
    /// a host callback or a graph allocation issued inside the region.
    /// `kind` is the driver's `CUgraphNodeType` value.
    NodeType {
        index: usize,
        kind: u32,
        /// Where the refused graph was written, every node named.
        graph: String,
    },
    /// The capture recorded a copy that touches host memory: a launcher
    /// issued an upload or a readback on the launch stream instead of through
    /// the device, which runs it eagerly in order.
    Copy {
        index: usize,
        from: String,
        to: String,
        bytes: usize,
        /// Where to look for the site: the kernel recorded last before it, or,
        /// when none was, where the graph was written.
        after: String,
    },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeCount { expected, recorded } => write!(
                f,
                "graph capture recorded {recorded} nodes where {expected} launches were issued"
            ),
            Self::NoGraph => write!(f, "graph capture ended without a graph"),
            Self::NodeType { index, kind, graph } => write!(
                f,
                "graph capture recorded node {index} of type {kind}; only kernels, memsets, \
                 device-to-device copies and event records are recorded {graph}"
            ),
            Self::Copy {
                index,
                from,
                to,
                bytes,
                after,
            } => write!(
                f,
                "graph capture recorded node {index}, a {bytes}-byte copy {from} -> {to} \
                 {after}; only device-to-device copies are recorded"
            ),
        }
    }
}

impl std::error::Error for GraphError {}
