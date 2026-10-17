//! An instantiated graph.

use super::{budget, ComputeStream, GraphError};
use crate::cuda_backend::WrapErr;
use crate::Result;
use cudarc::driver::{result, sys, CudaContext};
use std::ffi::{c_char, CStr, CString};
use std::mem::MaybeUninit;
use std::sync::Arc;
use tracing::Level;

/// A recorded run of launches, ready to replay into a [`ComputeStream`].
///
/// Topology and every kernel argument are fixed at capture: a replay runs the
/// same kernels on the same addresses with the same scalars. A wave whose
/// scalars or addresses differ is recaptured and folded into the same
/// executable with [`Self::update`], which keeps the instantiation and upload
/// and rewrites only the parameters.
pub struct GraphExec {
    exec: sys::CUgraphExec,
    nodes: usize,
    /// The [shape](audit) it was last instantiated or folded at.
    shape: u64,
    /// Its nodes' kinds and kernels, which a fold never changes — for naming
    /// where a recapture that cannot be folded departs from it.
    topology: Vec<NodeKey>,
    ctx: Arc<CudaContext>,
    /// Device memory it is charged against the library's budget
    /// ([`budget::charge`]).
    bytes: u64,
}

// SAFETY: a `CUgraphExec` is a driver handle usable from any thread once the
// context is bound, which every method does first; `launch` takes `&self` and
// the driver serialises launches of one executable on the stream it targets.
unsafe impl Send for GraphExec {}

/// Destroys a captured graph template when it goes out of scope.
pub(super) struct Template(pub(super) sys::CUgraph);

impl Drop for Template {
    fn drop(&mut self) {
        // SAFETY: a graph handed over by its capture, destroyed once.
        let _ = unsafe { result::graph::destroy(self.0) };
    }
}

/// Check `graph` holds exactly `expected` nodes, every one of them a kernel, a
/// memset, a device-to-device copy or an event record, and answer its shape.
///
/// A host copy, a host callback or a graph allocation recorded inside a region
/// is a launcher that uploaded, read back or allocated during the capture;
/// replaying it would repeat that side effect against stale host state.
///
/// **The shape** is everything about the graph but its arguments: each node's
/// kind in order, and for a kernel its function, grid, block and shared memory,
/// for a memset or copy its extent. Two captures of one shape differ only in
/// addresses and scalars, which is the cheap fold — `cuGraphExecUpdate` then
/// rewrites arguments and nothing else. Folding a capture into an executable
/// last shaped differently also succeeds, at several times the cost: measured
/// on Flash-Next's adaptive-depth verify, 200–500 µs a segment against 2–60.
pub(super) fn audit(graph: sys::CUgraph, expected: usize) -> Result<Audited> {
    let mut shape = ShapeFold::default();
    let mut key = ShapeFold::default();
    let mut topology = Vec::with_capacity(expected);
    let mut dims = Vec::with_capacity(expected);
    let mut recorded = 0usize;
    // SAFETY: a null node array asks only for the count.
    unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut recorded).result() }.w()?;
    if recorded != expected {
        return Err(crate::Error::wrap(GraphError::NodeCount {
            expected,
            recorded,
        }));
    }
    let mut nodes = vec![std::ptr::null_mut(); recorded];
    let mut filled = recorded;
    // SAFETY: `nodes` holds `filled` slots.
    unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut filled).result() }.w()?;
    nodes.truncate(filled);
    let all = &nodes;
    for (index, &node) in all.iter().enumerate() {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
        // SAFETY: a node of `graph`.
        unsafe { sys::cuGraphNodeGetType(node, &mut kind).result() }.w()?;
        shape.word(kind as u64);
        let mut node_key = NodeKey {
            kind: kind as u32,
            func: 0,
        };
        match kind {
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => {
                let mut p = MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::uninit();
                // SAFETY: a kernel node of `graph`; the driver fills the parameters.
                let p = unsafe {
                    sys::cuGraphKernelNodeGetParams_v2(node, p.as_mut_ptr())
                        .result()
                        .w()?;
                    p.assume_init()
                };
                shape.word(p.func as u64);
                node_key.func = p.func as u64;
                shape.word(u64::from(p.gridDimX) | (u64::from(p.gridDimY) << 32));
                shape.word(u64::from(p.gridDimZ) | (u64::from(p.blockDimX) << 32));
                shape.word(u64::from(p.blockDimY) | (u64::from(p.blockDimZ) << 32));
                shape.word(u64::from(p.sharedMemBytes));
                dims.push(NodeDims {
                    func: node_key.func,
                    words: [
                        p.gridDimX,
                        p.gridDimY,
                        p.gridDimZ,
                        p.blockDimX,
                        p.blockDimY,
                        p.blockDimZ,
                        p.sharedMemBytes,
                    ],
                });
            }
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => {
                let mut p = MaybeUninit::<sys::CUDA_MEMSET_NODE_PARAMS>::uninit();
                // SAFETY: a memset node of `graph`; the driver fills the parameters.
                let p = unsafe {
                    sys::cuGraphMemsetNodeGetParams(node, p.as_mut_ptr())
                        .result()
                        .w()?;
                    p.assume_init()
                };
                shape.word(u64::from(p.elementSize));
                shape.word(p.width as u64);
                shape.word(p.height as u64);
                dims.push(NodeDims::extent(p.elementSize, p.width, p.height, 1));
            }
            // An event-record node is a profiling span's timestamp, recorded
            // on purpose as external so it fires on every replay.
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD => {}
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => {
                let mut p = MaybeUninit::<sys::CUDA_MEMCPY3D>::uninit();
                // SAFETY: a memcpy node of `graph`; the driver fills every field.
                let p = unsafe {
                    sys::cuGraphMemcpyNodeGetParams(node, p.as_mut_ptr())
                        .result()
                        .w()?;
                    p.assume_init()
                };
                // Device to device reads nothing the host owns: it replays in
                // order like a kernel.
                if p.srcMemoryType == sys::CUmemorytype::CU_MEMORYTYPE_DEVICE
                    && p.dstMemoryType == sys::CUmemorytype::CU_MEMORYTYPE_DEVICE
                {
                    shape.word(p.WidthInBytes as u64);
                    shape.word(p.Height as u64);
                    shape.word(p.Depth as u64);
                    dims.push(NodeDims::extent(1, p.WidthInBytes, p.Height, p.Depth));
                    key.word(u64::from(node_key.kind));
                    topology.push(node_key);
                    continue;
                }
                let after = all[..index]
                    .iter()
                    .rev()
                    .filter_map(|&n| kernel_name(n))
                    .next()
                    .map(|name| format!("after kernel `{name}`"))
                    .unwrap_or_else(|| format!("before any kernel {}", dump(graph)));
                return Err(crate::Error::wrap(GraphError::Copy {
                    index,
                    from: format!("{:?}", p.srcMemoryType),
                    to: format!("{:?}", p.dstMemoryType),
                    bytes: p.WidthInBytes * p.Height.max(1) * p.Depth.max(1),
                    after,
                }));
            }
            other => {
                return Err(crate::Error::wrap(GraphError::NodeType {
                    index,
                    kind: other as u32,
                    graph: dump(graph),
                }))
            }
        }
        key.word(u64::from(node_key.kind));
        topology.push(node_key);
    }
    Ok(Audited {
        shape: shape.finish(),
        key: key.finish(),
        topology,
        dims,
    })
}

/// What [`audit`] learns of a recapture.
pub(super) struct Audited {
    /// Everything but its arguments — see [`audit`].
    pub(super) shape: u64,
    /// Its topology folded to one word: each node's kind, in order — not its
    /// kernel, grid or arguments, which the driver rewrites on a fold
    /// (`tests::a_recapture_with_another_kernel_at_a_node_folds_in_place`).
    /// Recaptures sharing it can be folded into each other's executables, so
    /// it is what an executable is found by (`library`).
    pub(super) key: u64,
    /// The topology itself, for naming where a refused fold departs.
    pub(super) topology: Vec<NodeKey>,
    /// What the shape folds, node by node (event records carry none), for
    /// naming the launch that made a shape new ([`reshaped_at`]).
    pub(super) dims: Vec<NodeDims>,
}

/// What one node contributes to a graph's [shape](audit): for a kernel its
/// function, grid, block and shared memory; for a memset or a device copy its
/// extent, with no function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NodeDims {
    func: u64,
    /// A kernel's grid x, y, z, block x, y, z and shared bytes; an extent's
    /// element size, width, height and depth.
    words: [u32; 7],
}

impl NodeDims {
    fn extent(element: u32, width: usize, height: usize, depth: usize) -> Self {
        Self {
            func: 0,
            words: [element, width as u32, height as u32, depth as u32, 0, 0, 0],
        }
    }

    fn describe(&self) -> String {
        let w = self.words;
        if self.func == 0 {
            return format!("extent {}x{}x{}x{}", w[0], w[1], w[2], w[3]);
        }
        format!(
            "grid {}x{}x{} block {}x{}x{} smem {}",
            w[0], w[1], w[2], w[3], w[4], w[5], w[6]
        )
    }
}

/// Where a recapture's shape first departs from `kept`'s — two captures of
/// one topology — naming the launch and its dimensions in each, and how many
/// nodes differ in all. `None` when they agree node for node.
pub(super) fn reshaped_at(recaptured: &[NodeDims], kept: &[NodeDims]) -> Option<String> {
    let differ = |(a, b): (&NodeDims, &NodeDims)| a != b;
    let at = recaptured.iter().zip(kept).position(differ)?;
    let count = recaptured.iter().zip(kept).filter(|&p| differ(p)).count();
    let name = |d: &NodeDims| match d.func {
        0 => "copy or memset".to_string(),
        f => func_name(f as sys::CUfunction).unwrap_or_else(|| format!("kernel {f:#x}")),
    };
    let (a, b) = (&recaptured[at], &kept[at]);
    Some(format!(
        "{count} node(s), first {at}: recaptured {} {} / kept {} {}",
        name(a),
        a.describe(),
        name(b),
        b.describe()
    ))
}

/// One node of a graph's topology: its kind and, for a kernel, its function.
/// Two graphs whose keys agree node for node can be folded into each other;
/// where they first disagree is why one could not be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NodeKey {
    kind: u32,
    /// The kernel's `CUfunction`, or 0 for a node that is not a kernel.
    func: u64,
}

impl NodeKey {
    /// The node as a reader would name it: a kernel by its function's name,
    /// anything else by its kind.
    fn describe(&self) -> String {
        if self.func == 0 {
            return format!("node kind {}", self.kind);
        }
        func_name(self.func as sys::CUfunction)
            .unwrap_or_else(|| format!("kernel {:#x}", self.func))
    }
}

/// Whether the graph's debug log is on — what the explanations below
/// ([`divergence`], [`reshaped_at`]) are for. They name kernels through the
/// driver and format strings; with the log off, computing them is host time
/// spent on every segment that misses, for lines nobody reads.
pub(super) fn explaining() -> bool {
    tracing::enabled!(target: "candle_core::graph", Level::DEBUG)
}

/// Where a recapture's topology first departs from an executable's, for the
/// log line that explains an instantiation: the node index, and what each
/// graph holds there (`end` past the shorter one).
pub(super) fn divergence(recaptured: &[NodeKey], kept: &[NodeKey]) -> Option<String> {
    let at = recaptured
        .iter()
        .zip(kept)
        .position(|(a, b)| a != b)
        .or_else(|| (recaptured.len() != kept.len()).then(|| recaptured.len().min(kept.len())))?;
    let name = |keys: &[NodeKey]| {
        keys.get(at)
            .map_or_else(|| "end".to_string(), NodeKey::describe)
    };
    let before = at
        .checked_sub(1)
        .and_then(|i| recaptured.get(i))
        .map_or_else(|| "start".to_string(), NodeKey::describe);
    Some(format!(
        "node {at} (after {before}): recaptured {} / kept {}",
        name(recaptured),
        name(kept)
    ))
}

/// The shape of a graph folded one word at a time (FNV-1a over the words). A
/// collision costs only an expensive fold, never a wrong one: the driver
/// decides whether a fold is possible, the shape only which executable is
/// offered it first.
struct ShapeFold(u64);

impl Default for ShapeFold {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl ShapeFold {
    fn word(&mut self, w: u64) {
        for b in w.to_le_bytes() {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// The function a kernel node launches, for naming the neighbour of a node
/// the audit refuses.
fn kernel_name(node: sys::CUgraphNode) -> Option<String> {
    let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY;
    // SAFETY: a node of a live graph.
    unsafe { sys::cuGraphNodeGetType(node, &mut kind) }
        .result()
        .ok()?;
    if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
        return None;
    }
    let mut p = MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::uninit();
    // SAFETY: a kernel node; the driver fills the parameters.
    let p = unsafe {
        sys::cuGraphKernelNodeGetParams_v2(node, p.as_mut_ptr())
            .result()
            .ok()?;
        p.assume_init()
    };
    func_name(p.func)
}

/// A kernel function's name.
fn func_name(func: sys::CUfunction) -> Option<String> {
    let mut name: *const c_char = std::ptr::null();
    // SAFETY: a function of a loaded module; the name is a driver-owned C string.
    unsafe { sys::cuFuncGetName(&mut name, func) }
        .result()
        .ok()?;
    // SAFETY: a NUL-terminated string the driver keeps for the module's life.
    Some(
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// Write `graph` as Graphviz DOT, every node named, beside the other temporary
/// files; returns where, for an error to point at.
fn dump(graph: sys::CUgraph) -> String {
    let path = std::env::temp_dir().join(format!("candle-refused-{}.dot", std::process::id()));
    let Ok(c) = CString::new(path.to_string_lossy().as_bytes()) else {
        return String::new();
    };
    // SAFETY: a live graph and a NUL-terminated path; flags select the verbose
    // node description, which includes kernel names.
    let wrote = unsafe { sys::cuGraphDebugDotPrint(graph, c.as_ptr(), 1) };
    if wrote == sys::CUresult::CUDA_SUCCESS {
        format!("(names in {})", path.display())
    } else {
        String::new()
    }
}

impl GraphExec {
    /// Audit `graph` ([`audit`]), instantiate it and upload it on `on`. The
    /// template is destroyed on every path.
    ///
    /// # Safety
    ///
    /// `graph` must be a valid graph owned by the caller.
    #[cfg(test)]
    pub(super) unsafe fn instantiate(
        graph: sys::CUgraph,
        on: &ComputeStream,
        expected_nodes: usize,
    ) -> Result<Self> {
        let template = Template(graph);
        let a = audit(template.0, expected_nodes)?;
        Self::instantiate_audited(template, on, expected_nodes, a.shape, a.topology)
    }

    /// Instantiate and upload a template [`audit`] has already passed as
    /// `shape` and `topology`.
    pub(super) fn instantiate_audited(
        template: Template,
        on: &ComputeStream,
        expected_nodes: usize,
        shape: u64,
        topology: Vec<NodeKey>,
    ) -> Result<Self> {
        on.bind()?;
        // What the driver has free on either side of the instantiation and its
        // upload is what this executable is charged, when the reading is
        // believable (`budget::charge`).
        let free_before = result::mem_get_info().ok().map(|(free, _)| free as u64);
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        // SAFETY: `graph` is valid; flags 0 is the default instantiation.
        unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, template.0, 0).result() }.w()?;
        drop(template);
        on.bind()?;
        let mut me = Self {
            exec,
            nodes: expected_nodes,
            shape,
            topology,
            ctx: on.context(),
            bytes: 0,
        };
        // Uploading ahead of the first launch keeps the one-time setup off the
        // launch that the wave is waiting on.
        // SAFETY: a valid executable and the compute stream.
        unsafe { sys::cuGraphUpload(me.exec, on.cu()).result().w()? };
        let free_after = result::mem_get_info().ok().map(|(free, _)| free as u64);
        let measured = free_before
            .zip(free_after)
            .map(|(before, after)| before.saturating_sub(after));
        me.bytes = budget::charge(measured, expected_nodes);
        Ok(me)
    }

    /// Fold a recapture of the same region into this executable.
    ///
    /// Returns `true` when the driver rewrote the parameters in place. A
    /// recapture whose topology or kernels differ cannot be folded; it is
    /// instantiated afresh and replaces this one, and `false` is returned.
    ///
    /// # Safety
    ///
    /// `graph` must be a valid graph owned by the caller.
    #[cfg(test)]
    pub(super) unsafe fn update(
        &mut self,
        graph: sys::CUgraph,
        on: &ComputeStream,
        expected_nodes: usize,
    ) -> Result<bool> {
        let template = Template(graph);
        let a = audit(template.0, expected_nodes)?;
        if self.try_fold(&template, on, expected_nodes, a.shape)? {
            return Ok(true);
        }
        *self = Self::instantiate_audited(template, on, expected_nodes, a.shape, a.topology)?;
        Ok(false)
    }

    /// Fold an audited `template` of `shape` into this executable's parameters
    /// in place. `false` when the driver refuses it — a different topology or
    /// different kernels — and the executable is left as it was.
    pub(super) fn try_fold(
        &mut self,
        template: &Template,
        on: &ComputeStream,
        expected_nodes: usize,
        shape: u64,
    ) -> Result<bool> {
        if expected_nodes != self.nodes {
            return Ok(false);
        }
        on.bind()?;
        let mut info = sys::CUgraphExecUpdateResultInfo {
            result: sys::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS,
            errorNode: std::ptr::null_mut(),
            errorFromNode: std::ptr::null_mut(),
        };
        // SAFETY: a valid executable and a valid graph of the same context.
        let folded = unsafe { sys::cuGraphExecUpdate_v2(self.exec, template.0, &mut info) };
        let ok = folded == sys::CUresult::CUDA_SUCCESS
            && info.result == sys::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS;
        if ok {
            self.shape = shape;
        }
        Ok(ok)
    }

    /// The [shape](audit) this executable was last instantiated or folded at.
    pub(super) fn shape(&self) -> u64 {
        self.shape
    }

    /// Its nodes' kinds and kernels, in order.
    pub(super) fn topology(&self) -> &[NodeKey] {
        &self.topology
    }

    /// Replay into `on`.
    pub fn launch(&self, on: &ComputeStream) -> Result<()> {
        on.bind()?;
        // SAFETY: a valid executable on a stream of the context it was made in.
        unsafe { result::graph::launch(self.exec, on.cu()).w() }
    }

    /// Nodes the graph holds — one per recorded launch.
    pub fn node_count(&self) -> usize {
        self.nodes
    }

    /// Device memory it is charged.
    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_ok() {
            // SAFETY: an executable this value owns, destroyed once.
            let _ = unsafe { result::graph::exec_destroy(self.exec) };
        }
    }
}

#[cfg(test)]
mod divergence_tests {
    use super::*;

    fn node(kind: u32) -> NodeKey {
        NodeKey { kind, func: 0 }
    }

    /// Two topologies that agree node for node have nowhere to diverge.
    #[test]
    fn matching_topologies_do_not_diverge() {
        assert_eq!(divergence(&[node(0), node(2)], &[node(0), node(2)]), None);
    }

    /// The first node that differs is named, with the one before it.
    #[test]
    fn the_first_differing_node_is_named() {
        assert_eq!(
            divergence(&[node(0), node(0), node(2)], &[node(0), node(2), node(2)]).as_deref(),
            Some("node 1 (after node kind 0): recaptured node kind 0 / kept node kind 2")
        );
    }

    /// A recapture that runs on past the kept graph diverges where the kept
    /// one ends.
    #[test]
    fn a_longer_recapture_diverges_at_the_kept_end() {
        assert_eq!(
            divergence(&[node(0), node(0), node(2)], &[node(0), node(0)]).as_deref(),
            Some("node 2 (after node kind 0): recaptured node kind 2 / kept end")
        );
    }

    /// Two captures of one shape have nowhere to reshape.
    #[test]
    fn matching_shapes_do_not_reshape() {
        let a = [NodeDims::extent(4, 64, 1, 1)];
        assert_eq!(reshaped_at(&a, &a), None);
    }

    /// The first node whose dimensions differ is named, with both sides and
    /// how many nodes differ in all.
    #[test]
    fn the_first_reshaped_node_is_named() {
        let kept = [
            NodeDims::extent(4, 64, 1, 1),
            NodeDims::extent(1, 256, 1, 1),
            NodeDims::extent(1, 8, 1, 1),
        ];
        let recaptured = [
            NodeDims::extent(4, 64, 1, 1),
            NodeDims::extent(1, 512, 1, 1),
            NodeDims::extent(1, 16, 1, 1),
        ];
        assert_eq!(
            reshaped_at(&recaptured, &kept).as_deref(),
            Some(
                "2 node(s), first 1: recaptured copy or memset extent 1x512x1x1 \
                 / kept copy or memset extent 1x256x1x1"
            )
        );
    }
}
