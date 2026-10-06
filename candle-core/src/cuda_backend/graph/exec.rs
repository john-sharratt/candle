//! An instantiated graph.

use super::{ComputeStream, GraphError};
use crate::cuda_backend::WrapErr;
use crate::Result;
use cudarc::driver::{result, sys, CudaContext};
use std::ffi::{c_char, CStr, CString};
use std::mem::MaybeUninit;
use std::sync::Arc;

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
    ctx: Arc<CudaContext>,
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
pub(super) fn audit(graph: sys::CUgraph, expected: usize) -> Result<u64> {
    let mut shape = ShapeFold::default();
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
                shape.word(u64::from(p.gridDimX) | (u64::from(p.gridDimY) << 32));
                shape.word(u64::from(p.gridDimZ) | (u64::from(p.blockDimX) << 32));
                shape.word(u64::from(p.blockDimY) | (u64::from(p.blockDimZ) << 32));
                shape.word(u64::from(p.sharedMemBytes));
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
    }
    Ok(shape.finish())
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
    let mut name: *const c_char = std::ptr::null();
    // SAFETY: the node's own function; the name is a driver-owned C string.
    unsafe { sys::cuFuncGetName(&mut name, p.func) }
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
        let shape = audit(template.0, expected_nodes)?;
        Self::instantiate_audited(template, on, expected_nodes, shape)
    }

    /// Instantiate and upload a template [`audit`] has already passed as
    /// `shape`.
    pub(super) fn instantiate_audited(
        template: Template,
        on: &ComputeStream,
        expected_nodes: usize,
        shape: u64,
    ) -> Result<Self> {
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        // SAFETY: `graph` is valid; flags 0 is the default instantiation.
        unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, template.0, 0).result() }.w()?;
        drop(template);
        on.bind()?;
        let me = Self {
            exec,
            nodes: expected_nodes,
            shape,
            ctx: on.context(),
        };
        // Uploading ahead of the first launch keeps the one-time setup off the
        // launch that the wave is waiting on.
        // SAFETY: a valid executable and the compute stream.
        unsafe { sys::cuGraphUpload(me.exec, on.cu()).result().w()? };
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
        let shape = audit(template.0, expected_nodes)?;
        if self.try_fold(&template, on, expected_nodes, shape)? {
            return Ok(true);
        }
        *self = Self::instantiate_audited(template, on, expected_nodes, shape)?;
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
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_ok() {
            // SAFETY: an executable this value owns, destroyed once.
            let _ = unsafe { result::graph::exec_destroy(self.exec) };
        }
    }
}
