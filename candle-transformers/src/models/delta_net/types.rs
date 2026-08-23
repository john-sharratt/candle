//! The two value types every part of the DeltaNet subsystem speaks:
//! the hybrid layer schedule's kind, and one layer's DeltaNet geometry.

/// What a decoder layer is, per the hybrid schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet: linear attention over a per-session recurrent state.
    /// No KV cache; the layer's memory is the delta-rule matrix + conv tail.
    DeltaNet,
    /// Full gated attention (GQA, `head_dim` from the attention metadata) over
    /// the paged KV cache.
    Attention,
}

/// Gated DeltaNet geometry for one layer (uniform across a model's layers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaNetDims {
    /// Width of one QK head and of one V head (`ssm.state_size`; the two are
    /// equal in this lineage, asserted at load).
    pub head_dim: usize,
    /// Number of QK heads (`ssm.group_count`). Divides `n_v_heads`; Q/K are
    /// broadcast across the V heads they serve, GQA-style.
    pub n_k_heads: usize,
    /// Number of V heads (`ssm.time_step_rank`).
    pub n_v_heads: usize,
    /// Causal-conv kernel width over the fused QKV channels (`ssm.conv_kernel`).
    pub conv_kernel: usize,
}

impl DeltaNetDims {
    /// Total QK width: `head_dim × n_k_heads`.
    pub fn key_dim(&self) -> usize {
        self.head_dim * self.n_k_heads
    }

    /// Total V width: `head_dim × n_v_heads` (equals `ssm.inner_size`).
    pub fn value_dim(&self) -> usize {
        self.head_dim * self.n_v_heads
    }

    /// Channels of the fused causal conv: `2 × key_dim + value_dim`
    /// (Q and K first, V last — the split order inside the conv output).
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// Elements of recurrent matrix state per layer at f32:
    /// one `[head_dim × head_dim]` matrix per V head.
    pub fn state_elems(&self) -> usize {
        self.n_v_heads * self.head_dim * self.head_dim
    }

    /// Elements of conv tail state per layer: the last `conv_kernel − 1`
    /// inputs of every conv channel.
    pub fn conv_state_elems(&self) -> usize {
        (self.conv_kernel - 1) * self.conv_dim()
    }
}
