//! What a Flash-Next pack stores for each dense weight it builds, and under
//! which names.
//!
//! The pack's GGUF part holds every int8 matmul weight as the KO twin the
//! engine runs — built once, at pack build, on the device
//! (`qwen4exp::pack_build`) — and the load copies each one to its destination
//! as stored. A tensor that is the twin of one checkpoint tensor keeps that
//! tensor's name; one built from several — a stack, a padded module, a fused
//! pair — takes a name of its own, and the checkpoint tensors it was built from
//! leave the pack.
//!
//! Both sides name a stored tensor through this module, so the build and the
//! load cannot disagree on where a weight is.

use super::config::Qwen4ExpConfig;
use crate::models::delta_net::LayerKind;

/// The trunk build's metadata key: the version of what `pack_build` writes.
/// A pack without it, or at another, is of the old family and is rebuilt.
pub const TRUNK_BUILD_KEY: &str = "zen.pack.trunk.version";

/// The version of the built trunk. Moves whenever what a stored tensor holds
/// changes at an unchanged name, shape and type — a new padding rule, a moved
/// fold — which no fingerprint of a repack pair would see.
pub const TRUNK_BUILD_VERSION: u32 = 1;

/// The draft head's own narrower LM head.
pub const DRAFT_LM_HEAD: &str = "output.draft.weight";

/// One hyper-connection module's two stored projections, under its prefix
/// (`blk.3.hc_attn`, `output_hc`, `blk.40.nextn.hc_head`).
pub fn hc_down(prefix: &str) -> String {
    format!("{prefix}_down.ko.weight")
}

/// See [`hc_down`].
pub fn hc_up(prefix: &str) -> String {
    format!("{prefix}_up.ko.weight")
}

/// The checkpoint tensors a hyper-connection module is built from; its norm
/// gain is read as stored, and is not among them.
pub fn hc_sources(prefix: &str, with_inject: bool) -> Vec<String> {
    let mut out = vec![
        format!("{prefix}_down.weight"),
        format!("{prefix}_up.weight"),
    ];
    if with_inject {
        out.push(format!("{prefix}_inject.weight"));
    }
    out
}

/// A DeltaNet layer's four input projections, stacked.
pub fn delta_net_stack(p: &str) -> String {
    format!("{p}.ssm_in.weight")
}

/// A full-attention layer's q|k|v, stacked.
pub fn attention_stack(p: &str) -> String {
    format!("{p}.attn_q_k_v.weight")
}

/// The parts of a block's stacked input projection, in stacking order.
pub fn stack_parts(p: &str, kind: LayerKind) -> Vec<String> {
    let parts: &[&str] = match kind {
        LayerKind::DeltaNet => &["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha"],
        LayerKind::Attention => &["attn_q", "attn_k", "attn_v"],
    };
    parts.iter().map(|s| format!("{p}.{s}.weight")).collect()
}

/// The shared expert's one-row gate, padded to a KO tile.
pub fn shared_gate(p: &str) -> String {
    format!("{p}.ffn_gate_inp_shexp.padded.weight")
}

/// The shared expert's gate and up projections, fused into one weight.
pub fn shared_gate_up(p: &str) -> String {
    format!("{p}.ffn_gate_up_shexp.weight")
}

/// The PLE block's key and value projections, stacked.
pub fn ple_key_value(p: &str) -> String {
    format!("{p}.ple_key_value.weight")
}

/// The draft head's `eh_proj`, cut into its embedding and hidden halves.
pub fn eh_proj_embedding(p: &str) -> String {
    format!("{p}.nextn.eh_proj.embedding.weight")
}

/// See [`eh_proj_embedding`].
pub fn eh_proj_hidden(p: &str) -> String {
    format!("{p}.nextn.eh_proj.hidden.weight")
}

/// The draft head's hyper-connection module's prefix.
pub fn head_mixer(p: &str) -> String {
    format!("{p}.nextn.hc_head")
}

/// The kind of block `li` is: a trunk layer's own, or full attention for the
/// draft head past them — which the head declares (`"mtp": {"layer_types":
/// ["full_attention"]}`).
pub fn block_kind(cfg: &Qwen4ExpConfig, li: usize) -> LayerKind {
    if li >= cfg.num_layers {
        LayerKind::Attention
    } else {
        cfg.layer_kinds[li]
    }
}

/// Every checkpoint tensor the built ones replace, so the pack does not also
/// store it.
pub fn replaced_sources(cfg: &Qwen4ExpConfig) -> Vec<String> {
    let mut out = hc_sources("output_hc", false);
    for li in 0..cfg.num_layers + cfg.num_mtp_layers {
        let p = format!("blk.{li}");
        out.extend(hc_sources(&format!("{p}.hc_attn"), true));
        out.extend(hc_sources(&format!("{p}.hc_ffn"), true));
        out.extend(stack_parts(&p, block_kind(cfg, li)));
        out.push(format!("{p}.ffn_gate_inp_shexp.weight"));
        out.push(format!("{p}.ffn_gate_shexp.weight"));
        out.push(format!("{p}.ffn_up_shexp.weight"));
        if li == cfg.ple.layer {
            out.push(format!("{p}.ple_key.weight"));
            out.push(format!("{p}.ple_value.weight"));
        }
    }
    if cfg.num_mtp_layers > 0 {
        let p = format!("blk.{}", cfg.num_layers);
        out.push(format!("{p}.nextn.eh_proj.weight"));
        out.extend(hc_sources(&head_mixer(&p), false));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stored name never collides with a checkpoint tensor it replaces, so a
    /// pack can never hold both under one name.
    #[test]
    fn stored_names_are_not_their_sources_names() {
        let p = "blk.7";
        let stored = [
            hc_down("blk.7.hc_attn"),
            hc_up("blk.7.hc_attn"),
            delta_net_stack(p),
            attention_stack(p),
            shared_gate(p),
            shared_gate_up(p),
            ple_key_value(p),
            eh_proj_embedding(p),
            eh_proj_hidden(p),
        ];
        let mut sources = hc_sources("blk.7.hc_attn", true);
        sources.extend(stack_parts(p, LayerKind::DeltaNet));
        sources.extend(stack_parts(p, LayerKind::Attention));
        for s in &stored {
            assert!(!sources.contains(s), "{s}");
        }
        assert_eq!(
            stored,
            [
                "blk.7.hc_attn_down.ko.weight",
                "blk.7.hc_attn_up.ko.weight",
                "blk.7.ssm_in.weight",
                "blk.7.attn_q_k_v.weight",
                "blk.7.ffn_gate_inp_shexp.padded.weight",
                "blk.7.ffn_gate_up_shexp.weight",
                "blk.7.ple_key_value.weight",
                "blk.7.nextn.eh_proj.embedding.weight",
                "blk.7.nextn.eh_proj.hidden.weight",
            ]
        );
    }

    /// The stacks take their parts in the order the block splits them.
    #[test]
    fn the_stacks_take_their_parts_in_order() {
        assert_eq!(
            stack_parts("blk.0", LayerKind::DeltaNet),
            [
                "blk.0.attn_qkv.weight",
                "blk.0.attn_gate.weight",
                "blk.0.ssm_beta.weight",
                "blk.0.ssm_alpha.weight"
            ]
        );
        assert_eq!(
            stack_parts("blk.3", LayerKind::Attention),
            [
                "blk.3.attn_q.weight",
                "blk.3.attn_k.weight",
                "blk.3.attn_v.weight"
            ]
        );
    }
}
