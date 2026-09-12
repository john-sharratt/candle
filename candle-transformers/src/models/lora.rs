//! PEFT LoRA adapters over a quantized base.
//!
//! # What an adapter is, mechanically
//!
//! A pair of small matrices per targeted projection. Where the base computes
//! `y = Wx`, the adapted model computes:
//!
//! ```text
//! y = Wx + (alpha / r) · B(A x)
//! ```
//!
//! `A` is `[r, in]` and `B` is `[out, r]`, so the round trip through rank `r`
//! costs two thin GEMMs instead of one wide one. At `r = 64` against a 4096-wide
//! hidden that is about 3% of the base projection's work.
//!
//! # The scale is not always `alpha / r`
//!
//! An adapter trained with **rank-stabilised** LoRA divides by `√r` instead, and
//! `adapter_config.json`'s `use_rslora` is the only place that fact is recorded.
//! The difference is a factor of `√r` — eight, at rank 64 — applied to the whole
//! adapter contribution, and it is invisible to every check that does not know
//! to look: the shapes agree, the tensors load, the model runs, and the output
//! is merely wrong. [`Adapter::load`] reads the flag.
//!
//! # Why this works over a GGUF
//!
//! The base weight stays quantized and is never touched. `Wx` is computed by the
//! existing `QMatMul` exactly as before, and the adapter term is a separate
//! full-precision computation added to the result. Nothing is merged, so one
//! loaded base serves adapted and unadapted sessions at the same time — which is
//! what makes per-conversation opt-in possible at all.
//!
//! Merging would be the other design: fold `BA` into `W` once and pay nothing
//! per token. It is rejected here because it destroys exactly the property the
//! engine needs — a merged base can only ever be the adapted model, so serving
//! both would mean two copies of a 8.8 GB checkpoint resident.
//!
//! # Naming
//!
//! PEFT writes HuggingFace module paths
//! (`base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight`); the
//! engine reads GGUF names (`blk.7.attn_q.weight`). [`Target`] is the mapping,
//! and it is exhaustive on purpose — an unrecognised module is reported rather
//! than skipped, because a silently-dropped adapter tensor is an adapter that
//! half-applies and nothing says so.

use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use candle::{DType, Device, LiveTensor, Result, Tensor};

/// A poisoned adapter lock means a panic while a dtype was being swapped, which
/// leaves the resident weights in an unknown state. Reported rather than
/// recovered: continuing would run whichever half of the swap landed.
fn lock_poisoned<T>(_: std::sync::PoisonError<T>) -> candle::Error {
    candle::Error::Msg("LoRA adapter lock poisoned — a dtype swap panicked".into())
}

/// Which projection an adapter pair applies to.
///
/// The GGUF's own roles, not PEFT's. A hybrid checkpoint names the same weight
/// differently depending on whether its layer mixes tokens through attention or
/// through a recurrent state, and the adapter has to land on the tensor the
/// engine will actually multiply by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Target {
    /// `attn_q` — attention layers only.
    AttnQ,
    /// `attn_k` — attention layers only.
    AttnK,
    /// `attn_v` — attention layers only.
    AttnV,
    /// `attn_output`.
    AttnOut,
    /// `ffn_gate` — every layer.
    FfnGate,
    /// `ffn_up`.
    FfnUp,
    /// `ffn_down`.
    FfnDown,
}

impl Target {
    /// The GGUF tensor role this adapter modifies.
    pub fn gguf_role(self) -> &'static str {
        match self {
            Target::AttnQ => "attn_q",
            Target::AttnK => "attn_k",
            Target::AttnV => "attn_v",
            Target::AttnOut => "attn_output",
            Target::FfnGate => "ffn_gate",
            Target::FfnUp => "ffn_up",
            Target::FfnDown => "ffn_down",
        }
    }

    /// The PEFT module name this corresponds to.
    fn from_peft(module: &str) -> Option<Target> {
        Some(match module {
            "q_proj" => Target::AttnQ,
            "k_proj" => Target::AttnK,
            "v_proj" => Target::AttnV,
            "o_proj" => Target::AttnOut,
            "gate_proj" => Target::FfnGate,
            "up_proj" => Target::FfnUp,
            "down_proj" => Target::FfnDown,
            _ => return None,
        })
    }
}

/// One projection's adapter, stored **transposed and pre-scaled** for the
/// forward path: `a_t` is `[in, r]` and `b_t` is `[r, out] · scale`.
///
/// PEFT writes `A [r, in]` and `B [out, r]`, which is the orientation a matmul
/// wants transposed. Doing that per call costs a copy per projection per layer
/// per token — `t()` returns a non-contiguous view and the matmul materialises
/// it — so both are transposed once at load, where the cost is paid on 256 small
/// tensors instead of on every token forever (hot-path invariant 2).
///
/// The scale rides in `B` for the same reason. Applying it at use is a
/// full-tensor multiply over the `[.., out]` result; folding it into `B` is the
/// same arithmetic over an `[r, out]` weight, once. [`Adapter::scale`] still
/// reports the value, so it stays legible in one place.
///
/// # The source is on the host, and the device copies are keyed by dtype
///
/// A projection's operand widths belong to the **session and the layer**, not to
/// the load, and within one layer they are not even all the same. A matmul
/// refuses mismatched operands outright, so this cannot be settled once:
///
/// * The residual stream flows in the session's activation dtype.
/// * Q/K/V are projected in the *KV arena's* dtype, which a model that computes
///   wider than it stores does not share with the stream.
/// * **The SwiGLU intermediates are promoted**: an F16 session runs `gate`/`up`
///   in BF16, because MLP intermediates can exceed F16's range. So inside one
///   MLP, `gate`'s adapter reads F16 and writes BF16 while `down`'s reads BF16
///   and writes F16 — opposite directions, same layer.
///
/// So `A` is materialised in every dtype an *input* may arrive in and `B` in
/// every dtype an *output* may be wanted in, keyed by dtype, all of it decided
/// eagerly at session creation and never inside a wave (hot-path invariant 1).
/// In practice that is one dtype for a BF16 deployment and two for an F16 one.
///
/// The F32 source stays on the **host**. It is twice the width of anything the
/// device wants, and holding ~465 MB of it in VRAM to serve a conversion that
/// happens once per session is memory a card with weights on it does not have.
#[derive(Debug)]
pub struct Pair {
    /// `[in, r]`, host, F32 — PEFT's `A`, transposed.
    a_src: Tensor,
    /// `[r, out] · scale`, host, F32 — PEFT's `B`, transposed and pre-scaled.
    b_src: Tensor,
    /// Device copies, keyed by dtype. Both halves are materialised for every
    /// dtype in the set, because which half needs which is a property of the
    /// call site rather than of the pair.
    ///
    /// Behind a lock because the set is chosen when a session is created, which
    /// happens through `&self` — and because a model is shared across threads.
    /// Replaced wholesale by [`Pair::materialise`], never added to during a wave.
    dev: RwLock<HashMap<DType, (Tensor, Tensor)>>,
}

impl Pair {
    /// Build from PEFT's orientation, folding in the scale and materialising for
    /// `device` in `dtype`.
    ///
    /// The transpose happens once, here. PEFT writes `A [r, in]` and `B [out, r]`,
    /// which is the orientation a matmul wants transposed; doing that per call
    /// costs a copy per projection per layer per token, because `t()` returns a
    /// non-contiguous view and the matmul materialises it (hot-path invariant 2).
    pub fn from_peft_weights(
        a: &Tensor,
        b: &Tensor,
        scale: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Pair> {
        let pair = Pair {
            a_src: a.t()?.contiguous()?.to_device(&Device::Cpu)?,
            b_src: (b.t()?.contiguous()? * scale)?.to_device(&Device::Cpu)?,
            dev: RwLock::new(HashMap::new()),
        };
        pair.materialise(&[dtype], device)?;
        Ok(pair)
    }

    /// Hold device copies for exactly `dtypes`, and nothing else.
    ///
    /// Idempotent: a dtype already resident is left alone, so the ordinary
    /// session — one whose widths match the last — does no work and moves no
    /// bytes. Dtypes not named are dropped, so a session cannot leave VRAM held
    /// for a width nothing will ask for again.
    ///
    /// Always rebuilds from the F32 host master rather than converting a
    /// resident copy: `BF16 → F16 → BF16` would compound rounding on every
    /// session, and the master cannot.
    pub fn materialise(&self, dtypes: &[DType], device: &Device) -> Result<()> {
        let mut m = self.dev.write().map_err(lock_poisoned)?;
        m.retain(|d, (a, _)| dtypes.contains(d) && a.device().same_device(device));
        for &d in dtypes {
            if m.contains_key(&d) {
                continue;
            }
            let a = self.a_src.to_device(device)?.to_dtype(d)?;
            let b = self.b_src.to_device(device)?.to_dtype(d)?;
            m.insert(d, (a, b));
        }
        Ok(())
    }

    /// The dtypes currently resident on the device, sorted for a stable message.
    pub fn resident_dtypes(&self) -> Result<Vec<DType>> {
        let m = self.dev.read().map_err(lock_poisoned)?;
        let mut v: Vec<DType> = m.keys().copied().collect();
        v.sort_by_key(|d| format!("{d:?}"));
        Ok(v)
    }

    /// The rank this pair rounds through.
    pub fn rank(&self) -> Result<usize> {
        self.a_src.dim(1)
    }

    /// `y = base + scale · B(A x)`, for an `x` of shape `[.., in]`.
    ///
    /// The base output is passed in and added to rather than returned separately
    /// so the caller cannot forget the addition — an adapter whose contribution
    /// is computed and dropped is the failure this signature makes impossible.
    ///
    /// Generic over `'w` because the activations on the batched path are
    /// wave-allocated and `Tensor` is `LiveTensor<'static>`: the adapter's own
    /// weights are `'static` and the intermediates come off whichever arena `x`
    /// was carved from, so the rank round-trip is wave-backed like every other
    /// transient in the layer rather than a heap allocation per projection.
    pub fn apply<'w>(&self, base: &LiveTensor<'w>, x: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        // One uncontended read lock per projection. The alternative — handing
        // the forward a borrowed tensor — would mean the resident set could not
        // be replaced through `&self`, which is how session creation reaches it.
        let m = self.dev.read().map_err(lock_poisoned)?;
        let (in_ty, out_ty) = (x.dtype(), base.dtype());
        let (a, _) = m.get(&in_ty).ok_or_else(|| self.missing(in_ty, "input"))?;
        let (_, b) = m
            .get(&out_ty)
            .ok_or_else(|| self.missing(out_ty, "output"))?;

        // [.., in] × [in, r] → [.., r] × [r, out] → [.., out]. Every operand is
        // contiguous in the orientation its matmul reads and already in the
        // right dtype, so no weight is converted or copied here — that was all
        // paid once, at load, from the F32 master.
        let down = x.broadcast_matmul(a)?;
        // **The one conversion on this path, and it is unavoidable.** A matmul's
        // result takes its operands' dtype, so when a projection's input and
        // output widths differ — an F16 session's `gate`, which reads the F16
        // stream and writes the promoted BF16 SwiGLU intermediate — something
        // between the two matmuls has to change width. This is the narrowest
        // point available: the rank-`r` bottleneck, 64 wide against the 4096 of
        // the input and the 12288 of the output, so it is ~64–192× smaller than
        // converting either end.
        //
        // It does not run at all when the two agree, which is every projection
        // of a BF16 session — the width production serves in.
        let down = if down.dtype() == out_ty {
            down
        } else {
            down.to_dtype(out_ty)?
        };
        base + down.broadcast_matmul(b)?
    }

    fn missing(&self, dtype: DType, side: &str) -> candle::Error {
        candle::Error::Msg(format!(
            "LoRA adapter has no {dtype:?} copy for a projection's {side} — resident: \
             {:?}. Session creation materialises every width the model can ask for, so \
             this means a width reached the forward that `maybe_change_dtype` was not \
             told about.",
            self.resident_dtypes().unwrap_or_default()
        ))
    }
}

/// One layer's adapter pairs, resolved once by [`Adapter::layer`].
///
/// This is what the forward path carries: `Copy`, seven borrowed options, no
/// lookups. [`Default`] is the unadapted layer, so a model running without an
/// adapter and a layer the adapter does not cover take the same path — there is
/// no second code path for the unadapted case, only pairs that are `None`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LayerLora<'a> {
    pub q: Option<&'a Pair>,
    pub k: Option<&'a Pair>,
    pub v: Option<&'a Pair>,
    pub o: Option<&'a Pair>,
    pub gate: Option<&'a Pair>,
    pub up: Option<&'a Pair>,
    pub down: Option<&'a Pair>,
}

impl LayerLora<'_> {
    /// Whether this layer is adapted at all.
    pub fn is_empty(&self) -> bool {
        self.q.is_none()
            && self.k.is_none()
            && self.v.is_none()
            && self.o.is_none()
            && self.gate.is_none()
            && self.up.is_none()
            && self.down.is_none()
    }
}

/// `y = base + scale · B(A x)` when `pair` is present, `base` untouched when it
/// is not.
///
/// The one place the forward path calls: every adapted projection is
/// `lora::adapt(self.lora.q, &base, &x)?` regardless of whether an adapter is
/// loaded, so the unadapted model runs the same line and pays a null check.
pub fn adapt<'w>(
    pair: Option<&Pair>,
    base: LiveTensor<'w>,
    x: &LiveTensor<'w>,
) -> Result<LiveTensor<'w>> {
    match pair {
        Some(p) => p.apply(&base, x),
        None => Ok(base),
    }
}

/// Everything one adapter carries, keyed by `(layer, target)`.
#[derive(Debug)]
pub struct Adapter {
    /// A stable name, for logs and for the per-conversation opt-in.
    pub name: String,
    pairs: HashMap<(usize, Target), Pair>,
    /// `alpha / r`, or `alpha / √r` when the adapter declares `use_rslora`.
    ///
    /// Reported, not applied: [`Pair`] carries it folded into `b_t`. Kept here
    /// because it is the number an operator checks when an adapter's effect
    /// looks too strong or too weak, and deriving it back out of a weight is not
    /// something anyone should have to do.
    scale: f64,
    pub rank: usize,
}

/// Why an adapter could not be loaded.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Config(String),
    Tensor(candle::Error),
    /// A tensor whose PEFT name this does not understand. Reported rather than
    /// skipped: a silently-dropped pair is an adapter that half-applies, which
    /// reads as a bad fine-tune rather than as a loading bug.
    UnknownTensor(String),
    /// `lora_A` without its `lora_B`, or the reverse.
    Unpaired(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "{e}"),
            LoadError::Config(s) => write!(f, "adapter_config.json: {s}"),
            LoadError::Tensor(e) => write!(f, "{e}"),
            LoadError::UnknownTensor(n) => write!(f, "unrecognised adapter tensor `{n}`"),
            LoadError::Unpaired(n) => write!(f, "`{n}` has no matching lora_A/lora_B"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Split a PEFT tensor name into `(layer, target, which)`.
///
/// `base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight`
///                              ^^           ^^^^^^      ^
///
/// Tolerant of the `base_model.model.` prefixes PEFT stacks up (they vary with
/// how the model was wrapped) by locating `layers.` rather than counting from
/// the front.
pub fn parse_name(name: &str) -> Option<(usize, Target, bool)> {
    let after = name.split("layers.").nth(1)?;
    let (idx, rest) = after.split_once('.')?;
    let layer: usize = idx.parse().ok()?;

    let is_a = if rest.contains(".lora_A") {
        true
    } else if rest.contains(".lora_B") {
        false
    } else {
        return None;
    };
    // The module is the segment before `.lora_`.
    let module = rest.split(".lora_").next()?.rsplit('.').next()?;
    Some((layer, Target::from_peft(module)?, is_a))
}

impl Adapter {
    /// Load a PEFT adapter directory — `adapter_config.json` +
    /// `adapter_model.safetensors`.
    ///
    /// `dtype` is the activation width the model computes in. The adapter is
    /// stored F32 by PEFT and converted once here rather than per token: a cast
    /// in the forward path would be a full-tensor pass per projection per layer
    /// per step, which is exactly the hot-path invariant against `to_dtype` in
    /// the loop.
    pub fn load(
        dir: &Path,
        name: impl Into<String>,
        dtype: DType,
        device: &Device,
    ) -> std::result::Result<Adapter, LoadError> {
        let cfg_text =
            std::fs::read_to_string(dir.join("adapter_config.json")).map_err(LoadError::Io)?;
        let cfg: serde_json::Value =
            serde_json::from_str(&cfg_text).map_err(|e| LoadError::Config(e.to_string()))?;
        let rank = cfg
            .get("r")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| LoadError::Config("no `r`".into()))? as usize;
        let alpha = cfg
            .get("lora_alpha")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| LoadError::Config("no `lora_alpha`".into()))?;
        if rank == 0 {
            return Err(LoadError::Config("rank 0".into()));
        }
        // **rsLoRA divides by √r, not r.** The two differ by √r — a factor of 8
        // at this adapter's rank — and nothing downstream can tell them apart:
        // both produce a running model, one of them with the adapter's
        // contribution multiplied by eight. `use_rslora` is the only record of
        // which was trained, so it is read rather than assumed.
        let rslora = cfg
            .get("use_rslora")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let divisor = if rslora {
            (rank as f64).sqrt()
        } else {
            rank as f64
        };

        // Read onto the **host**, in PEFT's own F32. That is where the source
        // copy lives for the life of the adapter (see [`Pair`]); the device gets
        // one materialised copy per pair, in `dtype`, built below.
        let weights = dir.join("adapter_model.safetensors");
        let loaded =
            candle::safetensors::load(&weights, &Device::Cpu).map_err(LoadError::Tensor)?;

        let mut a: HashMap<(usize, Target), Tensor> = HashMap::new();
        let mut b: HashMap<(usize, Target), Tensor> = HashMap::new();
        for (tensor_name, t) in loaded {
            let Some((layer, target, is_a)) = parse_name(&tensor_name) else {
                return Err(LoadError::UnknownTensor(tensor_name));
            };
            if is_a {
                a.insert((layer, target), t);
            } else {
                b.insert((layer, target), t);
            }
        }

        let scale = alpha / divisor;
        let mut pairs = HashMap::new();
        for (key, a_w) in a {
            let b_w = b
                .remove(&key)
                .ok_or_else(|| LoadError::Unpaired(format!("layer {} {:?}", key.0, key.1)))?;
            let pair = Pair::from_peft_weights(&a_w, &b_w, scale, dtype, device)
                .map_err(LoadError::Tensor)?;
            pairs.insert(key, pair);
        }
        // Anything left in `b` had no `A`. Same failure, other direction.
        if let Some((key, _)) = b.into_iter().next() {
            return Err(LoadError::Unpaired(format!("layer {} {:?}", key.0, key.1)));
        }

        Ok(Adapter {
            name: name.into(),
            pairs,
            scale,
            rank,
        })
    }

    /// The pair for one projection, if this adapter touches it.
    ///
    /// `None` is the common case and not an error: an adapter over a hybrid
    /// model has attention pairs on only the attention layers, so every
    /// DeltaNet layer asks and is told no.
    pub fn pair(&self, layer: usize, target: Target) -> Option<&Pair> {
        self.pairs.get(&(layer, target))
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Apply to one projection, or hand back the base output unchanged.
    ///
    /// The shape every call site wants: a projection does not have to know
    /// whether this adapter covers it.
    pub fn apply<'w>(
        &self,
        layer: usize,
        target: Target,
        base: &LiveTensor<'w>,
        x: &LiveTensor<'w>,
    ) -> Result<LiveTensor<'w>> {
        match self.pair(layer, target) {
            Some(p) => p.apply(base, x),
            None => Ok(base.clone()),
        }
    }

    /// Re-materialise every pair in the session's activation dtype.
    ///
    /// Called from the model's own `maybe_change_dtype` at session creation —
    /// the single place a dtype change happens, and never inside a wave. A
    /// session whose dtype already matches pays one lock read per pair and
    /// nothing else.
    ///
    /// **This is not optional.** A matmul refuses mismatched operands outright,
    /// so an adapter left at the loader's dtype fails the first adapted
    /// projection of an F16 session with `dtype mismatch in matmul, lhs: F16,
    /// rhs: BF16` — which is what happens when this is not wired up, and is how
    /// the need for it was found.
    pub fn maybe_change_dtype(&self, dtypes: &[DType], device: &Device) -> Result<()> {
        for pair in self.pairs.values() {
            pair.materialise(dtypes, device)?;
        }
        Ok(())
    }

    /// The dtypes the adapter is currently resident in, or empty when it holds
    /// no pairs.
    pub fn resident_dtypes(&self) -> Result<Vec<DType>> {
        match self.pairs.values().next() {
            Some(p) => p.resident_dtypes(),
            None => Ok(Vec::new()),
        }
    }

    /// Resolve every projection this layer might adapt, once.
    ///
    /// The forward path asks per layer, not per projection: seven hash lookups
    /// happen here, at the top of a layer, instead of inside `project_qkv` and
    /// the FFN where they would repeat per wave. What reaches the hot path is
    /// then a null check on a borrowed pointer.
    ///
    /// A layer the adapter does not cover yields an all-`None` [`LayerLora`],
    /// which is the ordinary case: on a 3:1 hybrid, three layers in four have no
    /// attention pairs, and the MTP head — which loads as `blk.{n_layers}` and
    /// runs the same attention layer type — has none at all, because a 32-layer
    /// adapter simply has nothing at index 32.
    pub fn layer(&self, layer: usize) -> LayerLora<'_> {
        LayerLora {
            q: self.pair(layer, Target::AttnQ),
            k: self.pair(layer, Target::AttnK),
            v: self.pair(layer, Target::AttnV),
            o: self.pair(layer, Target::AttnOut),
            gate: self.pair(layer, Target::FfnGate),
            up: self.pair(layer, Target::FfnUp),
            down: self.pair(layer, Target::FfnDown),
        }
    }

    /// Which layers carry an attention adapter — the attention layers of a
    /// hybrid, as the adapter itself saw them.
    ///
    /// Used to check the adapter and the checkpoint agree about the
    /// architecture: an adapter trained on a 3:1 hybrid applied to a stack with
    /// attention somewhere else would load and land its pairs on the wrong
    /// layers.
    pub fn attention_layers(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .pairs
            .keys()
            .filter(|(_, t)| matches!(t, Target::AttnQ))
            .map(|(l, _)| *l)
            .collect();
        v.sort_unstable();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: &str = "base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight";
    const GATE_B: &str = "base_model.model.model.layers.31.mlp.gate_proj.lora_B.weight";

    #[test]
    fn a_peft_name_splits_into_layer_target_and_side() {
        assert_eq!(parse_name(Q), Some((7, Target::AttnQ, true)));
        assert_eq!(parse_name(GATE_B), Some((31, Target::FfnGate, false)));
    }

    /// PEFT stacks `base_model.model.` prefixes differently depending on how the
    /// model was wrapped, so the parse locates `layers.` rather than counting
    /// segments from the front.
    #[test]
    fn the_parse_survives_a_different_prefix_depth() {
        assert_eq!(
            parse_name("model.layers.3.mlp.down_proj.lora_B.weight"),
            Some((3, Target::FfnDown, false))
        );
        assert_eq!(
            parse_name("base_model.model.model.model.layers.3.mlp.up_proj.lora_A.weight"),
            Some((3, Target::FfnUp, true))
        );
    }

    #[test]
    fn every_peft_module_maps_to_a_gguf_role() {
        for (module, want) in [
            ("q_proj", Target::AttnQ),
            ("k_proj", Target::AttnK),
            ("v_proj", Target::AttnV),
            ("o_proj", Target::AttnOut),
            ("gate_proj", Target::FfnGate),
            ("up_proj", Target::FfnUp),
            ("down_proj", Target::FfnDown),
        ] {
            assert_eq!(Target::from_peft(module), Some(want));
        }
        // Every role names a distinct GGUF tensor, or two adapters would land on
        // one weight.
        let mut roles: Vec<&str> = [
            Target::AttnQ,
            Target::AttnK,
            Target::AttnV,
            Target::AttnOut,
            Target::FfnGate,
            Target::FfnUp,
            Target::FfnDown,
        ]
        .iter()
        .map(|t| t.gguf_role())
        .collect();
        let n = roles.len();
        roles.sort_unstable();
        roles.dedup();
        assert_eq!(roles.len(), n);
    }

    /// **An unrecognised tensor is refused, not skipped.** A dropped pair is an
    /// adapter that half-applies, which reads as a bad fine-tune rather than as
    /// a loading bug — and the person debugging it has no reason to suspect the
    /// loader.
    #[test]
    fn an_unrecognised_module_is_not_silently_dropped() {
        assert_eq!(
            parse_name("model.layers.0.self_attn.rotary.lora_A.weight"),
            None
        );
        assert_eq!(parse_name("model.embed_tokens.weight"), None);
        // And a name with no `lora_` side is not a pair at all.
        assert_eq!(parse_name("model.layers.0.mlp.up_proj.weight"), None);
    }

    /// The maths, on a case small enough to check by hand.
    ///
    /// `A = [[1,0],[0,1]]` (r=2, in=2) and `B = [[2,0],[0,3]]` (out=2, r=2), so
    /// `B(Ax) = [2x₀, 3x₁]`, and with `scale = 0.5` the adapter contributes
    /// `[x₀, 1.5x₁]` on top of the base.
    #[test]
    fn the_adapter_term_is_added_to_the_base() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::new(&[[1f32, 0.], [0., 1.]], &dev)?;
        let b = Tensor::new(&[[2f32, 0.], [0., 3.]], &dev)?;
        let pair = Pair::from_peft_weights(&a, &b, 0.5, DType::F32, &dev)?;

        let x = Tensor::new(&[[1f32, 2.]], &dev)?;
        let base = Tensor::new(&[[10f32, 20.]], &dev)?;
        let out = pair.apply(&base, &x)?;

        // base + 0.5 · [2·1, 3·2] = [10+1, 20+3]
        assert_eq!(out.to_vec2::<f32>()?, vec![vec![11f32, 23.]]);
        Ok(())
    }

    /// **The adapter follows the session's activation dtype.**
    ///
    /// The dtype activations arrive in belongs to the *session*, not the load:
    /// one resident model serves an F16 session and a BF16 one. A matmul refuses
    /// mismatched operands outright, so an adapter left at the loader's width
    /// fails the first adapted projection with `dtype mismatch in matmul, lhs:
    /// F16, rhs: BF16`.
    ///
    /// That is not hypothetical — it is exactly how the 9B LoRA gate failed
    /// before `maybe_change_dtype` reached the adapters: the model was loaded
    /// BF16 and the gate's first row runs F16.
    ///
    /// Re-materialising also has to be *lossless with respect to the source*.
    /// Round-tripping the resident copy (BF16 → F16 → BF16) would compound
    /// rounding on every session; materialising from the host F32 master each
    /// time cannot. Asserted by going out to F16 and back and demanding the
    /// original values.
    #[test]
    fn the_adapter_re_materialises_in_the_sessions_dtype() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::new(&[[1f32, 0.], [0., 1.]], &dev)?;
        let b = Tensor::new(&[[2f32, 0.], [0., 3.]], &dev)?;
        let pair = Pair::from_peft_weights(&a, &b, 0.5, DType::F32, &dev)?;
        assert_eq!(pair.resident_dtypes()?, vec![DType::F32]);

        let x = Tensor::new(&[[1f32, 2.]], &dev)?;
        let base = Tensor::new(&[[10f32, 20.]], &dev)?;
        let before = pair.apply(&base, &x)?.to_vec2::<f32>()?;

        // A session that computes in F16.
        pair.materialise(&[DType::F16], &dev)?;
        assert_eq!(pair.resident_dtypes()?, vec![DType::F16]);
        let f16_out = pair.apply(&base.to_dtype(DType::F16)?, &x.to_dtype(DType::F16)?)?;
        assert_eq!(
            f16_out.dtype(),
            DType::F16,
            "the result follows the operands"
        );

        // And back. From the F32 master, so this is the original, not a
        // twice-rounded copy.
        pair.materialise(&[DType::F32], &dev)?;
        assert_eq!(pair.resident_dtypes()?, vec![DType::F32]);
        assert_eq!(pair.apply(&base, &x)?.to_vec2::<f32>()?, before);
        Ok(())
    }

    /// **A projection whose input and output widths differ.**
    ///
    /// This is the shape that broke the gate's second run: inside one MLP, an
    /// F16 session's `gate` reads the F16 residual stream and writes the
    /// promoted BF16 SwiGLU intermediate, and `down` then reads BF16 and writes
    /// F16 back. A single-width adapter cannot serve both, and the failure is
    /// `dtype mismatch in add, lhs: BF16, rhs: F16` — the adapter term arriving
    /// in the input's width when the base output is in the other.
    ///
    /// Both directions asserted, because they are genuinely different code
    /// paths through the same function and the first one working says nothing
    /// about the second.
    #[test]
    fn a_projection_may_read_one_width_and_write_another() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::new(&[[1f32, 0.], [0., 1.]], &dev)?;
        let b = Tensor::new(&[[2f32, 0.], [0., 3.]], &dev)?;
        let pair = Pair::from_peft_weights(&a, &b, 0.5, DType::F32, &dev)?;
        // Both widths a mixed-width layer can ask for, materialised together at
        // session creation — which is the whole point: the forward converts no
        // weight, it only picks the copy it needs.
        pair.materialise(&[DType::F16, DType::F32], &dev)?;

        let x16 = Tensor::new(&[[1f32, 2.]], &dev)?.to_dtype(DType::F16)?;
        let base32 = Tensor::new(&[[10f32, 20.]], &dev)?;

        // Narrow in, wide out — an F16 stream feeding a promoted intermediate.
        let widened = pair.apply(&base32, &x16)?;
        assert_eq!(widened.dtype(), DType::F32, "the result follows the base");
        assert_eq!(widened.to_vec2::<f32>()?, vec![vec![11f32, 23.]]);

        // Wide in, narrow out — the same layer's `down`, going back.
        let x32 = Tensor::new(&[[1f32, 2.]], &dev)?;
        let base16 = Tensor::new(&[[10f32, 20.]], &dev)?.to_dtype(DType::F16)?;
        let narrowed = pair.apply(&base16, &x32)?;
        assert_eq!(narrowed.dtype(), DType::F16);
        assert_eq!(
            narrowed.to_dtype(DType::F32)?.to_vec2::<f32>()?,
            vec![vec![11f32, 23.]]
        );
        Ok(())
    }

    /// A width the session never declared is refused by name, not guessed at.
    ///
    /// The alternative — converting on demand inside the wave — is the thing
    /// this design exists to avoid, and a silent conversion there would be a
    /// full-tensor pass per projection per layer per step that nothing reports.
    #[test]
    fn an_undeclared_width_is_refused_rather_than_converted() -> Result<()> {
        let dev = Device::Cpu;
        let pair = Pair::from_peft_weights(
            &Tensor::new(&[[1f32, 0.], [0., 1.]], &dev)?,
            &Tensor::new(&[[2f32, 0.], [0., 3.]], &dev)?,
            1.0,
            DType::F32,
            &dev,
        )?;
        let x = Tensor::new(&[[1f32, 2.]], &dev)?.to_dtype(DType::F16)?;
        let base = Tensor::new(&[[0f32, 0.]], &dev)?.to_dtype(DType::F16)?;
        let err = pair.apply(&base, &x).unwrap_err().to_string();
        assert!(err.contains("F16"), "{err}");
        assert!(err.contains("maybe_change_dtype"), "{err}");
        Ok(())
    }

    /// Re-materialising to the dtype already resident is a no-op, which is what
    /// every session after the first gets — the common case must not copy
    /// ~230 MB across PCIe for nothing.
    #[test]
    fn re_materialising_to_the_same_dtype_changes_nothing() -> Result<()> {
        let dev = Device::Cpu;
        let pair = Pair::from_peft_weights(
            &Tensor::new(&[[1f32, 0.], [0., 1.]], &dev)?,
            &Tensor::new(&[[2f32, 0.], [0., 3.]], &dev)?,
            1.0,
            DType::F32,
            &dev,
        )?;
        let x = Tensor::new(&[[1f32, 2.]], &dev)?;
        let base = Tensor::new(&[[0f32, 0.]], &dev)?;
        let before = pair.apply(&base, &x)?.to_vec2::<f32>()?;
        pair.materialise(&[DType::F32], &dev)?;
        assert_eq!(pair.apply(&base, &x)?.to_vec2::<f32>()?, before);
        Ok(())
    }

    /// A zero scale must leave the base exactly alone — the property an
    /// opted-out conversation relies on.
    #[test]
    fn a_zero_scale_leaves_the_base_untouched() -> Result<()> {
        let dev = Device::Cpu;
        let pair = Pair::from_peft_weights(
            &Tensor::new(&[[5f32, 7.]], &dev)?,
            &Tensor::new(&[[9f32], [11.]], &dev)?,
            0.0,
            DType::F32,
            &dev,
        )?;
        let x = Tensor::new(&[[1f32, 1.]], &dev)?;
        let base = Tensor::new(&[[3f32, 4.]], &dev)?;
        assert_eq!(
            pair.apply(&base, &x)?.to_vec2::<f32>()?,
            base.to_vec2::<f32>()?
        );
        Ok(())
    }

    /// A projection the adapter does not cover gets its base output back
    /// unchanged — the common case on a hybrid, where three layers in four have
    /// no attention pairs at all.
    #[test]
    fn an_uncovered_projection_passes_the_base_through() -> Result<()> {
        let dev = Device::Cpu;
        let adapter = Adapter {
            name: "t".into(),
            pairs: HashMap::new(),
            scale: 1.0,
            rank: 8,
        };
        let base = Tensor::new(&[[1f32, 2.]], &dev)?;
        let x = Tensor::new(&[[1f32, 1.]], &dev)?;
        let out = adapter.apply(0, Target::AttnQ, &base, &x)?;
        assert_eq!(out.to_vec2::<f32>()?, base.to_vec2::<f32>()?);
        Ok(())
    }

    /// Write a minimal PEFT adapter directory: one `q_proj` pair on layer 3,
    /// `r = 64, alpha = 64`, with `use_rslora` as given.
    fn write_adapter(tag: &str, rslora: bool) -> Result<std::path::PathBuf> {
        let dir = std::env::temp_dir().join(format!("candle-lora-{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("adapter_config.json"),
            format!(
                r#"{{"r": 64, "lora_alpha": 64, "use_rslora": {rslora}, "peft_type": "LORA"}}"#
            ),
        )
        .unwrap();
        let dev = Device::Cpu;
        let mut w: HashMap<String, Tensor> = HashMap::new();
        w.insert(
            "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight".into(),
            Tensor::zeros((64, 8), DType::F32, &dev)?,
        );
        w.insert(
            "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight".into(),
            Tensor::zeros((16, 64), DType::F32, &dev)?,
        );
        candle::safetensors::save(&w, dir.join("adapter_model.safetensors"))?;
        Ok(dir)
    }

    /// **`use_rslora` divides by `√r`, not `r`.**
    ///
    /// Both spellings load, both run, and the difference is the whole adapter
    /// contribution multiplied by eight at rank 64. The adapter npcd loads
    /// declares `use_rslora: true` with `r = alpha = 64`, so its true scale is
    /// 8.0 — reading the flag is the only thing standing between that and a
    /// silently eightfold-weak adapter.
    #[test]
    fn rank_stabilised_scaling_is_read_from_the_config() -> Result<()> {
        let dev = Device::Cpu;

        let plain = Adapter::load(&write_adapter("plain", false)?, "p", DType::F32, &dev).unwrap();
        assert_eq!(plain.scale(), 1.0, "alpha/r at r = alpha = 64");

        let stabilised = Adapter::load(&write_adapter("rs", true)?, "r", DType::F32, &dev).unwrap();
        assert_eq!(stabilised.scale(), 8.0, "alpha/√r at r = alpha = 64");

        // Both found the same single pair, on the layer the name encodes — the
        // scale is the only thing that differs.
        assert_eq!(plain.len(), 1);
        assert_eq!(stabilised.len(), 1);
        assert!(stabilised.pair(3, Target::AttnQ).is_some());
        assert_eq!(stabilised.attention_layers(), vec![3]);
        Ok(())
    }

    /// An adapter with no `use_rslora` key is a plain one — PEFT's own default,
    /// and the reading every adapter written before rsLoRA existed needs.
    #[test]
    fn an_absent_rslora_flag_means_plain_scaling() -> Result<()> {
        let dir = std::env::temp_dir().join("candle-lora-absent");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("adapter_config.json"),
            r#"{"r": 16, "lora_alpha": 32}"#,
        )
        .unwrap();
        let dev = Device::Cpu;
        let w: HashMap<String, Tensor> = HashMap::new();
        candle::safetensors::save(&w, dir.join("adapter_model.safetensors"))?;

        let a = Adapter::load(&dir, "a", DType::F32, &dev).unwrap();
        assert_eq!(a.scale(), 2.0, "alpha/r = 32/16");
        Ok(())
    }
}
