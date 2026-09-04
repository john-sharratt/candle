//! Weight adapters for Z-Image, and fusing one into a checkpoint.
//!
//! # What an adapter is for here
//!
//! Z-Image-Turbo is guidance-distilled, and distillation costs variety. Trained
//! to reach the modal answer in as few evaluations as it can, the model returns
//! **the same face whatever the seed** — measured here, one prompt at three
//! seeds gives the same person, at eight steps and at twelve alike, and the same
//! seed with a described subject gives someone else entirely. The identity is
//! decided by the conditioning; the noise barely touches it.
//!
//! No sampler setting is a lever on that. `cfg_truncation`, the parameter this
//! is usually blamed on, governs the guidance branch — and a model running at
//! `guidance_scale = 0.0` has no such branch. The fix has to be in the weights,
//! which is what an adapter is: a trained delta over the checkpoint. The SDA
//! diversity adapter ([`SDA_REPO`]) is the standing one; a deployment can fuse
//! others the same way.
//!
//! # The two factorisations
//!
//! An adapter file stores a delta in small factors, and two conventions cover
//! everything this fork fuses ([`Factors`]):
//!
//! | | Delta | Files carry |
//! |---|---|---|
//! | **LoKr** | `ΔW = kron(w1, w2)` | `….lokr_w1` / `….lokr_w2` |
//! | **LoRA** | `ΔW = B · A` | `….lora_B.weight` / `….lora_A.weight` |
//!
//! Both arrive under ai-toolkit's `diffusion_model.layers.N.…` naming, so one
//! module table places either. Scale is 1.0 for both, for two different
//! reasons, each of which looks like a bug until it is read:
//!
//! - **LoKr's `alpha` is a sentinel, not a multiplier.** A full-rank LoKr
//!   stores `alpha = lora_dim = 1e10` so LyCORIS's `alpha / lora_dim` cancels
//!   to 1 — the algorithm list says *"alpha is ignored when using full
//!   dimension"*. Reading it as a multiplier scales every delta by 10¹⁰.
//! - **A LoRA file with no `alpha` tensors is ai-toolkit's convention for
//!   `alpha = rank`,** and `alpha / rank` is again 1. A consumer that assumed
//!   diffusers' common `alpha < rank` would silently under-apply the adapter.
//!
//! [`relative_norm`] is the guard on both: a conversion reports `‖Δ‖ / ‖W‖`
//! per tensor, and a wrong scale is off by orders of magnitude there, not by a
//! rounding error.
//!
//! # Why it is fused rather than applied
//!
//! Either factorisation could be applied live at a fraction of the base
//! matmul's arithmetic. But the base runs int8 on the tensor cores and an
//! adapter path would run bf16, so a fraction of the FLOPs is not a fraction of
//! the time, and every step of every draw would pay it while the whole estate
//! waits.
//!
//! Fusing costs nothing at run time. The delta is a property of the weights, so
//! it belongs in the weights: [`fuse`] writes a checkpoint with the adapter
//! already in it, and a deployment chooses by pointing `guests.yaml` at one file
//! or another. There is no code path to select and nothing in the denoise loop
//! that knows this module exists.
//!
//! # The one wrinkle: qkv is fused here and not there
//!
//! An adapter names `to_q`, `to_k` and `to_v` separately, following the
//! diffusers convention. The GGUF fuses them into one `qkv` of three times the
//! width — see [`super::quantized_model`]. So three deltas stack row-wise into
//! one target, in the order the attention splits them, and a target is only
//! complete when all three have arrived. [`Fusion::deltas`] refuses a partial
//! stack rather than writing a third of an adapter into a projection.

use std::collections::HashMap;

use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{DType, Device, Result, Shape, Tensor};

/// Where the SDA diversity adapter is published.
pub const SDA_REPO: &str = "F16/z-image-turbo-sda";
pub const SDA_FILE: &str = "zit_sda_v1.safetensors";

/// The prefix every adapter tensor carries.
const PREFIX: &str = "diffusion_model.";

/// The strength to fuse at when nothing says otherwise.
///
/// An adapter's card asks for less than full strength when it is stacked with
/// other adapters. Fusing is the alone case, so this is 1.0 — but it is the
/// argument [`fuse`] takes rather than a constant it applies, because the
/// number is the whole reason to re-run a conversion.
pub const DEFAULT_STRENGTH: f64 = 1.0;

/// A delta ready to be added to one checkpoint tensor.
#[derive(Debug)]
pub struct Delta {
    /// The GGUF tensor this applies to, `.weight` included.
    pub name: String,
    /// Its full delta, already scaled, shaped exactly like the target.
    pub delta: Tensor,
}

/// One module's factor pair, in whichever factorisation its file used.
#[derive(Debug)]
pub enum Factors {
    /// LoKr: `ΔW = kron(w1, w2)`.
    Kron { w1: Tensor, w2: Tensor },
    /// LoRA: `ΔW = b · a`, with `a` `[rank, in]` and `b` `[out, rank]`.
    Low { a: Tensor, b: Tensor },
}

impl Factors {
    /// The full delta this pair expands to.
    pub fn delta(&self) -> Result<Tensor> {
        match self {
            Self::Kron { w1, w2 } => kron(w1, w2),
            Self::Low { a, b } => b.matmul(a),
        }
    }
}

/// Which half of which factorisation a tensor name is, if any.
///
/// The LoRA halves end in `.weight` and the LoKr halves do not — that is the
/// files' own naming, kept as-is so an adapter opened in a viewer reads the
/// same as this table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    KronW1,
    KronW2,
    LowA,
    LowB,
}

impl Part {
    const SUFFIXES: &'static [(&'static str, Part)] = &[
        (".lokr_w1", Part::KronW1),
        (".lokr_w2", Part::KronW2),
        (".lora_A.weight", Part::LowA),
        (".lora_B.weight", Part::LowB),
    ];
}

/// `diffusion_model.layers.0.attention.to_q.lokr_w1` →
/// `("layers.0.attention.to_q", KronW1)`. Anything else — `alpha` included, see
/// the module doc — is not a factor and answers `None`.
fn classify(name: &str) -> Option<(&str, Part)> {
    let rest = name.strip_prefix(PREFIX)?;
    Part::SUFFIXES
        .iter()
        .find_map(|(suffix, part)| Some((rest.strip_suffix(suffix)?, *part)))
}

/// The two halves seen so far for one module, before they are proven a pair.
#[derive(Debug, Default)]
struct Halves {
    kron_w1: Option<Tensor>,
    kron_w2: Option<Tensor>,
    low_a: Option<Tensor>,
    low_b: Option<Tensor>,
}

impl Halves {
    fn set(&mut self, part: Part, t: Tensor) {
        match part {
            Part::KronW1 => self.kron_w1 = Some(t),
            Part::KronW2 => self.kron_w2 = Some(t),
            Part::LowA => self.low_a = Some(t),
            Part::LowB => self.low_b = Some(t),
        }
    }

    /// The finished pair, or a named refusal. A module missing half its pair
    /// cannot produce a delta, and silently dropping it would fuse a partial
    /// adapter; one carrying both factorisations is a file this fork has never
    /// seen, and guessing which to apply is not a decision to make silently.
    fn finish(self, module: &str) -> Result<Factors> {
        let kron = self.kron_w1.is_some() || self.kron_w2.is_some();
        let low = self.low_a.is_some() || self.low_b.is_some();
        match (kron, low) {
            (true, true) => Err(candle::Error::Msg(format!(
                "the adapter's `{module}` carries both LoKr and LoRA factors — this fork does \
                 not know which to fuse"
            ))),
            (true, false) => match (self.kron_w1, self.kron_w2) {
                (Some(w1), Some(w2)) => Ok(Factors::Kron { w1, w2 }),
                _ => Err(candle::Error::Msg(format!(
                    "the adapter's `{module}` is missing one of `lokr_w1` / `lokr_w2`"
                ))),
            },
            (false, true) => match (self.low_a, self.low_b) {
                (Some(a), Some(b)) => Ok(Factors::Low { a, b }),
                _ => Err(candle::Error::Msg(format!(
                    "the adapter's `{module}` is missing one of `lora_A` / `lora_B`"
                ))),
            },
            (false, false) => Err(candle::Error::Msg(format!(
                "the adapter's `{module}` carries no factors at all"
            ))),
        }
    }
}

/// The adapter's factors, keyed by the module they belong to.
///
/// The key is the adapter's own module path with [`PREFIX`] removed —
/// `layers.0.attention.to_q` — because that is what the mapping below is written
/// against and it keeps the file's naming visible rather than translated on the
/// way in.
#[derive(Debug)]
pub struct Fusion {
    factors: HashMap<String, Factors>,
}

/// Where a module's delta lands in the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    /// The whole tensor, named by the `&'static str` suffix.
    Whole,
    /// A third of `attention.qkv`, at this row offset in units of the delta's
    /// own height. The attention splits `qkv` as plain thirds in the order
    /// q, k, v, so the offset *is* the split index.
    Qkv(usize),
}

/// Every adapter module this fork knows how to place, and where it goes.
///
/// A table rather than string surgery so an adapter carrying a module we cannot
/// place is a named failure instead of a silent omission — a fused checkpoint
/// missing a third of its adapter is not distinguishable by looking at it.
const MODULES: &[(&str, &str, Target)] = &[
    ("attention.to_q", "attention.qkv", Target::Qkv(0)),
    ("attention.to_k", "attention.qkv", Target::Qkv(1)),
    ("attention.to_v", "attention.qkv", Target::Qkv(2)),
    ("attention.to_out.0", "attention.out", Target::Whole),
    ("feed_forward.w1", "feed_forward.w1", Target::Whole),
    ("feed_forward.w2", "feed_forward.w2", Target::Whole),
    ("feed_forward.w3", "feed_forward.w3", Target::Whole),
    ("adaLN_modulation.0", "adaLN_modulation.0", Target::Whole),
];

/// The Kronecker product, `kron(a, b)`.
///
/// `a` is `[m, n]` and `b` is `[p, q]`; the result is `[m·p, n·q]` with
/// `out[i·p + k, j·q + l] = a[i, j] · b[k, l]`.
///
/// Built by broadcasting rather than by a loop: `a` viewed as `[m, 1, n, 1]`
/// against `b` as `[1, p, 1, q]` multiplies to exactly that indexing, and the
/// reshape back is free once it is contiguous. The axis order is the whole of
/// the correctness here — `[m, p, n, q]` flattens to the right thing and
/// `[p, m, q, n]` does not — so it is pinned by
/// [`tests::kron_matches_the_definition_elementwise`].
pub fn kron(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (m, n) = a.dims2()?;
    let (p, q) = b.dims2()?;
    a.reshape((m, 1, n, 1))?
        .broadcast_mul(&b.reshape((1, p, 1, q))?)?
        .contiguous()?
        .reshape((m * p, n * q))
}

impl Fusion {
    /// Read the adapter from a safetensors file.
    ///
    /// Everything is taken to f32 on the way in. The files are bf16 and the
    /// checkpoint they will be added to dequantises to f32, so converting once
    /// here keeps the arithmetic in one width instead of rounding a delta to
    /// bf16 after building it at full precision.
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        Self::from_tensors(candle::safetensors::load(path, device)?)
    }

    fn from_tensors(all: HashMap<String, Tensor>) -> Result<Self> {
        let mut halves: HashMap<String, Halves> = HashMap::new();
        for (name, t) in all.iter() {
            let Some((module, part)) = classify(name) else {
                continue;
            };
            halves
                .entry(module.to_string())
                .or_default()
                .set(part, t.to_dtype(DType::F32)?);
        }
        if halves.is_empty() {
            return Err(candle::Error::Msg(format!(
                "no `{PREFIX}*` factor tensors — this is not a Z-Image LoKr or LoRA adapter"
            )));
        }
        let factors = halves
            .into_iter()
            .map(|(module, h)| {
                let f = h.finish(&module)?;
                Ok((module, f))
            })
            .collect::<Result<_>>()?;
        Ok(Self { factors })
    }

    /// How many modules the adapter carries.
    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    /// Build every delta, keyed by the GGUF tensor it belongs to.
    ///
    /// `strength` multiplies the whole adapter — 1.0 is the trained strength,
    /// and a card asks for less only when it is stacked with other adapters.
    ///
    /// The three `qkv` thirds are stacked here rather than by the caller,
    /// because only this function knows they are thirds: it refuses a target
    /// whose stack is incomplete rather than writing a partial adapter into a
    /// projection, which nothing downstream could detect.
    pub fn deltas(&self, strength: f64) -> Result<Vec<Delta>> {
        // Per GGUF target, the pieces gathered so far. `Qkv` targets collect
        // three and `Whole` targets one.
        let mut staged: HashMap<String, Vec<(usize, Tensor)>> = HashMap::new();

        for (module, factors) in self.factors.iter() {
            let (layer, tail) = split_layer(module).ok_or_else(|| {
                candle::Error::Msg(format!("`{module}` is not a `layers.N.…` module"))
            })?;
            let (_, gguf_tail, target) = MODULES
                .iter()
                .find(|(adapter_tail, _, _)| *adapter_tail == tail)
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "the adapter carries `{tail}`, which this fork cannot place — fusing \
                         would silently drop it"
                    ))
                })?;
            let delta = (factors.delta()? * strength)?;
            let slot = match target {
                Target::Whole => 0,
                Target::Qkv(i) => *i,
            };
            staged
                .entry(format!("layers.{layer}.{gguf_tail}.weight"))
                .or_default()
                .push((slot, delta));
        }

        let mut out = Vec::with_capacity(staged.len());
        for (name, mut parts) in staged {
            parts.sort_by_key(|(slot, _)| *slot);
            let delta = if parts.len() == 1 {
                parts.pop().expect("one part").1
            } else {
                // Three thirds of `qkv`, in q/k/v order. Anything else means the
                // adapter and the table disagree about the split.
                if parts.len() != 3 || parts.iter().map(|(s, _)| *s).ne(0..3) {
                    return Err(candle::Error::Msg(format!(
                        "`{name}` gathered {} of its 3 parts — a partial stack would fuse a \
                         fraction of the adapter",
                        parts.len()
                    )));
                }
                let rows: Vec<Tensor> = parts.into_iter().map(|(_, t)| t).collect();
                Tensor::cat(&rows, 0)?
            };
            out.push(Delta { name, delta });
        }
        // Deterministic order, so a conversion's log reads the same twice.
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }
}

/// `layers.12.attention.to_q` → `(12, "attention.to_q")`.
fn split_layer(module: &str) -> Option<(usize, &str)> {
    let rest = module.strip_prefix("layers.")?;
    let (idx, tail) = rest.split_once('.')?;
    Some((idx.parse().ok()?, tail))
}

/// Add `delta` to a dequantised weight, checking the shapes agree.
///
/// Separate from the loop that calls it so the one thing that must not go wrong
/// — a delta landing on a tensor of a different shape — is a named error rather
/// than a broadcast that happens to succeed.
pub fn apply(weight: &Tensor, delta: &Tensor) -> Result<Tensor> {
    if weight.shape() != delta.shape() {
        return Err(candle::Error::Msg(format!(
            "the adapter's delta is {:?} but the weight is {:?}",
            delta.shape(),
            weight.shape()
        )));
    }
    weight + delta
}

/// The relative size of a delta against the weight it modifies, `‖Δ‖ / ‖W‖`.
///
/// A conversion reports this per tensor. It is the one number that catches the
/// scale being wrong — see the module doc's two scale conventions. Reading
/// LoKr's sentinel `alpha` as a multiplier scales every delta by 10¹⁰, which
/// produces a checkpoint that is all NaN on its first step rather than
/// anything a shape check would notice; an under-scaled LoRA shows up here as
/// a delta a hundredth of the size the card promises.
pub fn relative_norm(weight: &Tensor, delta: &Tensor) -> Result<f64> {
    let n = |t: &Tensor| -> Result<f64> {
        Ok(t.to_dtype(DType::F32)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()? as f64)
    };
    let w = n(weight)?.sqrt();
    let d = n(delta)?.sqrt();
    Ok(if w > 0.0 { d / w } else { f64::INFINITY })
}

/// The shape a LoKr module's delta will have, for a caller sizing work before
/// doing it — `kron` of an `[m, n]` and a `[p, q]` is `[m·p, n·q]`.
pub fn delta_shape(w1: &Shape, w2: &Shape) -> Result<(usize, usize)> {
    let (m, n) = w1.dims2()?;
    let (p, q) = w2.dims2()?;
    Ok((m * p, n * q))
}

/// What happened to one tensor, for a conversion to report.
#[derive(Clone, Debug)]
pub struct Fused {
    pub name: String,
    /// `‖Δ‖ / ‖W‖` — see [`relative_norm`].
    pub relative: f64,
    /// The format it was written back as, which is the one it arrived in.
    pub dtype: GgmlDType,
}

/// Read every tensor of `content`, add the adapter where it applies, and hand
/// back the whole set ready for [`gguf_file::write`].
///
/// # The rounding, stated plainly
///
/// A GGUF tensor arrives quantised. Adding to it means dequantising, adding, and
/// quantising again, so an adapted tensor goes through the quantiser **twice**
/// over its life and carries one extra rounding that an unadapted one does not.
/// That is the price of fusing, and it is small — the second rounding is the
/// same operation as the first, on a matrix that has barely moved — but it is
/// real, and it is why every tensor is written back in the format it arrived in
/// rather than being promoted: a conversion that silently changed the rung would
/// be reporting a quality difference that came from the rung and not the adapter.
///
/// Tensors the adapter does not touch are passed through **unchanged**, still
/// quantised, never dequantised at all. Most of the checkpoint takes that path.
///
/// # Nothing is dropped quietly
///
/// Every delta must find a tensor. A delta naming something the checkpoint does
/// not have is an error rather than a warning, because the result of ignoring it
/// is a checkpoint that loads, runs, and is missing part of its adapter — which
/// is not visible in the file, in the logs, or in one image.
pub fn fuse<R: std::io::Seek + std::io::Read>(
    content: &gguf_file::Content,
    reader: &mut R,
    fusion: &Fusion,
    strength: f64,
    device: &Device,
    mut report: impl FnMut(&Fused),
) -> Result<Vec<(String, QTensor)>> {
    let mut deltas: HashMap<String, Tensor> = fusion
        .deltas(strength)?
        .into_iter()
        .map(|d| (d.name, d.delta))
        .collect();

    for name in deltas.keys() {
        if !content.tensor_infos.contains_key(name) {
            return Err(candle::Error::Msg(format!(
                "the adapter targets `{name}`, which this checkpoint does not have — fusing \
                 would drop it silently"
            )));
        }
    }

    // The checkpoint's own order, so the output is a rewrite of the input rather
    // than a reordering of it.
    let mut names: Vec<&String> = content.tensor_infos.keys().collect();
    names.sort();

    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let q = content.tensor(reader, name, device)?;
        let Some(delta) = deltas.remove(name.as_str()) else {
            out.push((name.clone(), q));
            continue;
        };
        let dtype = q.dtype();
        let w = q.dequantize(device)?;
        let relative = relative_norm(&w, &delta)?;
        let fused = apply(&w, &delta)?;
        out.push((name.clone(), QTensor::quantize(&fused, dtype)?));
        report(&Fused {
            name: name.clone(),
            relative,
            dtype,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[f32], shape: (usize, usize)) -> Tensor {
        Tensor::from_vec(v.to_vec(), shape, &Device::Cpu).unwrap()
    }

    /// **The Kronecker product, against its definition, element by element.**
    ///
    /// Asserted as raw expected values rather than a tolerance: this is the
    /// whole of a LoKr's arithmetic, and the failure mode that matters is
    /// an axis order that produces a correctly-*shaped* matrix with the entries
    /// permuted. A norm check would pass that; this cannot.
    #[test]
    fn kron_matches_the_definition_elementwise() {
        let a = t(&[1., 2., 3., 4.], (2, 2));
        let b = t(&[10., 20., 30., 40., 50., 60.], (2, 3));
        let k = kron(&a, &b).unwrap();
        assert_eq!(k.dims(), &[4, 6]);
        assert_eq!(
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![
                10., 20., 30., 20., 40., 60., //
                40., 50., 60., 80., 100., 120., //
                30., 60., 90., 40., 80., 120., //
                120., 150., 180., 160., 200., 240.,
            ]
        );
    }

    /// The identity case, which pins the axis order a second way: `kron(I, B)`
    /// is `B` repeated down the diagonal, and any transposition of the axes puts
    /// it somewhere else.
    #[test]
    fn kron_with_the_identity_is_block_diagonal() {
        let i = t(&[1., 0., 0., 1.], (2, 2));
        let b = t(&[1., 2., 3., 4.], (2, 2));
        let k = kron(&i, &b).unwrap();
        assert_eq!(
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![
                1., 2., 0., 0., //
                3., 4., 0., 0., //
                0., 0., 1., 2., //
                0., 0., 3., 4.,
            ]
        );
    }

    /// **The LoRA delta is `B · A`, against the definition, element by
    /// element.** `a` is `[rank, in]` and `b` is `[out, rank]`, which is the
    /// orientation the files store — swapped, the matmul either fails on shape
    /// or, on a square module, produces the transpose of the trained delta.
    #[test]
    fn a_lora_delta_is_b_times_a_elementwise() {
        // rank 2, in 3, out 2.
        let a = t(&[1., 2., 3., 4., 5., 6.], (2, 3));
        let b = t(&[1., 0., 2., 1.], (2, 2));
        let f = Factors::Low { a, b };
        let d = f.delta().unwrap();
        assert_eq!(d.dims(), &[2, 3]);
        // Row 0: 1·[1,2,3] + 0·[4,5,6]; row 1: 2·[1,2,3] + 1·[4,5,6].
        assert_eq!(
            d.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1., 2., 3., 6., 9., 12.]
        );
    }

    /// The four factor suffixes classify to their halves, and everything else —
    /// `alpha` above all — is not a factor.
    #[test]
    fn a_factor_tensor_is_classified_by_its_suffix() {
        let m = "layers.0.attention.to_q";
        assert_eq!(
            classify("diffusion_model.layers.0.attention.to_q.lokr_w1"),
            Some((m, Part::KronW1))
        );
        assert_eq!(
            classify("diffusion_model.layers.0.attention.to_q.lokr_w2"),
            Some((m, Part::KronW2))
        );
        assert_eq!(
            classify("diffusion_model.layers.0.attention.to_q.lora_A.weight"),
            Some((m, Part::LowA))
        );
        assert_eq!(
            classify("diffusion_model.layers.0.attention.to_q.lora_B.weight"),
            Some((m, Part::LowB))
        );
        assert_eq!(
            classify("diffusion_model.layers.0.attention.to_q.alpha"),
            None
        );
        assert_eq!(classify("some_other_model.layers.0.to_q.lokr_w1"), None);
    }

    /// **A half without its other half is refused by name**, in both
    /// factorisations, and a module carrying both factorisations at once is
    /// refused rather than guessed at.
    #[test]
    fn a_module_missing_half_its_pair_is_refused() {
        let some = || Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let mut h = Halves::default();
        h.set(Part::KronW1, some());
        let err = h.finish("layers.0.attention.to_q").unwrap_err().to_string();
        assert!(err.contains("lokr_w2"), "{err}");

        let mut h = Halves::default();
        h.set(Part::LowB, some());
        let err = h.finish("layers.0.attention.to_q").unwrap_err().to_string();
        assert!(err.contains("lora_A"), "{err}");

        let mut h = Halves::default();
        h.set(Part::KronW1, some());
        h.set(Part::KronW2, some());
        h.set(Part::LowA, some());
        h.set(Part::LowB, some());
        let err = h.finish("layers.0.attention.to_q").unwrap_err().to_string();
        assert!(err.contains("both"), "{err}");
    }

    /// A complete LoRA pair loads end to end through the same path a file
    /// takes, and its delta reaches `deltas()` shaped like the module.
    #[test]
    fn a_lora_adapter_loads_and_produces_a_whole_target_delta() {
        let mut all = HashMap::new();
        all.insert(
            "diffusion_model.layers.3.feed_forward.w2.lora_A.weight".to_string(),
            t(&[1., 0., 0., 1.], (2, 2)),
        );
        all.insert(
            "diffusion_model.layers.3.feed_forward.w2.lora_B.weight".to_string(),
            t(&[2., 0., 0., 2.], (2, 2)),
        );
        let fusion = Fusion::from_tensors(all).unwrap();
        assert_eq!(fusion.len(), 1);
        let deltas = fusion.deltas(0.5).unwrap();
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].name, "layers.3.feed_forward.w2.weight");
        // B·A = 2·I, at strength 0.5 → I.
        assert_eq!(
            deltas[0]
                .delta
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1., 0., 0., 1.]
        );
    }

    /// The shapes the SDA adapter carries produce exactly the checkpoint's
    /// tensors. Cheap to state and it is the mapping's whole premise — a factor
    /// pair that does not multiply out to the target is a checkpoint that fails
    /// to load after a conversion that reported success.
    #[test]
    fn the_adapters_factor_shapes_rebuild_the_models_weights() {
        let s = |a: (usize, usize), b: (usize, usize)| {
            delta_shape(&Shape::from(a), &Shape::from(b)).unwrap()
        };
        // dim 3840, ffn hidden 10240, adaLN 4·dim by ADALN_EMBED_DIM.
        assert_eq!(
            s((8, 8), (480, 480)),
            (3840, 3840),
            "to_q / to_k / to_v / out"
        );
        assert_eq!(
            s((8, 8), (1280, 480)),
            (10240, 3840),
            "feed_forward.w1 / w3"
        );
        assert_eq!(s((8, 8), (480, 1280)), (3840, 10240), "feed_forward.w2");
        assert_eq!(s((8, 8), (1920, 32)), (15360, 256), "adaLN_modulation.0");
    }

    #[test]
    fn a_module_path_splits_into_a_layer_and_a_tail() {
        assert_eq!(
            split_layer("layers.12.attention.to_q"),
            Some((12, "attention.to_q"))
        );
        assert_eq!(
            split_layer("layers.0.feed_forward.w2"),
            Some((0, "feed_forward.w2"))
        );
        // The refiners are not `layers.N` and the adapters do not carry them;
        // anything that is not a numbered layer must not be guessed at.
        assert_eq!(split_layer("noise_refiner.0.attention.to_q"), None);
        assert_eq!(split_layer("layers.attention.to_q"), None);
    }

    /// **Every module an adapter carries has somewhere to go.**
    ///
    /// The table is exhaustive over the eight modules an ai-toolkit Z-Image
    /// adapter carries, and the three attention projections map onto the
    /// *one* fused `qkv` at distinct thirds. A duplicate slot would have two
    /// deltas overwrite each other.
    #[test]
    fn the_module_table_covers_the_adapter_and_the_qkv_thirds_are_distinct() {
        let names: Vec<&str> = MODULES.iter().map(|(a, _, _)| *a).collect();
        for expected in [
            "attention.to_q",
            "attention.to_k",
            "attention.to_v",
            "attention.to_out.0",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
            "adaLN_modulation.0",
        ] {
            assert!(names.contains(&expected), "`{expected}` has no target");
        }
        assert_eq!(
            MODULES.len(),
            8,
            "a new module needs a target, not a default"
        );

        let mut thirds: Vec<usize> = MODULES
            .iter()
            .filter_map(|(_, g, t)| match t {
                Target::Qkv(i) if *g == "attention.qkv" => Some(*i),
                _ => None,
            })
            .collect();
        thirds.sort_unstable();
        assert_eq!(
            thirds,
            vec![0, 1, 2],
            "the qkv thirds must be q, k, v exactly once"
        );
    }

    /// The three thirds stack in q/k/v order, which is the order
    /// `Attention::forward` narrows them in. Stacking them in any other order
    /// swaps two projections' adapters and produces a model that runs.
    #[test]
    fn the_qkv_thirds_stack_in_the_order_attention_splits_them() {
        let q = Tensor::full(1f32, (2, 2), &Device::Cpu).unwrap();
        let k = Tensor::full(2f32, (2, 2), &Device::Cpu).unwrap();
        let v = Tensor::full(3f32, (2, 2), &Device::Cpu).unwrap();
        let stacked = Tensor::cat(&[q, k, v], 0).unwrap();
        assert_eq!(
            stacked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.]
        );
        let by_slot: Vec<usize> = MODULES
            .iter()
            .filter_map(|(a, _, t)| match t {
                Target::Qkv(i) => Some((*i, *a)),
                _ => None,
            })
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .enumerate()
            .map(|(i, name)| {
                let want = ["attention.to_q", "attention.to_k", "attention.to_v"][i];
                assert_eq!(name, want, "third {i} is not {want}");
                i
            })
            .collect();
        assert_eq!(by_slot, vec![0, 1, 2]);
    }

    /// A delta must land on a weight of exactly its own shape.
    #[test]
    fn a_mismatched_delta_is_refused_rather_than_broadcast() {
        let w = Tensor::zeros((4, 6), DType::F32, &Device::Cpu).unwrap();
        assert!(apply(
            &w,
            &Tensor::zeros((4, 6), DType::F32, &Device::Cpu).unwrap()
        )
        .is_ok());
        // A shape that would broadcast is the dangerous one: candle would
        // happily add a row vector to every row.
        assert!(apply(
            &w,
            &Tensor::zeros((1, 6), DType::F32, &Device::Cpu).unwrap()
        )
        .is_err());
        assert!(apply(
            &w,
            &Tensor::zeros((6, 4), DType::F32, &Device::Cpu).unwrap()
        )
        .is_err());
    }

    #[test]
    fn the_relative_norm_is_the_ratio_of_the_two_frobenius_norms() {
        let w = t(&[3., 4., 0., 0.], (2, 2)); // ‖W‖ = 5
        let d = t(&[0., 0., 3., 4.], (2, 2)); // ‖Δ‖ = 5
        assert!((relative_norm(&w, &d).unwrap() - 1.0).abs() < 1e-6);
        let half = (d * 0.5).unwrap();
        assert!((relative_norm(&w, &half).unwrap() - 0.5).abs() < 1e-6);
    }

    /// **`alpha` is a sentinel here, not a multiplier.**
    ///
    /// `zit_sda_v1` stores `alpha = 1e10` on all 240 modules. LyCORIS computes
    /// `scale = alpha / lora_dim`, and a full-rank LoKr sets `lora_dim` to the
    /// same large number so the two cancel and the scale is 1 — the algorithm
    /// list says as much: *"alpha is ignored when using full dimension"*.
    ///
    /// This test exists because reading it as a multiplier is the single
    /// catastrophic mistake available in this file, and it fails in a way that
    /// looks like a broken model rather than a bad constant: every weight is
    /// scaled past f32's range on the first fused tensor.
    #[test]
    fn the_scale_is_one_for_a_full_rank_lokr_whatever_alpha_says() {
        let stored_alpha = 1e10f64;
        // What a consumer that trusted `alpha` blindly would apply, against what
        // a full-rank adapter actually means.
        let naive = stored_alpha;
        let correct = DEFAULT_STRENGTH;
        assert_eq!(correct, 1.0);
        assert!(
            naive / correct > 1e9,
            "if these were close the mistake would not matter"
        );
        // And a delta at the correct scale stays finite where the naive one
        // does not survive contact with a weight.
        let w = t(&[1., 1., 1., 1.], (2, 2));
        let d = t(&[0.1, -0.1, 0.05, 0.0], (2, 2));
        let fused = apply(&w, &(d.clone() * correct).unwrap()).unwrap();
        assert!(relative_norm(&w, &(d * naive).unwrap()).unwrap() > 1e8);
        assert!(relative_norm(&w, &fused.sub(&w).unwrap()).unwrap() < 1.0);
    }

    /// Strength scales the delta and nothing else — 0 is the stock checkpoint.
    #[test]
    fn strength_scales_the_delta_and_zero_is_the_original_weight() {
        let w = t(&[1., 2., 3., 4.], (2, 2));
        let d = t(&[10., 10., 10., 10.], (2, 2));
        let at = |s: f64| {
            apply(&w, &(d.clone() * s).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_eq!(at(0.0), vec![1., 2., 3., 4.]);
        assert_eq!(at(0.7), vec![8., 9., 10., 11.]);
        assert_eq!(at(1.0), vec![11., 12., 13., 14.]);
    }
}
