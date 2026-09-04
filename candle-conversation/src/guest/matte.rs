//! The matte guest: a salient-object network that says what the subject is.
//!
//! # What this replaced, and why
//!
//! Removing a background used to be a colour algorithm here — flood-fill the
//! backdrop in from the frame, fit a plane through it, refine the edge. It is a
//! reasonable thing to try and it is the wrong *shape* of answer, because it
//! cannot see a subject. All it knows is "connected to the edge and close to
//! the backdrop", so it refuses every picture with a real background in it, and
//! on the ones it accepts any subject whose colour approaches the wall's
//! defeats it. Measured on this daemon's own portraits: a light-grey knit came
//! within 24 of the wall behind it while the wall's own gradient spread 22.7.
//! Two overlapping distributions — no threshold separates them.
//!
//! Nobody solves this with colour. The field runs a network that emits a soft
//! alpha matte directly, and has since U²-Net in 2020. There is no trimap, no
//! refinement pass, and no green screen: the network looks at the picture and
//! says how opaque each pixel is.
//!
//! # Why the graph is ONNX and the weights still land in ground
//!
//! The published weights are somebody else's exported graph, and the export
//! folds batch-norm into the convolutions and renames every tensor to
//! `onnx::Conv_1896`. Rebuilding the architecture in candle would mean mapping
//! 238 anonymous tensors onto hand-written modules by position — a silent,
//! expensive mistake waiting to happen — so the graph is evaluated as it
//! shipped, through [`candle_onnx`].
//!
//! **That changes nothing about where the weights live.** They are placed in
//! span ground through [`place_bytes`], the same primitive
//! [`super::varground::GroundVars`] uses for the image guest's autoencoder, for
//! the same reason: the reservation is the budget, and a model's worth of pool
//! allocations is the largest competitor for the card the engine has. What
//! `GroundVars` adds on top — a `VarBuilder` backend — is only how *named*
//! tensors get requested, and an ONNX graph has no `VarBuilder` to be a backend
//! for. The placement underneath is shared.
//!
//! # Why it is a guest at all
//!
//! Because the alternative is a CPU. Measured on this box: IS-Net at 1024²
//! takes **11.9 s** on the CPU and **0.23 s** on the card, and the two agree to
//! within 1 of 255 on every alpha. A drain does stop the world — but for a
//! quarter of a second, against the ten to twenty-five an image draw already
//! costs, and the guest system is exactly the machinery for borrowing the card
//! between waves.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle::{DType, Device, Tensor};
use candle_onnx::onnx::ModelProto;
use image::imageops::FilterType;

use super::ground::GuestGround;
use super::model::GuestModel;
use super::progress::{GuestEvent, GuestSink};
use super::varground::place_bytes;
use super::work::{Guest, GuestMatte, GuestOutcome, GuestRequest, MatteRequest};

/// Which published network this is, and therefore how it wants its input.
///
/// **The weights encode an expectation about their input's distribution**, and
/// the two families disagree about it. Feeding one the other's statistics is
/// not a worse matte, it is a different function — measured here, IS-Net given
/// ImageNet's numbers returned a portrait with holes punched through the hair
/// and a green-screen shot that was 100% transparent, which reads as a broken
/// model rather than as a wrong constant.
///
/// So the family travels with the path in deployment config. There is no
/// sniffing it from the file name: a mis-detected family fails exactly the way
/// above, silently and expensively.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// IS-Net / DIS — centre on 0.5, do not rescale. The default, and what
    /// `isnet-general-use` wants.
    #[default]
    IsNet,
    /// U²-Net and its `silueta` relatives — ImageNet channel statistics.
    U2Net,
}

impl Family {
    fn mean(self) -> [f32; 3] {
        match self {
            Self::IsNet => [0.5, 0.5, 0.5],
            Self::U2Net => [0.485, 0.456, 0.406],
        }
    }

    fn std(self) -> [f32; 3] {
        match self {
            Self::IsNet => [1.0, 1.0, 1.0],
            Self::U2Net => [0.229, 0.224, 0.225],
        }
    }
}

/// Where the matte guest's checkpoint is, and what it is.
#[derive(Clone, Debug, PartialEq)]
pub struct MatteSpec {
    /// The exported graph.
    pub model: PathBuf,
    pub family: Family,
}

impl MatteSpec {
    pub fn new(model: impl Into<PathBuf>, family: Family) -> Self {
        Self {
            model: model.into(),
            family,
        }
    }
}

/// The guest. Holds no device memory until [`GuestModel::load`].
pub struct MatteGuest {
    spec: MatteSpec,
    loaded: Option<Loaded>,
}

struct Loaded {
    device: Device,
    graph: Arc<ModelProto>,
    /// The graph's weights, already on the card and viewing ground.
    weights: std::collections::HashMap<String, Tensor>,
    input: String,
    output: String,
    side: usize,
    family: Family,
}

impl MatteGuest {
    pub fn new(spec: MatteSpec) -> Self {
        Self { spec, loaded: None }
    }

    /// The graph, from the process-wide cache.
    ///
    /// **Not a field on this struct**, which is the point: a guest is rebuilt
    /// per drain by design, so a graph parsed in the constructor is 178 MB of
    /// protobuf decoded every time the card is borrowed. [`super::checkpoint`]
    /// is where artefacts that outlive a drain belong.
    fn graph(&self) -> Result<Arc<ModelProto>, String> {
        super::checkpoint::onnx(&self.spec.model)
    }
}

/// The side the graph declares it runs at.
///
/// Read rather than assumed, and required rather than defaulted: these exports
/// bake constant sizes into the decoder's resizes, so a graph run at any other
/// size fails partway through with a shape mismatch in a concatenation. A
/// missing declaration means nobody can know the right size, which is worth
/// saying at load rather than discovering per request.
fn declared_side(graph: &ModelProto) -> Result<(String, String, usize), String> {
    use candle_onnx::onnx::{tensor_shape_proto::dimension::Value as DimVal, type_proto};
    let g = graph
        .graph
        .as_ref()
        .ok_or_else(|| "matte guest: the file carries no graph".to_string())?;
    let input = g
        .input
        .first()
        .ok_or_else(|| "matte guest: the graph declares no input".to_string())?;
    let output = g
        .output
        .first()
        .ok_or_else(|| "matte guest: the graph declares no output".to_string())?
        .name
        .clone();
    let dims = input
        .r#type
        .as_ref()
        .and_then(|t| match &t.value {
            Some(type_proto::Value::TensorType(t)) => t.shape.as_ref(),
            _ => None,
        })
        .map(|s| &s.dim)
        .ok_or_else(|| {
            "matte guest: the graph does not declare a fixed input side, so the resize would be \
             a guess"
                .to_string()
        })?;
    if dims.len() != 4 {
        return Err(format!(
            "matte guest: the graph's input is rank {}, not [N, C, H, W]",
            dims.len()
        ));
    }
    let fixed = |i: usize| match dims[i].value {
        Some(DimVal::DimValue(v)) if v > 0 => Some(v as usize),
        _ => None,
    };
    // **Both spatial axes, and they must agree.** Reading only the last one and
    // using it for both silently squashes a non-square graph into a square —
    // which produces a matte for a picture nobody asked about, at no point
    // failing. Every export this runs is square; the check is what makes that
    // a fact rather than an assumption.
    let (h, w) = (fixed(2), fixed(3));
    match (h, w) {
        (Some(h), Some(w)) if h == w => Ok((input.name.clone(), output, h)),
        (Some(h), Some(w)) => Err(format!(
            "matte guest: the graph runs at {h}×{w}, and this resizes to a square — a \
             non-square field needs the fit to be written before it can be used"
        )),
        _ => Err(
            "matte guest: the graph does not declare a fixed input size, so the resize \
                  would be a guess"
                .to_string(),
        ),
    }
}

/// Bytes the graph's weights occupy.
fn weight_bytes(graph: &ModelProto) -> usize {
    graph.graph.as_ref().map_or(0, |g| {
        g.initializer
            .iter()
            .map(|t| {
                let elems = t.dims.iter().fold(1usize, |a, d| a * (*d).max(0) as usize);
                // `raw_data` when the export used it, and four bytes an element
                // otherwise — which is the width every weight in these graphs
                // is actually stored at.
                t.raw_data.len().max(elems * 4)
            })
            .sum()
    })
}

/// Ground for one matte's activations, at the graph's own side.
///
/// A headroom figure rather than a claim, like the image guest's: these come
/// from the CUDA pool, so what it sizes is the room the drain leaves by
/// evicting.
///
/// **A kilobyte per input pixel**, which sounds extravagant and is not. The
/// network's widest stage is its first — 64 channels at the full side, so 256
/// bytes a pixel for *one* tensor in f32 — and a U-shaped decoder holds several
/// live at once across a skip connection. Over-claiming costs the engine an
/// eviction it did not need; under-claiming is an out-of-memory discovered
/// after the eviction has already happened, which is the worse of the two.
const HEADROOM_PER_PIXEL: usize = 1024;

fn activation_headroom(side: usize) -> usize {
    (side * side * HEADROOM_PER_PIXEL).max(256 << 20)
}

impl GuestModel for MatteGuest {
    fn guest(&self) -> Guest {
        Guest::Matte
    }

    fn footprint_bytes(&self, _jobs: &[GuestRequest]) -> usize {
        // Sized from the graph itself, which the cache has after the first
        // drain and reads once before it. A claim that is short fails at the
        // last tensor, after the engine has already been evicted for it.
        //
        // The fallback is the file's own length: the same order as its weights,
        // and the only honest guess when the graph cannot be read at all —
        // whereupon `load` will say so properly a moment later.
        match self.graph() {
            Ok(g) => {
                let side = declared_side(&g).map_or(1024, |(_, _, s)| s);
                weight_bytes(&g) + activation_headroom(side)
            }
            Err(_) => {
                let weights = std::fs::metadata(&self.spec.model).map_or(0, |m| m.len() as usize);
                weights + activation_headroom(1024)
            }
        }
    }

    fn load(
        &mut self,
        device: &Device,
        ground: &Arc<Mutex<GuestGround>>,
        _jobs: &[GuestRequest],
    ) -> Result<(), String> {
        let graph = self.graph()?;
        let (input, output, side) = declared_side(&graph)?;

        // **Every weight into ground, none into the pool.** The same primitive
        // the image guest's autoencoder goes through; see [`place_bytes`].
        let mut weights = std::collections::HashMap::new();
        {
            let mut g = ground
                .lock()
                .map_err(|_| "matte guest: the placement lock was poisoned".to_string())?;
            let proto = graph
                .graph
                .as_ref()
                .ok_or_else(|| "matte guest: no graph".to_string())?;
            for t in proto.initializer.iter() {
                let host = candle_onnx::get_tensor(t, &t.name)
                    .map_err(|e| format!("matte guest: reading weight {}: {e}", t.name))?;
                let host = host
                    .contiguous()
                    .map_err(|e| format!("matte guest: {}: {e}", t.name))?;
                let dtype = host.dtype();
                let shape = host.shape().clone();
                let bytes =
                    raw_bytes(&host).map_err(|e| format!("matte guest: {}: {e}", t.name))?;
                let (placed, _) = place_bytes(device, &mut g, &bytes, dtype, shape)
                    .map_err(|e| format!("matte guest: placing {}: {e}", t.name))?;
                weights.insert(t.name.clone(), placed);
            }
        }

        // **One barrier for the whole load**, for the reason `GroundVars::with`
        // documents: every copy above read a host buffer that dies here, so
        // they must all have landed before anything reads the weights.
        if let Device::Cuda(cuda) = device {
            cuda.cuda_stream()
                .synchronize()
                .map_err(|e| format!("matte guest: {e}"))?;
        }

        tracing::debug!(
            target: "candle_conversation::guest",
            tensors = weights.len(),
            side,
            family = ?self.spec.family,
            "matte guest: weights placed"
        );

        self.loaded = Some(Loaded {
            device: device.clone(),
            graph,
            weights,
            input,
            output,
            side,
            family: self.spec.family,
        });
        Ok(())
    }

    fn run(&mut self, request: &GuestRequest, sink: &GuestSink) -> Result<GuestOutcome, String> {
        let GuestRequest::Matte(r) = request else {
            return Err(format!(
                "the matte guest was handed a {} job — the queue routed by kind and should not \
                 have",
                request.guest()
            ));
        };
        let loaded = self
            .loaded
            .as_mut()
            .ok_or_else(|| "the matte guest was asked to run before it loaded".to_string())?;
        loaded.cut(r, sink).map_err(|e| e.to_string())
    }

    fn unload(&mut self) {
        // Every weight views ground, and the drain hands it back the moment
        // this returns. The parsed graph is host memory and stays.
        self.loaded = None;
    }
}

impl Loaded {
    /// One picture in, one matted picture out.
    ///
    /// A matte is a single forward, so there is no interior count to report —
    /// the two units are the network and the encode, which is what a watcher
    /// sees.
    fn cut(&self, r: &MatteRequest, sink: &GuestSink) -> candle::Result<GuestOutcome> {
        let (w, h) = (r.width, r.height);
        sink.emit(GuestEvent::Step {
            done: 0,
            total: 2,
            what: "reading the picture",
        });

        // **Stretched to the network's square, not letterboxed.** The alpha is
        // stretched back the same way, so the geometry round-trips exactly;
        // padding would put bars inside the network's field of view and spend
        // part of its resolution describing them.
        // Borrowed, not cloned: the resize only reads, and the caller's buffer
        // is three quarters of a megabyte at 512² that would otherwise be
        // copied once per request for nothing.
        let src: image::ImageBuffer<image::Rgb<u8>, &[u8]> =
            image::ImageBuffer::from_raw(w, h, r.pixels.as_slice()).ok_or_else(|| {
                candle::Error::Msg("matte: the picture is not w*h*3 bytes".into())
            })?;
        let side = self.side as u32;
        let small = image::imageops::resize(&src, side, side, FilterType::Triangle);

        let n = self.side * self.side;
        let (mean, std) = (self.family.mean(), self.family.std());
        let mut chw = vec![0f32; 3 * n];
        for (i, px) in small.pixels().enumerate() {
            for c in 0..3 {
                chw[c * n + i] = (px[c] as f32 / 255.0 - mean[c]) / std[c];
            }
        }
        let input = Tensor::from_vec(chw, (1, 3, self.side, self.side), &self.device)?;

        // The placed weights are handed in rather than re-read from the proto,
        // which is what keeps them in ground: `simple_eval` would otherwise
        // parse its own copy onto the host and move it to the pool.
        let mut values = self.weights.clone();
        values.insert(self.input.clone(), input);
        let outputs = candle_onnx::simple_eval_on(&self.graph, values, &self.device)?;
        let out = outputs.get(&self.output).ok_or_else(|| {
            candle::Error::Msg(format!(
                "matte: the graph did not produce `{}`",
                self.output
            ))
        })?;

        sink.emit(GuestEvent::Step {
            done: 1,
            total: 2,
            what: "cutting out",
        });

        let mut flat = out.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        if flat.len() < n {
            candle::bail!(
                "matte: the graph produced {} values for a {}×{} field",
                flat.len(),
                self.side,
                self.side
            );
        }
        flat.truncate(n);
        normalise(&mut flat);

        let grey: image::GrayImage = image::ImageBuffer::from_raw(
            side,
            side,
            flat.iter().map(|a| (a * 255.0).round() as u8).collect(),
        )
        .ok_or_else(|| candle::Error::Msg("matte: the alpha is not S*S values".into()))?;
        // Lanczos on the way back up: this is an alpha channel and its edges
        // are the whole product, so a cheaper filter's stair-stepping would be
        // visible exactly where it matters.
        let alpha = image::imageops::resize(&grey, w, h, FilterType::Lanczos3);

        let mut rgba = vec![0u8; (w * h) as usize * 4];
        let mut lifted = 0usize;
        for (i, a) in alpha.as_raw().iter().enumerate() {
            rgba[i * 4] = r.pixels[i * 3];
            rgba[i * 4 + 1] = r.pixels[i * 3 + 1];
            rgba[i * 4 + 2] = r.pixels[i * 3 + 2];
            rgba[i * 4 + 3] = *a;
            if *a < 128 {
                lifted += 1;
            }
        }
        let png = encode_rgba(&rgba, w, h)?;
        Ok(GuestOutcome::Matte(GuestMatte {
            width: w,
            height: h,
            png,
            lifted: lifted as f32 / (w * h) as f32,
        }))
    }
}

/// How the network's own output becomes an alpha channel.
///
/// The published graphs emit **unbounded logits**, not probabilities — the
/// reference implementations min–max normalise the result rather than applying
/// a sigmoid, and doing it the other way produces a washed matte where
/// everything is faintly opaque. Normalising per picture is also what makes one
/// threshold work across a dark subject and a bright one.
fn normalise(v: &mut [f32]) {
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for x in v.iter() {
        if x.is_finite() {
            lo = lo.min(*x);
            hi = hi.max(*x);
        }
    }
    let span = hi - lo;
    if !span.is_finite() || span <= f32::EPSILON {
        // A constant output carries no matte. Opaque is the safe reading: the
        // caller gets their picture back rather than an empty frame.
        v.iter_mut().for_each(|x| *x = 1.0);
        return;
    }
    for x in v.iter_mut() {
        *x = ((*x - lo) / span).clamp(0.0, 1.0);
    }
}

/// A contiguous host tensor's raw bytes.
fn raw_bytes(t: &Tensor) -> candle::Result<Vec<u8>> {
    let flat = t.flatten_all()?;
    fn of<T: Copy>(v: &[T]) -> Vec<u8> {
        // SAFETY: `T` is a plain numeric type with no padding and no niches.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
            .to_vec()
    }
    Ok(match t.dtype() {
        DType::F32 => of(&flat.to_vec1::<f32>()?),
        DType::F16 => of(&flat.to_vec1::<half::f16>()?),
        DType::BF16 => of(&flat.to_vec1::<half::bf16>()?),
        DType::U8 => flat.to_vec1::<u8>()?,
        DType::U32 => of(&flat.to_vec1::<u32>()?),
        DType::I64 => of(&flat.to_vec1::<i64>()?),
        other => candle::bail!("matte: {other:?} has no host byte form here"),
    })
}

/// RGBA8 to PNG bytes.
///
/// **Encoded here, before the drain hands the ground back**, for the reason
/// [`super::image`]'s encoder is: what the caller gets has to be bytes rather
/// than a view of memory the KV side owns again the moment the drain ends.
fn encode_rgba(rgba: &[u8], w: u32, h: u32) -> candle::Result<Vec<u8>> {
    let buf: image::RgbaImage = image::ImageBuffer::from_raw(w, h, rgba.to_vec())
        .ok_or_else(|| candle::Error::Msg("matte: the buffer is not w*h*4 bytes".into()))?;
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(buf)
        .write_to(&mut out, image::ImageFormat::Png)
        .map_err(|e| candle::Error::Msg(format!("matte: encoding the PNG: {e}")))?;
    Ok(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The output is min–max normalised, not sigmoided.** Getting this wrong
    /// does not fail — it produces a matte where everything is faintly opaque,
    /// which reads as a model that does not work rather than as a bug here.
    #[test]
    fn the_matte_is_normalised_across_its_own_range() {
        let mut v = vec![-4.0, -2.0, 0.0, 2.0, 4.0];
        normalise(&mut v);
        assert_eq!(v, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    /// A network that answered one value everywhere carries no matte, and the
    /// safe reading is opaque: the caller gets their picture back rather than
    /// an empty frame.
    #[test]
    fn a_flat_output_is_read_as_fully_opaque() {
        let mut v = vec![0.7; 8];
        normalise(&mut v);
        assert_eq!(v, vec![1.0; 8]);
        let mut nan = vec![f32::NAN; 4];
        normalise(&mut nan);
        assert_eq!(nan, vec![1.0; 4]);
    }

    /// **The two families' statistics are different and must stay so.** Feeding
    /// one the other's is the failure that looks like a broken model: measured,
    /// IS-Net given ImageNet's numbers punched holes through a portrait's hair.
    #[test]
    fn the_families_normalise_differently() {
        assert_eq!(Family::IsNet.mean(), [0.5, 0.5, 0.5]);
        assert_eq!(Family::IsNet.std(), [1.0, 1.0, 1.0]);
        assert_ne!(Family::U2Net.mean(), Family::IsNet.mean());
        assert_ne!(Family::U2Net.std(), Family::IsNet.std());
        assert_eq!(Family::default(), Family::IsNet);
    }

    /// **A weight the caller supplied is the one the graph uses.**
    ///
    /// The whole reason this guest can put its model in span ground: if the
    /// runtime re-read the proto's initializers over the top, the ground copy
    /// would never be read, the parse would happen per evaluation, and the
    /// model would end up in the CUDA pool — which is the engine's. Nothing
    /// fails when that goes wrong, which is exactly why it is pinned here.
    ///
    /// Asserted on a hand-built graph rather than the real one so it runs
    /// without a checkpoint or a card: one node that returns its initializer.
    #[test]
    fn a_supplied_weight_is_not_overwritten_by_the_graphs_own() {
        use candle_onnx::onnx;

        let w = onnx::TensorProto {
            name: "w".into(),
            data_type: onnx::tensor_proto::DataType::Float as i32,
            dims: vec![2],
            float_data: vec![1.0, 2.0],
            ..Default::default()
        };
        let graph = onnx::GraphProto {
            node: vec![onnx::NodeProto {
                name: "id".into(),
                op_type: "Identity".into(),
                input: vec!["w".into()],
                output: vec!["out".into()],
                ..Default::default()
            }],
            initializer: vec![w],
            output: vec![onnx::ValueInfoProto {
                name: "out".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let model = onnx::ModelProto {
            graph: Some(graph),
            ..Default::default()
        };

        // Nothing supplied: the graph's own weight is used.
        let got = candle_onnx::simple_eval(&model, std::collections::HashMap::new()).unwrap();
        assert_eq!(
            got["out"].to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0],
            "the graph's own initializer was not used"
        );

        // Supplied: the caller's wins, untouched.
        let mine = Tensor::from_vec(vec![7f32, 9.], 2, &Device::Cpu).unwrap();
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("w".to_string(), mine);
        let got = candle_onnx::simple_eval(&model, inputs).unwrap();
        assert_eq!(
            got["out"].to_vec1::<f32>().unwrap(),
            vec![7.0, 9.0],
            "the runtime overwrote a weight the caller had already placed"
        );
    }

    /// **A non-square graph is refused rather than squashed.**
    ///
    /// The declared side drives a square resize on the way in and the inverse
    /// on the way out. Reading only the last dimension and using it for both —
    /// which this did — would run a 1024×512 network over a square field and
    /// return a matte for a picture nobody asked about, without failing
    /// anywhere. Every export in use is square; this is what makes that a fact.
    #[test]
    fn a_graph_that_is_not_square_is_refused_rather_than_squashed() {
        use candle_onnx::onnx;

        let input = |h: i64, w: i64| onnx::ValueInfoProto {
            name: "x".into(),
            r#type: Some(onnx::TypeProto {
                value: Some(onnx::type_proto::Value::TensorType(
                    onnx::type_proto::Tensor {
                        elem_type: onnx::tensor_proto::DataType::Float as i32,
                        shape: Some(onnx::TensorShapeProto {
                            dim: [1, 3, h, w]
                                .into_iter()
                                .map(|v| onnx::tensor_shape_proto::Dimension {
                                    value: Some(
                                        onnx::tensor_shape_proto::dimension::Value::DimValue(v),
                                    ),
                                    ..Default::default()
                                })
                                .collect(),
                        }),
                    },
                )),
                ..Default::default()
            }),
            ..Default::default()
        };
        let model = |h: i64, w: i64| onnx::ModelProto {
            graph: Some(onnx::GraphProto {
                input: vec![input(h, w)],
                output: vec![onnx::ValueInfoProto {
                    name: "y".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let (i, o, side) = declared_side(&model(1024, 1024)).expect("a square graph");
        assert_eq!((i.as_str(), o.as_str(), side), ("x", "y", 1024));

        let e = declared_side(&model(1024, 512)).unwrap_err();
        assert!(e.contains("1024×512"), "{e}");

        // A symbolic batch is fine; a symbolic *side* is not, because the
        // resize would then have nothing to resize to.
        let e = declared_side(&model(0, 0)).unwrap_err();
        assert!(e.contains("fixed input size"), "{e}");
    }

    /// The headroom has a floor and grows with the field the network runs over.
    #[test]
    fn the_headroom_scales_with_the_side_and_has_a_floor() {
        // 320² at a kilobyte a pixel is 100 MiB, under the floor.
        assert_eq!(
            activation_headroom(320),
            256 << 20,
            "a small field gets the floor"
        );
        // 1024² is a gibibyte, well clear of it, and four times 512²'s claim.
        assert_eq!(activation_headroom(1024), 1 << 30);
        assert_eq!(
            activation_headroom(1024),
            4 * activation_headroom(512),
            "the claim must follow the area, not the side"
        );
    }
}
