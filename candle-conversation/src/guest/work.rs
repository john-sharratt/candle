//! What a caller asks a guest model for, and what comes back.
//!
//! Deliberately plain data with no device types in it. A request crosses a
//! channel from an HTTP handler to the scheduler thread and an outcome crosses
//! back, so both have to be `Send` and neither may hold anything that borrows
//! the engine.

use serde::{Deserialize, Serialize};

/// The largest image a caller may ask for, per side.
///
/// A guest's ground is claimed from the KV side, so an unbounded latent is a
/// caller deciding how much of the engine's working set to evict. 2048 is four
/// times the models' native tile and already an eviction the scheduler will
/// report; past that the ask should be refused rather than served slowly.
pub const MAX_IMAGE_SIDE: u32 = 2048;

/// The largest number of denoising steps a caller may ask for.
///
/// Normal inference is blocked for the whole drain, so a step count is a
/// caller choosing how long every character in the world stops thinking for.
pub const MAX_IMAGE_STEPS: u32 = 100;

/// Which adapter-fused transformer a draw runs on.
///
/// # A LoRA here is a checkpoint, not a runtime patch
///
/// The image guest reloads its weights **every drain** — evicted, placed into
/// ground, dropped — so "swap the LoRA" is not a special operation, it is the
/// load the drain was going to do anyway pointed at a different file. Each
/// variant names a GGUF with the adapter already fused in
/// (`candle-examples/examples/z-image-fuse`), which is why choosing one costs
/// nothing per step: the denoise loop runs the same int8 KO path whatever was
/// chosen, and there is no bf16 side-path competing with it.
///
/// The alternative — applying `B(Ax)` per projection per step, the way the
/// LLM's [`candle_transformers::models::lora`] does — exists for the LLM
/// because *its* base stays resident and serves adapted and unadapted sessions
/// at once. The image guest has no resident base to share: every drain starts
/// from the file, so fusing ahead of time buys the same choice for free.
///
/// # Why an enum and not a path
///
/// A request that named a file would let any caller load arbitrary bytes into
/// guest ground. The variants are the deployment's **curated set**: each maps
/// to a path in `guests.yaml`, and a variant the deployment has not configured
/// is a named refusal rather than a guess.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ImageLora {
    /// The deployment's standing transformer — on this estate, the checkpoint
    /// with the SDA diversity adapter fused in. The default, so every caller
    /// that does not ask draws exactly what it drew before this enum existed.
    #[default]
    Diversity,
    /// An alternative checkpoint the deployment may provide. What is fused
    /// into it is the deployment's business — the enum only names the slot.
    Restricted,
}

impl ImageLora {
    /// The name a config file or a log line uses — the serde name, stated once.
    pub fn label(self) -> &'static str {
        match self {
            Self::Diversity => "diversity",
            Self::Restricted => "restricted",
        }
    }
}

impl std::fmt::Display for ImageLora {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// A picture the draw starts from, instead of starting from pure noise.
///
/// # What it is, stated exactly
///
/// This is **SDEdit**, not instruction editing. The reference is encoded to a
/// latent, noise is mixed into it at the strength [`Self::hold`] names, and the
/// walk joins the schedule there and finishes it. So the model is not told
/// "change this picture"; it is handed a picture that has been partly dissolved
/// and asked to complete it under the prompt.
///
/// What that buys is composition, pose, palette and framing. What it does not
/// buy is a targeted edit — "give the man on the left a hat" is not a thing
/// this can be asked, and any implementation that appears to do it is doing it
/// by luck. That capability is reference conditioning, which needs weights this
/// deployment does not have.
///
/// # Why the pixels arrive already fitted
///
/// The caller hands over RGB8 at exactly the draw's own size. Fitting an upload
/// — decoding whatever format it arrived in, refusing a bomb, cropping to the
/// output's aspect — is boundary work with its own failure modes, and doing it
/// here would mean a malformed upload was discovered *inside a drain*, after
/// the engine had been evicted for it. The invariant is checked by
/// [`GuestRequest::check`] at submission instead.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ImageReference {
    /// RGB8, exactly `width · height · 3` bytes, already fitted to the draw.
    pub pixels: Vec<u8>,
    /// How much of the picture survives, `0.0 ..= `[`MAX_REFERENCE_HOLD`].
    ///
    /// **A fraction of the schedule, not of sigma.** It is the share of the
    /// walk the reference is allowed to skip, so it is also the share of the
    /// step budget that is not spent — a hold of 0.25 runs three quarters of
    /// the steps asked for. The sigma that corresponds to is whatever the
    /// shift maps it to, which is very much not `1 − hold`; see
    /// [`candle_transformers::models::z_image::sampling::sigma_at`] for the
    /// measurement behind that.
    ///
    /// At `0.0` the reference contributes nothing and the draw is an ordinary
    /// one — deliberately continuous, so the dial has no special case at its
    /// bottom end.
    pub hold: f32,
}

/// What a reference holds when the caller does not say.
///
/// **Measured on Z-Image-Turbo rather than guessed**, by holding a portrait and
/// asking for an entirely different subject. At 0.25 the framing, scale and
/// ground carry across and the prompt still decides who is in the picture; by
/// 0.30 the reference's own subject is already contesting it, and past about
/// 0.45 the words can no longer change the face at all. The knee is that sharp,
/// so the default sits deliberately on the side where the prompt still wins —
/// somebody who wants more of their picture back slides up and sees it happen,
/// whereas a default past the knee reads as a prompt box that stopped working.
pub const DEFAULT_REFERENCE_HOLD: f32 = 0.25;

/// The most a reference may hold.
///
/// Not 1.0: at a full hold the walk has no distance to cover, so the draw is an
/// expensive way to be handed back the picture that was uploaded. Refusing it
/// is kinder than serving it, because the drain costs the whole estate its
/// thinking either way.
pub const MAX_REFERENCE_HOLD: f32 = 0.95;

/// The bounds a caller may move the schedule's shift within.
///
/// 1.0 is the un-shifted schedule — every step the same size in sigma — and the
/// deployment's default is 3.0, the value the model's step count is distilled
/// against. Past 8 the first step covers so much of the trajectory that the
/// rest have nothing left to decide.
pub const MIN_SHIFT: f64 = 1.0;
pub const MAX_SHIFT: f64 = 8.0;

/// An image to generate.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ImageRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    /// `None` draws a fresh seed. Pinning one makes a job reproducible, which
    /// is what an operator iterating on a prompt wants.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Which fused transformer to draw with. Defaults to the deployment's
    /// standing one, so the field is invisible to every caller that predates it.
    #[serde(default)]
    pub lora: ImageLora,
    /// A picture to start from. `None` starts from noise, which is the ordinary
    /// draw.
    #[serde(default)]
    pub reference: Option<ImageReference>,
    /// The schedule's shift, or the deployment's own when `None`.
    ///
    /// The one genuine sampler dial this model has — see
    /// [`candle_transformers::models::z_image::sampling::sigmas`]. There is no
    /// guidance scale to offer beside it, and deliberately no fake one: Turbo
    /// is guidance-distilled and runs at `guidance_scale = 0.0`, so a dial
    /// claiming to weigh the prompt against an unconditioned branch would be
    /// weighing against a branch that is not computed.
    #[serde(default)]
    pub shift: Option<f64>,
}

/// A picture to separate from its background.
///
/// # What the guest actually does
///
/// Runs a **salient-object network** over the picture and returns the alpha it
/// emits — how opaque each pixel is. That is the whole of it: no colour
/// threshold, no flood fill, no assumption about the backdrop. The network was
/// trained to know what a subject is, which is the thing a colour algorithm
/// cannot be told.
///
/// # Why the pixels arrive raw
///
/// RGB8 at exactly `width · height · 3`, decoded at the boundary for the same
/// reason [`ImageReference`] is: a drain evicts the engine's whole working set
/// before the guest loads, so a truncated upload must not be discovered inside
/// one.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct MatteRequest {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// A picture and the alpha the network gave it.
#[derive(Clone, Debug, PartialEq)]
pub struct GuestMatte {
    pub width: u32,
    pub height: u32,
    /// PNG bytes, RGBA. Encoded on the scheduler thread before the ground goes
    /// back, for the reason [`GuestImage::png`] is.
    pub png: Vec<u8>,
    /// The fraction of the picture the matte made at least half transparent.
    ///
    /// Reported so a caller can say something true about a picture where
    /// almost nothing was lifted, rather than showing an unchanged image and
    /// leaving somebody to wonder whether the button worked.
    pub lifted: f32,
}

/// One unit of guest work.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GuestRequest {
    Image(ImageRequest),
    Matte(MatteRequest),
}

impl GuestRequest {
    /// Which guest serves this. Jobs are drained one guest at a time, so this
    /// is also the grouping key.
    pub fn guest(&self) -> Guest {
        match self {
            Self::Image(_) => Guest::Image,
            Self::Matte(_) => Guest::Matte,
        }
    }

    /// Refuse a request that cannot be served, before anything is evicted for
    /// it.
    ///
    /// Checked at submission rather than at the drain: an ask that will be
    /// refused must not first cost the engine its working set, and the caller
    /// gets a synchronous answer instead of a queue position.
    pub fn check(&self) -> Result<(), String> {
        match self {
            Self::Image(r) => {
                if r.prompt.trim().is_empty() {
                    return Err("an image request needs a prompt".into());
                }
                if r.width == 0 || r.height == 0 {
                    return Err("an image needs a non-zero width and height".into());
                }
                if r.width > MAX_IMAGE_SIDE || r.height > MAX_IMAGE_SIDE {
                    return Err(format!(
                        "{}x{} is past the {MAX_IMAGE_SIDE} limit per side — a guest's ground is \
                         evicted from the KV side, so the size is how much of the engine's \
                         working set this costs",
                        r.width, r.height
                    ));
                }
                // The latent is the image over 8 and the transformer patches it
                // by 2, so the side has to survive both divisions. A remainder
                // is not refused downstream — it is rounded away somewhere in
                // between, and the caller gets back an image of a size it did
                // not ask for with nothing saying so.
                if r.width % 16 != 0 || r.height % 16 != 0 {
                    return Err(format!(
                        "{}x{} is not a multiple of 16 — the latent is the image over 8 and the \
                         transformer patches that by 2, and a remainder is rounded away rather \
                         than refused",
                        r.width, r.height
                    ));
                }
                if r.steps == 0 {
                    return Err("an image needs at least one denoising step".into());
                }
                if r.steps > MAX_IMAGE_STEPS {
                    return Err(format!(
                        "{} steps is past the {MAX_IMAGE_STEPS} limit — normal inference is \
                         blocked for the whole drain, so this is how long every character stops \
                         thinking for",
                        r.steps
                    ));
                }
                if let Some(s) = r.shift {
                    if !s.is_finite() || !(MIN_SHIFT..=MAX_SHIFT).contains(&s) {
                        return Err(format!(
                            "a shift of {s} is outside {MIN_SHIFT}..={MAX_SHIFT} — it re-weights \
                             the schedule toward composition or toward finish, and outside this \
                             range one end of the walk gets no steps at all"
                        ));
                    }
                }
                if let Some(reference) = &r.reference {
                    // **The invariant that makes the guest simple.** The guest
                    // reshapes these bytes into a `[3, height, width]` tensor
                    // without re-deriving the size, so a length that does not
                    // match is the one error that would reach the card.
                    let want = r.width as usize * r.height as usize * 3;
                    if reference.pixels.len() != want {
                        return Err(format!(
                            "the reference is {} bytes and a {}x{} RGB picture is {want} — it is \
                             fitted to the draw before it is queued, so a mismatch here is a \
                             caller that skipped the fitting",
                            reference.pixels.len(),
                            r.width,
                            r.height
                        ));
                    }
                    let hold = reference.hold;
                    if !hold.is_finite() || !(0.0..=MAX_REFERENCE_HOLD).contains(&hold) {
                        return Err(format!(
                            "a reference hold of {hold} is outside 0..={MAX_REFERENCE_HOLD} — at a \
                             full hold the walk has no distance to cover and the draw would cost \
                             a drain to hand back the picture it was given"
                        ));
                    }
                }
                Ok(())
            }
            Self::Matte(r) => {
                if r.width == 0 || r.height == 0 {
                    return Err("a matte needs a picture with pixels in it".into());
                }
                if r.width > MAX_IMAGE_SIDE || r.height > MAX_IMAGE_SIDE {
                    return Err(format!(
                        "{}x{} is past the {MAX_IMAGE_SIDE} limit per side",
                        r.width, r.height
                    ));
                }
                // The guest reshapes these bytes without re-deriving the size,
                // so a mismatch is the one malformed matte that would reach the
                // card — after the engine had been evicted for it.
                let want = r.width as usize * r.height as usize * 3;
                if r.pixels.len() != want {
                    return Err(format!(
                        "the picture is {} bytes and {}x{} RGB is {want}",
                        r.pixels.len(),
                        r.width,
                        r.height
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Which co-resident model serves a request.
///
/// The drain works one guest at a time and completely: a guest is loaded, every
/// job queued for it runs, and it is unloaded before the next is considered.
/// Loading is the expensive half — weights across the PCIe link into ground the
/// engine has just evicted — so amortising it across the whole backlog is the
/// entire reason the queue exists rather than a load per job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Guest {
    Image,
    Matte,
}

impl Guest {
    pub const ALL: [Guest; 2] = [Guest::Image, Guest::Matte];

    pub fn label(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Matte => "matte",
        }
    }
}

impl std::fmt::Display for Guest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// A generated image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestImage {
    pub width: u32,
    pub height: u32,
    /// PNG bytes. Encoded on the scheduler thread before the ground is handed
    /// back, because the pixels live in a buffer the give-back invalidates.
    pub png: Vec<u8>,
    /// The seed actually used, whether the caller pinned one or not — so an
    /// operator who liked a draw can ask for it again.
    pub seed: u64,
}

/// What a finished job produced.
#[derive(Clone, Debug, PartialEq)]
pub enum GuestOutcome {
    Image(GuestImage),
    Matte(GuestMatte),
}

impl GuestOutcome {
    pub fn guest(&self) -> Guest {
        match self {
            Self::Image(_) => Guest::Image,
            Self::Matte(_) => Guest::Matte,
        }
    }
}

/// Why a job did not produce anything.
///
/// Distinct from `ConversationError` because none of these are the main
/// engine's failures, and a caller reading one needs to know whether to retry
/// (`NoRoom`), fix the ask (`Refused`), or give up (`Unavailable`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuestError {
    /// The ask itself is not servable. Carries what to change.
    Refused(String),
    /// The engine could not free enough ground for this guest, having evicted
    /// everything it is willing to. Retryable — a quieter moment may serve it.
    NoRoom { wanted_mib: u64, freed_mib: u64 },
    /// The guest is not configured on this deployment.
    Unavailable(Guest),
    /// The guest ran and failed.
    Failed(String),
    /// The daemon shut down with the job still queued.
    Abandoned,
}

impl std::fmt::Display for GuestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(why) => write!(f, "{why}"),
            Self::NoRoom {
                wanted_mib,
                freed_mib,
            } => write!(
                f,
                "not enough room for the guest: wanted {wanted_mib} MiB, freed {freed_mib} MiB \
                 after evicting everything sheddable — try again when the engine is quieter"
            ),
            Self::Unavailable(g) => write!(f, "no {g} guest is configured on this deployment"),
            Self::Failed(e) => write!(f, "the guest failed: {e}"),
            Self::Abandoned => f.write_str("the daemon shut down before this job ran"),
        }
    }
}

impl std::error::Error for GuestError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32, steps: u32) -> GuestRequest {
        GuestRequest::Image(ImageRequest {
            prompt: "a lantern in the rain".into(),
            width,
            height,
            steps,
            seed: None,
            lora: ImageLora::default(),
            reference: None,
            shift: None,
        })
    }

    /// An image request carrying a correctly-sized reference at `hold`.
    fn with_reference(width: u32, height: u32, hold: f32) -> GuestRequest {
        let mut r = match image(width, height, 12) {
            GuestRequest::Image(r) => r,
            _ => unreachable!(),
        };
        r.reference = Some(ImageReference {
            pixels: vec![128u8; width as usize * height as usize * 3],
            hold,
        });
        GuestRequest::Image(r)
    }

    #[test]
    fn an_ordinary_ask_is_accepted() {
        assert!(image(512, 512, 20).check().is_ok());
    }

    /// **A request that does not mention a LoRA is the request it always was.**
    ///
    /// The field defaults, so every caller written before it existed — the
    /// portrait route, the console, anything scripted against the API — keeps
    /// drawing on the deployment's standing checkpoint without knowing the
    /// choice exists. A default that pointed anywhere else would silently move
    /// every one of them onto weights nobody chose.
    #[test]
    fn a_request_without_a_lora_takes_the_standing_checkpoint() {
        let r: ImageRequest =
            serde_json::from_str(r#"{"prompt":"a lantern","width":512,"height":512,"steps":8}"#)
                .unwrap();
        assert_eq!(r.lora, ImageLora::Diversity);
        assert_eq!(ImageLora::default(), ImageLora::Diversity);
    }

    /// The wire names are the labels, exactly — a config file, a request body
    /// and a log line all spell one variant one way.
    #[test]
    fn a_lora_round_trips_by_its_label() {
        for (lora, name) in [
            (ImageLora::Diversity, "\"diversity\""),
            (ImageLora::Restricted, "\"restricted\""),
        ] {
            assert_eq!(serde_json::to_string(&lora).unwrap(), name);
            assert_eq!(serde_json::from_str::<ImageLora>(name).unwrap(), lora);
            assert_eq!(format!("\"{lora}\""), name, "Display and serde disagree");
        }
        // An unknown variant is refused, not defaulted: a caller who asked for
        // an adapter this deployment never heard of must not silently draw on
        // a different one.
        assert!(serde_json::from_str::<ImageLora>("\"sepia\"").is_err());
    }

    /// **A reference of the wrong length is refused at submission.** The guest
    /// reshapes these bytes into `[3, height, width]` without re-deriving the
    /// size from them, so this is the one malformed reference that would reach
    /// the card — and it would reach it *after* the engine had been evicted.
    #[test]
    fn a_reference_that_is_not_the_draws_own_size_is_refused() {
        assert!(with_reference(512, 512, 0.45).check().is_ok());

        let mut r = match with_reference(512, 512, 0.45) {
            GuestRequest::Image(r) => r,
            _ => unreachable!(),
        };
        // One row short — the shape that would reshape into something plausible.
        r.reference.as_mut().unwrap().pixels.truncate(511 * 512 * 3);
        let e = GuestRequest::Image(r).check().unwrap_err();
        assert!(e.contains("512x512"), "{e}");
    }

    /// **Both ends of the hold dial.** Zero is a legal, ordinary draw — the
    /// dial is continuous at the bottom so the UI needs no special case — and a
    /// full hold is refused, because the drain would cost the whole estate its
    /// thinking to hand back the upload.
    #[test]
    fn the_reference_hold_is_continuous_at_zero_and_stops_short_of_one() {
        assert!(with_reference(512, 512, 0.0).check().is_ok());
        assert!(with_reference(512, 512, MAX_REFERENCE_HOLD).check().is_ok());
        for bad in [1.0, 0.99, -0.1, f32::NAN] {
            let e = with_reference(512, 512, bad).check().unwrap_err();
            assert!(e.contains("hold"), "hold {bad}: {e}");
        }
        assert!((0.0..=MAX_REFERENCE_HOLD).contains(&DEFAULT_REFERENCE_HOLD));
    }

    /// The shift is optional and bounded. Outside the range one end of the walk
    /// gets no steps, which is a picture rather than an error.
    #[test]
    fn the_shift_dial_is_optional_and_bounded() {
        let shifted = |s: Option<f64>| {
            let mut r = match image(512, 512, 12) {
                GuestRequest::Image(r) => r,
                _ => unreachable!(),
            };
            r.shift = s;
            GuestRequest::Image(r).check()
        };
        assert!(shifted(None).is_ok(), "the default must stay optional");
        assert!(shifted(Some(MIN_SHIFT)).is_ok());
        assert!(shifted(Some(3.0)).is_ok());
        assert!(shifted(Some(MAX_SHIFT)).is_ok());
        for bad in [0.5, 12.0, f64::NAN, f64::INFINITY] {
            assert!(shifted(Some(bad)).is_err(), "shift {bad} was accepted");
        }
    }

    /// A caller that says nothing gets the draw it got before either dial
    /// existed: no reference, the deployment's own shift.
    #[test]
    fn a_request_without_the_dials_is_an_ordinary_draw() {
        let r: ImageRequest =
            serde_json::from_str(r#"{"prompt":"a lantern","width":512,"height":512,"steps":8}"#)
                .unwrap();
        assert!(r.reference.is_none());
        assert!(r.shift.is_none());
    }

    /// The choice survives the trip through a whole request — the queue clones
    /// and serialises these, and a field lost in transit would load the wrong
    /// weights while every log said otherwise.
    #[test]
    fn the_lora_survives_a_request_round_trip() {
        let mut req = match image(512, 512, 8) {
            GuestRequest::Image(r) => r,
            _ => unreachable!(),
        };
        req.lora = ImageLora::Restricted;
        let json = serde_json::to_string(&req).unwrap();
        let back: ImageRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.lora, ImageLora::Restricted);
        assert_eq!(back, req);
    }

    /// **A refusal must cost nothing.** A drain evicts the engine's working set
    /// before the guest runs, so an ask that was never servable has to be
    /// refused at submission — otherwise a typo in a width costs every
    /// character its resident KV before anyone finds out.
    #[test]
    fn an_unservable_ask_is_refused_before_anything_is_evicted() {
        assert!(image(4096, 512, 20).check().is_err());
        assert!(image(512, 512, 500).check().is_err());
        assert!(image(0, 512, 20).check().is_err());
    }

    /// A size the decoder would round is refused rather than rounded: the
    /// caller otherwise gets an image of a size it did not ask for, and nothing
    /// in the response says which one it got.
    #[test]
    fn a_size_the_latent_cannot_represent_is_refused() {
        assert!(image(513, 512, 20).check().is_err());
        assert!(image(512, 500, 20).check().is_err());
        assert!(image(512, 512, 20).check().is_ok());
    }

    #[test]
    fn an_empty_prompt_is_refused() {
        let mut i = match image(512, 512, 20) {
            GuestRequest::Image(r) => r,
            _ => unreachable!(),
        };
        i.prompt = "   ".into();
        assert!(GuestRequest::Image(i).check().is_err());
    }

    /// The request's guest is what the drain groups by, so it must not be
    /// inferred anywhere else.
    #[test]
    fn a_request_names_the_guest_that_serves_it() {
        assert_eq!(image(512, 512, 20).guest(), Guest::Image);
    }

    #[test]
    fn a_request_round_trips_through_json() {
        let r = image(768, 512, 30);
        let text = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<GuestRequest>(&text).unwrap(), r);
    }
}
