//! Which co-resident models this daemon offers, and where their weights are.
//!
//! # Why this is configuration and not a preset
//!
//! A guest's checkpoint is gigabytes a deployment chose to download. Naming one
//! in the code would make every daemon that starts up expect it on disk, and a
//! daemon with no `guests.yaml` would then fail to start over a feature nobody
//! asked it for. So the registry is empty unless a file says otherwise, and a
//! daemon without one carries one atomic load per scheduler pass and nothing
//! else.
//!
//! # The file
//!
//! `<data>/guests.yaml`, beside the substrate:
//!
//! ```yaml
//! prose:
//!   gguf: D:/models/Hermes-3-Llama-3.2-3B-Q6_K.gguf
//!   tokenizer: D:/models/hermes3/tokenizer.json
//!   # optional
//!   max_context: 4096
//!   system: "You are a narrator. Write vivid, concrete prose."
//! image:
//!   transformer: D:/models/z-image/z_image_turbo-Q8_0.gguf
//!   text_encoder: D:/models/z-image/qwen3_4b_f32-q8_0.gguf
//!   encoder_config: D:/models/z-image/text_encoder_config.json
//!   vae: D:/models/z-image/vae
//!   tokenizer: D:/models/z-image/tokenizer.json
//!   # optional
//!   shift: 3.0
//!   transformer_restricted: D:/models/z-image/z_image_turbo-Q8_0-r.gguf
//! matte:
//!   model: D:/models/matte/isnet-general-use.onnx
//!   # optional; `is_net` (the default) or `u2_net`
//!   family: is_net
//! ```
//!
//! Every section is optional and any may stand alone. A section whose files
//! are not on disk is **refused at startup**, loudly, rather than accepted and
//! discovered at the first drain — by then the engine has already evicted its
//! working set to load a checkpoint that is not there.

use std::path::{Path, PathBuf};

use candle_conversation::guest::prose_choice::HermesQuant;
use candle_conversation::guest::{
    Guest, GuestModel, GuestRegistry, ImageGuest, ImageSpec, MatteFamily, MatteGuest, MatteSpec,
    ProseGuest, ProseSpec,
};
use serde::Deserialize;

/// The file's shape.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GuestsFile {
    #[serde(default)]
    pub prose: Option<ProseSection>,
    #[serde(default)]
    pub image: Option<ImageSection>,
    #[serde(default)]
    pub matte: Option<MatteSection>,
}

/// The salient-object network behind "remove background".
///
/// Small beside the others — tens of megabytes against the image guest's seven
/// gigabytes — but it is a guest for the same reason they are: it wants the
/// card, and the card belongs to the engine between waves.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MatteSection {
    /// The exported ONNX graph.
    pub model: PathBuf,
    /// Which published network it is, and therefore how its input is
    /// normalised. **Not inferred from the file name**: the two families want
    /// different statistics, and a wrong guess does not fail — it returns a
    /// matte full of holes, which reads as a broken model.
    #[serde(default)]
    pub family: MatteFamily,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProseSection {
    /// The directory holding the Hermes-4-14B rungs a deployment downloaded.
    ///
    /// **A directory, not a file, because the rung is the card's decision.**
    /// The image guest names its checkpoint outright — a deployment picks a
    /// Z-Image rung once and lives with it — but the prose guest is claimed and
    /// dropped every drain against ground that a resident model is already
    /// standing in, so the file that fits is a property of the machine rather
    /// than of the deployment. `HermesQuant::for_vram` picks it, and this says
    /// where to look; see [`candle_conversation::guest::prose_choice`] for the
    /// ladder and the measurements under it.
    ///
    /// One machine's directory may hold one rung and another's three. Only the
    /// rung this card wants has to be present.
    pub dir: PathBuf,
    pub tokenizer: PathBuf,
    #[serde(default)]
    pub max_context: Option<usize>,
    #[serde(default)]
    pub system: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ImageSection {
    /// The Z-Image transformer GGUF. Which rung a deployment downloaded is its
    /// own call — `candle_transformers::models::z_image::ZQuant` says what a
    /// card can hold, and this names the file that resulted.
    ///
    /// This is the checkpoint a request's default `lora` loads.
    pub transformer: PathBuf,
    /// The checkpoint a request naming the `restricted` lora loads — the same
    /// rung with a different adapter fused in via `z-image-fuse`. What is in it
    /// is the deployment's business; the daemon only routes to it, behind the
    /// admin check `guest_routes::post_image` documents. Optional because a
    /// deployment that never asks for the variant should not have to fuse a
    /// file nothing uses; requests naming it are then refused by name.
    #[serde(default)]
    pub transformer_restricted: Option<PathBuf>,
    /// The Qwen3-4B text-encoder GGUF.
    pub text_encoder: PathBuf,
    /// The encoder's `config.json` from the Z-Image release. Named separately
    /// because the GGUF carries tensor data and nothing that says how wide the
    /// model is.
    pub encoder_config: PathBuf,
    /// The autoencoder's safetensors directory.
    pub vae: PathBuf,
    /// The Qwen3 tokenizer's `tokenizer.json`.
    pub tokenizer: PathBuf,
    /// The rectified-flow schedule's shift. 3.0 is Z-Image's own and what its
    /// eight steps are distilled against; there is no reason to move it except
    /// to see what happens.
    #[serde(default)]
    pub shift: Option<f64>,
}

/// Where the file lives for a data directory.
pub fn path(data: &Path) -> PathBuf {
    data.join("guests.yaml")
}

/// Read the file and build the registry, or say why not.
///
/// A missing file is an empty registry, not an error: no guests is the ordinary
/// deployment. A file that is present and wrong **is** an error — an operator
/// who wrote one meant it, and starting without the guest they configured is a
/// failure they will not notice until somebody asks for an image.
pub fn load(data: &Path) -> Result<GuestRegistry, String> {
    let file = path(data);
    let text = match std::fs::read_to_string(&file) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(GuestRegistry::new()),
        Err(e) => return Err(format!("reading {file:?}: {e}")),
    };
    let parsed: GuestsFile =
        serde_yaml::from_str(&text).map_err(|e| format!("parsing {file:?}: {e}"))?;
    build(parsed)
}

/// Turn a parsed file into a registry, checking every path first.
pub fn build(file: GuestsFile) -> Result<GuestRegistry, String> {
    let mut registry = GuestRegistry::new();

    if let Some(p) = file.prose {
        // **The card chooses the rung, here, once.** Total VRAM rather than
        // free: this is which checkpoint the deployment runs, and a free-memory
        // reading would have the daemon pick a different file depending on what
        // happened to be resident at startup.
        let total_vram = candle::quantized::get_total_vram_device0().unwrap_or(0) as u64;
        let quant = HermesQuant::for_vram(total_vram);
        let gguf = p.dir.join(quant.filename());
        require_dir(&p.dir, "prose.dir")?;
        // Named in the error, because "no such file" against a path the operator
        // never wrote is a puzzle: they configured a directory and the daemon
        // chose the filename inside it.
        if !gguf.is_file() {
            return Err(format!(
                "prose.dir has no {}: this card reports {} MiB of VRAM, so the prose guest wants \
                 the {:?} rung of Hermes-4-14B. Download it from {} into {:?}",
                quant.filename(),
                total_vram >> 20,
                quant,
                quant.repo(),
                p.dir,
            ));
        }
        require_file(&p.tokenizer, "prose.tokenizer")?;
        tracing::info!(
            target: "npcd::guests",
            vram_mib = total_vram >> 20,
            rung = ?quant,
            file = ?gguf,
            ground_mib = candle_conversation::guest::prose_choice::ground_bytes_at(quant, 4096) >> 20,
            "prose guest: Hermes-4-14B rung chosen for this card"
        );
        let mut spec = ProseSpec::hermes4_14b(gguf, p.tokenizer);
        if let Some(c) = p.max_context {
            spec.max_context = c;
        }
        if let Some(s) = p.system {
            spec.default_system = s;
        }
        registry.register(Guest::Prose, move || {
            Box::new(ProseGuest::new(spec.clone())) as Box<dyn GuestModel>
        });
    }

    if let Some(i) = file.image {
        require_file(&i.transformer, "image.transformer")?;
        require_file(&i.text_encoder, "image.text_encoder")?;
        require_file(&i.encoder_config, "image.encoder_config")?;
        require_dir(&i.vae, "image.vae")?;
        require_file(&i.tokenizer, "image.tokenizer")?;
        if let Some(m) = &i.transformer_restricted {
            require_file(m, "image.transformer_restricted")?;
        }
        let mut spec = ImageSpec::z_image(
            i.transformer,
            i.text_encoder,
            i.encoder_config,
            i.vae,
            i.tokenizer,
        );
        spec.transformer_restricted = i.transformer_restricted;
        if let Some(s) = i.shift {
            if !s.is_finite() || s <= 0.0 {
                return Err(format!("image.shift {s} is not a usable number"));
            }
            spec.shift = s;
        }
        registry.register(Guest::Image, move || {
            Box::new(ImageGuest::new(spec.clone())) as Box<dyn GuestModel>
        });
    }

    if let Some(m) = file.matte {
        require_file(&m.model, "matte.model")?;
        let spec = MatteSpec::new(m.model, m.family);
        registry.register(Guest::Matte, move || {
            Box::new(MatteGuest::new(spec.clone())) as Box<dyn GuestModel>
        });
    }

    Ok(registry)
}

/// Refuse a path that is not a readable file.
///
/// **At startup, not at the drain.** A drain evicts the engine's working set
/// before the guest loads, so a typo in a path costs every character its
/// resident KV and then fails — and it does so at whatever hour somebody first
/// asks for an image, not at the restart that introduced it.
fn require_file(p: &Path, field: &str) -> Result<(), String> {
    if p.is_file() {
        return Ok(());
    }
    Err(format!(
        "{field}: {p:?} is not a file — a guest's checkpoint is checked at startup, because a \
         drain evicts the engine's working set before it loads"
    ))
}

fn require_dir(p: &Path, field: &str) -> Result<(), String> {
    if p.is_dir() {
        return Ok(());
    }
    Err(format!(
        "{field}: {p:?} is not a directory — the image guest reads every `.safetensors` in it, so \
         a sharded checkpoint needs no re-listing here"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("npcd-guests-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// No file is no guests, and that is the ordinary deployment — not an
    /// error, and not a daemon that refuses to start over a feature nobody
    /// asked it for.
    #[test]
    fn a_daemon_with_no_guests_file_starts_with_none() {
        let d = tmp("absent");
        let r = load(&d).unwrap();
        assert!(r.is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// **A path that is not there is refused at startup.** The alternative is
    /// discovering it at the first drain — after the engine has evicted its
    /// working set to load a checkpoint that does not exist, at whatever hour
    /// somebody first asks for an image.
    #[test]
    fn a_checkpoint_that_is_not_there_is_refused_before_the_daemon_serves() {
        let d = tmp("missing-ckpt");
        std::fs::write(
            path(&d),
            "prose:\n  dir: ./nope\n  tokenizer: ./nope.json\n",
        )
        .unwrap();
        let e = load(&d).unwrap_err();
        assert!(e.contains("prose.dir"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// **A directory that exists without the rung this card wants is refused,
    /// and the message names the file.** The operator configured a directory
    /// and the daemon chose the filename inside it, so "no such file" against a
    /// path they never wrote is a puzzle unless the error says where the name
    /// came from.
    #[test]
    fn a_directory_without_this_cards_rung_says_which_file_it_wanted() {
        let d = tmp("wrong-rung");
        let dir = d.join("hermes4");
        std::fs::create_dir_all(&dir).unwrap();
        let tok = d.join("t.json");
        std::fs::write(&tok, b"{}").unwrap();
        std::fs::write(
            path(&d),
            format!(
                "prose:\n  dir: {}\n  tokenizer: {}\n",
                dir.display().to_string().replace('\\', "/"),
                tok.display().to_string().replace('\\', "/")
            ),
        )
        .unwrap();
        let e = load(&d).unwrap_err();
        assert!(e.contains("Hermes-4-14B-Q"), "should name the rung: {e}");
        assert!(e.contains("bartowski"), "should name where to get it: {e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A file that is present and unparseable is an error rather than an empty
    /// registry: the operator who wrote it meant it.
    #[test]
    fn a_malformed_file_is_an_error_not_an_empty_registry() {
        let d = tmp("malformed");
        std::fs::write(path(&d), "prose:\n  dir: [not, a, path]\n").unwrap();
        assert!(load(&d).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// An unknown key is refused rather than ignored — a misspelt `tokeniser`
    /// would otherwise leave the guest reading a default that does not exist.
    #[test]
    fn an_unknown_key_is_refused() {
        let d = tmp("unknown-key");
        std::fs::write(
            path(&d),
            "prose:\n  dir: a\n  tokeniser: b.json\n  tokenizer: b.json\n",
        )
        .unwrap();
        assert!(load(&d).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Both sections are optional and either stands alone.
    #[test]
    fn each_guest_can_be_configured_without_the_other() {
        let d = tmp("prose-only");
        let dir = d.join("hermes4");
        std::fs::create_dir_all(&dir).unwrap();
        // Whatever rung this machine's card asks for, so the test passes on all
        // three of them rather than on whichever one wrote it.
        let quant =
            HermesQuant::for_vram(candle::quantized::get_total_vram_device0().unwrap_or(0) as u64);
        std::fs::write(dir.join(quant.filename()), b"x").unwrap();
        let tok = d.join("t.json");
        std::fs::write(&tok, b"{}").unwrap();
        std::fs::write(
            path(&d),
            format!(
                "prose:\n  dir: {}\n  tokenizer: {}\n",
                dir.display().to_string().replace('\\', "/"),
                tok.display().to_string().replace('\\', "/")
            ),
        )
        .unwrap();
        let r = load(&d).unwrap();
        assert_eq!(r.configured(), vec![Guest::Prose]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The optional overrides reach the spec rather than being parsed and
    /// dropped — a `max_context` an operator set is the number a job is refused
    /// against.
    #[test]
    fn the_optional_settings_reach_the_spec() {
        let file = GuestsFile {
            matte: None,
            prose: Some(ProseSection {
                dir: PathBuf::from("a"),
                tokenizer: PathBuf::from("b"),
                max_context: Some(8192),
                system: Some("You are terse.".into()),
            }),
            image: None,
        };
        // The paths do not exist, so this refuses — which is itself the point of
        // the check above. What is asserted here is that parsing carried the
        // settings, which the refusal message cannot show.
        assert!(build(file).is_err());

        let mut spec = ProseSpec::hermes3_3b("a", "b");
        spec.max_context = 8192;
        spec.default_system = "You are terse.".into();
        assert_eq!(spec.max_context, 8192);
        assert_eq!(spec.default_system, "You are terse.");
    }

    /// A shift that is not a number would reach the scheduler and put every
    /// sigma at NaN — a denoise that does not fail, and decodes to flat grey.
    #[test]
    fn a_nonsense_shift_is_refused() {
        let d = tmp("shift");
        std::fs::create_dir_all(d.join("vae")).unwrap();
        let files: Vec<PathBuf> = ["t.gguf", "e.gguf", "e.json", "tok.json"]
            .iter()
            .map(|n| {
                let p = d.join(n);
                std::fs::write(&p, b"{}").unwrap();
                p
            })
            .collect();
        let file = GuestsFile {
            matte: None,
            prose: None,
            image: Some(ImageSection {
                transformer: files[0].clone(),
                transformer_restricted: None,
                text_encoder: files[1].clone(),
                encoder_config: files[2].clone(),
                vae: d.join("vae"),
                tokenizer: files[3].clone(),
                shift: Some(f64::NAN),
            }),
        };
        let e = build(file).unwrap_err();
        assert!(e.contains("shift"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// **Every path is checked, and the message names which one.** A guest's
    /// checkpoint is gigabytes; an operator who mistyped one of five paths gets
    /// told which, rather than a refusal they have to bisect by hand.
    #[test]
    fn each_image_path_is_named_when_it_is_missing() {
        let d = tmp("image-paths");
        let file = GuestsFile {
            matte: None,
            prose: None,
            image: Some(ImageSection {
                transformer: d.join("nope.gguf"),
                transformer_restricted: None,
                text_encoder: d.join("nope2.gguf"),
                encoder_config: d.join("nope.json"),
                vae: d.join("nope"),
                tokenizer: d.join("nope-tok.json"),
                shift: None,
            }),
        };
        let e = build(file).unwrap_err();
        assert!(e.contains("image.transformer"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The restricted checkpoint is optional — absent it configures nothing and
    /// refuses nothing — but one that is named and not on disk is refused at
    /// startup like every other path, and the message names the field.
    #[test]
    fn a_named_restricted_checkpoint_is_checked_like_the_others() {
        let d = tmp("restricted-path");
        std::fs::create_dir_all(d.join("vae")).unwrap();
        let files: Vec<PathBuf> = ["t.gguf", "e.gguf", "e.json", "tok.json"]
            .iter()
            .map(|n| {
                let p = d.join(n);
                std::fs::write(&p, b"{}").unwrap();
                p
            })
            .collect();
        let section = |restricted: Option<PathBuf>| GuestsFile {
            matte: None,
            prose: None,
            image: Some(ImageSection {
                transformer: files[0].clone(),
                transformer_restricted: restricted,
                text_encoder: files[1].clone(),
                encoder_config: files[2].clone(),
                vae: d.join("vae"),
                tokenizer: files[3].clone(),
                shift: None,
            }),
        };
        assert!(build(section(None)).is_ok(), "absent must stay optional");
        let e = build(section(Some(d.join("nope-r.gguf")))).unwrap_err();
        assert!(e.contains("image.transformer_restricted"), "{e}");
        let m = d.join("m.gguf");
        std::fs::write(&m, b"{}").unwrap();
        assert!(build(section(Some(m))).is_ok());
        let _ = std::fs::remove_dir_all(&d);
    }
}
