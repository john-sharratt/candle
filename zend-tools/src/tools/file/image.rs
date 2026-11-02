//! file_image tool: read what a picture is, and make resized copies of it in the
//! repository.
//!
//! `code_run`'s files are text and the sandbox holds no image program, so a
//! task as ordinary as a site's icon set had no way to be done: a worker spent
//! twenty calls looking for one (2026-10-31). The bytes go through the file
//! store, so the copies are the conversation's changes like any other write.
//!
//! With no `outputs` it only reads: the picture's size, its bytes, and an
//! `.ico`'s frames. With no way to ask, a worker spent a third of a favicon task
//! probing with 0×0 copies, `od`, `magick` and scripts (2026-11-01).

use std::io::Cursor;
use std::path::Path;

use image::codecs::ico::{IcoEncoder, IcoFrame};
use image::imageops::FilterType;
use image::{DynamicImage, ExtendedColorType, GenericImageView, ImageFormat};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// The widest or tallest copy.
const MAX_SIDE: u32 = 4096;
/// The sizes an `.ico` holds unless `ico_sizes` says otherwise.
const ICO_SIZES: [u32; 3] = [16, 32, 48];
/// The widest an `.ico` frame may be, by its format.
const MAX_ICO_SIDE: u32 = 256;
/// How far a copy's shape may stray from what it is cut from, as a fraction:
/// rounding a size to whole pixels, and no more.
const SHAPE_SLACK: f64 = 0.01;

/// The part of the picture to use.
#[derive(Clone, Copy, Deserialize, JsonSchema, Validate)]
pub struct Crop {
    /// Left edge, in pixels.
    pub x: u32,
    /// Top edge, in pixels.
    pub y: u32,
    /// Its width in pixels.
    pub width: u32,
    /// Its height in pixels.
    pub height: u32,
}

/// One copy to make.
#[derive(Clone, Serialize, Deserialize, JsonSchema, Validate)]
pub struct ImageOutput {
    /// Where the copy goes, relative to the repository. Its extension picks the format: `.png`, `.jpg`, `.webp` or `.ico`. Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// Its width in pixels. Required.
    pub width: u32,
    /// Its height in pixels. Required.
    pub height: u32,
    /// For an `.ico`, the square sizes it holds, `width` and `height` aside; [16, 32, 48] when left out. Each at most 256.
    #[serde(default)]
    #[schemars(with = "Vec<u32>")]
    pub ico_sizes: Option<Vec<u32>>,
}

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ImageRequest {
    /// The repository the picture belongs to. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Path of the picture to copy from, relative to the repository: PNG, JPEG, WebP, GIF, BMP or TIFF. Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// The repository `path` is in, when it is not `repo`: how a picture gets from one repository into another. The copies always go in `repo`. Optional.
    #[serde(default)]
    #[schemars(with = "String")]
    pub source_repo: Option<String>,
    /// The part of the picture to use; all of it when left out. Use it when a copy's shape differs from the picture's. Optional.
    #[serde(default)]
    pub crop: Option<Crop>,
    /// The copies to make. Left out, nothing is written and the answer says what the picture is.
    #[serde(default)]
    #[validate(length(max = 32))]
    pub outputs: Vec<ImageOutput>,
}

#[derive(Serialize)]
pub struct WrittenImage {
    pub path: String,
    pub bytes: usize,
    pub width: u32,
    pub height: u32,
}

#[derive(Serialize)]
pub struct ImageResponse {
    pub repo: String,
    /// The picture's own size.
    pub width: u32,
    pub height: u32,
    /// The picture's file size in bytes.
    pub bytes: usize,
    /// The side of each frame an `.ico` holds, in its file's order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ico_sizes: Vec<u32>,
    pub written: Vec<WrittenImage>,
}

pub struct FileImage;

impl Tool for FileImage {
    const NAME: &'static str = "file_image";
    const DESCRIPTION: &'static str =
        "Make resized copies of a picture in the repository, keeping its transparency: PNG, \
         JPEG, WebP or a multi-size ICO, cropped first if you say so. The copies are among your \
         uncommitted changes. The picture may be in another repository (`source_repo`), which \
         is how a picture gets from one repository into another. With no `outputs` nothing is \
         written and it reads what a picture is: its size, its file's bytes and an ICO's frame \
         sizes. Use for: an icon set, a favicon, thumbnails, converting a picture to another \
         format, checking a picture's dimensions. Triggered by \"resize the image\", \"make a \
         favicon\", \"generate the icon sizes\", \"convert to webp\", \"how big is this image\". \
         Returns the picture's size and each copy's path, bytes and size. A copy whose shape \
         differs from the picture's is refused: crop first.";

    type Request = ImageRequest;
    type Response = ImageResponse;
    type Error = FileError;

    /// The same picture makes the same copies.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: ImageRequest) -> Result<ImageResponse, FileError> {
        let files = ctx.files.repo(&req.repo)?;
        let source = ctx
            .files
            .repo(req.source_repo.as_deref().unwrap_or(&req.repo))?
            .read_bytes(&req.path)?
            .ok_or_else(|| not_found(&req))?;
        refuse_overwriting_source(&req)?;
        let picture = decode(&source, &req.path)?;
        let (width, height) = picture.dimensions();
        let ico_sizes = ico_frames(&source);
        let cut = cut(picture, req.crop)?;
        // Every copy is made before any is written, so a bad one leaves none.
        let made = req
            .outputs
            .iter()
            .map(|output| Ok((output, encode(&cut, output)?)))
            .collect::<Result<Vec<_>, FileError>>()?;
        let mut written = Vec::with_capacity(made.len());
        for (output, bytes) in made {
            written.push(WrittenImage {
                path: output.path.clone(),
                bytes: bytes.len(),
                width: output.width,
                height: output.height,
            });
            files.write_bytes(&output.path, bytes)?;
        }
        Ok(ImageResponse {
            repo: req.repo,
            width,
            height,
            bytes: source.len(),
            ico_sizes,
            written,
        })
    }
}

/// The side of each frame in `bytes`' `.ico` directory, none when it is no
/// `.ico`: a six-byte header counting sixteen-byte entries, each opening with
/// its width, where 0 means 256.
fn ico_frames(bytes: &[u8]) -> Vec<u32> {
    if !bytes.starts_with(&[0, 0, 1, 0]) || bytes.len() < 6 {
        return Vec::new();
    }
    let count = usize::from(u16::from_le_bytes([bytes[4], bytes[5]]));
    bytes[6..]
        .iter()
        .step_by(16)
        .take(count.min((bytes.len() - 6) / 16))
        .map(|&side| match side {
            0 => MAX_ICO_SIDE,
            side => u32::from(side),
        })
        .collect()
}

/// The picture `req` names is not there, said with the way out when it may be
/// in another repository: told only "not found", a worker sent the same call
/// five times and lost its tools.
fn not_found(req: &ImageRequest) -> FileError {
    match &req.source_repo {
        Some(repo) => FileError::NotFound(format!("{} in repository `{repo}`", req.path)),
        None => FileError::NotFound(format!(
            "{} in repository `{}`. If the picture is in another repository, name it in \
             `source_repo`; the copies still go in `repo`",
            req.path, req.repo
        )),
    }
}

/// A copy written over the picture it is made from is refused: measured live,
/// a worker shrank `favicon.ico` to 16×16 and rewrote its 1024-pixel source art
/// this way, each time by naming the output after the input.
fn refuse_overwriting_source(req: &ImageRequest) -> Result<(), FileError> {
    let source_repo = req.source_repo.as_deref().unwrap_or(&req.repo);
    let same = |path: &str| source_repo == req.repo && same_path(path, &req.path);
    match req.outputs.iter().find(|output| same(&output.path)) {
        Some(output) => Err(FileError::InvalidArguments(format!(
            "{} is the picture these copies are made from: give the copy a path of its own",
            output.path
        ))),
        None => Ok(()),
    }
}

/// Whether two repository paths name one file, however they are spelled.
fn same_path(a: &str, b: &str) -> bool {
    let tidy = |p: &str| {
        p.replace('\\', "/")
            .trim_start_matches("./")
            .trim_start_matches('/')
            .to_ascii_lowercase()
    };
    tidy(a) == tidy(b)
}

fn decode(bytes: &[u8], path: &str) -> Result<DynamicImage, FileError> {
    image::load_from_memory(bytes)
        .map_err(|e| FileError::Unreadable(format!("{path} is not a picture this reads: {e}")))
}

/// `picture`, cut down to `crop`.
fn cut(picture: DynamicImage, crop: Option<Crop>) -> Result<DynamicImage, FileError> {
    let Some(c) = crop else {
        return Ok(picture);
    };
    let (width, height) = picture.dimensions();
    let inside = c.width > 0
        && c.height > 0
        && u64::from(c.x) + u64::from(c.width) <= u64::from(width)
        && u64::from(c.y) + u64::from(c.height) <= u64::from(height);
    if !inside {
        return Err(FileError::InvalidArguments(format!(
            "a crop of {}x{} at {},{} does not fit in a {width}x{height} picture",
            c.width, c.height, c.x, c.y
        )));
    }
    Ok(picture.crop_imm(c.x, c.y, c.width, c.height))
}

/// `picture` at `width`×`height`, refused when that would stretch it.
fn sized(picture: &DynamicImage, width: u32, height: u32) -> Result<DynamicImage, FileError> {
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(FileError::InvalidArguments(format!(
            "a copy of {width}x{height} is out of range: each side is 1 to {MAX_SIDE}. To read a \
             picture's size, call with no `outputs`"
        )));
    }
    let (from_width, from_height) = picture.dimensions();
    let from = f64::from(from_width) / f64::from(from_height);
    let to = f64::from(width) / f64::from(height);
    if (from / to - 1.0).abs() > SHAPE_SLACK {
        return Err(FileError::InvalidArguments(format!(
            "a copy of {width}x{height} would stretch a {from_width}x{from_height} picture: pass \
             `crop` to cut it to that shape first"
        )));
    }
    Ok(picture.resize_exact(width, height, FilterType::Lanczos3))
}

/// The bytes of `output`, made from `picture`.
fn encode(picture: &DynamicImage, output: &ImageOutput) -> Result<Vec<u8>, FileError> {
    let extension = Path::new(&output.path)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    let format = match extension.as_deref() {
        Some("png") => ImageFormat::Png,
        Some("jpg" | "jpeg") => ImageFormat::Jpeg,
        Some("webp") => ImageFormat::WebP,
        Some("ico") => return encode_ico(picture, output),
        _ => {
            return Err(FileError::InvalidArguments(format!(
                "{} names no format this writes: `.png`, `.jpg`, `.webp` or `.ico`",
                output.path
            )))
        }
    };
    let copy = sized(picture, output.width, output.height)?;
    let copy = match format {
        // JPEG holds no alpha, and the WebP encoder here takes eight-bit pixels.
        ImageFormat::Jpeg => DynamicImage::ImageRgb8(copy.to_rgb8()),
        ImageFormat::WebP => DynamicImage::ImageRgba8(copy.to_rgba8()),
        _ => copy,
    };
    let mut bytes = Cursor::new(Vec::new());
    copy.write_to(&mut bytes, format)
        .map_err(|e| FileError::Unwritable(format!("{}: {e}", output.path)))?;
    Ok(bytes.into_inner())
}

/// An `.ico` holding `output`'s sizes, each frame a PNG.
fn encode_ico(picture: &DynamicImage, output: &ImageOutput) -> Result<Vec<u8>, FileError> {
    let sizes = output.ico_sizes.as_deref().unwrap_or(&ICO_SIZES);
    if sizes.is_empty() || sizes.iter().any(|&s| s > MAX_ICO_SIDE) {
        return Err(FileError::InvalidArguments(format!(
            "an .ico holds 1 or more sizes of at most {MAX_ICO_SIDE}, not {sizes:?}"
        )));
    }
    let unwritable = |e: image::ImageError| FileError::Unwritable(format!("{}: {e}", output.path));
    let frames = sizes
        .iter()
        .map(|&side| {
            let frame = sized(picture, side, side)?.to_rgba8();
            IcoFrame::as_png(&frame, side, side, ExtendedColorType::Rgba8).map_err(unwritable)
        })
        .collect::<Result<Vec<_>, FileError>>()?;
    let mut bytes = Vec::new();
    IcoEncoder::new(&mut bytes)
        .encode_images(&frames)
        .map_err(unwritable)?;
    Ok(bytes)
}

pub const FILE_IMAGE: RegisteredTool = RegisteredTool::new::<FileImage>();

#[cfg(test)]
mod tests {
    use image::{Rgba, RgbaImage};

    use super::*;

    /// A square whose top half is opaque red and whose bottom half is clear.
    fn half_clear(side: u32) -> DynamicImage {
        DynamicImage::ImageRgba8(RgbaImage::from_fn(side, side, |_, y| {
            if y < side / 2 {
                Rgba([255, 0, 0, 255])
            } else {
                Rgba([0, 0, 0, 0])
            }
        }))
    }

    fn output(path: &str, width: u32, height: u32) -> ImageOutput {
        ImageOutput {
            path: path.into(),
            width,
            height,
            ico_sizes: None,
        }
    }

    /// **A PNG copy keeps the picture's transparency**, at the size asked.
    #[test]
    fn a_png_copy_keeps_its_transparency() {
        let bytes = encode(&half_clear(64), &output("a/icon.PNG", 16, 16)).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let copy = image::load_from_memory(&bytes).unwrap().to_rgba8();
        assert_eq!(copy.dimensions(), (16, 16));
        assert_eq!(copy.get_pixel(8, 1).0, [255, 0, 0, 255]);
        assert_eq!(copy.get_pixel(8, 14).0[3], 0);
    }

    /// **An `.ico` holds every size asked**, read back from its own directory:
    /// a zero-reserved, type-1 header, then one entry per frame, 256 written 0.
    #[test]
    fn an_ico_holds_each_size_as_a_png() {
        let mut ico = output("favicon.ico", 0, 0);
        ico.ico_sizes = Some(vec![16, 32, 256]);
        let bytes = encode(&half_clear(512), &ico).unwrap();
        assert_eq!(&bytes[..6], &[0, 0, 1, 0, 3, 0]);
        let sides: Vec<u8> = (0..3).map(|i| bytes[6 + i * 16]).collect();
        assert_eq!(sides, [16, 32, 0]);
        let first = u32::from_le_bytes(bytes[18..22].try_into().unwrap()) as usize;
        assert_eq!(&bytes[first..first + 8], b"\x89PNG\r\n\x1a\n");
    }

    /// **An `.ico` reads back as its frames**, 256 written 0 included, and
    /// anything else as none. An `.ico` also decodes as a picture, which is what
    /// reading one with no `outputs` depends on.
    #[test]
    fn an_ico_reads_back_as_its_frames() {
        let mut ico = output("favicon.ico", 0, 0);
        ico.ico_sizes = Some(vec![16, 32, 256]);
        let bytes = encode(&half_clear(512), &ico).unwrap();
        assert_eq!(ico_frames(&bytes), [16, 32, 256]);
        assert!(decode(&bytes, "favicon.ico").is_ok());
        let png = encode(&half_clear(64), &output("a.png", 16, 16)).unwrap();
        assert!(ico_frames(&png).is_empty());
        assert!(ico_frames(&[0, 0, 1, 0]).is_empty(), "cut off");
    }

    /// With no sizes named an `.ico` holds the usual three.
    #[test]
    fn an_ico_holds_the_usual_sizes_by_default() {
        let bytes = encode(&half_clear(64), &output("favicon.ico", 0, 0)).unwrap();
        assert_eq!(&bytes[..6], &[0, 0, 1, 0, 3, 0]);
    }

    /// JPEG has no alpha and WebP keeps it; both come back the size asked.
    #[test]
    fn jpeg_and_webp_copies_are_written() {
        let jpeg = encode(&half_clear(64), &output("a.jpg", 32, 32)).unwrap();
        assert_eq!(&jpeg[..3], &[0xFF, 0xD8, 0xFF]);
        let webp = encode(&half_clear(64), &output("a.webp", 32, 32)).unwrap();
        assert_eq!(&webp[..4], b"RIFF");
        assert_eq!(&webp[8..12], b"WEBP");
        let back = image::load_from_memory(&webp).unwrap().to_rgba8();
        assert_eq!(back.dimensions(), (32, 32));
        assert_eq!(back.get_pixel(16, 30).0[3], 0);
    }

    /// **A stretch is refused with the way out**, and a crop to the shape
    /// passes.
    #[test]
    fn a_copy_of_another_shape_needs_a_crop() {
        let picture = half_clear(64);
        let refused = encode(&picture, &output("a.png", 64, 32)).unwrap_err();
        assert!(refused.to_string().contains("pass `crop`"), "{refused}");
        let wide = cut(
            picture,
            Some(Crop {
                x: 0,
                y: 0,
                width: 64,
                height: 32,
            }),
        )
        .unwrap();
        assert_eq!(wide.dimensions(), (64, 32));
        assert!(encode(&wide, &output("a.png", 32, 16)).is_ok());
    }

    /// **A copy cannot be written over its own source**, however it is spelled;
    /// the same path in another repository is another file.
    #[test]
    fn a_copy_over_its_source_is_refused() {
        let request = |source_repo: Option<&str>, to: &str| ImageRequest {
            repo: "game".into(),
            path: "web/icon.png".into(),
            source_repo: source_repo.map(str::to_string),
            crop: None,
            outputs: vec![output("a.png", 8, 8), output(to, 8, 8)],
        };
        for to in ["web/icon.png", "./web/icon.png", "web\\ICON.png"] {
            assert!(
                refuse_overwriting_source(&request(None, to)).is_err(),
                "{to}"
            );
        }
        assert!(refuse_overwriting_source(&request(None, "web/other.png")).is_ok());
        let across = request(Some("art"), "web/icon.png");
        assert!(refuse_overwriting_source(&across).is_ok());
    }

    /// A crop past the picture's edge, a zero side and an unknown extension
    /// are refused rather than clamped.
    #[test]
    fn what_cannot_be_made_is_refused() {
        let outside = Crop {
            x: 60,
            y: 0,
            width: 8,
            height: 8,
        };
        assert!(cut(half_clear(64), Some(outside)).is_err());
        assert!(encode(&half_clear(64), &output("a.png", 0, 0)).is_err());
        assert!(encode(&half_clear(64), &output("a.svg", 8, 8)).is_err());
        let mut big = output("a.ico", 0, 0);
        big.ico_sizes = Some(vec![512]);
        assert!(encode(&half_clear(64), &big).is_err());
    }
}
