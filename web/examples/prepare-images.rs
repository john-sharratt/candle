//! Turn source artwork into web-sized copies, once.
//!
//! The cinematic stills are 2048×1536 PNGs of about 4 MB each. Neither number
//! belongs on a page: the site would ship twenty megabytes to show five
//! pictures, and it would ship them at four times the resolution any browser
//! will draw.
//!
//! This resizes and re-encodes them, and the *output* is what gets committed —
//! a few hundred kilobytes each. The site itself does no image work at runtime
//! and needs no image dependency; it serves files, as it does for everything
//! else. Re-run this only when the source art changes:
//!
//! ```sh
//! cargo run -p web --example prepare-images -- \
//!     ../battle-cities/assets/movie  content/tokera/img/bc  scene_2 scene_9 title
//! ```
//!
//! Sources stay in the game repo. Only the small derived copies live here, so
//! the DMZ box needs nothing but this crate's own content directory.

use std::path::{Path, PathBuf};

use image::imageops::FilterType;

/// Default cap. Override with `--width N`: a picture that sits beside the text
/// at 440 CSS pixels needs about 900 here for a 2× display, and a banner that
/// spans the column needs more.
const DEFAULT_WIDTH: u32 = 900;
/// Visually indistinguishable from the source at this scale; roughly a tenth
/// of the bytes of the equivalent PNG for photographic art.
const JPEG_QUALITY: u8 = 82;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    let mut max_width = DEFAULT_WIDTH;
    if let Some(i) = args.iter().position(|a| a == "--width") {
        max_width = args
            .get(i + 1)
            .ok_or("--width needs a number")?
            .parse()
            .map_err(|_| "--width needs a number")?;
        args.drain(i..=i + 1);
    }

    if args.len() < 3 {
        eprintln!(
            "usage: prepare-images [--width N] <source-dir> <out-dir> <stem>...\n\
             \n\
             Each <stem> names a file in <source-dir> without its extension;\n\
             `.png` and `.jpg` are both tried. Output is <out-dir>/<stem>.jpg."
        );
        std::process::exit(2);
    }
    let source = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out)?;

    let mut total_in = 0u64;
    let mut total_out = 0u64;

    for stem in &args[2..] {
        let Some(path) = find(&source, stem) else {
            eprintln!("  {stem}: not found in {}", source.display());
            continue;
        };
        let bytes_in = std::fs::metadata(&path)?.len();

        let img = image::open(&path)?;
        let (w, h) = (img.width(), img.height());
        // Only ever downscale. Enlarging a source would add bytes and no
        // detail, which is the opposite of the point.
        let img = if w > max_width {
            img.resize(max_width, u32::MAX, FilterType::Lanczos3)
        } else {
            img
        };

        let dest = out.join(format!("{stem}.jpg"));
        let mut file = std::fs::File::create(&dest)?;
        // RGB8: the source art has no transparency to preserve, and an alpha
        // channel is not encodable as JPEG anyway.
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut file, JPEG_QUALITY)
            .encode_image(&img.to_rgb8())?;
        drop(file);

        let bytes_out = std::fs::metadata(&dest)?.len();
        total_in += bytes_in;
        total_out += bytes_out;
        println!(
            "  {stem:<14} {w}×{h} {:>7} KB  →  {}×{} {:>6} KB",
            bytes_in / 1024,
            img.width(),
            img.height(),
            bytes_out / 1024
        );
    }

    if total_in > 0 {
        println!(
            "\n  {} KB → {} KB  ({}× smaller)",
            total_in / 1024,
            total_out / 1024,
            total_in / total_out.max(1)
        );
    }
    Ok(())
}

/// A stem plus whichever extension the source actually uses.
fn find(dir: &Path, stem: &str) -> Option<PathBuf> {
    ["png", "jpg", "jpeg", "webp"]
        .iter()
        .map(|ext| dir.join(format!("{stem}.{ext}")))
        .find(|p| p.is_file())
}
