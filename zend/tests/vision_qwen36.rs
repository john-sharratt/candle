//! Image turns on Qwen3.6-35B-A3B, end to end: a picture read by the model's
//! own vision tower, its rows placed over the turn's placeholders in the hybrid
//! wave, and an answer that could only come from seeing it
//! (`docs/vision_models.md` §4).
//!
//! ```text
//! cargo test -p zend --release --test vision_qwen36 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The picture is a 5×5 grid with one cell filled. Which cell is a fact the
//! text never states, so a right answer twice over, for two different cells,
//! is the tower's rows reaching the trunk — a model reading placeholders, or
//! another image's rows, names a cell by chance one time in twenty-five.

mod common;

use std::io::Cursor;

use candle::Device;
use candle_conversation::vision::{load_kept, turn_images, ImageSource, TurnImage};
use candle_conversation::{ConversationEngine, Sequence};
use common::{quiet_turn, Workspace};
use image::{ImageFormat, Rgb, RgbImage};
use regex::Regex;

/// The grid's side, in cells.
const CELLS: u32 = 5;

/// A cell's side, in pixels.
const CELL_PX: u32 = 96;

/// The grid lines' width, in pixels.
const LINE_PX: u32 = 4;

/// The tower the 35B's spec names: its rows are kept under this name.
const TOWER: &str = "qwen3.6-35b-a3b";

/// The reply budget: the answer and nothing else, thinking off.
const ANSWER_TOKENS: usize = 48;

const QUESTION: &str = "The picture is a 5 by 5 grid with exactly one cell filled in red. \
     Rows are numbered 1 to 5 from the top, columns 1 to 5 from the left. Which cell is \
     filled? Answer exactly in the form: row R, column C.";

/// A PNG of the grid with cell `(row, col)` (from 1, top-left) filled red.
fn grid_png(row: u32, col: u32) -> Vec<u8> {
    let side = CELLS * CELL_PX + LINE_PX;
    let mut img = RgbImage::from_pixel(side, side, Rgb([255, 255, 255]));
    for (x, y, px) in img.enumerate_pixels_mut() {
        let on_line = x % CELL_PX < LINE_PX || y % CELL_PX < LINE_PX;
        let in_cell = x / CELL_PX == col - 1 && y / CELL_PX == row - 1;
        if on_line {
            *px = Rgb([0, 0, 0]);
        } else if in_cell {
            *px = Rgb([220, 20, 20]);
        }
    }
    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, ImageFormat::Png)
        .expect("encode png");
    out.into_inner()
}

/// The `(row, column)` a reply names, if it names one.
fn cell_named(reply: &str) -> Option<(u32, u32)> {
    let re = Regex::new(r"(?i)row\s*(\d)\D{0,12}col(?:umn)?\s*(\d)").expect("regex");
    let c = re.captures(reply)?;
    Some((c[1].parse().ok()?, c[2].parse().ok()?))
}

/// The images `sources` read as on `engine`, against its substrate.
fn read(engine: &ConversationEngine, sources: Vec<ImageSource>) -> Vec<TurnImage> {
    let substrate = engine.conversation().shared();
    turn_images(engine, &substrate, sources).expect("turn images")
}

/// Ask `conv` which cell `images` fills.
fn ask(engine: &ConversationEngine, conv: &mut Sequence, images: Vec<TurnImage>) -> String {
    let mut options = quiet_turn(engine, ANSWER_TOKENS);
    options.images = images;
    conv.send_turn_with_options(QUESTION, options)
        .expect("image turn")
        .text
}

/// **The model names the filled cell, for two cells, and a kept image reads
/// back as its rows.** The second conversation is given the first picture by
/// id, so its rows come from the substrate, not the tower, and the answer has
/// to match again.
#[test]
#[ignore = "loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) and its vision tower; \
            three image turns (~4 min). Run with: cargo test -p zend --release \
            --test vision_qwen36 -- --ignored --nocapture --test-threads=1"]
fn the_35b_names_the_filled_cell_of_a_grid() {
    let device = Device::new_cuda(0).expect("cuda");
    let ws = Workspace::new();
    let session = ws.session(&device);
    let engine = session.engine();
    assert_eq!(
        engine.vision().map(|t| t.name),
        Some(TOWER),
        "the 35B's spec names no vision tower, or another's"
    );

    let mut first_id = None;
    for (row, col) in [(2, 4), (5, 1)] {
        let images = read(engine, vec![ImageSource::Bytes(grid_png(row, col))]);
        assert_eq!(images.len(), 1);
        first_id.get_or_insert_with(|| images[0].id.clone());
        let mut conv = session.start();
        let reply = ask(engine, &mut conv, images);
        eprintln!("cell ({row}, {col}): {reply:?}");
        assert_eq!(
            cell_named(&reply),
            Some((row, col)),
            "the model did not name cell ({row}, {col}): {reply:?}"
        );
    }

    let id = first_id.expect("an image was read");
    let substrate = engine.conversation().shared();
    let kept = load_kept(&substrate, &id, TOWER)
        .expect("kept rows")
        .expect("the image's rows were not kept for the 35B's tower");
    assert_eq!(
        kept.rows.width, 2048,
        "the 35B's tower writes 2048-wide rows"
    );

    let images = read(engine, vec![ImageSource::Kept(id)]);
    let mut conv = session.start();
    let reply = ask(engine, &mut conv, images);
    eprintln!("kept (2, 4): {reply:?}");
    assert_eq!(
        cell_named(&reply),
        Some((2, 4)),
        "the kept image did not read back: {reply:?}"
    );
}

#[test]
fn a_reply_names_its_cell() {
    assert_eq!(cell_named("Row 2, column 4."), Some((2, 4)));
    assert_eq!(cell_named("row 5, col 1"), Some((5, 1)));
    assert_eq!(
        cell_named("The filled cell is in row 3 and column 2"),
        Some((3, 2))
    );
    assert_eq!(cell_named("I cannot tell."), None);
}

#[test]
fn the_grid_fills_only_its_cell() {
    let png = grid_png(2, 4);
    let img = image::load_from_memory(&png).expect("decode").to_rgb8();
    let side = CELLS * CELL_PX + LINE_PX;
    assert_eq!((img.width(), img.height()), (side, side));
    let mid = |i: u32| (i - 1) * CELL_PX + CELL_PX / 2;
    let centre = |r: u32, c: u32| *img.get_pixel(mid(c), mid(r));
    assert_eq!(centre(2, 4), Rgb([220, 20, 20]));
    assert_eq!(centre(4, 2), Rgb([255, 255, 255]));
    assert_eq!(centre(1, 1), Rgb([255, 255, 255]));
    assert_eq!(*img.get_pixel(0, 0), Rgb([0, 0, 0]));
}
