//! One track's analysis, from its two files under rekordbox's `share`
//! folder: the grid and hot cues from `ANLZ0000.DAT`, the phrases from
//! `ANLZ0000.EXT` beside it.

use std::fs;
use std::path::{Path, PathBuf};

use super::anlz::{parse, Anlz};
use crate::music::{Grid, HotCue, Phrase};

/// What the analysis gives a profile.
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    pub grid: Grid,
    pub phrases: Vec<Phrase>,
    pub cues: Vec<HotCue>,
    /// The phrase analysis's mood: 1 high, 2 mid, 3 low.
    pub mood: Option<u16>,
}

/// The file `anlz` — as the library writes it, `/PIONEER/USBANLZ/…` — names
/// under `share`.
pub fn file(share: &Path, anlz: &str) -> PathBuf {
    share.join(anlz.trim_start_matches(['/', '\\']))
}

fn read(path: &Path) -> Result<Anlz, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// The analysis of the track whose `.DAT` file is `dat`; its `.EXT` is read
/// when it is there. A track with no grid has no analysis to plan with.
pub fn analyse(dat: &Path) -> Result<Analysis, String> {
    let main = read(dat)?;
    let (first_ms, bpm) = main
        .grid()
        .ok_or_else(|| format!("{}: no beat grid", dat.display()))?;
    let grid = Grid {
        bpm,
        first_beat_ms: first_ms,
    };
    let ext = dat.with_extension("EXT");
    let phrases = if ext.is_file() {
        read(&ext)?.phrases
    } else {
        None
    };
    Ok(Analysis {
        grid,
        cues: main.hot_cues(first_ms, grid.ms_per_beat()),
        mood: phrases.as_ref().map(|p| p.mood),
        phrases: phrases.map(|p| p.folded()).unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::anlz::build;
    use super::*;

    #[test]
    fn a_tracks_two_files_make_its_analysis() {
        let dir = tempfile::tempdir().unwrap();
        let share = dir.path();
        let rel = "/PIONEER/USBANLZ/a1b/2c3d/ANLZ0000.DAT";
        let dat = file(share, rel);
        fs::create_dir_all(dat.parent().unwrap()).unwrap();
        fs::write(
            &dat,
            build::file(&[
                build::grid(&[(4, 12000, 0), (1, 12000, 500), (2, 12000, 1000)]),
                build::cues(&[(2, 4500)]),
            ]),
        )
        .unwrap();
        let only_dat = analyse(&dat).unwrap();
        assert_eq!(
            only_dat.grid,
            Grid {
                bpm: 120.0,
                first_beat_ms: 500.0
            }
        );
        assert_eq!(only_dat.cues, vec![HotCue { pad: 2, beat: 8.0 }]);
        assert!(only_dat.phrases.is_empty() && only_dat.mood.is_none());

        fs::write(
            dat.with_extension("EXT"),
            build::file(&[build::phrases(1, 65, &[(1, 1), (33, 5)], true)]),
        )
        .unwrap();
        let both = analyse(&dat).unwrap();
        assert_eq!(both.mood, Some(1));
        assert_eq!(both.phrases.len(), 2);
        assert_eq!(both.phrases[1].start, 32);

        fs::write(&dat, build::file(&[build::path("x")])).unwrap();
        assert!(analyse(&dat).unwrap_err().ends_with("no beat grid"));
    }
}
