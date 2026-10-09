//! Read-ahead's policy on the host: how many rows ahead a launch reads, and the
//! link window it may spend.
//!
//! The device does the reading (`moe_bucketize.cu`, READ-AHEAD): a launch at
//! row `r` claims promotion-ring offers for the warm experts the host predicted
//! for rows `r + 2 ..= r + READ_AHEAD_DEPTH`, with what its own misses leave of
//! the window, and its gate launch's workers copy and publish them. The window
//! is in slot images per layer: the link's rate times a layer's time, over an
//! image's bytes, times [`READ_AHEAD_SHARE`].
//!
//! **Why a share, and why the median of the previous pass.** A read-ahead copy
//! the link has no idle time for lengthens its layer, and a window taken from
//! layer times that include those copies would grow with them. With `a`
//! read-ahead images per layer, `n` misses, a base layer time `T0` and `c` the
//! link time of one image, the window's own fixed point is `a = (s·T0/c − n) /
//! (1 − s)` for a share `s` — finite for any `s < 1`, and at ½ never more than
//! the link could move in the layer's own time less twice its misses. The
//! median of a pass's layer intervals ignores the gaps a host-side pause or a
//! pipeline thread catching up puts into a few of them.

/// Rows past the one after a launch that it reads ahead for: a launch at `r`
/// reads `r + 2 ..= r + READ_AHEAD_DEPTH`. The row right after is excluded —
/// its bucketize is usually in flight by the time the prediction is made.
pub(crate) const READ_AHEAD_DEPTH: u32 = 4;

/// The share of a layer's link time read-ahead may spend (see the module docs).
pub(crate) const READ_AHEAD_SHARE: f64 = 0.5;

/// The window, in slot images per layer: the link at `rate` bytes/s for
/// `layer_secs`, in images of `image_bytes`, at [`READ_AHEAD_SHARE`] — at most
/// `cap`, the most one bucketize may claim.
pub(crate) fn read_ahead_window(rate: f64, layer_secs: f64, image_bytes: usize, cap: usize) -> u32 {
    // A NaN fails every comparison, so it is caught here too.
    let positive = |x: f64| x.partial_cmp(&0.0) == Some(std::cmp::Ordering::Greater);
    if image_bytes == 0 || !positive(rate) || !positive(layer_secs) {
        return 0;
    }
    let images = (rate * layer_secs * READ_AHEAD_SHARE / image_bytes as f64).floor();
    images.min(cap as f64) as u32
}

/// Whether a launch that reads `row`'s listing has yet to begin, so the listing
/// is worth writing. A launch at `L` reads the lists of rows `L + 2 ..= L +
/// READ_AHEAD_DEPTH`, so the last to read `row`'s is the launch at `row − 2`,
/// and that one is still to come while it lies past the served row and the
/// device has begun no invocation of it enqueued after the served one — its
/// started word, `begun(row − 2)`, is at most `served_ticket`. The served row
/// and every row before it in the pass have begun (the served summary is
/// written after them); before a pass's first row is served `served_row` is
/// `None` and the served ticket is the previous pass's last, which every
/// invocation of the new pass is past. Rows 0 and 1 are read only by the
/// previous pass's last launches, which no prediction targets.
pub(crate) fn listing_readable(
    row: usize,
    served_row: Option<usize>,
    served_ticket: u64,
    begun: impl FnOnce(usize) -> u64,
) -> bool {
    let Some(reader) = row.checked_sub(2) else {
        return false;
    };
    served_row.is_none_or(|s| reader > s) && begun(reader) <= served_ticket
}

/// The median of `samples` (the lower middle for an even count), or `None`
/// when there are none.
pub(crate) fn median(samples: &mut [f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(samples[(samples.len() - 1) / 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 10 GB/s for 4 ms is 40 MB; half of it in 1.32 MiB images is 14.
    #[test]
    fn the_window_is_half_the_layers_link_time_in_whole_images() {
        let image = 1_384_448;
        assert_eq!(read_ahead_window(10e9, 0.004, image, 64), 14);
        assert_eq!(read_ahead_window(10e9, 0.012, image, 64), 43);
        assert_eq!(read_ahead_window(10e9, 0.100, image, 64), 64, "capped");
        assert_eq!(
            read_ahead_window(10e9, 0.0001, image, 64),
            0,
            "under one image"
        );
    }

    /// Nothing measured yet — no rate, no layer time, no image — reads nothing.
    #[test]
    fn an_unmeasured_window_is_zero() {
        assert_eq!(read_ahead_window(0.0, 0.004, 1 << 20, 64), 0);
        assert_eq!(read_ahead_window(10e9, 0.0, 1 << 20, 64), 0);
        assert_eq!(read_ahead_window(10e9, 0.004, 0, 64), 0);
        assert_eq!(read_ahead_window(f64::NAN, 0.004, 1 << 20, 64), 0);
    }

    /// A row's last reader is the launch two rows before it: once that row is
    /// served, or the device has begun an invocation of it past the served
    /// ticket, the listing is read by nobody.
    #[test]
    fn a_listing_is_readable_until_the_launch_two_rows_before_it_begins() {
        // Row 5 served at ticket 105; started words from an earlier pass.
        let earlier = |_: usize| 58;
        assert!(
            !listing_readable(7, Some(5), 105, earlier),
            "its reader is the served row"
        );
        assert!(
            !listing_readable(6, Some(5), 105, earlier),
            "its reader is before it"
        );
        assert!(listing_readable(8, Some(5), 105, earlier));
        assert!(listing_readable(9, Some(5), 105, |r| if r == 7 {
            70
        } else {
            200
        }));
        assert!(
            !listing_readable(8, Some(5), 105, |_| 106),
            "row 6 has begun this pass"
        );
        assert!(
            !listing_readable(1, Some(5), 105, earlier),
            "rows 0 and 1 have no reader"
        );
        assert!(!listing_readable(0, None, 0, earlier));
    }

    /// Before a pass's first row is served the served ticket is the previous
    /// pass's last: a row is readable from the third until the device begins
    /// the launch two rows before it in the new pass.
    #[test]
    fn a_new_pass_is_readable_from_its_third_row_until_the_device_begins_it() {
        assert!(listing_readable(2, None, 147, |_| 100));
        assert!(
            listing_readable(3, None, 147, |_| 147),
            "at most the served ticket"
        );
        assert!(!listing_readable(3, None, 147, |_| 148));
        assert!(!listing_readable(1, None, 147, |_| 0));
    }

    /// The median ignores a few outliers either way.
    #[test]
    fn the_median_ignores_outliers() {
        assert_eq!(median(&mut []), None);
        assert_eq!(
            median(&mut [0.004, 0.0001, 0.004, 0.5, 0.0041]),
            Some(0.004)
        );
        assert_eq!(median(&mut [0.003, 0.001]), Some(0.001));
    }
}
