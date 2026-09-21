//! The sampling kernel's banned-token buffer, one deny-list per row.
//!
//! The kernel takes its banned tokens in one of two shapes: a single list every
//! row shares (`banned_per_seq == 0`, `num_banned` entries), or one fixed-width
//! slot per row (`banned_per_seq == stride`, `-1` padding the unused entries).
//! The shared shape is only right when every row bans the same thing. A turn
//! that bans a token for itself alone — the structural `</think>` outside a
//! block, `<tool_call>` on the answer that closes a stuck tool loop — needs its
//! own row: taken from the wave's first row and applied to all, the ban either
//! lands on every other session in the wave or, when the banning row is not
//! first, on none of it.

/// `(buffer, num_banned, banned_per_seq)` for the kernel. Each row is its own
/// deny-list plus an optional extra id (the think-close ban).
pub(crate) fn banned_buffer(rows: &[(&[i32], Option<i32>)]) -> (Vec<i32>, i32, i32) {
    let first = rows.first().map_or(&[][..], |row| row.0);
    if rows
        .iter()
        .all(|&(list, extra)| extra.is_none() && list == first)
    {
        return (first.to_vec(), first.len() as i32, 0);
    }
    let stride = rows
        .iter()
        .map(|&(list, extra)| list.len() + usize::from(extra.is_some()))
        .max()
        .unwrap_or(0)
        .max(1);
    let mut flat = Vec::with_capacity(rows.len() * stride);
    for &(list, extra) in rows {
        let start = flat.len();
        flat.extend_from_slice(list);
        flat.extend(extra);
        flat.resize(start + stride, -1);
    }
    (flat, 0, stride as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows that all ban the same list share one list.
    #[test]
    fn identical_rows_share_one_list() {
        let list = [7, 9];
        assert_eq!(
            banned_buffer(&[(&list, None), (&list, None)]),
            (vec![7, 9], 2, 0)
        );
        assert_eq!(banned_buffer(&[]), (vec![], 0, 0));
    }

    /// **A ban one row carries is that row's alone** — the closing turn's
    /// `<tool_call>` (here 42) in the second row, beside a row with none.
    #[test]
    fn a_row_s_own_ban_stays_in_its_row() {
        let shared: [i32; 1] = [7];
        let closing: [i32; 2] = [7, 42];
        assert_eq!(
            banned_buffer(&[(&shared, None), (&closing, None)]),
            (vec![7, -1, 7, 42], 0, 2)
        );
    }

    /// The think-close ban rides in the row's slot after its list.
    #[test]
    fn the_think_close_ban_is_per_row() {
        let shared: [i32; 1] = [7];
        assert_eq!(
            banned_buffer(&[(&shared, Some(5)), (&shared, None)]),
            (vec![7, 5, 7, -1], 0, 2)
        );
        assert_eq!(banned_buffer(&[(&[], Some(5))]), (vec![5], 0, 1));
    }
}
