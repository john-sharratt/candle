//! Where a head's dims sit across its palette bands.
//!
//! A chunk's palette map gives every dim of a head its palette, 2 bits per dim
//! (`head_dim / 4` bytes per head; dims 4i..4i+3 in byte i, low bits first).
//! Within a palette, dims are stored in ascending order: a dim's RANK is the
//! number of lower dims in the same palette, and band p holds its ranks 0..sub.
//! This is the addressing every kernel uses (`pal_map_get` / `rank_in_pal` in
//! palette4_convert.cuh, the prefill's rank tables).

/// For each dim `d` of the head, its column in the palette-order row
/// `[band 0 | band 1 | … ]`: `palette(d) · sub + rank(d)`. `map` is the
/// head's `head_dim / 4` map bytes.
pub(crate) fn palette_columns(map: &[u8], head_dim: usize, n_palette: usize) -> Vec<u32> {
    let sub = head_dim / n_palette;
    let mut seen = vec![0u32; n_palette];
    (0..head_dim)
        .map(|d| {
            let p = ((map[d / 4] >> (2 * (d % 4))) & 0x3) as usize;
            let col = (p * sub) as u32 + seen[p];
            seen[p] += 1;
            col
        })
        .collect()
}

/// Whether `columns` is the identity (the map puts dims 0..sub in band 0,
/// sub..2·sub in band 1, and so on).
pub(crate) fn is_identity(columns: &[u32]) -> bool {
    columns.iter().enumerate().all(|(d, &c)| c as usize == d)
}
