// FFI bindings for the device-side KvHead record fill.
//
// Replaces a host serialize plus one upload per record: the record body is derivable
// from data the device already holds, so the host ships a descriptor table and the
// kernel writes the bytes. See `simple/kv_record_fill.cu` for the layout it must
// match, and `kv_cache::chunked::meta_pool::serialize_kv_heads` for the host
// reference the kernel is held against byte-for-byte.

use std::ffi::c_void;

/// One record's inputs, mirroring `KvRecordDesc` in the `.cu`.
///
/// `repr(C)` and field order are load-bearing — the kernel reinterprets this array.
/// An offset of `-1` means "not supplied, derive it": an absent palette map is the
/// identity routing and an absent scale is unity, which is what keeps the upload
/// smaller than the records it produces.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KvRecordDesc {
    /// Device address of the record to write.
    pub dst: u64,
    /// Offset into the packed gid array: `n_kv_head * n_palette * 2` entries.
    pub gid_off: i64,
    /// Offset into `k_pal`/`v_pal`, or `-1` for the identity map.
    pub pal_off: i64,
    /// Offset into `k_fmt`/`v_fmt`, or `-1` for `ArenaFormatTag::Invalid`.
    pub fmt_off: i64,
    /// Offset into `k_scale`/`v_scale`, or `-1` for unity.
    pub scale_off: i64,
}

/// One arena's slot geometry, mirroring `KvArenaExtent` in the `.cu`. Indexed by
/// arena index; `base == 0` marks a non-resident arena, whose bands resolve to null.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KvArenaExtent {
    pub base: u64,
    pub stride: i64,
}

/// Words per descriptor and per extent.
///
/// Callers upload these as flat `i64` arrays rather than as slices of the structs
/// above — `cudarc`'s `DeviceRepr` is not implemented for them, and every field is
/// eight bytes, so the flat form is bit-identical with no padding to reason about.
/// These constants and the assertions below are what keep that equivalence honest: if
/// a field is ever added, the build breaks here rather than the kernel reading a
/// misaligned table.
pub const DESC_WORDS: usize = 5;
pub const EXTENT_WORDS: usize = 2;

const _: () = assert!(std::mem::size_of::<KvRecordDesc>() == DESC_WORDS * 8);
const _: () = assert!(std::mem::size_of::<KvArenaExtent>() == EXTENT_WORDS * 8);
const _: () = assert!(std::mem::align_of::<KvRecordDesc>() == 8);
const _: () = assert!(std::mem::align_of::<KvArenaExtent>() == 8);

extern "C" {
    /// Write `n_records` `KvHead[n_kv_head]` records in one grid-strided launch.
    ///
    /// `gid_stride` is Rust's `GID_STRIDE`, passed rather than duplicated so the gid
    /// packing has exactly one definition. `invalid_tag` is
    /// `ArenaFormatTag::Invalid.as_u8()`, used for a band whose tag was not recorded —
    /// never a float tag, which would decode quantized bytes as floats.
    ///
    /// # Safety
    ///
    /// Every pointer must be device-resident and long enough for the offsets the
    /// descriptors name; each `dst` must be a writable record slot of
    /// `n_kv_head * (head_dim / 2 + n_palette * 26)` bytes.
    pub fn run_kv_record_fill(
        descs: *const c_void,
        gids: *const i64,
        k_pal: *const u8,
        v_pal: *const u8,
        k_fmt: *const u8,
        v_fmt: *const u8,
        k_scale: *const f32,
        v_scale: *const f32,
        extents: *const c_void,
        n_extents: i32,
        n_records: i32,
        n_kv_head: i32,
        head_dim: i32,
        n_palette: i32,
        gid_stride: i32,
        invalid_tag: i32,
        stream: *mut c_void,
    );
}
