//! The Windows walk: `VirtualQuery` over the whole user address space for the
//! committed regions, `QueryWorkingSetEx` for which of their pages are
//! resident, and the module / mapped-file name for images and views.

use super::{ProcessRam, RegionKind, RegionRecord};
use std::ffi::c_void;
use std::mem::size_of;

const MEM_COMMIT: u32 = 0x1000;
const MEM_IMAGE: u32 = 0x100_0000;
const MEM_MAPPED: u32 = 0x4_0000;
const PAGE: u64 = 4096;
/// Pages asked of `QueryWorkingSetEx` per call — bounds the query buffer at
/// 1 MiB whatever the region's size.
const BATCH_PAGES: usize = 65_536;

/// `MEMORY_BASIC_INFORMATION`, 64-bit layout.
#[repr(C)]
#[derive(Default)]
struct MemoryBasicInformation {
    base_address: usize,
    allocation_base: usize,
    allocation_protect: u32,
    partition_id: u16,
    region_size: usize,
    state: u32,
    protect: u32,
    kind: u32,
}

/// `PSAPI_WORKING_SET_EX_INFORMATION`: an address in, its attributes out. Bit 0
/// of the attributes is `Valid` — the page is in the working set.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct WorkingSetExInformation {
    virtual_address: usize,
    attributes: usize,
}

#[link(name = "kernel32")]
extern "system" {
    fn VirtualQuery(
        address: *const c_void,
        buffer: *mut MemoryBasicInformation,
        len: usize,
    ) -> usize;
    fn GetCurrentProcess() -> *mut c_void;
    fn K32QueryWorkingSetEx(process: *mut c_void, info: *mut c_void, len: u32) -> i32;
    fn K32GetMappedFileNameW(
        process: *mut c_void,
        address: *mut c_void,
        name: *mut u16,
        len: u32,
    ) -> u32;
    fn GetModuleFileNameW(module: *mut c_void, name: *mut u16, len: u32) -> u32;
}

/// Resident bytes of `[base, base + size)`.
fn resident_bytes(base: usize, size: usize) -> Option<u64> {
    let pages = size / PAGE as usize;
    let mut resident = 0u64;
    let mut buf = vec![WorkingSetExInformation::default(); pages.min(BATCH_PAGES)];
    let mut done = 0;
    while done < pages {
        let n = (pages - done).min(BATCH_PAGES);
        for (i, e) in buf[..n].iter_mut().enumerate() {
            *e = WorkingSetExInformation {
                virtual_address: base + (done + i) * PAGE as usize,
                attributes: 0,
            };
        }
        let ok = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                buf.as_mut_ptr() as *mut c_void,
                (n * size_of::<WorkingSetExInformation>()) as u32,
            )
        };
        if ok == 0 {
            return None;
        }
        resident += buf[..n].iter().filter(|e| e.attributes & 1 != 0).count() as u64 * PAGE;
        done += n;
    }
    Some(resident)
}

/// The file behind an image or a mapped view at `base`.
fn backing_name(base: usize, kind: RegionKind) -> Option<String> {
    let mut name = [0u16; 1024];
    let len = unsafe {
        match kind {
            RegionKind::Image => {
                GetModuleFileNameW(base as *mut c_void, name.as_mut_ptr(), name.len() as u32)
            }
            RegionKind::Mapped => K32GetMappedFileNameW(
                GetCurrentProcess(),
                base as *mut c_void,
                name.as_mut_ptr(),
                name.len() as u32,
            ),
            RegionKind::Private => 0,
        }
    };
    (len > 0).then(|| {
        let path = String::from_utf16_lossy(&name[..len as usize]);
        path.rsplit(['\\', '/']).next().unwrap_or(&path).to_string()
    })
}

pub(super) fn capture() -> Option<ProcessRam> {
    let mut records = Vec::new();
    let mut address = 0usize;
    loop {
        let mut mbi = MemoryBasicInformation::default();
        let got = unsafe {
            VirtualQuery(
                address as *const c_void,
                &mut mbi,
                size_of::<MemoryBasicInformation>(),
            )
        };
        if got == 0 || mbi.region_size == 0 {
            break;
        }
        if mbi.state == MEM_COMMIT {
            let kind = match mbi.kind {
                MEM_IMAGE => RegionKind::Image,
                MEM_MAPPED => RegionKind::Mapped,
                _ => RegionKind::Private,
            };
            records.push(RegionRecord {
                alloc_base: mbi.allocation_base as u64,
                kind,
                committed: mbi.region_size as u64,
                resident: resident_bytes(mbi.base_address, mbi.region_size)?,
            });
        }
        match mbi.base_address.checked_add(mbi.region_size) {
            Some(next) => address = next,
            None => break,
        }
    }
    let mut ram = ProcessRam::from_records(records);
    for a in &mut ram.allocations {
        a.name = backing_name(a.alloc_base as usize, a.kind);
    }
    Some(ram)
}
