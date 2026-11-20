//! `SystemProcessInformation`: every process's counters in one call.
//!
//! The record is `SYSTEM_PROCESS_INFORMATION`, at the x64 offsets every tool
//! that reads it uses (unchanged since Windows 7):
//!
//! ```text
//!   offset   0  NextEntryOffset          u32   0 on the last record
//!   offset   8  WorkingSetPrivateSize    i64
//!   offset  16  HardFaultCount           u32
//!   offset  56  ImageName.Length         u16   bytes, UTF-16
//!   offset  64  ImageName.Buffer         ptr   into the same buffer
//!   offset  80  UniqueProcessId          usize
//!   offset 144  WorkingSetSize           usize
//!   offset 184  PagefileUsage            usize  private committed bytes
//! ```
//!
//! The thread records that follow each process record are skipped by
//! `NextEntryOffset`.

use super::ProcessPaging;

const SYSTEM_PROCESS_INFORMATION: i32 = 5;
const STATUS_INFO_LENGTH_MISMATCH: i32 = 0xC000_0004_u32 as i32;

pub(super) const NEXT: usize = 0;
pub(super) const WS_PRIVATE: usize = 8;
pub(super) const HARD_FAULTS: usize = 16;
pub(super) const NAME_LEN: usize = 56;
pub(super) const NAME_PTR: usize = 64;
pub(super) const PID: usize = 80;
pub(super) const WS: usize = 144;
pub(super) const PRIVATE: usize = 184;
/// The record's fixed part, through `PagefileUsage`.
pub(super) const RECORD: usize = 192;

#[link(name = "ntdll")]
extern "system" {
    fn NtQuerySystemInformation(
        class: i32,
        info: *mut std::ffi::c_void,
        len: u32,
        ret_len: *mut u32,
    ) -> i32;
}

#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcessId() -> u32;
}

/// Every process's counters, or none when the call fails.
pub(super) fn all() -> Vec<ProcessPaging> {
    snapshot()
        .map(|buf| parse(as_bytes(&buf)))
        .unwrap_or_default()
}

/// This process's counters.
pub(super) fn own() -> Option<ProcessPaging> {
    // SAFETY: a plain Win32 call.
    let pid = unsafe { GetCurrentProcessId() };
    all().into_iter().find(|p| p.pid == pid)
}

/// The list, in a buffer grown until it fits — `u64`s, so the records'
/// 8-byte fields are aligned.
fn snapshot() -> Option<Vec<u64>> {
    let mut words = 64 * 1024;
    for _ in 0..8 {
        let mut buf = vec![0u64; words];
        let mut ret = 0u32;
        // SAFETY: the buffer is writable for the length passed.
        let status = unsafe {
            NtQuerySystemInformation(
                SYSTEM_PROCESS_INFORMATION,
                buf.as_mut_ptr().cast(),
                (buf.len() * 8) as u32,
                &mut ret,
            )
        };
        if status == STATUS_INFO_LENGTH_MISMATCH {
            words = (ret as usize / 8 + 1).max(words * 2);
            continue;
        }
        return (status >= 0).then_some(buf);
    }
    None
}

fn as_bytes(words: &[u64]) -> &[u8] {
    // SAFETY: any initialised `u64` slice is a valid byte slice of 8× its length.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 8) }
}

/// The records of a `SystemProcessInformation` buffer. A record running past
/// the buffer ends the walk; a name whose pointer is not inside the buffer reads
/// empty.
pub(super) fn parse(buf: &[u8]) -> Vec<ProcessPaging> {
    let u32_at = |at: usize| u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
    let u64_at = |at: usize| u64::from_le_bytes(buf[at..at + 8].try_into().unwrap());
    let base = buf.as_ptr() as u64;
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + RECORD <= buf.len() {
        let len = u32_at(at + NAME_LEN) as u16 as usize;
        let ptr = u64_at(at + NAME_PTR);
        let name = ptr
            .checked_sub(base)
            .map(|off| off as usize)
            .filter(|&off| len > 0 && off + len <= buf.len())
            .map(|off| {
                let units: Vec<u16> = buf[off..off + len]
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                String::from_utf16_lossy(&units)
            })
            .unwrap_or_default();
        out.push(ProcessPaging {
            pid: u64_at(at + PID) as u32,
            name,
            hard_faults: u64::from(u32_at(at + HARD_FAULTS)),
            working_set: u64_at(at + WS),
            private_working_set: u64_at(at + WS_PRIVATE),
            private_bytes: u64_at(at + PRIVATE),
        });
        let next = u32_at(at + NEXT) as usize;
        if next == 0 {
            break;
        }
        at += next;
    }
    out
}
