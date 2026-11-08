//! Reading another process's memory: rekordbox, found by its executable's
//! name, its module's base, and values read through pointer chains from
//! there (`offsets::Chain`).

use super::offsets::Chain;

/// What reading can fail at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemError {
    /// No process by that name is running.
    NotRunning,
    /// The process is running and its memory could not be read at `at`.
    Read { at: usize },
    /// The operating system refused to open the process.
    Open(String),
}

impl std::fmt::Display for MemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemError::NotRunning => f.write_str("rekordbox is not running"),
            MemError::Read { at } => write!(f, "rekordbox's memory at {at:#x} could not be read"),
            MemError::Open(why) => write!(f, "rekordbox could not be opened: {why}"),
        }
    }
}

/// Reads bytes of a process at an address.
pub trait Memory {
    /// The address the module's chains start from.
    fn base(&self) -> usize;
    fn read_into(&self, at: usize, out: &mut [u8]) -> Result<(), MemError>;

    fn read_u64(&self, at: usize) -> Result<u64, MemError> {
        let mut b = [0u8; 8];
        self.read_into(at, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// The address `chain` resolves to: from the base, each hop added and
    /// dereferenced, the last added.
    fn resolve(&self, chain: &Chain) -> Result<usize, MemError> {
        let mut at = self.base();
        for hop in &chain.hops {
            at = self.read_u64(at.wrapping_add(*hop))? as usize;
        }
        Ok(at.wrapping_add(chain.last))
    }

    fn f32_at(&self, chain: &Chain) -> Result<f32, MemError> {
        let mut b = [0u8; 4];
        self.read_into(self.resolve(chain)?, &mut b)?;
        Ok(f32::from_le_bytes(b))
    }

    fn i64_at(&self, chain: &Chain) -> Result<i64, MemError> {
        let mut b = [0u8; 8];
        self.read_into(self.resolve(chain)?, &mut b)?;
        Ok(i64::from_le_bytes(b))
    }

    fn u8_at(&self, chain: &Chain) -> Result<u8, MemError> {
        let mut b = [0u8; 1];
        self.read_into(self.resolve(chain)?, &mut b)?;
        Ok(b[0])
    }

    /// A NUL-terminated UTF-8 string of at most `max` bytes.
    fn text_at(&self, chain: &Chain, max: usize) -> Result<String, MemError> {
        let mut b = vec![0u8; max];
        self.read_into(self.resolve(chain)?, &mut b)?;
        let end = b.iter().position(|&x| x == 0).unwrap_or(max);
        Ok(String::from_utf8_lossy(&b[..end]).into_owned())
    }
}

#[cfg(windows)]
mod windows {
    use std::ffi::c_void;
    use std::mem::size_of;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Module32FirstW, Module32NextW, MODULEENTRY32W,
        TH32CS_SNAPMODULE, TH32CS_SNAPMODULE32,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ,
    };

    use super::{MemError, Memory};
    use crate::rekordbox::process::id_of;

    fn wide_name(raw: &[u16]) -> String {
        let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
        String::from_utf16_lossy(&raw[..end])
    }

    /// The base address of module `exe` in process `pid`.
    fn module_base(pid: u32, exe: &str) -> Option<usize> {
        // SAFETY: as `process_id`.
        unsafe {
            let snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid);
            if snap == INVALID_HANDLE_VALUE {
                return None;
            }
            let mut entry: MODULEENTRY32W = std::mem::zeroed();
            entry.dwSize = size_of::<MODULEENTRY32W>() as u32;
            let mut found = None;
            if Module32FirstW(snap, &mut entry) != 0 {
                loop {
                    if wide_name(&entry.szModule).eq_ignore_ascii_case(exe) {
                        found = Some(entry.modBaseAddr as usize);
                        break;
                    }
                    if Module32NextW(snap, &mut entry) == 0 {
                        break;
                    }
                }
            }
            CloseHandle(snap);
            found
        }
    }

    /// A process open for reading.
    pub struct Process {
        handle: HANDLE,
        base: usize,
    }

    // SAFETY: the handle is a kernel object handle, usable from any thread.
    unsafe impl Send for Process {}

    impl Process {
        /// The running process whose executable is `exe`.
        pub fn open(exe: &str) -> Result<Process, MemError> {
            let pid = id_of(exe).ok_or(MemError::NotRunning)?;
            let base = module_base(pid, exe)
                .ok_or_else(|| MemError::Open(format!("no module {exe} in process {pid}")))?;
            // SAFETY: the returned handle is checked and closed on drop.
            let handle =
                unsafe { OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, 0, pid) };
            if handle.is_null() {
                return Err(MemError::Open(std::io::Error::last_os_error().to_string()));
            }
            Ok(Process { handle, base })
        }
    }

    impl Drop for Process {
        fn drop(&mut self) {
            // SAFETY: the handle was opened by `open` and is closed once.
            unsafe {
                CloseHandle(self.handle);
            }
        }
    }

    impl Memory for Process {
        fn base(&self) -> usize {
            self.base
        }

        fn read_into(&self, at: usize, out: &mut [u8]) -> Result<(), MemError> {
            let mut read = 0usize;
            // SAFETY: `out` is a valid buffer of `out.len()` bytes; the
            // kernel checks the remote range.
            let ok = unsafe {
                ReadProcessMemory(
                    self.handle,
                    at as *const c_void,
                    out.as_mut_ptr() as *mut c_void,
                    out.len(),
                    &mut read,
                )
            };
            if ok == 0 || read != out.len() {
                return Err(MemError::Read { at });
            }
            Ok(())
        }
    }
}

#[cfg(windows)]
pub use windows::Process;

/// rekordbox runs on Windows here: elsewhere there is no process to open.
#[cfg(not(windows))]
pub struct Process;

#[cfg(not(windows))]
impl Process {
    pub fn open(_exe: &str) -> Result<Process, MemError> {
        Err(MemError::NotRunning)
    }
}

#[cfg(not(windows))]
impl Memory for Process {
    fn base(&self) -> usize {
        0
    }

    fn read_into(&self, at: usize, _out: &mut [u8]) -> Result<(), MemError> {
        Err(MemError::Read { at })
    }
}

#[cfg(test)]
pub mod fake {
    use std::collections::BTreeMap;

    use super::{MemError, Memory};

    /// A process image of a few written regions: what a test resolves
    /// chains through.
    #[derive(Default)]
    pub struct Image {
        pub base: usize,
        regions: BTreeMap<usize, Vec<u8>>,
    }

    impl Image {
        /// An empty image whose module is at `base`.
        pub fn at(base: usize) -> Image {
            Image {
                base,
                regions: BTreeMap::new(),
            }
        }

        pub fn write(&mut self, at: usize, bytes: &[u8]) {
            self.regions.insert(at, bytes.to_vec());
        }

        pub fn pointer(&mut self, at: usize, to: usize) {
            self.write(at, &(to as u64).to_le_bytes());
        }
    }

    impl Memory for Image {
        fn base(&self) -> usize {
            self.base
        }

        fn read_into(&self, at: usize, out: &mut [u8]) -> Result<(), MemError> {
            let (start, bytes) = self
                .regions
                .range(..=at)
                .next_back()
                .ok_or(MemError::Read { at })?;
            let from = at - start;
            let src = bytes
                .get(from..from + out.len())
                .ok_or(MemError::Read { at })?;
            out.copy_from_slice(src);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::Image;
    use super::*;

    /// **A chain resolves as rkbx_link's do**: each hop added and
    /// dereferenced, the last only added — and reads the value there.
    #[test]
    fn a_chain_resolves_through_its_hops() {
        let mut m = Image::at(0x1000);
        m.pointer(0x1000 + 0x40, 0x9000);
        m.pointer(0x9000 + 0x8, 0x5000);
        m.write(0x5000 + 0x1A0, &128.5f32.to_le_bytes());
        m.write(0x6000, b"/PIONEER/USBANLZ/abc/ANLZ0000.DAT\0junk");
        let chain = Chain {
            hops: vec![0x40, 0x8],
            last: 0x1A0,
        };
        assert_eq!(m.resolve(&chain), Ok(0x51A0));
        assert_eq!(m.f32_at(&chain), Ok(128.5));
        let text = Chain {
            hops: vec![],
            last: 0x5000,
        };
        assert_eq!(
            m.text_at(&text, 64),
            Err(MemError::Read { at: 0x6000 }),
            "fewer than 64 bytes written there"
        );
        assert_eq!(
            m.text_at(&text, 38),
            Ok("/PIONEER/USBANLZ/abc/ANLZ0000.DAT".into())
        );
        let broken = Chain {
            hops: vec![0x80],
            last: 0,
        };
        assert_eq!(m.resolve(&broken), Err(MemError::Read { at: 0x1080 }));
    }
}
