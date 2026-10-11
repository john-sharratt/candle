//! The operating system install's own identifier.
//!
//! A substrate's node is bound to the machine and directory it was minted in
//! (§3.7), and the machine half of that is the identifier the operating system
//! assigns at install: `MachineGuid` on Windows, `/etc/machine-id` elsewhere.
//! It survives reboots and renames, and a copied disk image is the one case
//! where two machines share it — which the directory half still tells apart
//! when the copy lands at another path.

use std::io;

/// The install identifier, or an error naming why it could not be read.
#[cfg(windows)]
pub fn machine_id() -> io::Result<String> {
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    use windows_sys::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ};

    let wide = |s: &str| s.encode_utf16().chain(Some(0)).collect::<Vec<u16>>();
    let key = wide(r"SOFTWARE\Microsoft\Cryptography");
    let value = wide("MachineGuid");
    let mut buf = [0u16; 128];
    let mut len = std::mem::size_of_val(&buf) as u32;
    // SAFETY: every pointer is to a live, NUL-terminated buffer or to `len`,
    // whose value is `buf`'s size in bytes as the call requires.
    let status = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            key.as_ptr(),
            value.as_ptr(),
            RRF_RT_REG_SZ,
            std::ptr::null_mut(),
            buf.as_mut_ptr().cast(),
            &mut len,
        )
    };
    if status != ERROR_SUCCESS {
        return Err(io::Error::from_raw_os_error(status as i32));
    }
    let chars = (len as usize / 2).saturating_sub(1);
    Ok(String::from_utf16_lossy(&buf[..chars]))
}

/// The install identifier, or an error naming why it could not be read.
#[cfg(not(windows))]
pub fn machine_id() -> io::Result<String> {
    for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"] {
        if let Ok(id) = std::fs::read_to_string(path) {
            let id = id.trim();
            if !id.is_empty() {
                return Ok(id.to_string());
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "no /etc/machine-id or /var/lib/dbus/machine-id",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The identifier reads, is not empty, and reads the same twice.
    #[test]
    fn the_install_identifier_is_stable() {
        let a = machine_id().expect("this machine has an install identifier");
        assert!(!a.is_empty());
        assert_eq!(machine_id().unwrap(), a);
    }
}
