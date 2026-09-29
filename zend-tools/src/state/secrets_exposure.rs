//! Whether a secrets file is private to the user running the daemon.
//!
//! A key file anyone else on the machine can read is already leaked, and one
//! anyone else can replace is theirs to fill; loading either quietly would
//! hide that. [`check`] returns why a file is not private, or `Ok` when it is.
//!
//! - **Unix.** The file must be owned by the daemon's user, readable and
//!   writable by nobody else (the rule OpenSSH applies to a private key),
//!   and sit in a folder nobody else can write to — otherwise another user
//!   could swap the file for their own.
//! - **Windows.** The file's access list must grant no read to the broad
//!   groups: Everyone, Authenticated Users and Users. A file under the
//!   user's profile inherits a private list; one placed elsewhere with
//!   `--secrets` is where this matters.

use std::path::Path;

#[cfg(unix)]
pub(crate) fn check(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let meta = std::fs::metadata(path).map_err(|e| format!("could not be inspected: {e}"))?;
    // SAFETY: `geteuid` has no preconditions and cannot fail.
    let me = unsafe { libc::geteuid() };
    if meta.uid() != me {
        return Err(format!(
            "is owned by user {} rather than the daemon's user {me}",
            meta.uid()
        ));
    }
    let mode = meta.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(format!(
            "can be read or written by other users (mode {mode:o}); chmod 600 it"
        ));
    }
    let folder = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let folder_mode = std::fs::metadata(folder)
        .map_err(|e| format!("its folder could not be inspected: {e}"))?
        .permissions()
        .mode();
    if folder_mode & 0o022 != 0 {
        return Err(format!(
            "sits in {} which other users can write to (mode {:o}); another user could replace it",
            folder.display(),
            folder_mode & 0o777
        ));
    }
    Ok(())
}

#[cfg(windows)]
pub(crate) fn check(path: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::null_mut;

    use windows_sys::Win32::Foundation::{LocalFree, ERROR_SUCCESS};
    use windows_sys::Win32::Security::Authorization::{GetNamedSecurityInfoW, SE_FILE_OBJECT};
    use windows_sys::Win32::Security::{
        AclSizeInformation, CreateWellKnownSid, EqualSid, GetAce, GetAclInformation,
        WinAuthenticatedUserSid, WinBuiltinUsersSid, WinWorldSid, ACCESS_ALLOWED_ACE, ACE_HEADER,
        ACL, ACL_SIZE_INFORMATION, DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR,
        SECURITY_MAX_SID_SIZE, WELL_KNOWN_SID_TYPE,
    };

    /// `ACCESS_ALLOWED_ACE_TYPE`.
    const ALLOWED: u8 = 0;
    /// `FILE_READ_DATA`, `GENERIC_READ` and `GENERIC_ALL`: any of them lets
    /// the holder read the file's contents.
    const READS: u32 = 0x0000_0001 | 0x8000_0000 | 0x1000_0000;

    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    let mut dacl: *mut ACL = null_mut();
    let mut descriptor: PSECURITY_DESCRIPTOR = null_mut();
    // SAFETY: `wide` is NUL-terminated; the out-pointers are valid; the
    // descriptor Windows returns is freed below with `LocalFree`, and `dacl`
    // points into it and is not used after that.
    unsafe {
        let status = GetNamedSecurityInfoW(
            wide.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            null_mut(),
            null_mut(),
            &mut dacl,
            null_mut(),
            &mut descriptor,
        );
        if status != ERROR_SUCCESS {
            return Err(format!(
                "its access list could not be read (error {status})"
            ));
        }
        let result = (|| {
            if dacl.is_null() {
                return Err("has no access list, which grants everyone full access".to_string());
            }
            let broad: Vec<(WELL_KNOWN_SID_TYPE, &str)> = vec![
                (WinWorldSid, "Everyone"),
                (WinAuthenticatedUserSid, "Authenticated Users"),
                (WinBuiltinUsersSid, "Users"),
            ];
            let mut sids = Vec::new();
            for (kind, name) in broad {
                let mut buf = vec![0u8; SECURITY_MAX_SID_SIZE as usize];
                let mut size = buf.len() as u32;
                if CreateWellKnownSid(kind, null_mut(), buf.as_mut_ptr().cast(), &mut size) == 0 {
                    return Err(format!("the {name} identity could not be built"));
                }
                sids.push((buf, name));
            }
            let mut info: ACL_SIZE_INFORMATION = std::mem::zeroed();
            if GetAclInformation(
                dacl,
                (&mut info as *mut ACL_SIZE_INFORMATION).cast(),
                std::mem::size_of::<ACL_SIZE_INFORMATION>() as u32,
                AclSizeInformation,
            ) == 0
            {
                return Err("its access list could not be walked".to_string());
            }
            for i in 0..info.AceCount {
                let mut ace: *mut std::ffi::c_void = null_mut();
                if GetAce(dacl, i, &mut ace) == 0 {
                    continue;
                }
                let header = &*(ace as *const ACE_HEADER);
                if header.AceType != ALLOWED {
                    continue;
                }
                let allowed = &*(ace as *const ACCESS_ALLOWED_ACE);
                if allowed.Mask & READS == 0 {
                    continue;
                }
                let sid = (&allowed.SidStart as *const u32).cast_mut().cast();
                for (broad, name) in &sids {
                    if EqualSid(sid, broad.as_ptr().cast_mut().cast()) != 0 {
                        return Err(format!(
                            "grants read access to {name}; restrict it to your own account"
                        ));
                    }
                }
            }
            Ok(())
        })();
        LocalFree(descriptor.cast());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn private_file() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = dir.path().join("secrets.yaml");
        std::fs::write(&path, b"k: v\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        (dir, path)
    }

    #[test]
    fn a_private_file_passes() {
        let (_dir, path) = private_file();
        assert_eq!(check(&path), Ok(()));
    }

    #[cfg(unix)]
    #[test]
    fn a_readable_file_or_a_writable_folder_fails() {
        use std::os::unix::fs::PermissionsExt;
        for mode in [0o644, 0o640, 0o604, 0o660] {
            let (_dir, path) = private_file();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            assert!(check(&path).is_err(), "mode {mode:o}");
        }
        let (dir, path) = private_file();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
        let e = check(&path).unwrap_err();
        assert!(e.contains("other users can write"), "{e}");
    }

    /// Granting Everyone read on the file is caught.
    #[cfg(windows)]
    #[test]
    fn a_file_everyone_can_read_fails() {
        let (_dir, path) = private_file();
        let out = std::process::Command::new("icacls")
            .arg(&path)
            .args(["/grant", "*S-1-1-0:R"])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let e = check(&path).unwrap_err();
        assert!(e.contains("Everyone"), "{e}");
    }
}
