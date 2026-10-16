//! Let the account keep runs as manage what keep starts.
//!
//! keep runs from a boot task as an S4U logon. A token from that logon carries
//! a default DACL granting its own logon session and SYSTEM — not the user's
//! interactive session, though it is the same account. Every object a child
//! creates is stamped with that DACL, so a shell the user opens cannot stop
//! motherd, read its threads' times, or sample them without elevating: every
//! restart and every look inside was a UAC prompt.
//!
//! [`grant_owner`] replaces this process's default DACL, before any child is
//! started, with one granting the token's own user and SYSTEM. Children inherit
//! the token, and with it the DACL on every process and thread they create.

use std::io;

use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE};
use windows_sys::Win32::Security::{
    AddAccessAllowedAce, CreateWellKnownSid, GetLengthSid, GetTokenInformation, InitializeAcl,
    SetTokenInformation, TokenDefaultDacl, TokenUser, WinLocalSystemSid, ACL, ACL_REVISION,
    SECURITY_MAX_SID_SIZE, TOKEN_ADJUST_DEFAULT, TOKEN_DEFAULT_DACL, TOKEN_QUERY, TOKEN_USER,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

/// Every right on the object.
const GENERIC_ALL: u32 = 0x1000_0000;

fn last_error(what: &str) -> io::Error {
    // SAFETY: reads the calling thread's last-error value.
    let code = unsafe { GetLastError() };
    io::Error::other(format!("{what}: Win32 error {code}"))
}

/// This process's token, opened for `access`; closed on drop.
struct Token(HANDLE);

impl Token {
    fn open(access: u32) -> io::Result<Token> {
        let mut handle: HANDLE = std::ptr::null_mut();
        // SAFETY: the pseudo-handle of the current process is always valid,
        // and `handle` is a valid out-pointer.
        if unsafe { OpenProcessToken(GetCurrentProcess(), access, &mut handle) } == 0 {
            return Err(last_error("OpenProcessToken"));
        }
        Ok(Token(handle))
    }

    /// The bytes of one information class.
    fn info(&self, class: i32) -> io::Result<Vec<u8>> {
        let mut len = 0u32;
        // SAFETY: a null buffer of length 0 asks only for the size.
        unsafe { GetTokenInformation(self.0, class, std::ptr::null_mut(), 0, &mut len) };
        let mut buf = vec![0u8; len as usize];
        // SAFETY: `buf` holds `len` bytes, the size the first call reported.
        if unsafe { GetTokenInformation(self.0, class, buf.as_mut_ptr().cast(), len, &mut len) }
            == 0
        {
            return Err(last_error("GetTokenInformation"));
        }
        Ok(buf)
    }
}

impl Drop for Token {
    fn drop(&mut self) {
        // SAFETY: the handle was opened by `Token::open` and is closed once.
        unsafe { CloseHandle(self.0) };
    }
}

/// Make this process's default DACL grant its token's user and SYSTEM every
/// right, so the processes and threads its children create can be managed by
/// that user from any of its sessions. See the module documentation.
pub fn grant_owner() -> io::Result<()> {
    let token = Token::open(TOKEN_QUERY | TOKEN_ADJUST_DEFAULT)?;
    let user = token.info(TokenUser)?;
    // SAFETY: a `TokenUser` query fills the buffer with a `TOKEN_USER`, whose
    // SID pointer points into the same buffer.
    let user_sid = unsafe { (*user.as_ptr().cast::<TOKEN_USER>()).User.Sid };

    let mut system = vec![0u8; SECURITY_MAX_SID_SIZE as usize];
    let mut system_len = system.len() as u32;
    // SAFETY: `system` holds `system_len` bytes, the largest SID there is.
    if unsafe {
        CreateWellKnownSid(
            WinLocalSystemSid,
            std::ptr::null_mut(),
            system.as_mut_ptr().cast(),
            &mut system_len,
        )
    } == 0
    {
        return Err(last_error("CreateWellKnownSid"));
    }
    let system_sid = system.as_mut_ptr().cast();

    // An ACL header, and per ACE its header and access mask, then the SID.
    // SAFETY: both SIDs are valid, initialised above.
    let sids_len = unsafe { GetLengthSid(user_sid) + GetLengthSid(system_sid) };
    let ace_fixed = 8u32;
    let acl_len =
        (std::mem::size_of::<ACL>() as u32 + 2 * ace_fixed + sids_len).next_multiple_of(4);
    // `u32`s, so the ACL the API writes into is DWORD-aligned.
    let mut acl = vec![0u32; acl_len.div_ceil(4) as usize];
    let pacl = acl.as_mut_ptr().cast::<ACL>();
    // SAFETY: `acl` holds `acl_len` bytes, sized for exactly these two ACEs.
    unsafe {
        if InitializeAcl(pacl, acl_len, ACL_REVISION) == 0 {
            return Err(last_error("InitializeAcl"));
        }
        if AddAccessAllowedAce(pacl, ACL_REVISION, GENERIC_ALL, user_sid) == 0 {
            return Err(last_error("AddAccessAllowedAce (user)"));
        }
        if AddAccessAllowedAce(pacl, ACL_REVISION, GENERIC_ALL, system_sid) == 0 {
            return Err(last_error("AddAccessAllowedAce (SYSTEM)"));
        }
    }
    let dacl = TOKEN_DEFAULT_DACL { DefaultDacl: pacl };
    // SAFETY: `dacl` points at the ACL above, which outlives the call; the
    // token copies it.
    if unsafe {
        SetTokenInformation(
            token.0,
            TokenDefaultDacl,
            (&dacl as *const TOKEN_DEFAULT_DACL).cast(),
            std::mem::size_of::<TOKEN_DEFAULT_DACL>() as u32,
        )
    } == 0
    {
        return Err(last_error("SetTokenInformation(TokenDefaultDacl)"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use windows_sys::Win32::Security::{EqualSid, GetAce, ACCESS_ALLOWED_ACE, PSID};

    /// After the grant, the default DACL holds exactly two ACEs, every right to
    /// the token's own user and then to SYSTEM.
    #[test]
    fn the_default_dacl_grants_the_user_and_system() {
        grant_owner().unwrap();
        let token = Token::open(TOKEN_QUERY).unwrap();
        let user = token.info(TokenUser).unwrap();
        // SAFETY: as in `grant_owner`.
        let user_sid = unsafe { (*user.as_ptr().cast::<TOKEN_USER>()).User.Sid };
        let dacl = token.info(TokenDefaultDacl).unwrap();
        // SAFETY: a `TokenDefaultDacl` query fills the buffer with a
        // `TOKEN_DEFAULT_DACL` pointing into the same buffer.
        let acl = unsafe { (*dacl.as_ptr().cast::<TOKEN_DEFAULT_DACL>()).DefaultDacl };
        // SAFETY: `acl` is the token's own, valid ACL.
        assert_eq!(unsafe { (*acl).AceCount }, 2);
        let ace = |i: u32| -> (u32, PSID) {
            let mut p: *mut core::ffi::c_void = std::ptr::null_mut();
            // SAFETY: `i` is below the ACE count asserted above.
            assert_ne!(unsafe { GetAce(acl, i, &mut p) }, 0);
            let ace = p.cast::<ACCESS_ALLOWED_ACE>();
            // SAFETY: both ACEs are access-allowed ACEs, the only kind added.
            unsafe { ((*ace).Mask, std::ptr::addr_of_mut!((*ace).SidStart).cast()) }
        };
        let (mask, sid) = ace(0);
        assert_eq!(mask, GENERIC_ALL);
        // SAFETY: both SIDs are valid.
        assert_ne!(
            unsafe { EqualSid(sid, user_sid) },
            0,
            "the first is the user"
        );
        assert_eq!(ace(1).0, GENERIC_ALL);
    }
}
