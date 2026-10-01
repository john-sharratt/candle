//! Make the daemon hear Ctrl-C however it was launched.
//!
//! Windows keeps a per-process "ignore Ctrl-C" flag, and a process inherits it
//! from the one that started it. A launcher that ignores Ctrl-C itself — a
//! PowerShell host driving `Start-Process`, a service wrapper, the self-heal
//! relaunch — therefore starts zend already ignoring it, and installing a handler
//! does not clear the flag: `tokio::signal::ctrl_c` registers its handler and
//! then never fires. The stop procedure delivers Ctrl-C to the daemon's console
//! to get the drain-and-flush shutdown, so a daemon deaf to it can only be
//! killed, which skips the flush. Observed on this deployment: two daemons in a
//! row received Ctrl-C and logged nothing, and both had to be force-stopped.
//!
//! Clearing the flag at startup — `SetConsoleCtrlHandler(NULL, FALSE)`, the
//! documented way to restore normal Ctrl-C processing — makes the handler
//! reachable whatever the launcher did.

/// Restore Ctrl-C delivery to this process. A no-op off Windows, where a
/// signal's disposition is reset by the handler `tokio` installs.
pub fn accept_ctrl_c() {
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn SetConsoleCtrlHandler(
                handler: Option<unsafe extern "system" fn(u32) -> i32>,
                add: i32,
            ) -> i32;
        }
        // SAFETY: a null handler with `add = FALSE` only clears the process's
        // ignore-Ctrl-C attribute; no callback is registered or dereferenced.
        let ok = unsafe { SetConsoleCtrlHandler(None, 0) };
        if ok == 0 {
            tracing::warn!(
                "could not restore Ctrl-C delivery — a stop will have to force-kill, \
                 skipping the substrate flush"
            );
        }
    }
}
