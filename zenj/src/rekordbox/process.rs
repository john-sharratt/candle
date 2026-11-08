//! rekordbox as a process (`docs/house_modes.md` §3): whether it runs, opening
//! it, and closing it the way a person does — asked through its window, so it
//! saves its library first — never killed. It is opened from the console
//! session `zenj` itself runs in, the only one with the desktop, the FLX4 and
//! the screens.
//!
//! Both waits block: callers run them off the runtime. Each is inside the
//! longest the gateway's bus holds an ask open (30 s), so the house's mode
//! switch hears how it went rather than a timeout.

use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

/// How long an opened rekordbox has to appear as a process.
pub const OPEN_WITHIN: Duration = Duration::from_secs(25);

/// How long a rekordbox asked to close has to go: it saves its library first.
pub const CLOSE_WITHIN: Duration = Duration::from_secs(25);

/// How often a wait looks.
const LOOK: Duration = Duration::from_millis(250);

/// Whether a process whose executable is `exe` runs.
pub fn running(exe: &str) -> bool {
    id_of(exe).is_some()
}

/// Why closing rekordbox is refused, if it is: a deck is playing, and the
/// person asking did not say to close it anyway.
pub fn refuse_close(playing: bool, force: bool) -> Option<String> {
    (playing && !force).then(|| {
        "a deck is playing: rekordbox is closed only once the music has stopped, or when a \
         person says to close it anyway"
            .to_string()
    })
}

/// Open `program`, rekordbox's executable, unless `exe` already runs; the
/// line saying what came of it.
pub fn open(program: &Path, exe: &str) -> Result<String, String> {
    if running(exe) {
        return Ok("rekordbox is already open".into());
    }
    let mut cmd = Command::new(program);
    if let Some(dir) = program.parent() {
        cmd.current_dir(dir);
    }
    cmd.spawn()
        .map_err(|e| format!("rekordbox could not be started from {}: {e}", program.display()))?;
    match until(|| running(exe), OPEN_WITHIN) {
        true => Ok("rekordbox is open".into()),
        false => Err(format!(
            "rekordbox was started and has not appeared within {} seconds",
            OPEN_WITHIN.as_secs()
        )),
    }
}

/// Ask the running `exe` to close through its windows, and wait for it to go;
/// the line saying what came of it.
pub fn close(exe: &str) -> Result<String, String> {
    let Some(pid) = id_of(exe) else {
        return Ok("rekordbox is already closed".into());
    };
    if ask_to_close(pid) == 0 {
        return Err("rekordbox shows no window to close".into());
    }
    match until(|| !running(exe), CLOSE_WITHIN) {
        true => Ok("rekordbox is closed".into()),
        false => Err(format!(
            "rekordbox was asked to close and is still open after {} seconds — it may be \
             asking something on its screen",
            CLOSE_WITHIN.as_secs()
        )),
    }
}

/// Whether `done` holds within `within`, looked at every [`LOOK`].
fn until(done: impl Fn() -> bool, within: Duration) -> bool {
    let start = Instant::now();
    loop {
        if done() {
            return true;
        }
        if start.elapsed() >= within {
            return false;
        }
        thread::sleep(LOOK);
    }
}

#[cfg(windows)]
mod windows {
    use std::mem::size_of;

    use windows_sys::Win32::Foundation::{CloseHandle, BOOL, HWND, INVALID_HANDLE_VALUE, LPARAM};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W,
        TH32CS_SNAPPROCESS,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetWindow, GetWindowThreadProcessId, IsWindowVisible, PostMessageW,
        GW_OWNER, WM_CLOSE,
    };

    fn wide_name(raw: &[u16]) -> String {
        let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
        String::from_utf16_lossy(&raw[..end])
    }

    /// The id of the first process whose executable is `exe`.
    pub fn id_of(exe: &str) -> Option<u32> {
        // SAFETY: a snapshot handle is checked before use and closed after;
        // the entry is sized as the API requires.
        unsafe {
            let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if snap == INVALID_HANDLE_VALUE {
                return None;
            }
            let mut entry: PROCESSENTRY32W = std::mem::zeroed();
            entry.dwSize = size_of::<PROCESSENTRY32W>() as u32;
            let mut found = None;
            if Process32FirstW(snap, &mut entry) != 0 {
                loop {
                    if wide_name(&entry.szExeFile).eq_ignore_ascii_case(exe) {
                        found = Some(entry.th32ProcessID);
                        break;
                    }
                    if Process32NextW(snap, &mut entry) == 0 {
                        break;
                    }
                }
            }
            CloseHandle(snap);
            found
        }
    }

    struct Asking {
        pid: u32,
        asked: usize,
    }

    unsafe extern "system" fn each(hwnd: HWND, lparam: LPARAM) -> BOOL {
        // SAFETY: `lparam` is the `Asking` `ask_to_close` lent for the
        // duration of the enumeration.
        let asking = unsafe { &mut *(lparam as *mut Asking) };
        let mut pid = 0u32;
        // SAFETY: `hwnd` is a window the enumeration handed over.
        unsafe {
            GetWindowThreadProcessId(hwnd, &mut pid);
            // Its own top-level windows, as a person sees them: closing the
            // main one is closing the application.
            if pid == asking.pid
                && IsWindowVisible(hwnd) != 0
                && GetWindow(hwnd, GW_OWNER).is_null()
                && PostMessageW(hwnd, WM_CLOSE, 0, 0) != 0
            {
                asking.asked += 1;
            }
        }
        1
    }

    /// Post a close to each visible top-level window of process `pid`; how
    /// many were asked.
    pub fn ask_to_close(pid: u32) -> usize {
        let mut asking = Asking { pid, asked: 0 };
        // SAFETY: `each` reads `asking` only while `EnumWindows` runs.
        unsafe {
            EnumWindows(Some(each), &mut asking as *mut Asking as LPARAM);
        }
        asking.asked
    }
}

#[cfg(windows)]
pub use windows::id_of;
#[cfg(windows)]
use windows::ask_to_close;

/// rekordbox runs on Windows here: elsewhere there is no process to find.
#[cfg(not(windows))]
pub fn id_of(_exe: &str) -> Option<u32> {
    None
}

#[cfg(not(windows))]
fn ask_to_close(_pid: u32) -> usize {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A playing rekordbox is closed only when a person says so**; a quiet
    /// one is closed when asked.
    #[test]
    fn a_playing_rekordbox_is_closed_only_when_forced() {
        assert_eq!(refuse_close(false, false), None);
        assert_eq!(refuse_close(false, true), None);
        assert_eq!(refuse_close(true, true), None);
        assert!(refuse_close(true, false)
            .unwrap()
            .starts_with("a deck is playing"));
    }

    /// **A process that is not running is found as not running, and closing
    /// it is already done** — nothing is started or asked.
    #[test]
    fn what_is_not_running_is_closed_already() {
        let exe = "zenj-no-such-process.exe";
        assert!(!running(exe));
        assert_eq!(close(exe), Ok("rekordbox is already closed".into()));
    }

    #[test]
    fn a_wait_ends_when_its_condition_holds_or_its_time_is_up() {
        assert!(until(|| true, Duration::ZERO));
        let start = Instant::now();
        assert!(!until(|| false, Duration::from_millis(300)));
        assert!(start.elapsed() >= Duration::from_millis(300));
    }

    /// **A program that cannot be started says so**, naming where it was
    /// looked for.
    #[test]
    fn a_program_that_is_not_there_is_refused() {
        let err = open(
            Path::new("Z:/no/such/rekordbox.exe"),
            "zenj-no-such-process.exe",
        )
        .unwrap_err();
        assert!(err.starts_with("rekordbox could not be started from Z:/no/such/rekordbox.exe"));
    }
}
