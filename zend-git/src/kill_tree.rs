//! Killing a git process together with everything it started.
//!
//! `git push` runs `ssh`, and a hook or alias runs a shell: killing only the
//! `git` process on a timeout would leave those running, holding the
//! connection or the lock. On Windows the child is put in a Job Object that
//! kills its members when terminated and when the job's handle is closed; on
//! Unix the child leads its own process group, and the group is signalled.
//!
//! On Windows the child is assigned to the job just after it starts, so a
//! grandchild spawned in the first microseconds of its life would escape the
//! job. git starts `ssh` and hook shells far later than that.

use std::io;
use std::process::{Child, Command};

#[cfg(windows)]
mod imp {
    use std::io;
    use std::mem::{size_of, zeroed};
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    use std::process::{Child, Command};
    use std::ptr::null;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    /// No console window for a child of a daemon that has none.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    pub fn prepare(command: &mut Command) {
        command.creation_flags(CREATE_NO_WINDOW);
    }

    pub struct Tree {
        job: HANDLE,
    }

    // A job handle may be used from any thread.
    unsafe impl Send for Tree {}
    unsafe impl Sync for Tree {}

    impl Tree {
        pub fn adopt(child: &Child) -> io::Result<Self> {
            // SAFETY: plain Win32 calls on a handle this function owns, and on
            // the child's process handle, which `child` keeps open.
            unsafe {
                let job = CreateJobObjectW(null(), null());
                if job.is_null() {
                    return Err(io::Error::last_os_error());
                }
                let tree = Self { job };
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                if SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ) == 0
                {
                    return Err(io::Error::last_os_error());
                }
                if AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE) == 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(tree)
            }
        }

        pub fn kill(&self) {
            // SAFETY: `job` is a live job handle owned by `self`.
            unsafe {
                TerminateJobObject(self.job, 1);
            }
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            // Closing the last handle kills whatever is still in the job.
            // SAFETY: `job` is owned by `self` and closed exactly once.
            unsafe {
                CloseHandle(self.job);
            }
        }
    }
}

#[cfg(unix)]
mod imp {
    use std::io;
    use std::os::unix::process::CommandExt;
    use std::process::{Child, Command};

    pub fn prepare(command: &mut Command) {
        command.process_group(0);
    }

    pub struct Tree {
        group: i32,
    }

    impl Tree {
        pub fn adopt(child: &Child) -> io::Result<Self> {
            Ok(Self {
                group: child.id() as i32,
            })
        }

        pub fn kill(&self) {
            // SAFETY: signalling a process group this process created.
            unsafe {
                libc::kill(-self.group, libc::SIGKILL);
            }
        }
    }
}

/// Set `command` up to start a tree [`ProcessTree::adopt`] can kill.
pub(crate) fn prepare(command: &mut Command) {
    imp::prepare(command);
}

/// A started child and everything it goes on to start.
pub(crate) struct ProcessTree(imp::Tree);

impl ProcessTree {
    pub(crate) fn adopt(child: &Child) -> io::Result<Self> {
        imp::Tree::adopt(child).map(Self)
    }

    /// Kill every process in the tree.
    pub(crate) fn kill(&self) {
        self.0.kill();
    }
}
