//! Killing a process together with everything it started.
//!
//! `git push` runs `ssh`, and a hook or alias runs a shell: killing only the
//! `git` process on a timeout would leave those running, holding the
//! connection or the lock. A sandboxed command is killed the same way, so
//! nothing it started outlives its run on the checkout. On Windows the child is put in a Job Object that
//! kills its members when terminated and when the job's handle is closed; on
//! Unix the child leads its own process group, and the group is signalled.
//!
//! On Windows the child is assigned to the job just after it starts, so a
//! grandchild spawned in the first microseconds of its life would escape the
//! job. git starts `ssh` and hook shells far later than that.
//!
//! A command can be started below normal priority ([`Priority::Below`]): on
//! Windows with the below-normal creation flag, which what it starts
//! inherits, and the job's priority-class limit, which holds every process in
//! the job there; on Unix with a nice of ten taken before the program runs,
//! which what it starts inherits and none of it can lower again without
//! privilege.

use std::io;
use std::process::{Child, Command};

use crate::sandbox::Priority;

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
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PRIORITY_CLASS,
    };
    use windows_sys::Win32::System::Threading::BELOW_NORMAL_PRIORITY_CLASS;

    use super::Priority;

    /// No console window for a child of a daemon that has none.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    pub fn prepare(command: &mut Command, priority: Priority) {
        let class = match priority {
            Priority::Normal => 0,
            Priority::Below => BELOW_NORMAL_PRIORITY_CLASS,
        };
        command.creation_flags(CREATE_NO_WINDOW | class);
    }

    pub struct Tree {
        job: HANDLE,
    }

    // A job handle may be used from any thread.
    unsafe impl Send for Tree {}
    unsafe impl Sync for Tree {}

    impl Tree {
        pub fn adopt(child: &Child, priority: Priority) -> io::Result<Self> {
            Self::adopt_handle(child.as_raw_handle() as HANDLE, priority)
        }

        pub fn adopt_async(child: &tokio::process::Child, priority: Priority) -> io::Result<Self> {
            let handle = child.raw_handle().ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "the child has already exited")
            })?;
            Self::adopt_handle(handle as HANDLE, priority)
        }

        /// Put the process behind `process` — a handle its owner keeps open —
        /// into a new job, held at `priority`.
        fn adopt_handle(process: HANDLE, priority: Priority) -> io::Result<Self> {
            let (flags, class) = match priority {
                Priority::Normal => (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, 0),
                Priority::Below => (
                    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_PRIORITY_CLASS,
                    BELOW_NORMAL_PRIORITY_CLASS,
                ),
            };
            // SAFETY: plain Win32 calls on a handle this function owns, and on
            // the child's process handle, which the caller keeps open; `info`
            // outlives the call that reads it.
            unsafe {
                let job = CreateJobObjectW(null(), null());
                if job.is_null() {
                    return Err(io::Error::last_os_error());
                }
                let tree = Self { job };
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
                info.BasicLimitInformation.LimitFlags = flags;
                info.BasicLimitInformation.PriorityClass = class;
                if SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ) == 0
                {
                    return Err(io::Error::last_os_error());
                }
                if AssignProcessToJobObject(job, process) == 0 {
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

    use super::Priority;

    /// The nice a command started below normal priority takes.
    const BELOW_NICE: libc::c_int = 10;

    pub fn prepare(command: &mut Command, priority: Priority) {
        command.process_group(0);
        if priority == Priority::Below {
            // SAFETY: runs in the child between fork and exec, and calls only
            // `setpriority` and `errno`, which are async-signal-safe.
            unsafe {
                command.pre_exec(|| {
                    if libc::setpriority(libc::PRIO_PROCESS as _, 0, BELOW_NICE) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
    }

    pub struct Tree {
        group: i32,
    }

    impl Tree {
        pub fn adopt(child: &Child, _priority: Priority) -> io::Result<Self> {
            Ok(Self {
                group: child.id() as i32,
            })
        }

        pub fn adopt_async(child: &tokio::process::Child, _priority: Priority) -> io::Result<Self> {
            let id = child.id().ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "the child has already exited")
            })?;
            Ok(Self { group: id as i32 })
        }

        pub fn kill(&self) {
            // SAFETY: signalling a process group this process created.
            unsafe {
                libc::kill(-self.group, libc::SIGKILL);
            }
        }
    }

    impl Drop for Tree {
        /// Whatever is still in the group goes with the last handle on it, as
        /// a Windows job's does — so a run whose future is dropped part way
        /// leaves nothing of its command running.
        fn drop(&mut self) {
            self.kill();
        }
    }
}

/// Set `command` up to start, at `priority`, a tree [`ProcessTree::adopt`]
/// can kill.
pub(crate) fn prepare(command: &mut Command, priority: Priority) {
    imp::prepare(command, priority);
}

/// A started child and everything it goes on to start.
pub(crate) struct ProcessTree(imp::Tree);

impl ProcessTree {
    /// The tree of `child`, started by [`prepare`] at `priority`, which the
    /// tree holds what it starts at.
    pub(crate) fn adopt(child: &Child, priority: Priority) -> io::Result<Self> {
        imp::Tree::adopt(child, priority).map(Self)
    }

    /// [`Self::adopt`] for a child started by tokio. Fails once the child has
    /// been reaped.
    pub(crate) fn adopt_async(
        child: &tokio::process::Child,
        priority: Priority,
    ) -> io::Result<Self> {
        imp::Tree::adopt_async(child, priority).map(Self)
    }

    /// Kill every process in the tree.
    pub(crate) fn kill(&self) {
        self.0.kill();
    }
}
