//! The launch error pending on the calling thread (`simple/launch_error.cu`).

extern "C" {
    /// Take the CUDA error pending on the calling thread, if any, and clear
    /// it: 0 when there is none.
    ///
    /// A launcher that does not read its launch's result leaves a failure
    /// pending for the next reader on the thread. Work that runs many launches
    /// takes it after each step, so a failure is named as that step's — never
    /// left for the next model on the thread, whose own launcher would refuse
    /// to run on it.
    pub fn take_pending_launch_error() -> i32;
}
