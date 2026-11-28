// The launch error pending on the calling thread.
//
// Most launchers in this crate launch with `<<<>>>` and never read the result,
// so a launch that fails — an empty grid, an oversized block — leaves its
// error pending until the next `cudaGetLastError()` on the thread, which
// belongs to whoever happens to call it. A caller that runs many launches for
// one piece of work takes it after each step, so the failure is named as that
// step's and not left for the next model on the thread to trip over.

#include <cuda_runtime.h>

// Take the error pending on the calling thread, if any, and clear it:
// `cudaSuccess` (0) when there is none.
extern "C" int take_pending_launch_error() {
    return (int)cudaGetLastError();
}
