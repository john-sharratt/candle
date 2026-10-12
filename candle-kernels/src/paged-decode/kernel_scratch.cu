// The allocator the launchers' grow-only scratch is taken from — see
// `grow_scratch.cuh`. Installed once by the host, before any launch.

#include "../grow_scratch.cuh"

extern "C" {

candle_kernel_scratch_fn g_candle_kernel_scratch = nullptr;

void candle_set_kernel_scratch(candle_kernel_scratch_fn take) {
    g_candle_kernel_scratch = take;
}

}
