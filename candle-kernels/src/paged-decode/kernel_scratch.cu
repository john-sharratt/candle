// The allocator the launchers' scratch is taken from, the release a replaced
// block goes back through, and the epoch a block is valid for — see
// `grow_scratch.cuh`. Installed once by the host, before any launch.

#include "../grow_scratch.cuh"

extern "C" {

candle_kernel_scratch_fn g_candle_kernel_scratch = nullptr;
candle_kernel_scratch_release_fn g_candle_kernel_scratch_release = nullptr;
unsigned long long g_candle_kernel_scratch_epoch = 0;

void candle_set_kernel_scratch(candle_kernel_scratch_fn take,
                               candle_kernel_scratch_release_fn release) {
    g_candle_kernel_scratch = take;
    g_candle_kernel_scratch_release = release;
}

void candle_set_kernel_scratch_epoch(unsigned long long epoch) {
    g_candle_kernel_scratch_epoch = epoch;
}

}
