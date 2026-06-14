@echo off
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d C:\Users\johna\prog\candle
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe" -k "regex:int8_decode_stripe_kernel" -c 1 --section SpeedOfLight --section Occupancy --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis .\target\release\examples\decode_ab.exe compare --scenarios perf_b8_ctx2048 --formats rq-uni-q8_0-L0 > ncu_full.txt 2>&1
echo NCU_DONE exit=%errorlevel% >> ncu_full.txt
