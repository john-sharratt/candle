# INSTALL — Windows build setup for `zend` / candle (CUDA)

This guide takes a **fresh Windows 11 machine** to a working
`cargo build -p zend --release`. The `zend` daemon (and most of this workspace's
production code) hard-depends on the candle `cuda` feature, so a working
**NVIDIA CUDA toolchain + MSVC host compiler** is mandatory — there is no
CPU-only build of `zend`.

The single most important rule: **use CUDA Toolkit 12.9. Do not use 12.4 or
13.x.** See [Step 4](#step-4--cuda-toolkit-129) for why.

---

## 0. Hardware / driver prerequisites

- An NVIDIA GPU. Development is on an **RTX PRO 5000 (Blackwell, sm_120)**; any
  recent NVIDIA card works for building (kernels compile to `sm_89` and JIT to
  your GPU at runtime).
- A current **NVIDIA display driver** (this machine: `596.59`, exposes CUDA 13.2
  runtime). Install the driver from GeForce/NVIDIA Studio separately and keep it
  current — **do not** let the CUDA Toolkit installer downgrade it (see Step 4).

Verify the driver:

```bash
nvidia-smi
```

You should see your GPU and a `CUDA Version: 13.x` ceiling (that's the *driver's*
max runtime — it does **not** mean you should install the 13.x toolkit).

---

## Step 1 — Rust (rustup + MSVC toolchain)

> Done manually on this machine; documented here for a clean setup.

1. Install **rustup** from <https://rustup.rs> (download and run `rustup-init.exe`).
   When prompted, accept the default host triple **`x86_64-pc-windows-msvc`**.
   The MSVC ABI is required — do **not** use the `gnu` toolchain; cudarc and the
   CUDA libs link against MSVC.

2. Confirm:

   ```bash
   rustc --version   # rustc 1.96.0 or newer
   cargo --version
   rustup show       # active: stable-x86_64-pc-windows-msvc
   ```

There is no `rust-toolchain.toml` pin in this repo, so stable is fine.

You also need **Git** (<https://git-scm.com/download/win>) — Git for Windows
ships the `bash` shell these instructions assume.

---

## Step 2 — Visual Studio 2022 + MSVC + Windows SDK

NVCC needs a host C++ compiler (`cl.exe`) plus the C++ standard headers/libs and
the Windows SDK. Install **Visual Studio 2022** (Community is fine) **or** the
standalone **Build Tools for Visual Studio 2022**:

- Download: <https://visualstudio.microsoft.com/downloads/> → "Build Tools for
  Visual Studio 2022" (or the Community installer).
- In the installer, select the **"Desktop development with C++"** workload. That
  brings in:
  - **MSVC v143 — VS 2022 C++ x64/x86 build tools** (provides `cl.exe`; this
    machine has `14.44.35207`)
  - **Windows 10/11 SDK** (this machine has `10.0.26100.0`)

This is what provides `cl.exe`, `INCLUDE`, `LIB`, and `rc.exe`. Without it, nvcc
fails with `Cannot find compiler 'cl.exe' in PATH`.

---

## Step 3 — confirm there is no CUDA toolkit yet

A clean machine has **no `nvcc`**:

```bash
nvcc --version    # 'program not found' on a clean box — expected
```

If an older/newer CUDA toolkit is already installed, note its version; you want
exactly **12.9** on `PATH` for this build (Step 4).

---

## Step 4 — CUDA Toolkit 12.9

### Why 12.9 specifically (read this)

The usable nvcc window is **≥ 12.8 and ≤ 13.0**. 12.9 is the pick.

- **The kernels now compile native SASS for Blackwell (`sm_120`)** in addition
  to Ada (`sm_89`) — see the `-gencode=arch=compute_120,…` flags in
  `candle-kernels/build_utils.rs`. `compute_120` codegen **requires nvcc 12.8 or
  newer**; CUDA **12.4 now fails** the build with
  *`nvcc fatal: Unsupported gpu architecture 'compute_120'`*. (This is the
  change from the old 12.4 guidance.)
- **cudarc 0.17.8** (pinned in the workspace `Cargo.toml`, `dynamic-linking` +
  `cuda-version-from-build-system`) only recognizes nvcc versions **up to 13.0**.
  A 13.x toolkit makes its build script panic with
  *"Unsupported cuda toolkit version: 13.x"* unless overridden, and 13.x CCCL
  headers also reject MSVC's traditional preprocessor
  (`#error: MSVC/cl.exe with traditional preprocessor is used…`).
- The earlier objection that *"several `paged-decode`/`paged-prefill` kernels
  exceed the 48 KB static-shared-memory `ptxas` limit"* **no longer applies** —
  the prefill triple-buffering path was moved to **dynamic** shared memory
  (opted-in via `cudaFuncSetAttribute`), so it no longer trips the 48 KB static
  cap. The remaining reason to avoid 13.x is the cudarc/CCCL incompatibility
  above, not the kernels.

> `winget install Nvidia.CUDA` installs the newest (13.x) — too new. Use the
> direct 12.9.1 installer below instead.

### Install (toolkit only — never the driver)

1. Download the CUDA **12.9.1** local installer:

   ```bash
   curl -L --ssl-no-revoke -o /d/cuda_12.9.1_installer.exe \
     "https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_576.57_windows.exe"
   ```

   (`--ssl-no-revoke` works around a common Windows schannel
   `CRYPT_E_NO_REVOCATION_CHECK` error on corporate/VPN networks.)

2. Install **toolkit components only**, explicitly excluding `Display.Driver`.
   CUDA 12.9.1 bundles driver `576.57`, which would **downgrade** your modern
   driver and can break a Blackwell GPU — so we list subpackages and omit the
   driver. From a shell (a UAC elevation prompt will appear):

   ```bash
   /d/cuda_12.9.1_installer.exe -s \
     nvcc_12.9 cudart_12.9 cuobjdump_12.9 nvprune_12.9 cuxxfilt_12.9 \
     nvfatbin_12.9 nvjitlink_12.9 cuda_profiler_api_12.9 \
     nvrtc_12.9 nvrtc_dev_12.9 cublas_12.9 cublas_dev_12.9 \
     curand_12.9 curand_dev_12.9 cusparse_12.9 cusparse_dev_12.9 \
     cusolver_12.9 cusolver_dev_12.9 cufft_12.9 cufft_dev_12.9 \
     thrust_12.9 nvtx_12.9
   ```

   > If running from Git Bash, prefix with `MSYS_NO_PATHCONV=1` and call via
   > `cmd.exe /c "D:\cuda_12.9.1_installer.exe -s …"` so the flags aren't mangled.

3. Verify the toolkit installed and the driver is **unchanged**:

   ```bash
   "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe" --version
   #   → Cuda compilation tools, release 12.9, V12.9.xx
   nvidia-smi --query-gpu=driver_version --format=csv,noheader
   #   → still your original driver (e.g. 596.59)
   ```

---

## Step 5 — environment variables

The build needs three things wired up: the CUDA toolkit on `PATH`/`CUDA_PATH`,
the cudarc version override, and the MSVC host-compiler environment.

### CUDA (permanent, machine/user scope)

The 12.9 installer sets `CUDA_PATH` and `CUDA_PATH_V12_9`, but if any other CUDA
version is also installed, make sure **v12.9 comes first on `PATH`** (this
machine also has stray v12.4 / v13.0 dirs that must not win). Set these
permanently (run PowerShell **as Administrator** for the `Machine` scope line):

```powershell
$v129 = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
# Put v12.9 bin first; drop any stray v12.4 / v13.x dirs from Machine PATH:
$parts = ([Environment]::GetEnvironmentVariable('Path','Machine')).Split(';') |
         Where-Object { $_ -ne '' -and $_ -notmatch 'CUDA\\v13' -and $_ -notmatch 'CUDA\\v12\.4' -and $_ -ne "$v129\bin" -and $_ -ne "$v129\libnvvp" }
[Environment]::SetEnvironmentVariable('Path', (@("$v129\bin","$v129\libnvvp") + $parts) -join ';', 'Machine')
[Environment]::SetEnvironmentVariable('CUDA_PATH', $v129, 'Machine')

# Optional: pin cudarc's detected toolkit version. 12.9 is within cudarc's
# recognized range so this is usually unnecessary, but harmless if other CUDA
# installs confuse nvcc detection:
[Environment]::SetEnvironmentVariable('CUDARC_CUDA_VERSION', '12090', 'User')
```

`CUDARC_CUDA_VERSION=12090` forces cudarc's build script to target CUDA 12.9
regardless of what `nvcc --version` reports.

### MSVC host compiler — pick ONE approach

**Approach A (recommended, robust): build from the VS Developer prompt.**
Open **"x64 Native Tools Command Prompt for VS 2022"** from the Start menu. It
runs `vcvars64.bat` automatically, setting `cl.exe` on `PATH` plus `INCLUDE` /
`LIB` / `LIBPATH`. Then just run the build (Step 6) from there. This is
version-independent and survives VS updates.

**Approach B (what this machine uses): persist the MSVC env so *any* shell
works.** This lets a plain `cargo build` succeed without a Developer prompt, at
the cost of pinning version-specific paths (re-run if VS updates the toolset).
Capture and persist the vcvars environment (PowerShell):

```powershell
# 1. Dump the vcvars64 environment
$bat = "$env:TEMP\dumpvc.bat"
@"
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set
"@ | Set-Content -Encoding Ascii $bat
$vc = @{}
cmd /c $bat | ForEach-Object { $i = $_.IndexOf('='); if ($i -gt 0) { $vc[$_.Substring(0,$i)] = $_.Substring($i+1) } }

# 2. Persist INCLUDE / LIB / LIBPATH and the VS+SDK bin dirs to the User env
foreach ($n in 'INCLUDE','LIB','LIBPATH') { [Environment]::SetEnvironmentVariable($n, $vc[$n], 'User') }
$userPath = [Environment]::GetEnvironmentVariable('Path','User'); if (-not $userPath) { $userPath = '' }
$existing = $userPath.Split(';') | Where-Object { $_ -ne '' }
$add = $vc['PATH'].Split(';') | Where-Object { $_ -match 'Microsoft Visual Studio' -or $_ -match 'Windows Kits' }
$new = $add | Where-Object { $_ -ne '' -and $existing -notcontains $_ }
[Environment]::SetEnvironmentVariable('Path', (($existing + $new) -join ';'), 'User')
```

> Adjust the Visual Studio edition in the path (`Community` →
> `BuildTools` / `Professional` / `Enterprise`) to match your install.

After setting permanent env vars, **open a new shell** so they take effect
(env changes don't apply to already-running shells).

---

## Step 6 — build

```bash
cargo build -p zend --release
```

- First build compiles 90+ CUDA kernels (a few minutes) and the full Rust
  dependency graph (~8–9 min total cold). Subsequent builds are incremental and
  reuse the cached kernel archives.
- Output: `target/release/zend.exe`.

Other useful commands:

```bash
cargo build --features cuda            # candle crates with cuda
cargo build --features cuda,cudnn      # + cuDNN
cargo test -p candle-core              # CPU tests
make clean-ptx                         # force full CUDA kernel recompile
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `` `nvcc --version` failed `` / `program not found` | CUDA toolkit not installed or not on `PATH`. Do Step 4; ensure `v12.9\bin` is on `PATH`. |
| `nvcc fatal : Unsupported gpu architecture 'compute_120'` | Your nvcc is **older than 12.8** (e.g. 12.4) and can't target Blackwell `sm_120`. Install **12.9** (Step 4). |
| `Unsupported cuda toolkit version: 13.x` (cudarc build) | You have CUDA 13.x. Install 12.9 and set `CUDARC_CUDA_VERSION=12090` (Step 4–5). |
| `Cannot find compiler 'cl.exe' in PATH` | MSVC not on `PATH`. Use the VS Developer prompt (Approach A) or persist the MSVC env (Approach B). |
| `#error: MSVC/cl.exe with traditional preprocessor is used` | A CUDA **13.x** toolkit is being used. Switch to 12.9 — do not patch the kernels. |
| `package ID specification 'zend' did not match any packages` | `vcvars64.bat` changed the working directory. `cd` back to the repo root before `cargo build`, or build from the Developer prompt opened at the repo. |
| `curl … CRYPT_E_NO_REVOCATION_CHECK` | TLS revocation check failing. Use `curl --ssl-no-revoke`; for cargo set `CARGO_HTTP_CHECK_REVOKE=false`. |
| Build links but `zend.exe` fails to load CUDA libs at runtime | Ensure `cuda.dll`, `cublas64_12.dll`, `curand64_10.dll` (from `v12.9\bin`) are on `PATH`. |

---

## Quick reference — final machine state

| Component | Version / location |
|---|---|
| Rust | stable `1.96.0`, host `x86_64-pc-windows-msvc` |
| Visual Studio | 2022 Community; MSVC `14.44.35207`; Win SDK `10.0.26100.0` |
| CUDA Toolkit | **12.9.1**, nvcc `V12.9.xx` at `…\CUDA\v12.9` |
| NVIDIA driver | `596.59` (Blackwell-capable; **not** downgraded by the toolkit install) |
| `CUDA_PATH` (Machine) | `…\NVIDIA GPU Computing Toolkit\CUDA\v12.9` |
| `CUDARC_CUDA_VERSION` (User) | `12090` |
| MSVC env (User) | `INCLUDE` / `LIB` / `LIBPATH` + VS & SDK bin dirs on `PATH` |
