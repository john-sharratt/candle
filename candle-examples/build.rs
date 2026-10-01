#![allow(unused)]
use anyhow::{Context, Result};
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};

struct KernelDirectories {
    kernel_glob: &'static str,
    rust_target: &'static str,
    include_dirs: &'static [&'static str],
}

const KERNEL_DIRS: [KernelDirectories; 1] = [KernelDirectories {
    kernel_glob: "examples/custom-ops/kernels/*.cu",
    rust_target: "examples/custom-ops/cuda_kernels.rs",
    include_dirs: &[],
}];

/// Put an older, `cl.exe`-carrying MSVC toolset ahead of `PATH` for the
/// child `nvcc` process, if more than one is installed.
///
/// Best-effort: absent `vswhere.exe` (no Visual Studio at all, or not
/// Windows), or a machine with only one toolset installed, this changes
/// nothing and `nvcc` resolves `cl.exe` exactly as it always did. Editing the
/// build's own `PATH` rather than the process-wide one, so this never leaks
/// into any other tool the build shells out to.
#[cfg(windows)]
fn prepend_a_usable_msvc_toolset_to_path() {
    let pf86 =
        env::var("ProgramFiles(x86)").unwrap_or_else(|_| r"C:\Program Files (x86)".to_string());
    let vswhere = Path::new(&pf86).join(r"Microsoft Visual Studio\Installer\vswhere.exe");
    let Ok(out) = std::process::Command::new(vswhere)
        .args([
            "-all",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
    else {
        return;
    };

    // Every toolset of every installation that actually carries a `cl.exe`,
    // oldest first by the directory name `VC\Tools\MSVC\<version>` sorts on
    // (`14.44.35207` < `14.51.36231`, ordinary string order, since a leading
    // `14.` keeps the comparison numeric in practice).
    let mut found: Vec<(String, PathBuf)> = Vec::new();
    for root in String::from_utf8_lossy(&out.stdout).lines() {
        let root = root.trim();
        if root.is_empty() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(Path::new(root).join(r"VC\Tools\MSVC")) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let bin = path.join(r"bin\Hostx64\x64");
            if bin.join("cl.exe").is_file() {
                if let Some(version) = path.file_name().and_then(|n| n.to_str()) {
                    found.push((version.to_string(), bin));
                }
            }
        }
    }
    // Only one toolset is nothing to choose between — leave `PATH` alone.
    if found.len() < 2 {
        return;
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    let (_, oldest) = &found[0];

    let path = env::var_os("PATH").unwrap_or_default();
    let mut dirs = vec![oldest.clone()];
    dirs.extend(std::env::split_paths(&path));
    if let Ok(joined) = std::env::join_paths(dirs) {
        println!("cargo:info=preferring the older MSVC toolset at {oldest:?} for nvcc");
        // SAFETY: build scripts run single-threaded before any other code in
        // this process spawns; nothing else reads or writes `PATH` here.
        unsafe {
            env::set_var("PATH", joined);
        }
    }
}

#[cfg(not(windows))]
fn prepend_a_usable_msvc_toolset_to_path() {}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    {
        // Added: Get the safe output directory from the environment.
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        // `candle-kernels/build_utils.rs` solves this generally (it parses
        // CUDA's own `host_config.h` for the exact MSVC range it accepts);
        // this is the compact version for a single demo kernel: `nvcc` locates
        // its host compiler on `PATH`, not through anything cargo sets up, so
        // a machine that keeps a newer Visual Studio ahead on `PATH` (or
        // running outside a Developer Command Prompt at all — an ordinary
        // terminal, CI) can hand it a toolset the installed CUDA rejects with
        // `error C1189` before a single kernel compiles, even though a CUDA-
        // compatible toolset sits right next to it. `-allow-unsupported-
        // compiler` only silences that one check; the STL headers a newer
        // toolset ships still reject the mismatch on their own terms, so the
        // real fix is handing nvcc a compiler it actually accepts.
        prepend_a_usable_msvc_toolset_to_path();

        for kdir in KERNEL_DIRS.iter() {
            let builder = bindgen_cuda::Builder::default().kernel_paths_glob(kdir.kernel_glob);
            println!("cargo:info={builder:?}");
            let bindings = builder.build_ptx().unwrap();

            // Changed: This now writes to a safe path inside $OUT_DIR.
            let safe_target = out_dir.join(
                Path::new(kdir.rust_target)
                    .file_name()
                    .context("Failed to get filename from rust_target")?,
            );
            bindings.write(safe_target).unwrap()
        }
    }
    Ok(())
}
