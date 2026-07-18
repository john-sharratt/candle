//! The Windows/WDDM measurement backend: DXGI per-process budget.
//!
//! On WDDM the OS virtualizes VRAM and `cuMemGetInfo`'s `free` is polluted, so we
//! read the OS-authoritative per-process signal instead:
//! `IDXGIAdapter3::QueryVideoMemoryInfo` on the LOCAL segment gives `Budget`
//! (how much VRAM the OS will let us keep resident before it pages us) and
//! `CurrentUsage`. `Budget − CurrentUsage` is the real headroom. The adapter is
//! matched to the CUDA device by LUID. See `docs/vram_governor_design.md` §6.

use super::reading::{BudgetWatchHandle, ProbeKind, VramProbe, VramReading};
use crate::cuda_backend::CudaDevice;
use crate::{Error, Result};

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, BOOL, HANDLE, LUID, WAIT_OBJECT_0};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory1, DXGI_ADAPTER_DESC1,
    DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

/// Reads VRAM via DXGI's per-process budget (WDDM-honest).
pub struct DxgiProbe {
    adapter: IDXGIAdapter3,
    total: u64,
    node: u32,
}

// DXGI adapter objects are free-threaded (agile), so the probe is safe to share
// across threads even though the raw COM interface isn't marked Send/Sync.
unsafe impl Send for DxgiProbe {}
unsafe impl Sync for DxgiProbe {}

impl DxgiProbe {
    /// Build a probe for the DXGI adapter whose LUID matches `cuda`'s device.
    pub fn for_cuda_device(cuda: &CudaDevice) -> Result<Self> {
        let want = cuda_luid(cuda)?;
        let (adapter, total) = find_adapter_by_luid(want)?;
        Ok(Self {
            adapter,
            total,
            node: 0,
        })
    }
}

impl VramProbe for DxgiProbe {
    fn read(&self) -> Result<VramReading> {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe {
            self.adapter
                .QueryVideoMemoryInfo(self.node, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
        }
        .map_err(|e| Error::Msg(format!("QueryVideoMemoryInfo failed: {e}")))?;
        let headroom = info.Budget.saturating_sub(info.CurrentUsage);
        Ok(VramReading::new(headroom, self.total, ProbeKind::Dxgi))
    }

    fn budget_change_event(&self) -> Option<Box<dyn BudgetWatchHandle>> {
        // Auto-reset, initially non-signalled event.
        let hevent = unsafe { CreateEventW(None, BOOL(0), BOOL(0), None) }.ok()?;
        let cookie = unsafe {
            self.adapter
                .RegisterVideoMemoryBudgetChangeNotificationEvent(hevent)
        }
        .ok()?;
        Some(Box::new(DxgiBudgetWatch {
            hevent,
            adapter: self.adapter.clone(),
            cookie,
        }))
    }
}

/// The registered budget-change event. Holds the adapter alive so the
/// registration stays valid; unregisters and closes the handle on drop.
pub struct DxgiBudgetWatch {
    hevent: HANDLE,
    adapter: IDXGIAdapter3,
    cookie: u32,
}

// The event handle and agile adapter are safe to move to the watcher thread.
unsafe impl Send for DxgiBudgetWatch {}

impl BudgetWatchHandle for DxgiBudgetWatch {
    fn wait(&self, timeout_ms: u32) -> bool {
        unsafe { WaitForSingleObject(self.hevent, timeout_ms) == WAIT_OBJECT_0 }
    }
}

impl Drop for DxgiBudgetWatch {
    fn drop(&mut self) {
        unsafe {
            self.adapter
                .UnregisterVideoMemoryBudgetChangeNotification(self.cookie);
            let _ = CloseHandle(self.hevent);
        }
    }
}

/// The 8-byte LUID of `cuda`'s device, via `cuDeviceGetLuid`.
fn cuda_luid(cuda: &CudaDevice) -> Result<[u8; 8]> {
    use cudarc::driver::sys;
    let cu_dev = cuda.cuda_context().cu_device();
    let mut luid = [0i8; 8];
    let mut node_mask: ::std::os::raw::c_uint = 0;
    unsafe { sys::cuDeviceGetLuid(luid.as_mut_ptr(), &mut node_mask, cu_dev) }
        .result()
        .map_err(|e| Error::Msg(format!("cuDeviceGetLuid failed: {e:?}")))?;
    let mut out = [0u8; 8];
    for (o, &b) in out.iter_mut().zip(luid.iter()) {
        *o = b as u8;
    }
    Ok(out)
}

/// The 8-byte little-endian encoding of a DXGI adapter LUID.
fn luid_bytes(luid: &LUID) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&luid.LowPart.to_ne_bytes());
    out[4..].copy_from_slice(&luid.HighPart.to_ne_bytes());
    out
}

/// Enumerate DXGI adapters and return the one whose LUID matches `want`, plus its
/// dedicated VRAM total.
fn find_adapter_by_luid(want: [u8; 8]) -> Result<(IDXGIAdapter3, u64)> {
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }
        .map_err(|e| Error::Msg(format!("CreateDXGIFactory1 failed: {e}")))?;
    let mut i = 0u32;
    loop {
        let adapter1 = match unsafe { factory.EnumAdapters1(i) } {
            Ok(a) => a,
            Err(_) => break, // DXGI_ERROR_NOT_FOUND: end of enumeration
        };
        i += 1;
        let mut desc = DXGI_ADAPTER_DESC1::default();
        if unsafe { adapter1.GetDesc1(&mut desc) }.is_err() {
            continue;
        }
        if luid_bytes(&desc.AdapterLuid) != want {
            continue;
        }
        let adapter3: IDXGIAdapter3 = adapter1
            .cast()
            .map_err(|e| Error::Msg(format!("adapter lacks IDXGIAdapter3: {e}")))?;
        return Ok((adapter3, desc.DedicatedVideoMemory as u64));
    }
    Err(Error::Msg(
        "no DXGI adapter matched the CUDA device LUID".into(),
    ))
}
