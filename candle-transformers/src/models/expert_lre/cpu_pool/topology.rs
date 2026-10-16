//! The machine's physical cores, as the pool places workers on them.
//!
//! A worker per *physical* core: SMT siblings share the core's load ports and
//! add contention rather than bandwidth, so each core is listed once, with all
//! of its logical processors. On a hybrid part the cores also differ in kind —
//! Windows reports an `EfficiencyClass` per logical processor, higher meaning
//! faster — and the pool starts from the fastest class.
//!
//! The grouping is a pure function of the per-logical-processor records
//! ([`group_cores`]), so it is tested on synthetic layouts; [`physical_cores`]
//! only reads the records from the OS.

use std::collections::BTreeMap;

/// One logical processor, as the OS reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalCpu {
    /// The CPU-set id `SetThreadSelectedCpuSets` takes.
    pub cpu_set: u32,
    /// Processor group.
    pub group: u16,
    /// The physical core it belongs to, unique within its group.
    pub core: u8,
    /// Higher is a faster core. Uniform parts report one class for every core.
    pub efficiency_class: u8,
    /// The last-level cache it shares — a CCD on Zen, one ring on client Intel.
    pub llc: u8,
    pub numa: u8,
}

/// One physical core and every logical processor on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalCore {
    pub group: u16,
    pub core: u8,
    pub efficiency_class: u8,
    pub llc: u8,
    pub numa: u8,
    /// CPU-set ids, ascending. The first is the one a single worker pins to.
    pub cpu_sets: Vec<u32>,
}

impl PhysicalCore {
    /// The CPU set a worker on this core pins to.
    pub fn primary(&self) -> u32 {
        self.cpu_sets[0]
    }
}

/// Folds logical processors into physical cores, fastest class first, then by
/// group and core index — the order the pool takes them in.
pub fn group_cores(cpus: &[LogicalCpu]) -> Vec<PhysicalCore> {
    let mut by_core: BTreeMap<(u16, u8), PhysicalCore> = BTreeMap::new();
    for c in cpus {
        let entry = by_core
            .entry((c.group, c.core))
            .or_insert_with(|| PhysicalCore {
                group: c.group,
                core: c.core,
                efficiency_class: c.efficiency_class,
                llc: c.llc,
                numa: c.numa,
                cpu_sets: Vec::new(),
            });
        entry.cpu_sets.push(c.cpu_set);
    }
    let mut cores: Vec<PhysicalCore> = by_core.into_values().collect();
    for core in &mut cores {
        core.cpu_sets.sort_unstable();
    }
    cores.sort_by(|a, b| {
        b.efficiency_class
            .cmp(&a.efficiency_class)
            .then(a.group.cmp(&b.group))
            .then(a.core.cmp(&b.core))
    });
    cores
}

/// The cores of the fastest efficiency class, in pool order.
pub fn performance_cores(cores: &[PhysicalCore]) -> Vec<PhysicalCore> {
    let Some(top) = cores.iter().map(|c| c.efficiency_class).max() else {
        return Vec::new();
    };
    cores
        .iter()
        .filter(|c| c.efficiency_class == top)
        .cloned()
        .collect()
}

/// This machine's physical cores, fastest class first.
#[cfg(windows)]
pub fn physical_cores() -> std::io::Result<Vec<PhysicalCore>> {
    Ok(group_cores(&windows_cpus::logical_cpus()?))
}

#[cfg(windows)]
mod windows_cpus {
    use std::io;
    use std::mem::size_of;
    use std::ptr::null_mut;

    use windows_sys::Win32::System::SystemInformation::{
        CpuSetInformation, GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    use super::LogicalCpu;

    /// Every logical processor the process may run on, from
    /// `GetSystemCpuSetInformation`. The records are variable-length (each
    /// carries its own `Size`), so the buffer is walked by that, not by the
    /// struct's size.
    pub(super) fn logical_cpus() -> io::Result<Vec<LogicalCpu>> {
        let process = unsafe { GetCurrentProcess() };
        let mut needed: u32 = 0;
        // Sizing call: fails with ERROR_INSUFFICIENT_BUFFER and reports the size.
        unsafe { GetSystemCpuSetInformation(null_mut(), 0, &mut needed, process, 0) };
        if needed == 0 {
            return Err(io::Error::last_os_error());
        }
        // u64 backing keeps every record 8-byte aligned, as the struct needs.
        let mut buf = vec![0u64; (needed as usize).div_ceil(8)];
        let ok = unsafe {
            GetSystemCpuSetInformation(
                buf.as_mut_ptr().cast::<SYSTEM_CPU_SET_INFORMATION>(),
                needed,
                &mut needed,
                process,
                0,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        let bytes = buf.as_ptr().cast::<u8>();
        let mut cpus = Vec::new();
        let mut off = 0usize;
        while off + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= needed as usize {
            let rec = unsafe { &*bytes.add(off).cast::<SYSTEM_CPU_SET_INFORMATION>() };
            if rec.Size == 0 {
                break;
            }
            if rec.Type == CpuSetInformation {
                let s = unsafe { rec.Anonymous.CpuSet };
                cpus.push(LogicalCpu {
                    cpu_set: s.Id,
                    group: s.Group,
                    core: s.CoreIndex,
                    efficiency_class: s.EfficiencyClass,
                    llc: s.LastLevelCacheIndex,
                    numa: s.NumaNodeIndex,
                });
            }
            off += rec.Size as usize;
        }
        Ok(cpus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu(cpu_set: u32, core: u8, class: u8) -> LogicalCpu {
        LogicalCpu {
            cpu_set,
            group: 0,
            core,
            efficiency_class: class,
            llc: 0,
            numa: 0,
        }
    }

    /// Meteor Lake H's shape: 6 P-cores with SMT (class 1), 8 E-cores and 2
    /// LP E-cores without it (class 0). SMT siblings fold into one core, and the
    /// P-cores come first.
    #[test]
    fn hybrid_part_folds_siblings_and_puts_performance_cores_first() {
        let mut cpus = Vec::new();
        for core in 0..6u8 {
            cpus.push(cpu(256 + 2 * core as u32, core, 1));
            cpus.push(cpu(257 + 2 * core as u32, core, 1));
        }
        for core in 6..16u8 {
            cpus.push(cpu(262 + core as u32, core, 0));
        }
        let cores = group_cores(&cpus);
        assert_eq!(cores.len(), 16);
        assert!(cores[..6]
            .iter()
            .all(|c| c.efficiency_class == 1 && c.cpu_sets.len() == 2));
        assert!(cores[6..]
            .iter()
            .all(|c| c.efficiency_class == 0 && c.cpu_sets.len() == 1));
        assert_eq!(cores[0].cpu_sets, [256, 257]);
        assert_eq!(cores[0].primary(), 256);
        assert_eq!(performance_cores(&cores).len(), 6);
    }

    /// A uniform part reports one class: every core is a performance core.
    #[test]
    fn uniform_part_is_all_performance_cores() {
        let cpus: Vec<_> = (0..8u8)
            .flat_map(|core| {
                [
                    cpu(256 + 2 * core as u32, core, 0),
                    cpu(257 + 2 * core as u32, core, 0),
                ]
            })
            .collect();
        let cores = group_cores(&cpus);
        assert_eq!(cores.len(), 8);
        assert_eq!(performance_cores(&cores).len(), 8);
    }

    /// Records arriving out of order still give ascending sibling lists and a
    /// stable core order.
    #[test]
    fn out_of_order_records_group_the_same() {
        let cpus = [
            cpu(259, 1, 1),
            cpu(256, 0, 1),
            cpu(258, 1, 1),
            cpu(257, 0, 1),
        ];
        let cores = group_cores(&cpus);
        assert_eq!(cores[0].cpu_sets, [256, 257]);
        assert_eq!(cores[1].cpu_sets, [258, 259]);
    }

    #[test]
    fn no_records_no_performance_cores() {
        assert!(performance_cores(&group_cores(&[])).is_empty());
    }
}
