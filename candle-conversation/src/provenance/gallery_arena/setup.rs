//! Creates the gallery arena for one device, with the locked folded-signature
//! geometry.
//!
//! Each page is one slot of the device's own slot arenas, sized by the fold groups
//! it holds. A zero page, claimed on first need and never evicted, is what a
//! (page, group) entry points at when the page does not hold that group, so a scan
//! reads a valid address rather than nothing. The arena reports its device and its
//! word widths for the scan kernels that read its pages.

use super::*;

impl GalleryArena {
    /// A gallery arena on `device` for the locked folded-signature geometry
    /// (`wpt` words per token, `n_groups` layer-groups). For the production fold
    /// that is `wpt = 24`, `n_groups = 3`.
    pub fn new(device: &Device, wpt: usize, n_groups: usize) -> Result<Self> {
        assert!(wpt > 0 && n_groups > 0 && wpt.is_multiple_of(n_groups));
        if !matches!(device, Device::Cuda(_)) {
            candle::bail!("gallery arena requires a CUDA device");
        }
        assert!(n_groups <= GroupSet::MAX, "{n_groups} fold groups");
        let Device::Cuda(cuda) = device else {
            candle::bail!("gallery arena requires a CUDA device");
        };
        let (_, total) = cuda.mem_get_info()?;
        Ok(Self {
            residency: Mutex::new(HashMap::new()),
            lru_clock: AtomicU64::new(0),
            residency_gen: AtomicU64::new(0),
            next_run_id: AtomicU64::new(0),
            keep_within_ns: AtomicU64::new(u64::MAX),
            upload_us_per_byte: AtomicU64::new(0),
            index_cache: Mutex::new(HashMap::new()),
            indexes: ArcSwap::default(),
            changes: AtomicU64::new(0),
            born: Instant::now(),
            zero_page: Mutex::new(None),
            dedup: Dedup::new(wpt),
            tensor_caps: OnceLock::new(),
            index_cache_bytes: index_cache_budget(total as u64),
            device: device.clone(),
            wpt,
            n_groups,
            gw: wpt / n_groups,
        })
    }

    /// Words a token holds per fold group.
    #[inline]
    pub(in super::super) fn gw(&self) -> usize {
        self.gw
    }

    /// Every fold group of this arena's signatures.
    #[inline]
    pub fn all_groups(&self) -> GroupSet {
        GroupSet::all(self.n_groups)
    }

    /// Bytes of one page holding `groups`.
    #[inline]
    pub(super) fn page_bytes_for(&self, groups: GroupSet) -> u64 {
        (page_words(self.gw, groups) * std::mem::size_of::<u64>()) as u64
    }

    /// Bytes of the pages `rt` holds.
    pub(super) fn run_bytes(&self, rt: &ResidentTurn) -> u64 {
        pages_for(rt.mask.kept()) as u64 * self.page_bytes_for(rt.groups)
    }

    /// The device address of the arena's zero page, claimed and written on
    /// first use — one group wide, which is all a (page, group) entry reads.
    pub(in super::super) fn zero_page_addr(&self) -> Result<u64> {
        let mut zero = self.zero_page.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = zero.as_ref() {
            return Ok(slot.ptr());
        }
        let one = GroupSet::default().with(0);
        let slots = claim_arena_slots(
            &self.device,
            SlotTenant::Gallery,
            self.page_bytes_for(one) as usize,
            1,
        )?;
        let words = slots[0].stride() / std::mem::size_of::<u64>();
        page_io::write_pages(&self.device, &slots, &vec![0u64; words])?;
        let addr = slots[0].ptr();
        *zero = slots.into_iter().next();
        Ok(addr)
    }

    /// The CUDA device the arena's slabs live on.
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Words per token (signature width).
    #[inline]
    pub fn wpt(&self) -> usize {
        self.wpt
    }

    /// Layer-groups per signature.
    #[inline]
    pub fn n_groups(&self) -> usize {
        self.n_groups
    }
}
