//! Typed binary activation-vector corpus loaded into the inference session.

use candle::{DType, Device, Storage, Tensor};
#[cfg(feature = "cuda")]
use candle_kernels::simple::weighted_row_accum::{run_weighted_row_accum, WeightedRowAccumDType};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct PersonalityVectorFile {
    pub magic: String,
    pub version: u32,
    pub source_records: usize,
    pub downshift_records: usize,
    pub similarity_threshold: f32,
    pub default_scale: f32,
    pub clusters: Vec<PersonalityVectorCluster>,
}

#[derive(Debug, Deserialize)]
pub struct PersonalityVectorCluster {
    pub layer: usize,
    pub relative_position: i8,
    pub members: usize,
    pub scale: f32,
    pub hidden: usize,
    pub centroid: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct DevicePersonalityVectors {
    pub values: Tensor,
    pub scales: Tensor,
    pub layers: Vec<usize>,
    pub relative_positions: Vec<i8>,
    pub members: Vec<usize>,
    layer_ranges: Vec<(usize, usize, usize)>,
}

impl PersonalityVectorFile {
    pub fn into_device(
        mut self,
        device: &candle::Device,
    ) -> candle::Result<DevicePersonalityVectors> {
        if self.magic != "CANDLE-PERSONALITY-VECTORS" || self.version != 1 {
            candle::bail!("unsupported personality vector artifact");
        }
        let Some(first) = self.clusters.first() else {
            candle::bail!("personality vector artifact contains no vectors");
        };
        let hidden = first.hidden;
        self.clusters
            .sort_by_key(|cluster| (cluster.layer, cluster.relative_position));
        if hidden == 0
            || self.clusters.iter().any(|cluster| {
                cluster.hidden != hidden
                    || cluster.centroid.len() != hidden
                    || cluster.relative_position != -1
            })
        {
            candle::bail!(
                "personality vector artifact has inconsistent widths or unsupported relative positions"
            );
        }
        let mut values = Vec::with_capacity(self.clusters.len() * hidden);
        let mut layers = Vec::with_capacity(self.clusters.len());
        let mut relative_positions = Vec::with_capacity(self.clusters.len());
        let mut members = Vec::with_capacity(self.clusters.len());
        let mut scales = Vec::with_capacity(self.clusters.len());
        for cluster in self.clusters {
            values.extend(cluster.centroid);
            layers.push(cluster.layer);
            relative_positions.push(cluster.relative_position);
            members.push(cluster.members);
            scales.push(cluster.scale);
        }
        let mut layer_ranges = Vec::new();
        for (index, &layer) in layers.iter().enumerate() {
            if layer_ranges
                .last()
                .map_or(true, |&(previous, _, _)| previous != layer)
            {
                layer_ranges.push((layer, index, index + 1));
            } else {
                layer_ranges.last_mut().unwrap().2 = index + 1;
            }
        }
        let values =
            Tensor::from_vec(values, (layers.len(), hidden), device)?.to_dtype(DType::F32)?;
        let scales = Tensor::from_vec(scales, layers.len(), device)?.to_dtype(DType::F32)?;
        Ok(DevicePersonalityVectors {
            values,
            scales,
            layers,
            relative_positions,
            members,
            layer_ranges,
        })
    }
}

impl DevicePersonalityVectors {
    #[cfg(feature = "cuda")]
    pub fn apply_to_activation(
        &self,
        layer: usize,
        activation: &Tensor,
        threshold: f32,
    ) -> candle::Result<()> {
        let Some(&(_, start, end)) = self
            .layer_ranges
            .iter()
            .find(|&&(vector_layer, _, _)| vector_layer == layer)
        else {
            return Ok(());
        };
        let activation_dims = activation.dims();
        if activation_dims.last().copied() != Some(self.values.dim(1)?) {
            candle::bail!(
                "personality layer {layer}: activation shape {:?} does not match hidden {}",
                activation_dims,
                self.values.dim(1)?
            );
        }
        let vectors = self.values.narrow(0, start, end - start)?;
        let scales = self.scales.narrow(0, start, end - start)?;
        if !vectors.is_contiguous() || !scales.is_contiguous() {
            candle::bail!("personality layer {layer}: vector operands must be contiguous");
        }
        let device = match activation.device() {
            Device::Cuda(device) => device,
            _ => candle::bail!("personality steering requires CUDA"),
        };
        let dtype = match activation.dtype() {
            DType::F32 => WeightedRowAccumDType::F32,
            DType::F16 => WeightedRowAccumDType::F16,
            DType::BF16 => WeightedRowAccumDType::BF16,
            dtype => {
                candle::bail!("personality layer {layer}: unsupported activation dtype {dtype:?}")
            }
        };
        let (vectors_storage, vectors_layout) = vectors.storage_and_layout();
        let (scales_storage, scales_layout) = scales.storage_and_layout();
        let (mut activation_storage, activation_layout) =
            unsafe { activation.storage_mut_and_layout() };
        let stream = device.cuda_stream();
        let vectors_ptr =
            match &*vectors_storage {
                Storage::Cuda(storage) => storage.slice.device_ptr(&stream).saturating_add(
                    (vectors_layout.start_offset() * std::mem::size_of::<f32>()) as u64,
                ) as *const core::ffi::c_void,
                _ => candle::bail!("personality vectors must be CUDA tensors"),
            };
        let scales_ptr =
            match &*scales_storage {
                Storage::Cuda(storage) => storage.slice.device_ptr(&stream).saturating_add(
                    (scales_layout.start_offset() * std::mem::size_of::<f32>()) as u64,
                ) as *const f32,
                _ => candle::bail!("personality scales must be CUDA tensors"),
            };
        let activation_ptr = match &mut *activation_storage {
            Storage::Cuda(storage) => unsafe {
                storage
                    .slice
                    .device_ptr_mut(&stream)?
                    .cast::<u8>()
                    .add(activation_layout.start_offset() * activation.dtype().size_in_bytes())
                    .cast::<core::ffi::c_void>()
            },
            _ => candle::bail!("personality activation must be a CUDA tensor"),
        };
        let code = unsafe {
            run_weighted_row_accum(
                dtype as i32,
                vectors_ptr,
                scales_ptr,
                activation_ptr,
                (end - start) as i32,
                activation_dims.last().copied().unwrap() as i32,
                vectors_layout.stride()[0] as i64,
                scales_layout.stride()[0] as i64,
                activation_layout.stride().last().copied().unwrap() as i64,
                threshold,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("personality kernel failed with CUDA error code {code}");
        }
        Ok(())
    }
}
