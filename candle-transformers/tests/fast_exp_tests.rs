//! Integration tests for fast_exp batch operations
//!
//! Tests the fast exponential library against reference implementations
//! across various input ranges and edge cases.

#[cfg(feature = "cuda")]
mod fast_exp_cuda {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::{DType, Device, Result, Tensor};
    use candle_kernels::simple::fast_exp::{
        FastActivation, FastExpDType, FastExpMode, FastExpPrecision,
    };
    use std::ffi::c_void;

    // Import FFI functions
    extern "C" {
        fn run_fast_exp_batch(
            mode: i32,
            precision: i32,
            dtype: i32,
            inp: *const c_void,
            out: *mut c_void,
            numel: usize,
        );

        fn run_fast_activation_batch(
            op: i32,
            dtype: i32,
            inp: *const c_void,
            out: *mut c_void,
            numel: usize,
        );
    }

    /// Test configuration
    struct TestConfig {
        num_values: usize,
        tolerance: f64,
        include_edge_cases: bool,
    }

    impl Default for TestConfig {
        fn default() -> Self {
            TestConfig {
                num_values: 100_000,
                tolerance: 0.02, // 2% relative error tolerance (covers linear precision)
                include_edge_cases: true,
            }
        }
    }

    /// Generate test values covering a wide range with edge cases
    fn generate_test_values(config: &TestConfig) -> Vec<f32> {
        let mut values = Vec::with_capacity(config.num_values);

        // 1. Edge cases (if enabled)
        if config.include_edge_cases {
            // Exact boundaries
            values.extend_from_slice(&[
                0.0,
                -0.0,
                1.0,
                -1.0,
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
            ]);

            // Near clamp boundaries
            values.extend_from_slice(&[-88.0, -87.99, -87.5, 88.0, 87.99, 87.5]);

            // Extreme values (should be clamped)
            values
                .extend_from_slice(&[-100.0, -200.0, -500.0, -1000.0, 100.0, 200.0, 500.0, 1000.0]);
        }

        // 2. Linear sweep across the valid range
        let sweep_count = config.num_values - values.len();
        let range_start = -100.0f32;
        let range_end = 100.0f32;
        let step = (range_end - range_start) / (sweep_count as f32);

        for i in 0..sweep_count {
            values.push(range_start + (i as f32) * step);
        }

        values
    }

    /// Generate test values in softmax range (x <= 0)
    fn generate_softmax_test_values(config: &TestConfig) -> Vec<f32> {
        let mut values = Vec::with_capacity(config.num_values);

        if config.include_edge_cases {
            values.extend_from_slice(&[0.0, -0.0, -1.0, -f32::MIN_POSITIVE]);
            values.extend_from_slice(&[-88.0, -87.99, -87.5, -50.0, -10.0, -1.0, -0.001]);
            values.extend_from_slice(&[-100.0, -200.0, -500.0, -1000.0]);
        }

        let sweep_count = config.num_values - values.len();
        let range_start = -100.0f32;
        let range_end = 0.0f32;
        let step = (range_end - range_start) / (sweep_count as f32);

        for i in 0..sweep_count {
            values.push(range_start + (i as f32) * step);
        }

        values
    }

    /// Reference exponential computation on CPU
    fn reference_exp(x: f32) -> f32 {
        x.exp()
    }

    /// Reference sigmoid
    fn reference_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Reference SiLU
    fn reference_silu(x: f32) -> f32 {
        x * reference_sigmoid(x)
    }

    /// Reference GELU (fast approximation)
    fn reference_gelu(x: f32) -> f32 {
        x * reference_sigmoid(1.702 * x)
    }

    /// Compare results with tolerance
    ///
    /// For clamped exp functions:
    /// - When ref is +inf and output > 1e30, that's correct (clamped to avoid overflow)
    /// - When ref is 0 and output is very small (< 1e-30), that's correct (clamped underflow)
    fn compare_results(
        input: &[f32],
        output: &[f32],
        reference: &[f32],
        name: &str,
        tolerance: f64,
    ) -> Result<()> {
        let mut max_rel_error = 0.0f64;
        let mut max_abs_error = 0.0f64;
        let mut error_count = 0usize;
        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        let mut clamped_high = 0usize;
        let mut clamped_low = 0usize;

        for (i, ((inp, out), ref_val)) in input
            .iter()
            .zip(output.iter())
            .zip(reference.iter())
            .enumerate()
        {
            // Handle special cases
            if ref_val.is_nan() {
                // For NaN inputs, accept any finite non-negative output (clamped behavior)
                if out.is_nan() || (!out.is_finite() && *out < 0.0) {
                    error_count += 1;
                    if error_count <= 10 {
                        println!(
                            "[{}] Input={}, Output={}, Expected=NaN or clamped",
                            i, inp, out
                        );
                    }
                }
                nan_count += 1;
                continue;
            }

            if ref_val.is_infinite() && *ref_val > 0.0 {
                // For +inf reference, accept inf OR a very large finite value (clamped)
                // exp(88) ≈ 1.65e38, so anything > 1e30 is reasonable for clamped overflow
                if *out > 1e30 || out.is_infinite() {
                    clamped_high += 1;
                    // This is correct behavior
                } else {
                    error_count += 1;
                    if error_count <= 10 {
                        println!(
                            "[{}] Input={}, Output={}, Expected=inf or >1e30",
                            i, inp, out
                        );
                    }
                }
                inf_count += 1;
                continue;
            }

            if ref_val.is_infinite() && *ref_val < 0.0 {
                // -inf reference shouldn't happen for exp()
                error_count += 1;
                continue;
            }

            // For very small reference values (underflow region)
            if *ref_val < 1e-38 && *ref_val >= 0.0 {
                // Accept any very small non-negative value
                if *out >= 0.0 && *out < 1e-30 {
                    clamped_low += 1;
                    continue;
                }
            }

            let abs_error = (out - ref_val).abs() as f64;
            let rel_error = if ref_val.abs() > 1e-10 {
                abs_error / ref_val.abs() as f64
            } else {
                abs_error
            };

            max_abs_error = max_abs_error.max(abs_error);
            max_rel_error = max_rel_error.max(rel_error);

            if rel_error > tolerance && abs_error > 1e-6 {
                error_count += 1;
                if error_count <= 10 {
                    println!(
                        "[{}] Input={:.6}, Output={:.6}, Expected={:.6}, RelErr={:.4}%",
                        i,
                        inp,
                        out,
                        ref_val,
                        rel_error * 100.0
                    );
                }
            }
        }

        println!(
            "{}: max_rel_error={:.4}%, max_abs_error={:.6e}, errors={}/{}, clamped_high={}, clamped_low={}, nan={}, inf={}",
            name,
            max_rel_error * 100.0,
            max_abs_error,
            error_count,
            input.len(),
            clamped_high,
            clamped_low,
            nan_count,
            inf_count
        );

        if error_count > input.len() / 100 {
            // Allow up to 1% errors
            candle::bail!(
                "{} had too many errors: {}/{}",
                name,
                error_count,
                input.len()
            );
        }

        Ok(())
    }

    /// Run exp test with specified mode and precision
    fn run_exp_test(
        device: &Device,
        mode: FastExpMode,
        precision: FastExpPrecision,
        config: &TestConfig,
    ) -> Result<()> {
        let input_values = if mode == FastExpMode::Softmax {
            generate_softmax_test_values(config)
        } else {
            generate_test_values(config)
        };

        let numel = input_values.len();

        // Create tensors
        let input_tensor = Tensor::from_vec(input_values.clone(), numel, device)?;
        let output_tensor = Tensor::zeros(numel, DType::F32, device)?;

        // Get CUDA device and stream
        let cuda_device = match device {
            Device::Cuda(d) => d,
            _ => candle::bail!("Expected CUDA device"),
        };

        // Get raw pointers - need to keep storage refs alive
        let (input_storage, _) = input_tensor.storage_and_layout();
        let (output_storage, _) = output_tensor.storage_and_layout();

        // Extract pointers while keeping guards in scope
        {
            let stream = cuda_device.cuda_stream();

            let input_ptr = match &*input_storage {
                candle::Storage::Cuda(cuda_storage) => {
                    let slice = cuda_storage.as_cuda_slice::<f32>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *const c_void
                }
                _ => candle::bail!("Expected CUDA storage"),
            };

            let output_ptr = match &*output_storage {
                candle::Storage::Cuda(cuda_storage) => {
                    let slice = cuda_storage.as_cuda_slice::<f32>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *mut c_void
                }
                _ => candle::bail!("Expected CUDA storage"),
            };

            // Run kernel while guards are alive
            unsafe {
                run_fast_exp_batch(
                    mode as i32,
                    precision as i32,
                    FastExpDType::F32 as i32,
                    input_ptr,
                    output_ptr,
                    numel,
                );
            }
        }

        // Synchronize
        device.synchronize()?;

        // Get results
        let output_vec: Vec<f32> = output_tensor.to_vec1()?;

        // Compute reference
        let reference: Vec<f32> = input_values.iter().map(|&x| reference_exp(x)).collect();

        // Compare
        let name = format!("{:?}/{:?}", mode, precision);
        compare_results(
            &input_values,
            &output_vec,
            &reference,
            &name,
            config.tolerance,
        )?;

        Ok(())
    }

    /// Run activation test
    fn run_activation_test(
        device: &Device,
        activation: FastActivation,
        config: &TestConfig,
    ) -> Result<()> {
        let input_values = generate_test_values(config);
        let numel = input_values.len();

        // Create tensors
        let input_tensor = Tensor::from_vec(input_values.clone(), numel, device)?;
        let output_tensor = Tensor::zeros(numel, DType::F32, device)?;

        // Get CUDA device
        let cuda_device = match device {
            Device::Cuda(d) => d,
            _ => candle::bail!("Expected CUDA device"),
        };

        // Get raw pointers
        let (input_storage, _) = input_tensor.storage_and_layout();
        let (output_storage, _) = output_tensor.storage_and_layout();

        {
            let stream = cuda_device.cuda_stream();

            let input_ptr = match &*input_storage {
                candle::Storage::Cuda(cuda_storage) => {
                    let slice = cuda_storage.as_cuda_slice::<f32>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *const c_void
                }
                _ => candle::bail!("Expected CUDA storage"),
            };

            let output_ptr = match &*output_storage {
                candle::Storage::Cuda(cuda_storage) => {
                    let slice = cuda_storage.as_cuda_slice::<f32>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *mut c_void
                }
                _ => candle::bail!("Expected CUDA storage"),
            };

            // Run kernel
            unsafe {
                run_fast_activation_batch(
                    activation as i32,
                    FastExpDType::F32 as i32,
                    input_ptr,
                    output_ptr,
                    numel,
                );
            }
        }

        // Synchronize
        device.synchronize()?;

        // Get results
        let output_vec: Vec<f32> = output_tensor.to_vec1()?;

        // Compute reference based on activation type
        let reference: Vec<f32> = match activation {
            FastActivation::Sigmoid => input_values.iter().map(|&x| reference_sigmoid(x)).collect(),
            FastActivation::SiLU => input_values.iter().map(|&x| reference_silu(x)).collect(),
            FastActivation::GELU => input_values.iter().map(|&x| reference_gelu(x)).collect(),
        };

        // Compare
        let name = format!("{:?}", activation);
        compare_results(
            &input_values,
            &output_vec,
            &reference,
            &name,
            config.tolerance,
        )?;

        Ok(())
    }

    // ========================================================================
    // TESTS
    // ========================================================================

    #[test]
    fn test_fast_exp_generic_high() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config = TestConfig {
            tolerance: 0.001, // 0.1% for high precision
            ..Default::default()
        };
        run_exp_test(
            &device,
            FastExpMode::Generic,
            FastExpPrecision::High,
            &config,
        )
    }

    #[test]
    fn test_fast_exp_generic_medium() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config = TestConfig {
            tolerance: 0.01, // 1% for medium precision (documented as ~0.08%)
            ..Default::default()
        };
        run_exp_test(
            &device,
            FastExpMode::Generic,
            FastExpPrecision::Medium,
            &config,
        )
    }

    #[test]
    fn test_fast_exp_generic_low() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config = TestConfig {
            tolerance: 0.035, // 3.5% for low precision (documented as ~1.5%)
            ..Default::default()
        };
        run_exp_test(
            &device,
            FastExpMode::Generic,
            FastExpPrecision::Low,
            &config,
        )
    }

    #[test]
    fn test_fast_exp_softmax_high() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config = TestConfig {
            tolerance: 0.001,
            ..Default::default()
        };
        run_exp_test(
            &device,
            FastExpMode::Softmax,
            FastExpPrecision::High,
            &config,
        )
    }

    #[test]
    fn test_fast_exp_softmax_medium() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config = TestConfig {
            tolerance: 0.01, // 1% for medium precision
            ..Default::default()
        };
        run_exp_test(
            &device,
            FastExpMode::Softmax,
            FastExpPrecision::Medium,
            &config,
        )
    }

    #[test]
    fn test_fast_exp_softmax_low() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config = TestConfig {
            tolerance: 0.035, // 3.5% for low precision
            ..Default::default()
        };
        run_exp_test(
            &device,
            FastExpMode::Softmax,
            FastExpPrecision::Low,
            &config,
        )
    }

    #[test]
    fn test_fast_sigmoid() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config = TestConfig {
            tolerance: 0.002,
            ..Default::default()
        };
        run_activation_test(&device, FastActivation::Sigmoid, &config)
    }

    #[test]
    fn test_fast_silu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config = TestConfig {
            tolerance: 0.002,
            ..Default::default()
        };
        run_activation_test(&device, FastActivation::SiLU, &config)
    }

    #[test]
    fn test_fast_gelu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config = TestConfig {
            tolerance: 0.002,
            ..Default::default()
        };
        run_activation_test(&device, FastActivation::GELU, &config)
    }

    #[test]
    fn test_monotonicity() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let cuda_device = match &device {
            Device::Cuda(d) => d,
            _ => candle::bail!("Expected CUDA device"),
        };

        // Generate monotonically increasing values
        let input_values: Vec<f32> = (0..10000).map(|i| -50.0 + (i as f32) * 0.01).collect();
        let numel = input_values.len();

        let input_tensor = Tensor::from_vec(input_values.clone(), numel, &device)?;
        let output_tensor = Tensor::zeros(numel, DType::F32, &device)?;

        let (input_storage, _) = input_tensor.storage_and_layout();
        let (output_storage, _) = output_tensor.storage_and_layout();

        {
            let stream = cuda_device.cuda_stream();

            let input_ptr = match &*input_storage {
                candle::Storage::Cuda(cuda_storage) => {
                    let slice = cuda_storage.as_cuda_slice::<f32>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *const c_void
                }
                _ => candle::bail!("Expected CUDA storage"),
            };

            let output_ptr = match &*output_storage {
                candle::Storage::Cuda(cuda_storage) => {
                    let slice = cuda_storage.as_cuda_slice::<f32>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *mut c_void
                }
                _ => candle::bail!("Expected CUDA storage"),
            };

            unsafe {
                run_fast_exp_batch(
                    FastExpMode::Generic as i32,
                    FastExpPrecision::High as i32,
                    FastExpDType::F32 as i32,
                    input_ptr,
                    output_ptr,
                    numel,
                );
            }
        }

        device.synchronize()?;

        let output_vec: Vec<f32> = output_tensor.to_vec1()?;

        // Check monotonicity
        let mut violations = 0;
        for i in 1..output_vec.len() {
            if output_vec[i] < output_vec[i - 1]
                && !output_vec[i].is_nan()
                && !output_vec[i - 1].is_nan()
            {
                violations += 1;
                if violations <= 5 {
                    println!(
                        "Monotonicity violation at {}: f({})={} < f({})={}",
                        i,
                        input_values[i],
                        output_vec[i],
                        input_values[i - 1],
                        output_vec[i - 1]
                    );
                }
            }
        }

        println!(
            "Monotonicity test: {} violations out of {} pairs",
            violations,
            numel - 1
        );

        if violations > 0 {
            candle::bail!("Monotonicity violated {} times", violations);
        }

        Ok(())
    }

    #[test]
    fn test_softmax_stability() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let cuda_device = match &device {
            Device::Cuda(d) => d,
            _ => candle::bail!("Expected CUDA device"),
        };

        // Test with very negative values that could cause underflow
        let input_values: Vec<f32> = vec![
            -1000.0, -500.0, -200.0, -100.0, -88.7, -88.0, -50.0, -10.0, -1.0, 0.0,
        ];
        let numel = input_values.len();

        let input_tensor = Tensor::from_vec(input_values.clone(), numel, &device)?;
        let output_tensor = Tensor::zeros(numel, DType::F32, &device)?;

        let (input_storage, _) = input_tensor.storage_and_layout();
        let (output_storage, _) = output_tensor.storage_and_layout();

        {
            let stream = cuda_device.cuda_stream();

            let input_ptr = match &*input_storage {
                candle::Storage::Cuda(cuda_storage) => {
                    let slice = cuda_storage.as_cuda_slice::<f32>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *const c_void
                }
                _ => candle::bail!("Expected CUDA storage"),
            };

            let output_ptr = match &*output_storage {
                candle::Storage::Cuda(cuda_storage) => {
                    let slice = cuda_storage.as_cuda_slice::<f32>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *mut c_void
                }
                _ => candle::bail!("Expected CUDA storage"),
            };

            unsafe {
                run_fast_exp_batch(
                    FastExpMode::Softmax as i32,
                    FastExpPrecision::High as i32,
                    FastExpDType::F32 as i32,
                    input_ptr,
                    output_ptr,
                    numel,
                );
            }
        }

        device.synchronize()?;

        let output_vec: Vec<f32> = output_tensor.to_vec1()?;

        // All outputs should be finite (no inf, no nan) and non-negative
        for (i, (inp, out)) in input_values.iter().zip(output_vec.iter()).enumerate() {
            if !out.is_finite() {
                candle::bail!("Output {} is not finite for input {}: {}", i, inp, out);
            }
            if *out < 0.0 {
                candle::bail!("Output {} is negative for input {}: {}", i, inp, out);
            }
        }

        // exp(0) should be exactly 1.0
        let exp_zero = output_vec[output_vec.len() - 1];
        if (exp_zero - 1.0).abs() > 0.001 {
            candle::bail!("exp(0) = {} != 1.0", exp_zero);
        }

        println!("Softmax stability test passed");
        println!("Results: {:?}", output_vec);

        Ok(())
    }
}
