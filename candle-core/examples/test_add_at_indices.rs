use candle_core::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
    println!("Testing add_at_indices operations...\n");

    let device = Device::Cpu;

    // Test 1: Basic addition
    println!("Test 1: Basic addition");
    let t = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0, 50.0], &device)?;
    println!("Original tensor: {:?}", t.to_vec1::<f32>()?);

    let indices = [0u32, 2u32, 4u32];
    let result = t.add_at_indices(&indices, 5.0)?;
    println!(
        "After adding 5.0 at indices [0, 2, 4]: {:?}",
        result.to_vec1::<f32>()?
    );
    assert_eq!(result.to_vec1::<f32>()?, &[15.0, 20.0, 35.0, 40.0, 55.0]);
    println!("✓ Test 1 passed\n");

    // Test 2: Repeated indices (compound addition)
    println!("Test 2: Repeated indices");
    let t = Tensor::new(&[2.0f32, 2.0, 2.0, 2.0], &device)?;
    println!("Original tensor: {:?}", t.to_vec1::<f32>()?);

    let indices = [1u32, 1u32, 2u32];
    let result = t.add_at_indices(&indices, 3.0)?;
    println!(
        "After adding 3.0 at indices [1, 1, 2]: {:?}",
        result.to_vec1::<f32>()?
    );
    // Index 1 added twice: 2 + 3 + 3 = 8, Index 2 once: 2 + 3 = 5
    assert_eq!(result.to_vec1::<f32>()?, &[2.0, 8.0, 5.0, 2.0]);
    println!("✓ Test 2 passed\n");

    // Test 3: Mutable API (faster)
    println!("Test 3: Mutable API");
    let mut t = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &device)?;
    println!("Original tensor: {:?}", t.to_vec1::<f32>()?);

    let indices = [0u32, 2u32];
    t.add_at_indices_mut(&indices, 5.0)?;
    println!(
        "After add_at_indices_mut with 5.0 at indices [0, 2]: {:?}",
        t.to_vec1::<f32>()?
    );
    assert_eq!(t.to_vec1::<f32>()?, &[15.0, 20.0, 35.0, 40.0]);
    println!("✓ Test 3 passed\n");

    // Test 4: Different dtypes
    println!("Test 4: Different dtypes");

    // F16
    let t_f16 = Tensor::new(&[2.0f32, 4.0, 6.0, 8.0], &device)?.to_dtype(DType::F16)?;
    let indices = [0u32, 3u32];
    let result = t_f16.add_at_indices(&indices, 1.5)?;
    let result_f32 = result.to_dtype(DType::F32)?;
    println!("F16: {:?}", result_f32.to_vec1::<f32>()?);
    assert_eq!(result_f32.to_vec1::<f32>()?, &[3.5, 4.0, 6.0, 9.5]);

    // F64
    let t_f64 = Tensor::new(&[10.0f64, 20.0, 30.0], &device)?.to_dtype(DType::F64)?;
    let indices = [0u32, 2u32];
    let result = t_f64.add_at_indices(&indices, 1.5)?;
    println!("F64: {:?}", result.to_vec1::<f64>()?);
    assert_eq!(result.to_vec1::<f64>()?, &[11.5, 20.0, 31.5]);
    println!("✓ Test 4 passed\n");

    // Test 5: Negative addition (subtraction)
    println!("Test 5: Negative addition");
    let mut t = Tensor::new(&[10.0f32, 20.0, 30.0], &device)?;
    println!("Original tensor: {:?}", t.to_vec1::<f32>()?);

    let indices = [1u32];
    t.add_at_indices_mut(&indices, -5.0)?;
    println!("After adding -5.0 at index [1]: {:?}", t.to_vec1::<f32>()?);
    assert_eq!(t.to_vec1::<f32>()?, &[10.0, 15.0, 30.0]);
    println!("✓ Test 5 passed\n");

    // Test 6: Large batch
    println!("Test 6: Large batch of indices");
    let t = Tensor::ones((1000,), DType::F32, &device)?.affine(2.0, 0.0)?;
    let indices: Vec<u32> = (0..500).map(|i| i * 2).collect();
    let result = t.add_at_indices(&indices, 3.0)?;
    let result_vec = result.to_vec1::<f32>()?;

    // Verify even indices are 5.0 (2.0 + 3.0), odd indices are 2.0
    let mut success = true;
    for (i, &val) in result_vec.iter().enumerate() {
        let expected = if i % 2 == 0 { 5.0 } else { 2.0 };
        if (val - expected).abs() > 1e-6 {
            success = false;
            break;
        }
    }
    assert!(success);
    println!("✓ Test 6 passed\n");

    // Test 7: Empty indices
    println!("Test 7: Empty indices");
    let t = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
    let indices: [u32; 0] = [];
    let result = t.add_at_indices(&indices, 10.0)?;
    assert_eq!(result.to_vec1::<f32>()?, &[1.0, 2.0, 3.0]);
    println!("✓ Test 7 passed\n");

    println!("✓ All add_at_indices tests passed!");

    Ok(())
}
