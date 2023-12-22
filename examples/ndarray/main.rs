use gpgpu::{layout, BufOps, GpuBufferUsage::*};

// Simple compute example that multiplies 2 square arrays (matrixes)  A and B, storing the result in another array C using ndarray.
fn main() {
    let fw = gpgpu::Framework::default();
    let shader = gpgpu::Shader::from_wgsl_file(&fw, "examples/ndarray/shader.wgsl").unwrap();

    let kernel = gpgpu::Kernel::new(
        &fw,
        &shader,
        "main_fn_2",
        layout!(
            UniformBuffer,
            Buffer(ReadOnly),
            Buffer(ReadOnly),
            Buffer(ReadWrite)
        ),
    );

    let dims = (3200, 3200); // X and Y dimensions. Must be multiple of the enqueuing dimensions

    let array_src = ndarray::Array::<i32, _>::ones(dims) * 2;
    let src_view = array_src.view();

    let gpu_arrays_len = gpgpu::GpuUniformBuffer::from_slice(&fw, &[dims.0, dims.1]); // Send the ndarray dimensions

    let gpu_array_a = gpgpu::GpuArray::from_array(&fw, src_view).unwrap(); // Array A
    let gpu_array_b = gpgpu::GpuArray::from_array(&fw, src_view).unwrap(); // Array B
    let gpu_array_c = gpgpu::GpuArray::from_array(&fw, ndarray::Array::zeros(dims).view()).unwrap(); // Array C: result storage

    kernel.run(
        &fw,
        vec![&gpu_arrays_len, &gpu_array_a, &gpu_array_b, &gpu_array_c],
        dims.0 as u32 / 32,
        dims.1 as u32 / 32,
        1,
    ); // Kernel main_fn 2. Enqueuing in x and y dimensions (array dimensions are needed)

    let array_output = gpu_array_c.read_blocking().unwrap();

    for (cpu, gpu) in array_src.iter().zip(array_output) {
        assert_eq!(2 * cpu, gpu)
    }
}
