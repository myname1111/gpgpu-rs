use std::sync::Arc;

use gpgpu::{layout, BufOps};

// Framework is required to be static because of std::thread::spawn lifetime requirements.
// By using crossbeam ScopedThreads this could be avoided.
lazy_static::lazy_static! {
    static ref FW: gpgpu::Framework = gpgpu::Framework::default();
}

fn main() {
    let shader = Arc::new(
        gpgpu::Shader::from_wgsl_file(&FW, "examples/parallel-compute/shader.wgsl").unwrap(),
    );

    let kernel = Arc::new(gpgpu::Kernel::new(
        &FW,
        &shader,
        "main",
        layout![
            Buffer(gpgpu::GpuBufferUsage::ReadOnly),
            Buffer(gpgpu::GpuBufferUsage::ReadOnly),
            Buffer(gpgpu::GpuBufferUsage::ReadWrite)
        ],
    ));

    let threading = 4; // Threading level
    let size = 32000; // Must be multiple of 32

    let cpu_data = (0..size).into_iter().collect::<Vec<u32>>();
    let shader_input_buffer = Arc::new(gpgpu::GpuBuffer::from_slice(&FW, &cpu_data)); // Data shared across threads shader invocations

    let mut handles = Vec::with_capacity(threading);
    for _ in 0..threading {
        let local_shader_input_buffer = shader_input_buffer.clone();
        let kernel = kernel.clone();

        // Threads spawn
        let handle = std::thread::spawn(move || {
            // Current thread GPU objects
            let local_cpu_data = (0..size).into_iter().collect::<Vec<u32>>();
            let local_input_buffer = gpgpu::GpuBuffer::from_slice(&FW, &local_cpu_data);
            let local_output_buffer = gpgpu::GpuBuffer::<u32>::with_capacity(&FW, size as u64);

            kernel.run(
                &FW,
                vec![
                    local_shader_input_buffer.as_ref(),
                    &local_input_buffer,
                    &local_output_buffer,
                ],
                size / 32,
                1,
                1,
            );

            local_output_buffer.read_vec_blocking().unwrap()
        });

        handles.push(handle);
    }

    // Join threads
    for handle in handles {
        let output = handle.join().unwrap();

        for (idx, a) in cpu_data.iter().enumerate() {
            let cpu_mult = a.pow(2);
            let gpu_mult = output[idx];

            assert_eq!(cpu_mult, gpu_mult);
        }
    }
}
