use crate::{bindings::SetBindings, entry_type::EntryType, primitives::BindGroupEntryType, *};

/// Used to enqueue the execution of a shader with the bidings provided.
///
/// Equivalent to OpenCL's Kernel.
pub struct Kernel<'a> {
    pipeline: wgpu::ComputePipeline,
    entry_types: Vec<EntryType>,
    layout: wgpu::BindGroupLayout,
    function_name: &'a str,
}

impl<'a> Kernel<'a> {
    /// Creates a [`Kernel`] from a [`Program`].
    pub fn new<'sha, 'fw>(
        fw: &'fw Framework,
        shader: &'sha Shader,
        function_name: &'a str,
        layout: SetLayout,
    ) -> Self {
        let entry_types = layout.entry_type.clone();

        // Compute pipeline bindings
        let layout = fw
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &layout.layout_entry,
            });

        let pipeline_layout = fw
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });

        let pipeline = fw
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                module: &shader.0,
                entry_point: function_name,
                layout: Some(&pipeline_layout),
            });

        Self {
            pipeline,
            entry_types,
            layout,
            function_name,
        }
    }

    /// executes this [`Kernel`] with the give bindings.
    ///
    /// [`Kernel`] will dispatch `x`, `y` and `z` workgroups per dimension.
    pub fn run(
        &self,
        fw: &Framework,
        arguments: Vec<&'a dyn BindGroupEntryType>,
        x: u32,
        y: u32,
        z: u32,
    ) {
        let binding = SetBindings::new(arguments);

        let mut encoder = fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Kernel::enqueue"),
            });

        let bind_group = binding.to_bind_group(fw, &self.layout, &self.entry_types);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Kernel::enqueue"),
                timestamp_writes: None,
                // Figure out what timestamp_writes means
            });

            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker(self.function_name);
            cpass.dispatch_workgroups(x, y, z);
        }

        fw.queue.submit(Some(encoder.finish()));
    }
}
