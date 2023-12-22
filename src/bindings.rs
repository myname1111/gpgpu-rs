//! Provides the [`SetBindings`] struct used to bind the data to the gpu
use crate::{entry_type::EntryType, primitives::*, *};

/// Binds data to bindings that will be sent to the gpu
///
/// # Example
/// ```
/// # use gpgpu::*;
/// let fw = Framework::default();
///
/// let data = (0..10000).into_iter().collect::<Vec<u32>>();
/// let scalar = 10u32;
///
/// // Create the buffer
/// let buffer = GpuBuffer::from_slice(&fw, &data);
///
/// // Create the uniform
/// let uniform = GpuUniformBuffer::from_slice(&fw, &[scalar]);
///
/// let binds = SetBindings::default()
///     .add_buffer(0, &buffer) // Binds data to the gpu with id 0
///     .add_uniform_buffer(1, &uniform); // Bind a uniform with id 1
/// ```
#[derive(Clone, Default)]
pub struct SetBindings<'res> {
    pub(crate) bindings: Vec<wgpu::BindGroupEntry<'res>>,
    pub(crate) entry_type: Vec<EntryType>,
    bind_id: u32,
}

impl<'res> SetBindings<'res> {
    pub(crate) fn new(data: Vec<&'res dyn BindGroupEntryType>) -> Self {
        let mut out = Self::default();
        for data in data {
            out.add_arg(data)
        }
        out
    }

    fn add_arg<T>(&mut self, data: &'res T)
    where
        T: BindGroupEntryType + ?Sized,
    {
        let bind = wgpu::BindGroupEntry {
            binding: self.bind_id,
            resource: data.as_binding_resource(),
        };

        self.bindings.push(bind);
        self.entry_type.push(data.entry_type());
        self.bind_id += 1;
    }

    pub(crate) fn to_bind_group(
        &self,
        fw: &Framework,
        layout: &wgpu::BindGroupLayout,
        entry_types: &[EntryType],
    ) -> wgpu::BindGroup {
        if self.entry_type.len() != entry_types.len() {
            panic!("Bindings must have the same layout as Layout")
        }

        for entry_type in self.entry_type.iter().zip(entry_types.iter()) {
            if entry_type.0 != entry_type.1 {
                panic!(
                    "A binding of type {:?} is different from {:?} which was previouly defined in the layout",
                    entry_type.0, entry_type.1
                )
            }
        }

        fw.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &self.bindings,
        })
    }
}
