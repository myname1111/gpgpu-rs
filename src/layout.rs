use crate::{entry_type::EntryType, primitives::PixelInfo, GpuBufferUsage};

#[derive(Default, Clone)]
pub struct SetLayout {
    pub(crate) layout_entry: Vec<wgpu::BindGroupLayoutEntry>,
    pub(crate) entry_type: Vec<EntryType>,
    bind_id: u32,
}

/// Creates a new [SetLayout]
///
/// Types:
/// * UniformBuffer: Use it for uniforms
/// * Buffer: Use it for buffers or arrays
/// * ConstImage: Use it for images that cannot be changed
/// * Image: Use it for images that can be changed
///
/// Example:
/// ```
/// use gpgpu::{new_set_layout, primitives::pixels::Rgba8Uint, BufOps};
/// new_set_layout!(
///     0: UniformBuffer,
///     1: Image<Rgba8Uint>, // Replace Rgba8Uint with anything that implements [PixelInfo]
///     2: ConstImage<Rgba8Uint>,
///     3: Buffer(gpgpu::GpuBufferUsage::ReadOnly)
/// );
/// ```
#[macro_export]
macro_rules! layout {
    (@add_entry $usage:expr, $layout:expr, Buffer) => {
        $layout.add_buffer($usage);
    };
    (@add_entry $layout:expr, UniformBuffer) => {
        $layout.add_uniform_buffer();
    };
    (@add_entry $p:ty, $layout:expr, Image) => {
        $layout.add_image::<$p>();
    };
    (@add_entry $p:ty, $layout:expr, ConstImage) => {
        $layout.add_const_image::<$p>();
    };
    ($($ty:tt$(<$p:ty>)?$(($usage:expr))?),+) => {{
        let mut layout = $crate::layout::SetLayout::default();

        $(
            $crate::layout!(@add_entry $($p, )? $($usage, )? layout, $ty);
        )+

        layout
    }};
}

impl SetLayout {
    pub fn add_buffer(&mut self, usage: GpuBufferUsage) {
        let entry = wgpu::BindGroupLayoutEntry {
            binding: self.bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Storage {
                    read_only: usage == GpuBufferUsage::ReadOnly,
                },
            },
            count: None,
        };

        self.layout_entry.push(entry);
        self.entry_type.push(EntryType::Buffer);
        self.bind_id += 1;
    }

    pub fn add_uniform_buffer(&mut self) {
        let entry = wgpu::BindGroupLayoutEntry {
            binding: self.bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Uniform,
            },
            count: None,
        };

        self.layout_entry.push(entry);
        self.entry_type.push(EntryType::Uniform);
        self.bind_id += 1;
    }

    pub fn add_image<P: PixelInfo>(&mut self) {
        let entry = wgpu::BindGroupLayoutEntry {
            binding: self.bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: P::wgpu_format(),
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        self.layout_entry.push(entry);
        self.entry_type.push(EntryType::Image);
        self.bind_id += 1;
    }

    pub fn add_const_image<P: PixelInfo>(&mut self) {
        let entry = wgpu::BindGroupLayoutEntry {
            binding: self.bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: P::wgpu_texture_sample(),
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        self.layout_entry.push(entry);
        self.entry_type.push(EntryType::ConstImage);
        self.bind_id += 1;
    }
}
