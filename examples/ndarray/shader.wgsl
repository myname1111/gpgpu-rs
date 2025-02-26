struct Dims {
    x: u32,
    y: u32,
}

struct Array {
    data: array<i32>,
}

@group(0) @binding(0) var<uniform> dims: Dims;   // Array dimensions
@group(0) @binding(1) var<storage, read> a: Array;           
@group(0) @binding(2) var<storage, read> b: Array;           
@group(0) @binding(3) var<storage, read_write> c: Array;  

// // Example without workgroups
// fn main_fn_1([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
//     let idx = global_id.x; 
//     c.data[idx] = a.data[idx] + b.data[idx];
// }

// Example with workgroups
@compute @workgroup_size(32, 32)
fn main_fn_2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = (global_id.x * dims.x) + global_id.y;
    c.data[id] = a.data[id] + b.data[id];
}
