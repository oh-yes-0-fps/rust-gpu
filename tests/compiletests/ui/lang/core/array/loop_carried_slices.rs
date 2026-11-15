// Test slices, and structs and arrays holding slices, carried from one loop iteration to the
// next: the pointers inside them have to become indices for logical addressing.

// build-pass

use spirv_std::spirv;

#[derive(Clone, Copy)]
struct Window<'a> {
    rows: &'a [u32],
    stride: usize,
    _marker: (),
}

#[spirv(compute(threads(32)))]
pub fn main(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; 64],
    #[spirv(local_invocation_index)] i: u32,
) {
    let i = i as usize;
    shared[i] = i as u32;
    let mut rest = &input[..];
    let mut windows = [Window { rows: &shared[..], stride: 1, _marker: () }; 2];
    let mut sum = 0;
    let mut k = 0;
    while k < 4 {
        sum += rest[i] + windows[0].rows[i] + windows[1].rows[i * windows[1].stride];
        rest = &rest[8..];
        let next = Window { rows: &shared[k..], stride: 1, _marker: () };
        windows = [windows[1], next];
        k += 1;
    }
    out[i] = sum;
}
