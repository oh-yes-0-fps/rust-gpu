// Test unsizing arrays to slices and sub-slicing slices, for function-local, workgroup and
// storage-buffer memory, with the slices passed on to other functions.

// build-pass

use spirv_std::spirv;

fn sum(s: &[u32], i: usize) -> u32 {
    s[i] + s[i + 1]
}

fn fill(s: &mut [u32], v: u32) {
    let mut i = 0;
    while i < s.len() {
        s[i] = v;
        i += 1;
    }
}

#[spirv(compute(threads(32)))]
pub fn main(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; 64],
    #[spirv(local_invocation_index)] i: u32,
) {
    let i = i as usize;
    let mut local = [0u32; 8];
    fill(&mut local, 3);
    fill(&mut shared[32..], 7);
    shared[i] = i as u32;
    out[i] =
        sum(&shared[..], i) + sum(&input[4..], i) + sum(&shared[8..40], i) + sum(&local, i % 7);
}
