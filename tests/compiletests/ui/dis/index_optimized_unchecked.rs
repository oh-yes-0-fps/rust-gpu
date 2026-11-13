#![crate_name = "index_optimized_unchecked"]

// Optimized builds treat panics as unreachable, so the bounds check of `slice[i]`
// leaves no branch behind (compare `index_user_dst`, built unoptimized).

// build-pass
// compile-flags: -C opt-level=3 -C codegen-units=1 -C llvm-args=--disassemble-entry=main

use spirv_std::spirv;

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] slice: &mut [f32],
    #[spirv(push_constant)] i: &u32,
) {
    slice[0] = slice[*i as usize];
}
