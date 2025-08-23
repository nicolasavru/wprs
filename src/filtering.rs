// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::buffer_pointer::BufferPointer;
use crate::prelude::*;
use crate::transpose;
use crate::vec4u8::Vec4u8;
use crate::vec4u8::Vec4u8s;

// TODO: benchmarks, enable avx2 for auto-vectorization:
// https://doc.rust-lang.org/beta/core/arch/index.html#examples

#[instrument(skip_all, level = "debug")]
pub fn filter(data: BufferPointer<u8>, output_buf: &mut Vec4u8s) {
    assert!(data.len().is_multiple_of(4)); // data is a buffer of argb or xrgb pixels.
                                           // SAFETY: Vec4u8 is a repr(C, packed) wrapper around [u8; 4].
    let data = unsafe { data.cast::<Vec4u8>() };
    transpose::vec4u8_aos_to_soa(data, output_buf);
    // filter_argb8888(output_buf);
}

#[instrument(skip_all, level = "debug")]
pub fn unfilter(data: &mut Vec4u8s, output_buf: &mut [u8]) {
    let output_buf = bytemuck::cast_slice_mut(output_buf);
    // unfilter_argb8888(data);
    transpose::vec4u8_soa_to_aos(data, output_buf);
}

// https://afrantzis.com/pixel-format-guide/wayland_drm.html
