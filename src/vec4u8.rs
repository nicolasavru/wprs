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

//! vec4 u8 data structures.

use bytemuck::Pod;
use bytemuck::Zeroable;
use static_assertions::assert_eq_size;

/// Convenience layer for operating on arrays of u8s that represent 4-vectors.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C, packed)]
pub struct Vec4u8(pub u8, pub u8, pub u8, pub u8);

assert_eq_size!(Vec4u8, [u8; 4]);

// SAFETY:
// * is inhabited
// * all bit patterns, including all-zeroes, are valid
// * has no uninit/padding bytes
// * u8 is Pod
// * is repr(C)
// * contains no pointer types or interior mutability
// * a shared reference allows only read-only access
unsafe impl Zeroable for Vec4u8 {}
unsafe impl Pod for Vec4u8 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pointer::BufferPointer;

    // The main value from these two tests is running the cast code through miri.
    #[test]
    fn test_cast_vec4u8_to_u8() {
        let v: Vec<Vec4u8> = vec![
            Vec4u8(0, 0, 0, 0),
            Vec4u8(1, 1, 1, 1),
            Vec4u8(2, 2, 2, 2),
            Vec4u8(3, 3, 3, 3),
        ];
        let v_ptr = v.as_ptr();
        let v_bufptr = unsafe { BufferPointer::new(&v_ptr, v.len()) };
        let v_u8 = unsafe { v_bufptr.cast::<u8>() };
        assert_eq!(
            v_u8.into_iter().collect::<Vec<u8>>(),
            vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        );
    }

    #[test]
    fn test_cast_u8_to_vec4u8() {
        let v: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
        let v_ptr = v.as_ptr();
        let v_bufptr = unsafe { BufferPointer::new(&v_ptr, v.len()) };
        let v_vec4u8 = unsafe { v_bufptr.cast::<Vec4u8>() };

        assert_eq!(
            v_vec4u8.into_iter().collect::<Vec<Vec4u8>>(),
            vec![
                Vec4u8(0, 0, 0, 0),
                Vec4u8(1, 1, 1, 1),
                Vec4u8(2, 2, 2, 2),
                Vec4u8(3, 3, 3, 3)
            ]
        );
    }
}
