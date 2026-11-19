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

use std::arch::x86_64::_mm_extract_epi16;
use std::arch::x86_64::_mm_load_si128;
use std::arch::x86_64::_mm_or_si128;
use std::arch::x86_64::_mm_slli_si128;
use std::arch::x86_64::_mm_srli_si128;
use std::arch::x86_64::_mm_store_si128;

pub use crate::simd::sse2_base::*;

#[target_feature(enable = "sse2")]
#[inline]
pub fn _mm_extract_epi8<const INDEX: i32>(a: __m128i) -> i32 {
    // TODO: revisit this when generic_const_exprs graduates from nightly
    // _mm_extract_epi16 is available in SSE2
    let word = match INDEX / 2 {
        0 => _mm_extract_epi16::<0>(a),
        1 => _mm_extract_epi16::<1>(a),
        2 => _mm_extract_epi16::<2>(a),
        3 => _mm_extract_epi16::<3>(a),
        4 => _mm_extract_epi16::<4>(a),
        5 => _mm_extract_epi16::<5>(a),
        6 => _mm_extract_epi16::<6>(a),
        7 => _mm_extract_epi16::<7>(a),
        _ => unreachable!(),
    };

    let is_high_byte = (INDEX % 2) != 0;
    if is_high_byte {
        (word >> 8) & 0xFF
    } else {
        word & 0xFF
    }
}

/// Emulates SSSE3 _mm_shuffle_epi8 using non SIMD instructions
#[inline]
#[target_feature(enable = "sse2")]
fn _mm_shuffle_epi8(a: __m128i, b: __m128i) -> __m128i {
    #[repr(align(16))]
    struct AlignedM128([u8; 16]);

    // Copy to arrays to access individual bytes (SSE2 doesn't have a direct
    // variable byte-shuffler like PSHUFB).
    let mut src = AlignedM128([0u8; 16]);
    let mut mask = AlignedM128([0u8; 16]);
    // SAFETY: a and b are __m128i so it is safe to write [u8;16] = 128 bits to it.
    unsafe {
        _mm_store_si128(src.0.as_mut_ptr().cast::<__m128i>(), a);
        _mm_store_si128(mask.0.as_mut_ptr().cast::<__m128i>(), b);
    }

    let mut res = AlignedM128([0u8; 16]);

    for i in 0..16 {
        // SSSE3/AVX2 Logic:
        // If bit 7 of the mask byte is set, the result is 0.
        // Otherwise, use the lower 4 bits as an index into the source lane.
        if (mask.0[i] & 0x80) == 0 {
            let index = (mask.0[i] & 0x0F) as usize;
            res.0[i] = src.0[index];
        } else {
            res.0[i] = 0;
        }
    }

    // SAFETY: res is __m128i so it is safe to read [u8;16] = 128 bits from it.
    unsafe { _mm_load_si128(res.0.as_ptr().cast::<__m128i>()) }
}

#[target_feature(enable = "sse2")]
#[inline]
pub fn _mm256_shuffle_epi8(a: __m256i, b: __m256i) -> __m256i {
    _mm256_shuffle_epi8!(a, b)
}

#[target_feature(enable = "sse2")]
#[inline]
fn _mm_alignr_epi8_sse2<const SHIFT: i32>(a: __m128i, b: __m128i) -> __m128i {
    match SHIFT {
        0 => b,
        1 => _mm_or_si128(_mm_slli_si128::<15>(a), _mm_srli_si128::<1>(b)),
        2 => _mm_or_si128(_mm_slli_si128::<14>(a), _mm_srli_si128::<2>(b)),
        3 => _mm_or_si128(_mm_slli_si128::<13>(a), _mm_srli_si128::<3>(b)),
        4 => _mm_or_si128(_mm_slli_si128::<12>(a), _mm_srli_si128::<4>(b)),
        5 => _mm_or_si128(_mm_slli_si128::<11>(a), _mm_srli_si128::<5>(b)),
        6 => _mm_or_si128(_mm_slli_si128::<10>(a), _mm_srli_si128::<6>(b)),
        7 => _mm_or_si128(_mm_slli_si128::<9>(a), _mm_srli_si128::<7>(b)),
        8 => _mm_or_si128(_mm_slli_si128::<8>(a), _mm_srli_si128::<8>(b)),
        9 => _mm_or_si128(_mm_slli_si128::<7>(a), _mm_srli_si128::<9>(b)),
        10 => _mm_or_si128(_mm_slli_si128::<6>(a), _mm_srli_si128::<10>(b)),
        11 => _mm_or_si128(_mm_slli_si128::<5>(a), _mm_srli_si128::<11>(b)),
        12 => _mm_or_si128(_mm_slli_si128::<4>(a), _mm_srli_si128::<12>(b)),
        13 => _mm_or_si128(_mm_slli_si128::<3>(a), _mm_srli_si128::<13>(b)),
        14 => _mm_or_si128(_mm_slli_si128::<2>(a), _mm_srli_si128::<14>(b)),
        15 => _mm_or_si128(_mm_slli_si128::<1>(a), _mm_srli_si128::<15>(b)),
        16 => a,
        17..=31 => match SHIFT - 16 {
            1 => _mm_srli_si128::<1>(a),
            2 => _mm_srli_si128::<2>(a),
            3 => _mm_srli_si128::<3>(a),
            4 => _mm_srli_si128::<4>(a),
            5 => _mm_srli_si128::<5>(a),
            6 => _mm_srli_si128::<6>(a),
            7 => _mm_srli_si128::<7>(a),
            8 => _mm_srli_si128::<8>(a),
            9 => _mm_srli_si128::<9>(a),
            10 => _mm_srli_si128::<10>(a),
            11 => _mm_srli_si128::<11>(a),
            12 => _mm_srli_si128::<12>(a),
            13 => _mm_srli_si128::<13>(a),
            14 => _mm_srli_si128::<14>(a),
            15 => _mm_srli_si128::<15>(a),
            _ => unreachable!(),
        },
        _ => _mm_setzero_si128(),
    }
}

#[target_feature(enable = "sse2")]
#[inline]
pub fn _mm256_alignr_epi8<const SHIFT: i32>(a: __m256i, b: __m256i) -> __m256i {
    __m256i {
        low: _mm_alignr_epi8_sse2::<SHIFT>(a.low, b.low),
        high: _mm_alignr_epi8_sse2::<SHIFT>(a.high, b.high),
    }
}
