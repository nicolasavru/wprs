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

use std::env;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::num::NonZeroUsize;
use std::path::Path;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use png::BitDepth;
use png::ColorType;
use png::Decoder;
use rkyv::util::AlignedVec;
use wprs::buffer_pointer::BufferPointer;
use wprs::filtering;
use wprs::serialization::CompressionMode;
use wprs::serialization::wayland::BufferFormat;
use wprs::serialization::wayland::BufferMetadata;
use wprs::sharding_compression::ShardingCompressor;
use wprs::sharding_compression::ShardingDecompressor;

fn reorder_channels(data: &mut [u8]) {
    for pixel in data.chunks_mut(4) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];
        let a = pixel[3];

        // https://afrantzis.com/pixel-format-guide/wayland_drm.html
        pixel[0] = b;
        pixel[1] = g;
        pixel[2] = r;
        pixel[3] = a;
    }
}

fn read_png(path: &Path) -> (Vec<u8>, u32, u32) {
    println!("reading png {}", path.display());
    let decoder = Decoder::new(BufReader::new(File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    println!("INFO {info:?}");
    assert_eq!(info.color_type, ColorType::Rgba);
    assert_eq!(info.bit_depth, BitDepth::Eight);
    let mut data = buf[..info.buffer_size()].to_vec();
    reorder_channels(&mut data);
    (data, info.width, info.height)
}

fn filter_compress_png(c: &mut Criterion, path: &Path) {
    let (data, width, height) = read_png(path);
    let data_ptr = data.as_ptr();
    let buf_ptr = unsafe { BufferPointer::new(&data_ptr, data.len()) };
    let metadata = BufferMetadata {
        width: width as i32,
        height: height as i32,
        stride: (width * 4) as i32,
        format: BufferFormat::Argb8888,
    };

    let uncompressed_size = data.len();

    let n_compressors = NonZeroUsize::new(16).unwrap();
    let mut compressor = ShardingCompressor::new(n_compressors, 1).unwrap();

    let mut compressed_tiles = filtering::filter_and_compress(
        buf_ptr,
        metadata,
        CompressionMode::Lossless,
        &mut compressor,
    );

    c.bench_function(&format!("filter and compress: {}", path.display()), |b| {
        b.iter(|| {
            compressed_tiles = filtering::filter_and_compress(
                buf_ptr,
                metadata,
                CompressionMode::Lossless,
                &mut compressor,
            );
        })
    });

    let compressed_size = compressed_tiles.size();

    let n_decompressors = NonZeroUsize::new(8).unwrap();
    let mut sharding_decompressor = ShardingDecompressor::new(n_decompressors).unwrap();
    let mut output_buf = AlignedVec::<64>::with_capacity(uncompressed_size);
    output_buf.resize(uncompressed_size, 0);
    filtering::decompress_and_unfilter(
        &compressed_tiles,
        &mut sharding_decompressor,
        &mut output_buf[..],
    )
    .unwrap();
    assert!(output_buf[..] == data[..], "unfilter roundtrip mismatch");

    c.bench_function(
        &format!("unfilter and decompress: {}", path.display()),
        |b| {
            b.iter(|| {
                filtering::decompress_and_unfilter(
                    &compressed_tiles,
                    &mut sharding_decompressor,
                    &mut output_buf[..],
                )
                .unwrap();
            })
        },
    );

    let compression_ratio = uncompressed_size as f64 / compressed_size as f64;
    let compression_rate = compressed_size as f64 / uncompressed_size as f64;
    println!("compression ratio (higher is better): {compression_ratio:.1}");
    println!(
        "compression rate (lower is better): {:.1}%",
        compression_rate * 100.0
    );
}

fn compression_benchmark(c: &mut Criterion) {
    wprs::utils::exit_on_thread_panic();

    let image_dir: String =
        env::var("WPRS_BENCH_IMAGE_DIR").expect("WPRS_BENCH_IMAGE_DIR env var must be set.");
    if image_dir.is_empty() {
        panic!("WPRS_BENCH_IMAGE_DIR env var must be non-empty.")
    }

    let files = fs::read_dir(image_dir)
        .unwrap()
        .map(|dirent| dirent.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "png"));
    for file in files {
        filter_compress_png(c, &file);
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("");
    }
}

criterion_group!(benches, compression_benchmark);
criterion_main!(benches);
