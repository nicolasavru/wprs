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

use std::alloc;
use std::alloc::Layout;
use std::any::Any;
use std::num::NonZeroUsize;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::thread;
use std::thread::JoinHandle;

use zstd::bulk::Compressor;
use zstd::bulk::Decompressor;
use zstd::zstd_safe;

use crate::prelude::*;
use crate::serialization::framing::TILE_PIXELS;
use crate::serialization::framing::TileChannel;

#[repr(C, align(64))]
pub(crate) struct TileSoaScratch(pub(crate) [[u8; TILE_PIXELS]; 4]);

impl TileSoaScratch {
    fn new_boxed() -> Box<Self> {
        let layout = Layout::new::<Self>();
        // SAFETY: `TileSoaScratch` is a plain byte array with non-zero size, so
        // an all-zero allocation is valid for `Self`.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        // SAFETY: `ptr` was allocated with `Layout::new::<Self>()` and
        // initialized to valid zero bytes.
        unsafe { Box::from_raw(ptr.cast::<Self>()) }
    }
}

struct PoolJob<S> {
    n_active_target: usize,
    func: unsafe fn(*const (), &mut S),
    ctx: usize,
}

impl<S> Copy for PoolJob<S> {}

impl<S> Clone for PoolJob<S> {
    fn clone(&self) -> Self {
        *self
    }
}

struct PoolSync<S> {
    job: Option<PoolJob<S>>,
    claimed_workers: usize,
    active_workers: usize,
    shutdown: bool,
    panic_payload: Option<Box<dyn Any + Send + 'static>>,
}

struct ActiveJobGuard<'a, S> {
    shared: &'a (Mutex<PoolSync<S>>, Condvar, Condvar),
}

impl<S> Drop for ActiveJobGuard<'_, S> {
    fn drop(&mut self) {
        let (mutex, _, done_cv) = self.shared;
        let mut sync = mutex.lock().unwrap_or_else(PoisonError::into_inner);
        sync.job = None;
        while sync.active_workers > 0 {
            sync = done_cv.wait(sync).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

unsafe fn call_closure<F, S>(ctx: *const (), state: &mut S)
where
    F: Fn(&mut S) + Sync,
{
    // SAFETY: `ctx` points to `f` on `WorkerPool::execute`'s stack frame, which
    // outlives all worker executions via `ActiveJobGuard`.
    unsafe { (&*ctx.cast::<F>())(state) };
}

fn worker_loop<S>(mut state: S, shared: Arc<(Mutex<PoolSync<S>>, Condvar, Condvar)>) {
    let (mutex, work_cv, done_cv) = &*shared;

    loop {
        let job = {
            let mut sync = mutex.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                if sync.shutdown {
                    return;
                }
                if let Some(job) = sync.job
                    && sync.claimed_workers < job.n_active_target
                {
                    sync.claimed_workers += 1;
                    sync.active_workers += 1;
                    break job;
                }
                sync = work_cv.wait(sync).unwrap_or_else(PoisonError::into_inner);
            }
        };

        let res = panic::catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: `job.ctx` was created from `&f` in `WorkerPool::execute`,
            // and `active_workers > 0` prevents `ActiveJobGuard` from returning
            // while this call is in flight.
            unsafe { (job.func)(job.ctx as *const (), &mut state) };
        }));

        let mut sync = mutex.lock().unwrap_or_else(PoisonError::into_inner);
        if let Err(payload) = res
            && sync.panic_payload.is_none()
        {
            sync.panic_payload = Some(payload);
        }
        sync.active_workers -= 1;
        if sync.active_workers == 0 {
            done_cv.notify_one();
        }
    }
}

pub(crate) struct WorkerPool<S> {
    shared: Arc<(Mutex<PoolSync<S>>, Condvar, Condvar)>,
    inline_state: S,
    n_workers: usize,
    handles: Vec<JoinHandle<()>>,
}

impl<S: Send + 'static> WorkerPool<S> {
    fn new(
        total_parallelism: NonZeroUsize,
        mut make_state: impl FnMut() -> Result<S>,
    ) -> Result<Self> {
        let inline_state = make_state().location(loc!())?;
        let n_workers = total_parallelism.get().saturating_sub(1);
        // Build every worker state before spawning so a `make_state` error
        // leaks no parked threads.
        let mut worker_states = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            worker_states.push(make_state().location(loc!())?);
        }
        let shared = Arc::new((
            Mutex::new(PoolSync {
                job: None,
                claimed_workers: 0,
                active_workers: 0,
                shutdown: false,
                panic_payload: None,
            }),
            Condvar::new(),
            Condvar::new(),
        ));

        let mut handles = Vec::with_capacity(n_workers);
        for state in worker_states {
            let shared_clone = shared.clone();
            handles.push(thread::spawn(move || {
                worker_loop(state, shared_clone);
            }));
        }

        Ok(Self {
            shared,
            inline_state,
            n_workers,
            handles,
        })
    }

    /// Runs `f` concurrently across the caller thread and up to
    /// `(max_tasks - 1).min(self.n_workers)` background worker threads, passing
    /// each participant its exclusive `&mut S` state. A worker may re-enter `f`
    /// within the same job while `claimed_workers < n_active_target`, so `f`
    /// must self-schedule work items (e.g. via an `AtomicUsize` counter) until
    /// no work remains. Does nothing if `max_tasks == 0`. Blocks until all
    /// participants return. If the caller thread's `f` panics, that panic
    /// propagates after all workers finish; otherwise the first worker panic is
    /// re-raised.
    pub(crate) fn execute<F>(&mut self, max_tasks: usize, f: F)
    where
        F: Fn(&mut S) + Sync,
    {
        if max_tasks == 0 {
            return;
        }

        let n_active_target = (max_tasks - 1).min(self.n_workers);
        let job = PoolJob {
            n_active_target,
            func: call_closure::<F, S>,
            ctx: (&raw const f) as usize,
        };

        {
            let (mutex, work_cv, _) = &*self.shared;
            let mut sync = mutex.lock().unwrap_or_else(PoisonError::into_inner);
            sync.panic_payload = None;
            sync.claimed_workers = 0;
            sync.job = Some(job);
            if n_active_target >= self.n_workers {
                work_cv.notify_all();
            } else {
                for _ in 0..n_active_target {
                    work_cv.notify_one();
                }
            }
        }

        {
            let _guard = ActiveJobGuard {
                shared: &self.shared,
            };
            f(&mut self.inline_state);
        }

        let mut sync = self.shared.0.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(payload) = sync.panic_payload.take() {
            drop(sync);
            panic::resume_unwind(payload);
        }
    }
}

impl<S> Drop for WorkerPool<S> {
    fn drop(&mut self) {
        {
            let (mutex, work_cv, _) = &*self.shared;
            let mut sync = mutex.lock().unwrap_or_else(PoisonError::into_inner);
            sync.shutdown = true;
            work_cv.notify_all();
        }
        for handle in self.handles.drain(..) {
            _ = handle.join();
        }
    }
}

pub(crate) struct CompressorWorkerState {
    pub(crate) compressor: Compressor<'static>,
    pub(crate) soa_scratch: Box<TileSoaScratch>,
    pub(crate) zstd_scratch: Vec<u8>,
}

impl CompressorWorkerState {
    fn new(compression_level: i32) -> Result<Self> {
        let compressor = Compressor::new(compression_level).location(loc!())?;
        let zstd_scratch = Vec::with_capacity(zstd_safe::compress_bound(TILE_PIXELS));
        Ok(Self {
            compressor,
            soa_scratch: TileSoaScratch::new_boxed(),
            zstd_scratch,
        })
    }
}

/// Maximum filtered channel-slice length sent uncompressed without attempting
/// zstd (`512 B` compresses ~4.4x in ~4.4 us at level 1, ~730 Mbps breakeven).
const MIN_SIZE_TO_COMPRESS: usize = 512;

pub(crate) fn compress_channel_slice(
    compressor: &mut Compressor<'static>,
    zstd_scratch: &mut Vec<u8>,
    data: &[u8],
) -> TileChannel {
    if data.len() > MIN_SIZE_TO_COMPRESS {
        let bound = zstd_safe::compress_bound(data.len());
        zstd_scratch.clear();
        zstd_scratch.reserve(bound);
        let n = compressor
            .compress_to_buffer(data, zstd_scratch)
            .expect("dst reserved to compress_bound");
        if n < data.len() {
            return TileChannel::Zstd(zstd_scratch[..n].to_vec());
        }
    }
    TileChannel::Uncompressed(data.to_vec())
}

pub(crate) struct DecompressorWorkerState {
    pub(crate) decompressor: Decompressor<'static>,
    pub(crate) soa_scratch: Box<TileSoaScratch>,
}

impl DecompressorWorkerState {
    fn new() -> Result<Self> {
        Ok(Self {
            decompressor: Decompressor::new().location(loc!())?,
            soa_scratch: TileSoaScratch::new_boxed(),
        })
    }
}

pub struct ShardingCompressor {
    pub(crate) pool: WorkerPool<CompressorWorkerState>,
}

impl ShardingCompressor {
    pub fn new(n_compressors: NonZeroUsize, compression_level: i32) -> Result<Self> {
        let pool = WorkerPool::new(n_compressors, || {
            CompressorWorkerState::new(compression_level)
        })
        .location(loc!())?;
        Ok(Self { pool })
    }
}

pub struct ShardingDecompressor {
    pub(crate) pool: WorkerPool<DecompressorWorkerState>,
}

impl ShardingDecompressor {
    pub fn new(n_decompressors: NonZeroUsize) -> Result<Self> {
        let pool =
            WorkerPool::new(n_decompressors, DecompressorWorkerState::new).location(loc!())?;
        Ok(Self { pool })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use super::*;

    #[test]
    fn test_tile_soa_scratch_alignment_and_zero_init() {
        let mut scratch = TileSoaScratch::new_boxed();
        let addr = (&raw const *scratch) as usize;
        assert_eq!(addr % 64, 0);
        assert_eq!(scratch.0[0][0], 0);
        assert_eq!(scratch.0[3][TILE_PIXELS - 1], 0);
        scratch.0[2][123] = 0xAB;
        assert_eq!(scratch.0[2][123], 0xAB);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compress_channel_slice_selects_codec() {
        let mut compressor = Compressor::new(1).unwrap();
        let mut scratch = Vec::new();
        let mut noise = vec![0u8; TILE_PIXELS];
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for chunk in noise.as_chunks_mut::<8>().0 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            chunk.copy_from_slice(&z.to_le_bytes());
        }
        let compressible: Vec<u8> = (0..TILE_PIXELS).map(|i| (i / 64) as u8).collect();

        assert_eq!(
            compress_channel_slice(
                &mut compressor,
                &mut scratch,
                &compressible[..MIN_SIZE_TO_COMPRESS]
            ),
            TileChannel::Uncompressed(compressible[..MIN_SIZE_TO_COMPRESS].to_vec())
        );
        let above_threshold = &compressible[..MIN_SIZE_TO_COMPRESS + 1];
        let TileChannel::Zstd(compressed_above) =
            compress_channel_slice(&mut compressor, &mut scratch, above_threshold)
        else {
            panic!("expected Zstd above MIN_SIZE_TO_COMPRESS");
        };
        assert_eq!(
            zstd::bulk::decompress(&compressed_above, above_threshold.len()).unwrap(),
            above_threshold
        );

        assert_eq!(
            compress_channel_slice(&mut compressor, &mut scratch, &noise),
            TileChannel::Uncompressed(noise.clone())
        );
        let TileChannel::Zstd(compressed) =
            compress_channel_slice(&mut compressor, &mut scratch, &compressible)
        else {
            panic!("expected Zstd for compressible input");
        };
        assert_eq!(
            zstd::bulk::decompress(&compressed, compressible.len()).unwrap(),
            compressible
        );
    }

    #[test]
    fn test_worker_pool_dispatch_and_reuse() {
        let mut next_id = 0usize;
        let mut pool = WorkerPool::new(NonZeroUsize::new(4).unwrap(), || {
            let id = next_id;
            next_id += 1;
            Ok(id)
        })
        .unwrap();

        // max_tasks = 0 does not run the closure.
        let calls = AtomicUsize::new(0);
        pool.execute(0, |_| {
            calls.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        // max_tasks = 1 runs inline on worker 0 only.
        pool.execute(1, |id| {
            assert_eq!(*id, 0);
            calls.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Verify 1, 2 (`1 < n_active_target < n_workers`), and all 3 background
        // workers wake deterministically alongside the inline caller thread.
        for (max_tasks, expected_parties, iters) in [(2, 2, 8), (3, 3, 4), (12, 4, 2)] {
            for _ in 0..iters {
                let barrier = Barrier::new(expected_parties);
                let participants = AtomicUsize::new(0);
                pool.execute(max_tasks, |_| {
                    participants.fetch_add(1, Ordering::SeqCst);
                    barrier.wait();
                });
                assert_eq!(participants.load(Ordering::SeqCst), expected_parties);
            }
        }
    }

    #[test]
    fn test_worker_pool_panic_and_partial_init_error() {
        struct Counted(Arc<AtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        // `WorkerPool::new` builds all states before spawning threads, so a
        // mid-init error drops already-constructed states immediately without
        // leaving background threads holding them.
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut created = 0usize;
        let err_res: Result<WorkerPool<Counted>> =
            WorkerPool::new(NonZeroUsize::new(4).unwrap(), || {
                created += 1;
                if created == 3 {
                    bail!("synthetic init failure");
                }
                Ok(Counted(dropped.clone()))
            });
        assert!(err_res.is_err());
        assert_eq!(dropped.load(Ordering::SeqCst), 2);

        let mut id = 0usize;
        let mut pool = WorkerPool::new(NonZeroUsize::new(3).unwrap(), || {
            let cur = id;
            id += 1;
            Ok(cur)
        })
        .unwrap();

        // Worker panic propagates cleanly.
        let res = panic::catch_unwind(AssertUnwindSafe(|| {
            let barrier = Barrier::new(2);
            pool.execute(2, |worker_id| {
                barrier.wait();
                if *worker_id != 0 {
                    panic!("worker boom");
                }
            });
        }));
        assert!(res.is_err());

        // Dual inline + worker panic must not leave stale `panic_payload` for
        // subsequent jobs.
        let res_dual = panic::catch_unwind(AssertUnwindSafe(|| {
            let barrier = Barrier::new(2);
            pool.execute(2, |_| {
                barrier.wait();
                panic!("dual boom");
            });
        }));
        assert!(res_dual.is_err());

        let ok_calls = AtomicUsize::new(0);
        pool.execute(3, |_| {
            ok_calls.fetch_add(1, Ordering::SeqCst);
        });
        assert!(ok_calls.load(Ordering::SeqCst) >= 1);
    }
}
