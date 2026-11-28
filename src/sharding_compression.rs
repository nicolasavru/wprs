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

use std::any::Any;
use std::cell::UnsafeCell;
use std::ffi::CStr;
use std::ffi::c_int;
use std::hint;
use std::num::NonZeroUsize;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::ptr;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
#[cfg(all(target_os = "linux", not(miri)))]
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::thread;
use std::thread::JoinHandle;
#[cfg(not(miri))]
use std::time::Instant;

use zstd::bulk::Compressor;
use zstd::bulk::Decompressor;
use zstd::zstd_safe;

use crate::filtering::palette::PaletteScratch;
use crate::prelude::*;
use crate::serialization::framing::EncodedBytes;
use crate::serialization::framing::JpegSubsamp;
use crate::serialization::framing::TILE_PIXELS;
use crate::serialization::framing::TILE_W;

// Packed `job_state` layout (`u64`):
//   bits  0..32 : low 32 bits of the published `epoch`
//   bits 32..48 : `remaining` unclaimed worker slots (`0..=MAX_WORKERS`)
//   bits 48..62 : `active` workers currently executing `f` (`0..=MAX_WORKERS`)
//   bit  62     : `CALLER_WAITING_BIT` (set under `done.mutex` before sleeping)
//   bit  63     : `CLOSED_BIT` (set by `ActiveJobGuard::drop`)
const EPOCH_MASK: u64 = 0xFFFF_FFFF;
const REMAINING_SHIFT: u32 = 32;
const REMAINING_MASK: u64 = 0xFFFF << REMAINING_SHIFT;
const REMAINING_ONE: u64 = 1 << REMAINING_SHIFT;
const ACTIVE_SHIFT: u32 = 48;
const ACTIVE_MASK: u64 = 0x3FFF << ACTIVE_SHIFT;
const ACTIVE_ONE: u64 = 1 << ACTIVE_SHIFT;
const CALLER_WAITING_BIT: u64 = 1 << 62;
const CLOSED_BIT: u64 = 1 << 63;
const SHUTDOWN_EPOCH: u64 = u64::MAX;
const MAX_WORKERS: usize = 0x3FFF;

// Brief wall-clock spin budgets on uncontended CPUs so consecutive intra-frame
// barriers (Phase A -> Phase B, decode Step 1 -> Step 2, back-to-back frames)
// dispatch with zero syscalls while idle workers park within ~120 us (0% idle
// CPU). Under Miri, use minimal iteration counts to keep interpretation fast.
#[cfg(not(miri))]
const WORKER_SPIN_US: u64 = 120;
#[cfg(not(miri))]
const CALLER_SPIN_US: u64 = 20;
#[cfg(not(miri))]
const SPIN_CHUNK_ITERS: u32 = 64;

#[cfg(miri)]
const MIRI_SPIN_ITERS: u32 = 2;

#[cfg(all(target_os = "linux", not(miri)))]
const SHUTDOWN_FUTEX_WORD: u32 = u32::MAX;
#[cfg(all(target_os = "linux", not(miri)))]
const CONTENTION_POLL_STRIDE: u32 = 4;
#[cfg(all(target_os = "linux", not(miri)))]
const CONTENTION_SAMPLE_INTERVAL_US: u64 = 50_000;
// Process-wide involuntary context-switch rate (preemptions/sec) thresholds
// with hysteresis: pause spinning when external load contends for CPU cores,
// resume once preemptions subside.
#[cfg(all(target_os = "linux", not(miri)))]
const CONTENTION_DISABLE_NIVCSW_PER_SEC: u64 = 80;
#[cfg(all(target_os = "linux", not(miri)))]
const CONTENTION_ENABLE_NIVCSW_PER_SEC: u64 = 40;

#[cfg(all(target_os = "linux", not(miri)))]
#[inline]
const fn epoch_futex_word(epoch: u64) -> u32 {
    if epoch == 0 {
        0
    } else if epoch == SHUTDOWN_EPOCH {
        SHUTDOWN_FUTEX_WORD
    } else {
        // Map live epochs into `1..u32::MAX` so they never collide with `0`
        // (initial idle) or `u32::MAX` (shutdown).
        ((epoch - 1) % ((u32::MAX - 1) as u64)) as u32 + 1
    }
}

#[cfg(all(target_os = "linux", not(miri)))]
#[inline]
fn futex_wait(word: &AtomicU32, expected: u32) {
    // SAFETY: `word.as_ptr()` points to a live, 4-byte-aligned `AtomicU32`
    // inside `PoolShared` (kept alive by `Arc` until the worker exits), and
    // `FUTEX_WAIT | FUTEX_PRIVATE_FLAG` with null timeout only reads `*word`
    // atomically.
    unsafe {
        nix::libc::syscall(
            nix::libc::SYS_futex,
            word.as_ptr(),
            nix::libc::FUTEX_WAIT | nix::libc::FUTEX_PRIVATE_FLAG,
            expected,
            ptr::null::<nix::libc::timespec>(),
            ptr::null::<u32>(),
            0u32,
        );
    }
}

#[cfg(all(target_os = "linux", not(miri)))]
#[inline]
fn futex_wake(word: &AtomicU32, count: i32) {
    // SAFETY: `word.as_ptr()` points to a live, 4-byte-aligned `AtomicU32`.
    unsafe {
        nix::libc::syscall(
            nix::libc::SYS_futex,
            word.as_ptr(),
            nix::libc::FUTEX_WAKE | nix::libc::FUTEX_PRIVATE_FLAG,
            count,
            ptr::null::<nix::libc::timespec>(),
            ptr::null::<u32>(),
            0u32,
        );
    }
}

#[cfg(all(target_os = "linux", not(miri)))]
#[inline]
fn process_involuntary_csw() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<nix::libc::rusage>::zeroed();
    // SAFETY: `RUSAGE_SELF` writes a valid `rusage` struct into `usage`.
    let rc = unsafe { nix::libc::getrusage(nix::libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc == 0 {
        // SAFETY: `getrusage` returned 0, initializing `usage`.
        let u = unsafe { usage.assume_init() };
        Some(u.ru_nivcsw.max(0) as u64)
    } else {
        None
    }
}

#[cfg(all(target_os = "linux", not(miri)))]
struct ContentionMonitor {
    poll_countdown: u32,
    last_sample: Instant,
    last_nivcsw: u64,
    enabled: bool,
}

#[cfg(all(target_os = "linux", not(miri)))]
impl ContentionMonitor {
    fn new(enabled: bool) -> Self {
        Self {
            poll_countdown: CONTENTION_POLL_STRIDE,
            last_sample: Instant::now(),
            last_nivcsw: if enabled {
                process_involuntary_csw().unwrap_or(0)
            } else {
                0
            },
            enabled,
        }
    }

    #[inline]
    fn poll(&mut self, spin_enabled: &AtomicBool) {
        if !self.enabled {
            return;
        }
        self.poll_countdown -= 1;
        if self.poll_countdown > 0 {
            return;
        }
        self.poll_countdown = CONTENTION_POLL_STRIDE;

        let elapsed_us = self.last_sample.elapsed().as_micros() as u64;
        if elapsed_us < CONTENTION_SAMPLE_INTERVAL_US {
            return;
        }
        self.last_sample = Instant::now();
        let Some(cur_nivcsw) = process_involuntary_csw() else {
            return;
        };
        let delta = cur_nivcsw.saturating_sub(self.last_nivcsw);
        self.last_nivcsw = cur_nivcsw;
        let rate_per_sec = delta.saturating_mul(1_000_000) / elapsed_us.max(1);
        let currently_on = spin_enabled.load(Ordering::Relaxed);
        if currently_on && rate_per_sec >= CONTENTION_DISABLE_NIVCSW_PER_SEC {
            spin_enabled.store(false, Ordering::Relaxed);
        } else if !currently_on && rate_per_sec <= CONTENTION_ENABLE_NIVCSW_PER_SEC {
            spin_enabled.store(true, Ordering::Relaxed);
        }
    }
}

#[inline]
const fn pack_job_state(epoch: u64, n_active_target: usize) -> u64 {
    ((n_active_target as u64) << REMAINING_SHIFT) | (epoch & EPOCH_MASK)
}

#[inline]
const fn active_count(state: u64) -> u64 {
    (state & ACTIVE_MASK) >> ACTIVE_SHIFT
}

struct PoolJob<S> {
    epoch: u64,
    func: unsafe fn(*const (), &mut S),
    ctx: *const (),
}

impl<S> Copy for PoolJob<S> {}

impl<S> Clone for PoolJob<S> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe fn noop_job<S>(_ctx: *const (), _state: &mut S) {}

/// Caller-published dispatch line: written only by the `&mut WorkerPool<S>`
/// caller and read-only by spinning/active workers (stays Shared in L1d).
#[repr(align(64))]
struct PublishLine<S> {
    epoch: AtomicU64,
    #[cfg(all(target_os = "linux", not(miri)))]
    futex_word: AtomicU32,
    spin_enabled: AtomicBool,
    job: UnsafeCell<PoolJob<S>>,
}

/// Hot worker-contended atomic claim/active counter, isolated on its own cache
/// line so worker RMWs never false-share with `PublishLine` or `ParkLine`.
#[repr(align(64))]
struct JobStateLine {
    state: AtomicU64,
}

/// Worker sleep-tracking line, isolated from `JobStateLine` so `parked_workers`
/// stays clean/Shared in the caller's L1d while workers are spinning.
#[repr(align(64))]
struct ParkLine {
    #[cfg(all(target_os = "linux", not(miri)))]
    parked_workers: AtomicU32,
    #[cfg(not(all(target_os = "linux", not(miri))))]
    word0: AtomicU64,
    #[cfg(not(all(target_os = "linux", not(miri))))]
    extra: Box<[AtomicU64]>,
}

/// Cold completion-wait and panic-reporting line, only touched when the caller
/// sleeps for straggler workers or when a worker unwinds.
#[repr(align(64))]
struct DoneLine {
    mutex: Mutex<()>,
    cv: Condvar,
    has_panic: AtomicBool,
    panic_payload: Mutex<Option<Box<dyn Any + Send + 'static>>>,
}

struct PoolShared<S> {
    publish: PublishLine<S>,
    job_state: JobStateLine,
    park: ParkLine,
    done: DoneLine,
}

#[cfg(not(all(target_os = "linux", not(miri))))]
impl<S> PoolShared<S> {
    #[inline]
    fn parked_word(&self, word_idx: usize) -> &AtomicU64 {
        if word_idx == 0 {
            &self.park.word0
        } else {
            &self.park.extra[word_idx - 1]
        }
    }
}

// SAFETY: `PoolShared<S>` owns no `S` values. `publish.job` is written solely
// by the `&mut WorkerPool<S>` caller while `CLOSED_BIT` is set and
// `active_count == 0`, released through `job_state.state` (`Release`), and read
// only by workers holding an `active >= 1` claim (`AcqRel`) on the same epoch.
unsafe impl<S> Send for PoolShared<S> {}
// SAFETY: See `Send` impl above.
unsafe impl<S> Sync for PoolShared<S> {}

struct ActiveJobGuard<'a, S> {
    shared: &'a PoolShared<S>,
    caller_completed: bool,
}

impl<S> Drop for ActiveJobGuard<'_, S> {
    fn drop(&mut self) {
        let prev = self
            .shared
            .job_state
            .state
            .fetch_or(CLOSED_BIT, Ordering::AcqRel);
        if active_count(prev) != 0 {
            let mut done = false;
            if self.shared.publish.spin_enabled.load(Ordering::Relaxed) {
                #[cfg(not(miri))]
                {
                    let start = Instant::now();
                    loop {
                        for _ in 0..SPIN_CHUNK_ITERS {
                            if active_count(self.shared.job_state.state.load(Ordering::Acquire))
                                == 0
                            {
                                done = true;
                                break;
                            }
                            hint::spin_loop();
                        }
                        if done || start.elapsed().as_micros() as u64 >= CALLER_SPIN_US {
                            break;
                        }
                    }
                }
                #[cfg(miri)]
                {
                    for _ in 0..MIRI_SPIN_ITERS {
                        if active_count(self.shared.job_state.state.load(Ordering::Acquire)) == 0 {
                            done = true;
                            break;
                        }
                        hint::spin_loop();
                    }
                }
            }

            if !done {
                let mut guard = self
                    .shared
                    .done
                    .mutex
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                // Set `CALLER_WAITING_BIT` while holding `done.mutex` so the
                // last active worker's `fetch_sub(ACTIVE_ONE, AcqRel)` either
                // precedes this RMW (giving `active_count(prev) == 0` below) or
                // follows it (observing `CLOSED_BIT | CALLER_WAITING_BIT` and
                // notifying `done.cv` under `done.mutex`).
                let prev = self
                    .shared
                    .job_state
                    .state
                    .fetch_or(CALLER_WAITING_BIT, Ordering::AcqRel);
                if active_count(prev) > 0 {
                    while active_count(self.shared.job_state.state.load(Ordering::Acquire)) > 0 {
                        guard = self
                            .shared
                            .done
                            .cv
                            .wait(guard)
                            .unwrap_or_else(PoisonError::into_inner);
                    }
                }
            }
        }

        // If the caller's closure panicked, discard any worker panic recorded
        // for this job so the pool stays clean for subsequent `execute` calls.
        if !self.caller_completed && self.shared.done.has_panic.load(Ordering::Acquire) {
            *self
                .shared
                .done
                .panic_payload
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = None;
            self.shared.done.has_panic.store(false, Ordering::Release);
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

/// Waits until `publish.epoch` advances past `last_seen_epoch`, returning
/// `(new_epoch, woke_hot)` where `woke_hot` is `true` if the epoch change was
/// observed without entering the kernel sleep path.
fn wait_for_epoch<S>(
    shared: &PoolShared<S>,
    #[cfg_attr(all(target_os = "linux", not(miri)), allow(unused_variables))] worker_idx: usize,
    last_seen_epoch: u64,
    allow_spin: bool,
) -> (u64, bool) {
    let cur = shared.publish.epoch.load(Ordering::Acquire);
    if cur != last_seen_epoch {
        return (cur, true);
    }

    if allow_spin && shared.publish.spin_enabled.load(Ordering::Relaxed) {
        #[cfg(not(miri))]
        {
            let start = Instant::now();
            loop {
                for _ in 0..SPIN_CHUNK_ITERS {
                    hint::spin_loop();
                    let cur = shared.publish.epoch.load(Ordering::Acquire);
                    if cur != last_seen_epoch {
                        return (cur, true);
                    }
                }
                if start.elapsed().as_micros() as u64 >= WORKER_SPIN_US {
                    break;
                }
            }
        }
        #[cfg(miri)]
        {
            for _ in 0..MIRI_SPIN_ITERS {
                hint::spin_loop();
                let cur = shared.publish.epoch.load(Ordering::Acquire);
                if cur != last_seen_epoch {
                    return (cur, true);
                }
            }
        }
    }

    #[cfg(all(target_os = "linux", not(miri)))]
    {
        let expected_fw = epoch_futex_word(last_seen_epoch);
        // Publish parked count before re-checking `epoch` (`SeqCst` pairs with
        // caller's `epoch.store(..., SeqCst)` + `parked_workers.load(SeqCst)`).
        shared.park.parked_workers.fetch_add(1, Ordering::SeqCst);
        loop {
            let cur = shared.publish.epoch.load(Ordering::SeqCst);
            if cur != last_seen_epoch {
                shared.park.parked_workers.fetch_sub(1, Ordering::Relaxed);
                return (cur, false);
            }
            futex_wait(&shared.publish.futex_word, expected_fw);
        }
    }

    #[cfg(not(all(target_os = "linux", not(miri))))]
    {
        let word = shared.parked_word(worker_idx / 64);
        let bit = 1u64 << (worker_idx % 64);
        loop {
            // Publish our parked bit before checking `epoch` (`SeqCst` pairs
            // with caller's `epoch.store(..., SeqCst)` + `word.load(SeqCst)`).
            word.fetch_or(bit, Ordering::SeqCst);
            let cur = shared.publish.epoch.load(Ordering::SeqCst);
            if cur != last_seen_epoch {
                word.fetch_and(!bit, Ordering::Relaxed);
                return (cur, false);
            }
            thread::park();
            let cur = shared.publish.epoch.load(Ordering::Acquire);
            if cur != last_seen_epoch {
                if (word.load(Ordering::Relaxed) & bit) != 0 {
                    word.fetch_and(!bit, Ordering::Relaxed);
                }
                return (cur, false);
            }
        }
    }
}

fn worker_loop<S>(worker_idx: usize, mut state: S, shared: Arc<PoolShared<S>>) {
    let mut last_seen_epoch = 0u64;
    let mut allow_spin = false;

    'outer: loop {
        let (cur_epoch, woke_hot) =
            wait_for_epoch(&shared, worker_idx, last_seen_epoch, allow_spin);
        if cur_epoch == SHUTDOWN_EPOCH {
            return;
        }

        let cur_epoch_lo = cur_epoch & EPOCH_MASK;
        let mut s = shared.job_state.state.load(Ordering::Acquire);
        loop {
            if (s & EPOCH_MASK) != cur_epoch_lo {
                // Preempted past `cur_epoch`; advance `last_seen_epoch` and
                // immediately inspect the newer epoch without parking.
                last_seen_epoch = cur_epoch;
                allow_spin = false;
                continue 'outer;
            }
            if (s & CLOSED_BIT) != 0 || (s & REMAINING_MASK) == 0 {
                last_seen_epoch = cur_epoch;
                // Keep already-hot workers spinning briefly across a short
                // low-parallelism barrier (e.g. a 1..3-tile Phase B between
                // two full-width Phase A scans) while letting cold wakes park
                // immediately.
                allow_spin = woke_hot;
                continue 'outer;
            }
            // Atomically claim a slot: `remaining -= 1`, `active += 1`.
            // `remaining > 0` prevents borrow from `active`, and
            // `active < n_workers <= MAX_WORKERS` prevents overflow into
            // `CALLER_WAITING_BIT`.
            let next = (s - REMAINING_ONE) + ACTIVE_ONE;
            match shared.job_state.state.compare_exchange_weak(
                s,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => s = actual,
            }
        }

        // SAFETY: Our successful `AcqRel` CAS synchronized with caller's
        // `Release` store to `job_state.state`, and `active >= 1` prevents
        // `ActiveJobGuard::drop` from returning or overwriting `publish.job`
        // until our `fetch_sub(ACTIVE_ONE, AcqRel)` below.
        let job = unsafe { *shared.publish.job.get() };
        last_seen_epoch = job.epoch;

        let res = panic::catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: `job.ctx` points to `f` in `WorkerPool::execute`, kept
            // alive by `active >= 1` until `ActiveJobGuard::drop` completes.
            unsafe { (job.func)(job.ctx, &mut state) };
        }));

        if let Err(payload) = res {
            let mut slot = shared
                .done
                .panic_payload
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if slot.is_none() {
                *slot = Some(payload);
                shared.done.has_panic.store(true, Ordering::Release);
            }
        }

        let prev = shared
            .job_state
            .state
            .fetch_sub(ACTIVE_ONE, Ordering::AcqRel);
        if (prev & (CLOSED_BIT | CALLER_WAITING_BIT)) == (CLOSED_BIT | CALLER_WAITING_BIT)
            && active_count(prev) == 1
        {
            let _guard = shared
                .done
                .mutex
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            shared.done.cv.notify_one();
        }
        allow_spin = true;
    }
}

pub(crate) struct WorkerPool<S> {
    shared: Arc<PoolShared<S>>,
    inline_state: S,
    n_workers: usize,
    next_epoch: u64,
    #[cfg(all(target_os = "linux", not(miri)))]
    contention: ContentionMonitor,
    handles: Vec<JoinHandle<()>>,
}

impl<S: Send + 'static> WorkerPool<S> {
    fn new(
        total_parallelism: NonZeroUsize,
        mut make_state: impl FnMut() -> Result<S>,
    ) -> Result<Self> {
        let n_workers = total_parallelism.get().saturating_sub(1);
        if n_workers > MAX_WORKERS {
            bail!(
                "WorkerPool supports at most {} workers, got {n_workers}",
                MAX_WORKERS
            );
        }
        let inline_state = make_state().location(loc!())?;
        // Build every worker state before spawning so a `make_state` error
        // leaks no parked threads.
        let mut worker_states = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            worker_states.push(make_state().location(loc!())?);
        }

        #[cfg(not(all(target_os = "linux", not(miri))))]
        let extra = {
            let extra_words = n_workers.div_ceil(64).saturating_sub(1);
            (0..extra_words)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };

        // Disable spinning up front when the pool alone oversubscribes the
        // machine's available CPU quota (e.g. cgroup/taskset core caps).
        let allow_spin = total_parallelism.get()
            <= thread::available_parallelism()
                .map(NonZeroUsize::get)
                .unwrap_or(1);
        let shared = Arc::new(PoolShared {
            publish: PublishLine {
                epoch: AtomicU64::new(0),
                #[cfg(all(target_os = "linux", not(miri)))]
                futex_word: AtomicU32::new(0),
                spin_enabled: AtomicBool::new(allow_spin),
                job: UnsafeCell::new(PoolJob {
                    epoch: 0,
                    func: noop_job::<S>,
                    ctx: ptr::null(),
                }),
            },
            job_state: JobStateLine {
                state: AtomicU64::new(CLOSED_BIT),
            },
            park: ParkLine {
                #[cfg(all(target_os = "linux", not(miri)))]
                parked_workers: AtomicU32::new(0),
                #[cfg(not(all(target_os = "linux", not(miri))))]
                word0: AtomicU64::new(0),
                #[cfg(not(all(target_os = "linux", not(miri))))]
                extra,
            },
            done: DoneLine {
                mutex: Mutex::new(()),
                cv: Condvar::new(),
                has_panic: AtomicBool::new(false),
                panic_payload: Mutex::new(None),
            },
        });

        let mut handles = Vec::with_capacity(n_workers);
        for (worker_idx, state) in worker_states.into_iter().enumerate() {
            let shared_clone = shared.clone();
            handles.push(thread::spawn(move || {
                worker_loop(worker_idx, state, shared_clone);
            }));
        }

        Ok(Self {
            shared,
            inline_state,
            n_workers,
            next_epoch: 1,
            #[cfg(all(target_os = "linux", not(miri)))]
            contention: ContentionMonitor::new(allow_spin),
            handles,
        })
    }

    /// Total number of threads participating in [`Self::execute`] (caller plus
    /// background workers).
    #[inline]
    pub(crate) fn parallelism(&self) -> usize {
        self.n_workers + 1
    }

    #[inline]
    fn wake_workers(&self, to_wake: usize) {
        #[cfg(all(target_os = "linux", not(miri)))]
        {
            let parked = self.shared.park.parked_workers.load(Ordering::SeqCst);
            if parked > 0 {
                let count = if to_wake >= self.n_workers || to_wake >= parked as usize {
                    i32::MAX
                } else {
                    to_wake as i32
                };
                futex_wake(&self.shared.publish.futex_word, count);
            }
        }

        #[cfg(not(all(target_os = "linux", not(miri))))]
        {
            let mut to_wake = to_wake;
            let total_words = 1 + self.shared.park.extra.len();
            if to_wake >= self.n_workers {
                for word_idx in 0..total_words {
                    let word = self.shared.parked_word(word_idx);
                    if word.load(Ordering::SeqCst) == 0 {
                        continue;
                    }
                    let mut bits = word.swap(0, Ordering::AcqRel);
                    let base = word_idx * 64;
                    while bits != 0 {
                        let tz = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        self.handles[base + tz].thread().unpark();
                    }
                }
                return;
            }

            for word_idx in 0..total_words {
                let word = self.shared.parked_word(word_idx);
                let mut cur = word.load(Ordering::SeqCst);
                while cur != 0 && to_wake > 0 {
                    let clear_mask = if (cur.count_ones() as usize) <= to_wake {
                        cur
                    } else {
                        let mut m = 0u64;
                        let mut rem = cur;
                        for _ in 0..to_wake {
                            let lsb = rem & rem.wrapping_neg();
                            m |= lsb;
                            rem ^= lsb;
                        }
                        m
                    };
                    let mut bits = word.fetch_and(!clear_mask, Ordering::AcqRel) & clear_mask;
                    let base = word_idx * 64;
                    while bits != 0 {
                        let tz = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        self.handles[base + tz].thread().unpark();
                        to_wake -= 1;
                    }
                    if to_wake == 0 {
                        return;
                    }
                    cur = word.load(Ordering::Acquire);
                }
            }
        }
    }

    /// Runs `f` concurrently across the caller thread and up to
    /// `(max_tasks - 1).min(self.n_workers)` background worker threads, passing
    /// each participant its exclusive `&mut S` state. Each worker enters `f` at
    /// most once per call, so `f` must self-schedule work items (e.g. via an
    /// `AtomicUsize` counter) until no work remains. Does nothing if
    /// `max_tasks == 0`. Blocks until all participants return. If the caller
    /// thread's `f` panics, that panic propagates after all workers finish;
    /// otherwise the first worker panic is re-raised.
    pub(crate) fn execute<F>(&mut self, max_tasks: usize, f: F)
    where
        F: Fn(&mut S) + Sync,
    {
        if max_tasks == 0 {
            return;
        }

        let n_active_target = (max_tasks - 1).min(self.n_workers);
        if n_active_target == 0 {
            f(&mut self.inline_state);
            return;
        }

        let epoch = self.next_epoch;
        self.next_epoch = if epoch == SHUTDOWN_EPOCH - 1 {
            1
        } else {
            epoch + 1
        };

        let job = PoolJob {
            epoch,
            func: call_closure::<F, S>,
            ctx: (&raw const f).cast::<()>(),
        };

        // SAFETY: `&mut self` guarantees exclusive caller access, and the
        // previous job's `ActiveJobGuard::drop` ensured `CLOSED_BIT` is set and
        // `active_count == 0`, so no worker can read `publish.job` right now.
        unsafe {
            *self.shared.publish.job.get() = job;
        }
        self.shared
            .job_state
            .state
            .store(pack_job_state(epoch, n_active_target), Ordering::Release);
        #[cfg(all(target_os = "linux", not(miri)))]
        self.shared
            .publish
            .futex_word
            .store(epoch_futex_word(epoch), Ordering::Relaxed);
        self.shared.publish.epoch.store(epoch, Ordering::SeqCst);
        self.wake_workers(n_active_target);

        {
            let mut guard = ActiveJobGuard {
                shared: &self.shared,
                caller_completed: false,
            };
            f(&mut self.inline_state);
            guard.caller_completed = true;
        }

        #[cfg(all(target_os = "linux", not(miri)))]
        self.contention.poll(&self.shared.publish.spin_enabled);

        if self.shared.done.has_panic.load(Ordering::Acquire) {
            self.shared.done.has_panic.store(false, Ordering::Relaxed);
            if let Some(payload) = self
                .shared
                .done
                .panic_payload
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
            {
                panic::resume_unwind(payload);
            }
        }
    }
}

impl<S> Drop for WorkerPool<S> {
    fn drop(&mut self) {
        #[cfg(all(target_os = "linux", not(miri)))]
        self.shared
            .publish
            .futex_word
            .store(SHUTDOWN_FUTEX_WORD, Ordering::Relaxed);
        self.shared
            .publish
            .epoch
            .store(SHUTDOWN_EPOCH, Ordering::SeqCst);
        #[cfg(all(target_os = "linux", not(miri)))]
        futex_wake(&self.shared.publish.futex_word, i32::MAX);
        #[cfg(not(all(target_os = "linux", not(miri))))]
        for handle in &self.handles {
            handle.thread().unpark();
        }
        for handle in self.handles.drain(..) {
            _ = handle.join();
        }
    }
}

fn tj_error(handle: turbojpeg::raw::tjhandle) -> anyhow::Error {
    // SAFETY: `tj3GetErrorStr` returns a valid NUL-terminated error string for
    // either a live handle or `NULL`.
    let msg = unsafe { CStr::from_ptr(turbojpeg::raw::tj3GetErrorStr(handle)) };
    anyhow!("turbojpeg: {}", msg.to_string_lossy())
}

struct RawJpegCompressor {
    handle: turbojpeg::raw::tjhandle,
}

// SAFETY: Each libjpeg-turbo handle owns independent compression state and is
// moved into a single worker thread slot (never shared concurrently).
unsafe impl Send for RawJpegCompressor {}

impl Drop for RawJpegCompressor {
    fn drop(&mut self) {
        // SAFETY: `self.handle` was created by `tj3Init` and is destroyed once.
        unsafe { turbojpeg::raw::tj3Destroy(self.handle) };
    }
}

impl RawJpegCompressor {
    fn new() -> Result<Self> {
        // SAFETY: `tj3Init` takes a valid `TJINIT` enum value.
        let handle =
            unsafe { turbojpeg::raw::tj3Init(turbojpeg::raw::TJINIT_TJINIT_COMPRESS as c_int) };
        if handle.is_null() {
            return Err(tj_error(ptr::null_mut()));
        }
        let mut this = Self { handle };
        this.set(turbojpeg::raw::TJPARAM_TJPARAM_NOREALLOC, 1)?;
        Ok(this)
    }

    fn set(&mut self, param: turbojpeg::raw::TJPARAM, value: c_int) -> Result<()> {
        // SAFETY: `self.handle` is a valid live compressor handle.
        let res = unsafe { turbojpeg::raw::tj3Set(self.handle, param as c_int, value) };
        if res != 0 {
            return Err(tj_error(self.handle));
        }
        Ok(())
    }
}

struct RawJpegDecompressor {
    handle: turbojpeg::raw::tjhandle,
}

// SAFETY: Each libjpeg-turbo handle owns independent decompression state and is
// moved into a single worker thread slot (never shared concurrently).
unsafe impl Send for RawJpegDecompressor {}

impl Drop for RawJpegDecompressor {
    fn drop(&mut self) {
        // SAFETY: `self.handle` was created by `tj3Init` and is destroyed once.
        unsafe { turbojpeg::raw::tj3Destroy(self.handle) };
    }
}

impl RawJpegDecompressor {
    fn new() -> Result<Self> {
        // SAFETY: `tj3Init` takes a valid `TJINIT` enum value.
        let handle =
            unsafe { turbojpeg::raw::tj3Init(turbojpeg::raw::TJINIT_TJINIT_DECOMPRESS as c_int) };
        if handle.is_null() {
            return Err(tj_error(ptr::null_mut()));
        }
        let this = Self { handle };
        // SAFETY: `self.handle` is a valid live decompressor handle.
        let res = unsafe {
            turbojpeg::raw::tj3Set(
                this.handle,
                turbojpeg::raw::TJPARAM_TJPARAM_SCANLIMIT as c_int,
                MAX_JPEG_PROGRESSIVE_SCANS as c_int,
            )
        };
        if res != 0 {
            return Err(tj_error(this.handle));
        }
        Ok(this)
    }
}

pub(crate) struct CompressorWorkerState {
    pub(crate) compressor: Compressor<'static>,
    pub(crate) zstd_scratch: Vec<u8>,
    pub(crate) filt_bgr: Vec<u8>,
    pub(crate) raw_row_a: Vec<u8>,
    pub(crate) raw_row_b: Vec<u8>,
    pub(crate) raw_alpha: Vec<u8>,
    pub(crate) filt_alpha: Vec<u8>,
    pub(crate) palette: PaletteScratch,
    jpeg_compressor: Option<RawJpegCompressor>,
    jpeg_quality: u8,
    jpeg_subsamp: JpegSubsamp,
    pub(crate) jpeg_pixels: Vec<u8>,
    jpeg_out: Vec<u8>,
    pub(crate) worker_idx: usize,
}

impl CompressorWorkerState {
    fn new(worker_idx: usize, compression_level: i32) -> Result<Self> {
        let mut compressor = Compressor::new(compression_level).location(loc!())?;
        if compression_level == 1 {
            // Raising minMatch from Zstd L1's default 6 to 7 avoids short hash
            // collisions on peaked image residuals, improving both ratio and speed.
            compressor
                .set_parameter(zstd_safe::CParameter::MinMatch(7))
                .location(loc!())?;
        }
        let max_bgr = TILE_PIXELS * 3;
        Ok(Self {
            compressor,
            zstd_scratch: Vec::with_capacity(zstd_safe::compress_bound(max_bgr)),
            filt_bgr: vec![0u8; max_bgr],
            raw_row_a: vec![0u8; TILE_W * 3],
            raw_row_b: vec![0u8; TILE_W * 3],
            raw_alpha: Vec::new(),
            filt_alpha: Vec::new(),
            palette: PaletteScratch::new(),
            jpeg_compressor: None,
            jpeg_quality: 0,
            jpeg_subsamp: JpegSubsamp::Sub444,
            jpeg_pixels: Vec::new(),
            jpeg_out: Vec::new(),
            worker_idx,
        })
    }

    /// Compresses a `width x height` BGRX rectangle starting at `src` with row
    /// stride `pitch` bytes directly via `tj3Compress8`, lazily initializing
    /// `RawJpegCompressor` and reusing `self.jpeg_out` across tiles.
    ///
    /// # Safety
    /// `src` must point to `height` readable, initialized rows of `width * 4`
    /// bytes separated by `pitch >= width * 4` bytes, and no other thread may
    /// concurrently mutate bytes within `[row * pitch .. row * pitch + width * 4)`.
    /// Sibling workers mutating disjoint columns in the same `shadow` rows are
    /// safe because libjpeg-turbo strictly reads `[0..width * 4]` per row and
    /// no spanning Rust `&[u8]` slice is created.
    pub(crate) unsafe fn compress_jpeg_strided_raw(
        &mut self,
        src: *const u8,
        width: usize,
        pitch: usize,
        height: usize,
        quality: u8,
        subsamp: JpegSubsamp,
    ) -> Result<Vec<u8>> {
        if width == 0 || height == 0 || pitch < width * 4 {
            bail!("invalid JPEG encode geometry {width}x{height} pitch={pitch}");
        }
        let width_i: c_int = width
            .try_into()
            .context(loc!(), "JPEG width exceeds c_int")?;
        let pitch_i: c_int = pitch
            .try_into()
            .context(loc!(), "JPEG pitch exceeds c_int")?;
        let height_i: c_int = height
            .try_into()
            .context(loc!(), "JPEG height exceeds c_int")?;

        let tj_subsamp: turbojpeg::Subsamp = subsamp.into();
        let needed = turbojpeg::compressed_buf_len(width, height, tj_subsamp).location(loc!())?;
        if self.jpeg_out.len() < needed {
            self.jpeg_out.resize(needed, 0);
        }
        let newly_created = self.jpeg_compressor.is_none();
        let comp = match &mut self.jpeg_compressor {
            Some(c) => c,
            None => self
                .jpeg_compressor
                .insert(RawJpegCompressor::new().location(loc!())?),
        };
        if newly_created || self.jpeg_quality != quality {
            comp.set(
                turbojpeg::raw::TJPARAM_TJPARAM_QUALITY,
                c_int::from(quality),
            )
            .location(loc!())?;
            self.jpeg_quality = quality;
        }
        if newly_created || self.jpeg_subsamp != subsamp {
            comp.set(turbojpeg::raw::TJPARAM_TJPARAM_SUBSAMP, tj_subsamp as c_int)
                .location(loc!())?;
            self.jpeg_subsamp = subsamp;
        }
        let mut out_ptr = self.jpeg_out.as_mut_ptr();
        let mut out_len: turbojpeg::raw::size_t = needed
            .try_into()
            .context(loc!(), "JPEG output buffer len exceeds size_t")?;
        // SAFETY: Caller guarantees `src` is valid for `height` rows of
        // `width * 4` bytes at stride `pitch`, `comp.handle` has
        // `TJPARAM_NOREALLOC = 1`, and `out_ptr` has `needed` writable bytes.
        let res = unsafe {
            turbojpeg::raw::tj3Compress8(
                comp.handle,
                src,
                width_i,
                pitch_i,
                height_i,
                turbojpeg::raw::TJPF_TJPF_BGRX,
                &mut out_ptr,
                &mut out_len,
            )
        };
        if res != 0 {
            return Err(tj_error(comp.handle)).location(loc!());
        }
        let written = out_len as usize;
        if out_ptr.is_null() || written > needed {
            bail!("turbojpeg returned invalid output size {written} (cap {needed})");
        }
        Ok(self.jpeg_out[..written].to_vec())
    }

    /// Compresses the tightly packed `width * height * 4`-byte BGRX tile stored
    /// in `self.jpeg_pixels` into a JPEG byte vector.
    pub(crate) fn compress_jpeg_from_scratch(
        &mut self,
        width: usize,
        height: usize,
        quality: u8,
        subsamp: JpegSubsamp,
    ) -> Result<Vec<u8>> {
        let byte_len = width * height * 4;
        assert!(self.jpeg_pixels.len() >= byte_len);
        let src = self.jpeg_pixels.as_ptr();
        // SAFETY: `self.jpeg_pixels` owns at least `width * height * 4`
        // initialized bytes and is not mutated during `compress_jpeg_strided_raw`.
        unsafe { self.compress_jpeg_strided_raw(src, width, width * 4, height, quality, subsamp) }
    }
}

/// Maximum payload length sent uncompressed without attempting zstd (`512 B`
/// compresses ~4.4x in ~4.4 us at level 1, ~730 Mbps breakeven).
const MIN_SIZE_TO_COMPRESS: usize = 512;

pub(crate) fn compress_bytes(
    compressor: &mut Compressor<'static>,
    zstd_scratch: &mut Vec<u8>,
    data: &[u8],
) -> EncodedBytes {
    if data.len() > MIN_SIZE_TO_COMPRESS {
        let bound = zstd_safe::compress_bound(data.len());
        zstd_scratch.clear();
        zstd_scratch.reserve(bound);
        let n = compressor
            .compress_to_buffer(data, zstd_scratch)
            .expect("dst reserved to compress_bound");
        if n < data.len() {
            return EncodedBytes::Zstd(zstd_scratch[..n].to_vec());
        }
    }
    EncodedBytes::Uncompressed(data.to_vec())
}

/// Progressive scan limit for [`RawJpegDecompressor`] (baseline JPEGs use 1
/// scan; capping prevents crafted multi-scan progressive DoS bombs).
const MAX_JPEG_PROGRESSIVE_SCANS: u32 = 100;

pub(crate) struct DecompressorWorkerState {
    pub(crate) worker_idx: usize,
    pub(crate) decompressor: Decompressor<'static>,
    pub(crate) filt_bgr: Vec<u8>,
    pub(crate) dec_row_a: Vec<u8>,
    pub(crate) dec_row_b: Vec<u8>,
    pub(crate) alpha_filt: Vec<u8>,
    pub(crate) alpha_row_a: Vec<u8>,
    pub(crate) alpha_row_b: Vec<u8>,
    jpeg_decompressor: Option<RawJpegDecompressor>,
    pub(crate) jpeg_pixels: Vec<u8>,
}

impl DecompressorWorkerState {
    fn new(worker_idx: usize) -> Result<Self> {
        Ok(Self {
            worker_idx,
            decompressor: Decompressor::new().location(loc!())?,
            filt_bgr: vec![0u8; TILE_PIXELS * 3],
            dec_row_a: vec![0u8; TILE_W * 3],
            dec_row_b: vec![0u8; TILE_W * 3],
            alpha_filt: Vec::new(),
            alpha_row_a: Vec::new(),
            alpha_row_b: Vec::new(),
            jpeg_decompressor: None,
            jpeg_pixels: Vec::new(),
        })
    }

    /// Validates the JPEG header matches `expected_w x expected_h` in a single
    /// `tj3DecompressHeader` call and decodes `data` directly into `dst` (with
    /// row stride `pitch >= expected_w * 4` bytes) as `TJPF_BGRA` (guaranteeing
    /// byte 3 is `0xFF` for every decoded pixel).
    ///
    /// # Safety
    /// `dst` must be writable (may be uninitialized) for `expected_h` rows of
    /// `expected_w * 4` bytes at stride `pitch`, and no other thread may
    /// concurrently read or write bytes within
    /// `[row * pitch .. row * pitch + expected_w * 4)`. Sibling workers writing
    /// disjoint columns on the same frame rows are safe because libjpeg-turbo
    /// strictly writes `[0..expected_w * 4]` per row and no spanning Rust
    /// `&mut [u8]` slice is created.
    pub(crate) unsafe fn decompress_jpeg_into_raw(
        &mut self,
        data: &[u8],
        expected_w: usize,
        expected_h: usize,
        dst: *mut u8,
        pitch: usize,
    ) -> Result<()> {
        if expected_w == 0 || expected_h == 0 || pitch < expected_w * 4 {
            bail!("invalid JPEG decode geometry {expected_w}x{expected_h} pitch={pitch}");
        }
        let pitch_i: c_int = pitch
            .try_into()
            .context(loc!(), "JPEG decode pitch exceeds c_int")?;
        let data_len: turbojpeg::raw::size_t = data
            .len()
            .try_into()
            .context(loc!(), "JPEG payload len exceeds size_t")?;

        let decomp = match &mut self.jpeg_decompressor {
            Some(d) => d,
            None => self
                .jpeg_decompressor
                .insert(RawJpegDecompressor::new().location(loc!())?),
        };
        // SAFETY: `decomp.handle` is a valid decompressor handle and `data` is
        // readable for `data_len` bytes.
        let hdr_res =
            unsafe { turbojpeg::raw::tj3DecompressHeader(decomp.handle, data.as_ptr(), data_len) };
        if hdr_res != 0 {
            return Err(tj_error(decomp.handle)).location(loc!());
        }
        // SAFETY: `tj3DecompressHeader` succeeded on `decomp.handle`.
        let (w, h) = unsafe {
            (
                turbojpeg::raw::tj3Get(
                    decomp.handle,
                    turbojpeg::raw::TJPARAM_TJPARAM_JPEGWIDTH as c_int,
                ),
                turbojpeg::raw::tj3Get(
                    decomp.handle,
                    turbojpeg::raw::TJPARAM_TJPARAM_JPEGHEIGHT as c_int,
                ),
            )
        };
        if w < 0 || h < 0 || w as usize != expected_w || h as usize != expected_h {
            bail!("JPEG dimensions {w}x{h} do not match tile rect {expected_w}x{expected_h}",);
        }
        // SAFETY: Header dimensions equal `expected_w x expected_h`, and caller
        // guarantees `dst` is writable for `expected_h` rows of `expected_w * 4`
        // bytes at stride `pitch`.
        let res = unsafe {
            turbojpeg::raw::tj3Decompress8(
                decomp.handle,
                data.as_ptr(),
                data_len,
                dst,
                pitch_i,
                turbojpeg::raw::TJPF_TJPF_BGRA,
            )
        };
        if res != 0 {
            return Err(tj_error(decomp.handle)).location(loc!());
        }
        Ok(())
    }

    /// Validates the JPEG header matches `expected_w x expected_h` and decodes
    /// `data` into `self.jpeg_pixels[..expected_w * expected_h * 4]` as BGRA
    /// (byte 3 guaranteed `0xFF`).
    pub(crate) fn decompress_jpeg_to_scratch(
        &mut self,
        data: &[u8],
        expected_w: usize,
        expected_h: usize,
    ) -> Result<()> {
        let byte_len = expected_w * expected_h * 4;
        if self.jpeg_pixels.len() < byte_len {
            self.jpeg_pixels.resize(byte_len, 0);
        }
        let dst = self.jpeg_pixels.as_mut_ptr();
        // SAFETY: `self.jpeg_pixels` owns at least `expected_w * expected_h * 4`
        // writable bytes and is not accessed by `decompress_jpeg_into_raw`.
        unsafe { self.decompress_jpeg_into_raw(data, expected_w, expected_h, dst, expected_w * 4) }
    }
}

pub struct ShardingCompressor {
    pub(crate) pool: WorkerPool<CompressorWorkerState>,
}

impl ShardingCompressor {
    pub fn new(n_compressors: NonZeroUsize, compression_level: i32) -> Result<Self> {
        let mut next_worker_idx = 0usize;
        let pool = WorkerPool::new(n_compressors, || {
            let worker_idx = next_worker_idx;
            next_worker_idx += 1;
            CompressorWorkerState::new(worker_idx, compression_level)
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
        let mut next_worker_idx = 0usize;
        let pool = WorkerPool::new(n_decompressors, || {
            let worker_idx = next_worker_idx;
            next_worker_idx += 1;
            DecompressorWorkerState::new(worker_idx)
        })
        .location(loc!())?;
        Ok(Self { pool })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compress_bytes_selects_codec() {
        let mut state = CompressorWorkerState::new(0, 1).unwrap();
        let mut noise = vec![0u8; TILE_PIXELS];
        let mut prng = 0x9E37_79B9_7F4A_7C15u64;
        for chunk in noise.as_chunks_mut::<8>().0 {
            prng = prng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = prng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            chunk.copy_from_slice(&z.to_le_bytes());
        }
        let compressible: Vec<u8> = (0..TILE_PIXELS).map(|i| (i / 64) as u8).collect();

        assert_eq!(
            compress_bytes(
                &mut state.compressor,
                &mut state.zstd_scratch,
                &compressible[..MIN_SIZE_TO_COMPRESS]
            ),
            EncodedBytes::Uncompressed(compressible[..MIN_SIZE_TO_COMPRESS].to_vec())
        );
        let above_threshold = &compressible[..MIN_SIZE_TO_COMPRESS + 1];
        let EncodedBytes::Zstd(compressed_above) = compress_bytes(
            &mut state.compressor,
            &mut state.zstd_scratch,
            above_threshold,
        ) else {
            panic!("expected Zstd above MIN_SIZE_TO_COMPRESS");
        };
        assert_eq!(
            zstd::bulk::decompress(&compressed_above, above_threshold.len()).unwrap(),
            above_threshold
        );

        assert_eq!(
            compress_bytes(&mut state.compressor, &mut state.zstd_scratch, &noise),
            EncodedBytes::Uncompressed(noise.clone())
        );
        let EncodedBytes::Zstd(compressed) = compress_bytes(
            &mut state.compressor,
            &mut state.zstd_scratch,
            &compressible,
        ) else {
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

        let calls = AtomicUsize::new(0);
        pool.execute(0, |_| {
            calls.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        let caller_tid = thread::current().id();
        pool.execute(1, |id| {
            assert_eq!(*id, 0);
            assert_eq!(thread::current().id(), caller_tid);
            calls.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);

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
    fn test_worker_pool_single_thread_fast_path() {
        let mut pool = WorkerPool::new(NonZeroUsize::new(1).unwrap(), || Ok(42usize)).unwrap();
        assert_eq!(pool.parallelism(), 1);
        let caller_tid = thread::current().id();
        let sum = AtomicUsize::new(0);
        for max_tasks in [0, 1, 2, 16] {
            pool.execute(max_tasks, |state| {
                assert_eq!(thread::current().id(), caller_tid);
                assert_eq!(*state, 42);
                sum.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(sum.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_worker_pool_rapid_barriers() {
        #[cfg(not(miri))]
        const THREADS: usize = 8;
        #[cfg(miri)]
        const THREADS: usize = 4;

        #[cfg(not(miri))]
        const ITERS: usize = 10_000;
        #[cfg(miri)]
        const ITERS: usize = 64;

        let mut next_id = 0usize;
        let mut pool = WorkerPool::new(NonZeroUsize::new(THREADS).unwrap(), || {
            let id = next_id;
            next_id += 1;
            Ok(id)
        })
        .unwrap();

        let task_schedule = [0usize, 1, 2, 3, 4, 8, 16, 1, 2, 12];
        for iter in 0..ITERS {
            let max_tasks = task_schedule[iter % task_schedule.len()];
            let n_items = if max_tasks == 0 { 0 } else { max_tasks * 3 };
            let next_item = AtomicUsize::new(0);
            let done_items = AtomicUsize::new(0);
            pool.execute(max_tasks, |_id| {
                loop {
                    let idx = next_item.fetch_add(1, Ordering::Relaxed);
                    if idx >= n_items {
                        break;
                    }
                    done_items.fetch_add(1, Ordering::Relaxed);
                }
            });
            assert_eq!(done_items.load(Ordering::SeqCst), n_items);
        }
    }

    #[test]
    fn test_worker_pool_idle_park_wake_cycles() {
        #[cfg(not(miri))]
        const CYCLES: usize = 20;
        #[cfg(miri)]
        const CYCLES: usize = 3;

        #[cfg(not(miri))]
        const THREADS: usize = 8;
        #[cfg(miri)]
        const THREADS: usize = 4;

        let mut next_id = 0usize;
        let mut pool = WorkerPool::new(NonZeroUsize::new(THREADS).unwrap(), || {
            let id = next_id;
            next_id += 1;
            Ok(id)
        })
        .unwrap();

        for cycle in 0..CYCLES {
            // Allow all workers to exhaust their spin budget and park.
            thread::sleep(Duration::from_millis(2));

            // Partial wake from full idle park.
            let barrier2 = Barrier::new(2);
            let count2 = AtomicUsize::new(0);
            pool.execute(2, |_| {
                count2.fetch_add(1, Ordering::SeqCst);
                barrier2.wait();
            });
            assert_eq!(count2.load(Ordering::SeqCst), 2);

            // Immediate follow-up full wake (mix of spinning and parked workers).
            let barrier_all = Barrier::new(THREADS);
            let count_all = AtomicUsize::new(0);
            pool.execute(THREADS + 4, |_| {
                count_all.fetch_add(1, Ordering::SeqCst);
                barrier_all.wait();
            });
            assert_eq!(count_all.load(Ordering::SeqCst), THREADS);

            // Rapid back-to-back burst while workers are hot.
            for step in 0..8 {
                let items = (cycle + step + 1) * 4;
                let cursor = AtomicUsize::new(0);
                let done = AtomicUsize::new(0);
                pool.execute(items, |_| {
                    while cursor.fetch_add(1, Ordering::Relaxed) < items {
                        done.fetch_add(1, Ordering::Relaxed);
                    }
                });
                assert_eq!(done.load(Ordering::SeqCst), items);
            }
        }
    }

    #[test]
    fn test_worker_pool_drop_while_spinning_and_parked() {
        // Drop immediately without ever executing a job.
        let pool_unused = WorkerPool::new(NonZeroUsize::new(4).unwrap(), || Ok(())).unwrap();
        drop(pool_unused);

        // Drop immediately after a job while workers are still spinning.
        let mut pool_hot = WorkerPool::new(NonZeroUsize::new(4).unwrap(), || Ok(())).unwrap();
        let barrier = Barrier::new(4);
        pool_hot.execute(4, |_| {
            barrier.wait();
        });
        drop(pool_hot);

        // Drop after sleeping so all workers are parked.
        let mut pool_parked = WorkerPool::new(NonZeroUsize::new(4).unwrap(), || Ok(())).unwrap();
        let barrier = Barrier::new(4);
        pool_parked.execute(4, |_| {
            barrier.wait();
        });
        thread::sleep(Duration::from_millis(5));
        drop(pool_parked);
    }

    #[test]
    fn test_worker_pool_caller_panic_waits_for_active_workers() {
        let mut next_id = 0usize;
        let mut pool = WorkerPool::new(NonZeroUsize::new(3).unwrap(), || {
            let id = next_id;
            next_id += 1;
            Ok(id)
        })
        .unwrap();

        let worker_finished = AtomicUsize::new(0);
        let res = panic::catch_unwind(AssertUnwindSafe(|| {
            let barrier = Barrier::new(2);
            // Stack-local state referenced by `f`; if `ActiveJobGuard` unwound
            // before the worker finished, writing to `local_flag` after return
            // would corrupt the stack (and fail immediately under Miri).
            let local_flag = AtomicUsize::new(0);
            pool.execute(2, |worker_id| {
                barrier.wait();
                if *worker_id == 0 {
                    panic!("caller boom");
                }
                thread::sleep(Duration::from_millis(5));
                local_flag.store(1, Ordering::SeqCst);
                worker_finished.store(local_flag.load(Ordering::SeqCst), Ordering::SeqCst);
            });
        }));
        assert!(res.is_err());
        assert_eq!(worker_finished.load(Ordering::SeqCst), 1);

        // Pool must remain fully usable afterwards.
        let barrier = Barrier::new(3);
        let ran = AtomicUsize::new(0);
        pool.execute(3, |_| {
            ran.fetch_add(1, Ordering::SeqCst);
            barrier.wait();
        });
        assert_eq!(ran.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_worker_pool_panic_and_partial_init_error() {
        struct Counted(Arc<AtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

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

        let res = panic::catch_unwind(AssertUnwindSafe(|| {
            let barrier = Barrier::new(2);
            pool.execute(2, |worker_id| {
                barrier.wait();
                if *worker_id != 0 {
                    panic!("worker boom");
                }
            });
        }));
        let payload = res.expect_err("expected worker panic to propagate");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"worker boom"));

        let res_dual = panic::catch_unwind(AssertUnwindSafe(|| {
            let barrier = Barrier::new(2);
            pool.execute(2, |worker_id| {
                barrier.wait();
                if *worker_id == 0 {
                    panic!("caller dual boom");
                }
                panic!("worker dual boom");
            });
        }));
        let dual_payload = res_dual.expect_err("expected caller panic to propagate");
        assert_eq!(
            dual_payload.downcast_ref::<&str>(),
            Some(&"caller dual boom")
        );

        // Stale worker panic from the dual-panic call must not bleed into the
        // next job.
        let barrier = Barrier::new(3);
        let ok_calls = AtomicUsize::new(0);
        pool.execute(3, |_| {
            ok_calls.fetch_add(1, Ordering::SeqCst);
            barrier.wait();
        });
        assert_eq!(ok_calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_worker_pool_parks_when_idle() {
        const THREADS: usize = 4;
        let mut pool = WorkerPool::new(NonZeroUsize::new(THREADS).unwrap(), || Ok(())).unwrap();
        let barrier = Barrier::new(THREADS);
        pool.execute(THREADS, |_| {
            barrier.wait();
        });

        // After the brief spin window expires, every background worker must
        // park cleanly (0% idle CPU).
        let expected_parked = THREADS - 1;
        let mut all_parked = false;
        for _ in 0..200 {
            #[cfg(all(target_os = "linux", not(miri)))]
            let parked = pool.shared.park.parked_workers.load(Ordering::SeqCst) as usize;
            #[cfg(not(all(target_os = "linux", not(miri))))]
            let parked = pool.shared.park.word0.load(Ordering::SeqCst).count_ones() as usize;

            if parked == expected_parked {
                all_parked = true;
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert!(
            all_parked,
            "expected all {expected_parked} workers to park when idle"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_worker_pool_many_workers() {
        // Exercises > 64 workers (exceeds inline 64-bit mask on non-Linux
        // fallback and stress-tests large futex wakeups on Linux).
        const THREADS: usize = 70;
        let mut pool = WorkerPool::new(NonZeroUsize::new(THREADS).unwrap(), || Ok(())).unwrap();
        thread::sleep(Duration::from_millis(3));

        let barrier2 = Barrier::new(2);
        pool.execute(2, |_| {
            barrier2.wait();
        });

        let barrier_all = Barrier::new(THREADS);
        let ran = AtomicUsize::new(0);
        pool.execute(THREADS, |_| {
            ran.fetch_add(1, Ordering::SeqCst);
            barrier_all.wait();
        });
        assert_eq!(ran.load(Ordering::SeqCst), THREADS);
    }
}
