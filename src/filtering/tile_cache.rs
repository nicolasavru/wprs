// Copyright 2026 Google LLC
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

//! Server-driven connection-wide inter-frame tile cache (`wprsd` metadata
//! directory + `wprsc` bounded decoded pixel array).
//!
//! - `wprsd` stores **only** metadata (keyed 128-bit SIMD hashes, slot
//!   dimensions, alpha/quality keys, and an intrusive O(1) LRU list) and never
//!   duplicates decoded tile pixels in RAM.
//! - `wprsc` holds at most [`MAX_TILE_CACHE_SLOTS`] (`1024`) slots and strictly
//!   enforces [`MAX_TILE_CACHE_BYTES`] (`64 MiB`), enough for two full 4K
//!   screens (`270` `256x256` tiles = `63.3 MiB`) or `1024` `128x128` tiles.
//! - Splittable (`> 128`px) 2D tiles hash their `<= 128x128` quadrants first
//!   and combine the quadrant hashes into the parent tile hash, indexing both
//!   the full tile and its quadrants so later partial updates (popup close,
//!   scroll-back, or static-header tab switches) hit immediately on the first
//!   return.
//! - Two-pass allocation pins all Pass-1 `CacheHit` slots and Pass-2 `CachePut`
//!   slots for the current frame so Step-1 parallel decodes (`CachePut`) and
//!   Step-2 parallel blits (`CacheHit`) on `wprsc` are lock-free and race-free.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use crate::filtering::BaseScanEntry;
use crate::filtering::FrameRows;
use crate::filtering::SlotDirty;
use crate::filtering::merge_cropped_quad_slots;
use crate::filtering::quad_grid_slot;
use crate::filtering::tile_hash::TileHash;
use crate::filtering::tile_hash::TileHashKey;
use crate::filtering::tile_hash::combine_quad_hashes;
use crate::filtering::tile_quadrants;
use crate::prelude::*;
use crate::serialization::framing::CacheHitRef;
use crate::serialization::framing::CacheHitRegion;
use crate::serialization::framing::CompressionMode;
use crate::serialization::framing::JpegSubsamp;
use crate::serialization::framing::LeafPayload;
use crate::serialization::framing::MAX_TILE_CACHE_BYTES;
use crate::serialization::framing::MAX_TILE_CACHE_SLOTS;
use crate::serialization::framing::SUBTILE_H;
use crate::serialization::framing::SUBTILE_W;
use crate::serialization::framing::TilePayload;
use crate::serialization::framing::TileRect;
use crate::vec4u8::Vec4u8;

/// Promotion bitmask flags applied after Phase B in [`CompressionMode::Hybrid`]
/// when a tile or sub-quadrant encoded losslessly (`Solid`, `Palette`, or
/// `AosZstd`) rather than via TurboJPEG.
pub(crate) const PROMOTE_PRIMARY: u8 = 1 << 0;
pub(crate) const PROMOTE_TOP_LEFT: u8 = 1 << 1;
pub(crate) const PROMOTE_TOP_RIGHT: u8 = 1 << 2;
pub(crate) const PROMOTE_BOTTOM_LEFT: u8 = 1 << 3;
pub(crate) const PROMOTE_BOTTOM_RIGHT: u8 = 1 << 4;
pub(crate) const PROMOTE_ALL: u8 = PROMOTE_PRIMARY
    | PROMOTE_TOP_LEFT
    | PROMOTE_TOP_RIGHT
    | PROMOTE_BOTTOM_LEFT
    | PROMOTE_BOTTOM_RIGHT;

const NONE_SLOT: u16 = u16::MAX;

#[inline]
pub(crate) fn region_promote_bit(region: CacheHitRegion) -> u8 {
    match region {
        CacheHitRegion::Full => PROMOTE_PRIMARY,
        CacheHitRegion::TopLeft => PROMOTE_TOP_LEFT,
        CacheHitRegion::TopRight => PROMOTE_TOP_RIGHT,
        CacheHitRegion::BottomLeft => PROMOTE_BOTTOM_LEFT,
        CacheHitRegion::BottomRight => PROMOTE_BOTTOM_RIGHT,
    }
}

/// Computes the exact post-Phase-B lossless promotion bitmask for `payload`.
#[inline]
pub(crate) fn payload_lossless_promotion_mask(rect: TileRect, payload: &TilePayload) -> u8 {
    match payload {
        TilePayload::Leaf(LeafPayload::Jpeg { .. }) | TilePayload::CacheHit(_) => 0,
        TilePayload::Leaf(
            LeafPayload::Solid(_) | LeafPayload::Palette { .. } | LeafPayload::AosZstd { .. },
        ) => PROMOTE_ALL,
        TilePayload::Split(subs) => {
            let mut mask = 0u8;
            for sub in subs {
                if !matches!(sub.payload, LeafPayload::Jpeg { .. }) {
                    mask |= region_promote_bit(quadrant_region(rect, sub.rect));
                }
            }
            mask
        },
    }
}

/// Maps a quadrant `quad` inside its parent `parent` to its [`CacheHitRegion`].
#[inline]
pub(crate) fn quadrant_region(parent: TileRect, quad: TileRect) -> CacheHitRegion {
    match (quad.x - parent.x, quad.y - parent.y) {
        (0, 0) => CacheHitRegion::TopLeft,
        (dx, 0) if dx == SUBTILE_W as u32 => CacheHitRegion::TopRight,
        (0, dy) if dy == SUBTILE_H as u32 => CacheHitRegion::BottomLeft,
        (dx, dy) if dx == SUBTILE_W as u32 && dy == SUBTILE_H as u32 => CacheHitRegion::BottomRight,
        _ => unreachable!("quadrant {quad:?} not aligned to 128px grid in parent {parent:?}"),
    }
}

const FX_MIX_K: u64 = 0x517C_C1B7_2722_0A95;

/// Fast per-instance randomly-seeded hasher for cache directories keyed on
/// keyed 128-bit [`TileHash`] digests.
#[derive(Debug, Copy, Clone)]
struct TileHashBuildHasher {
    seed: u64,
}

impl TileHashBuildHasher {
    fn new() -> Self {
        Self {
            seed: RandomState::new().hash_one(0u64),
        }
    }
}

struct TileHashHasher {
    state: u64,
}

impl TileHashHasher {
    #[inline(always)]
    fn mix(&mut self, word: u64) {
        self.state = (self.state.rotate_left(5) ^ word).wrapping_mul(FX_MIX_K);
    }
}

impl Hasher for TileHashHasher {
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rem) = bytes.as_chunks::<8>();
        for &c in chunks {
            self.mix(u64::from_ne_bytes(c));
        }
        if !rem.is_empty() {
            let mut tail = [0u8; 8];
            tail[..rem.len()].copy_from_slice(rem);
            self.mix(u64::from_ne_bytes(tail));
        }
    }

    #[inline(always)]
    fn write_u8(&mut self, i: u8) {
        self.mix(u64::from(i));
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.mix(u64::from(i));
    }

    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }

    #[inline(always)]
    fn write_usize(&mut self, i: usize) {
        self.mix(i as u64);
    }

    #[inline(always)]
    fn finish(&self) -> u64 {
        self.state
    }
}

impl BuildHasher for TileHashBuildHasher {
    type Hasher = TileHashHasher;

    #[inline(always)]
    fn build_hasher(&self) -> Self::Hasher {
        TileHashHasher { state: self.seed }
    }
}

/// Disjoint per-slot hash output table populated in parallel during Phase A,
/// carrying the immutable per-connection [`TileHashKey`].
pub(crate) struct SlotHashes {
    slots: Vec<UnsafeCell<TileHash>>,
    key: TileHashKey,
}

// SAFETY: Each `slot_idx` in `0..slots.len()` is owned by a single Phase-A
// task (`Strip1D`, `RowBand2D` column, or full-frame tile) and written at most
// once before `pool.execute` joins; `key` is immutable.
unsafe impl Sync for SlotHashes {}

impl SlotHashes {
    pub(crate) fn new(len: usize, key: TileHashKey) -> Self {
        Self {
            slots: (0..len).map(|_| UnsafeCell::new([0u8; 16])).collect(),
            key,
        }
    }

    #[inline]
    pub(crate) fn key(&self) -> &TileHashKey {
        &self.key
    }

    /// # Safety
    /// Caller must guarantee exclusive ownership of `slot_idx < self.slots.len()`.
    #[inline]
    pub(crate) unsafe fn write(&self, slot_idx: usize, hash: TileHash) {
        // SAFETY: Caller upholds exclusive ownership of `slot_idx`.
        unsafe { *self.slots[slot_idx].get() = hash };
    }

    pub(crate) fn into_vec(self) -> Vec<TileHash> {
        self.slots.into_iter().map(UnsafeCell::into_inner).collect()
    }
}

/// Quality class of a cached tile or sub-quadrant.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) enum TileQuality {
    /// Bit-exact decoded pixels (`Solid`, `Palette`, or `AosZstd`).
    Lossless,
    /// Lossy TurboJPEG decoded pixels at `(quality, subsamp)`.
    Lossy { quality: u8, subsamp: JpegSubsamp },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct CacheKey {
    hash: TileHash,
    w: u32,
    h: u32,
    has_alpha: bool,
    quality: TileQuality,
}

impl CacheKey {
    const EMPTY: Self = Self {
        hash: [0u8; 16],
        w: 0,
        h: 0,
        has_alpha: false,
        quality: TileQuality::Lossless,
    };
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct SubEntry {
    key: CacheKey,
    region: CacheHitRegion,
}

#[derive(Debug, Copy, Clone)]
struct ServerSlot {
    occupied: bool,
    bytes: usize,
    primary_key: CacheKey,
    sub_keys: [Option<SubEntry>; 4],
    n_sub_keys: usize,
    lru_prev: u16,
    lru_next: u16,
    pinned_gen: u64,
}

impl ServerSlot {
    const EMPTY: Self = Self {
        occupied: false,
        bytes: 0,
        primary_key: CacheKey::EMPTY,
        sub_keys: [None; 4],
        n_sub_keys: 0,
        lru_prev: NONE_SLOT,
        lru_next: NONE_SLOT,
        pinned_gen: 0,
    };
}

/// Action chosen by [`ServerTileCache::plan_frame`] for each transmitted tile.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum TileCacheAction {
    /// Tile pixels match an already-cached (or intra-frame `Put`) slot.
    Hit(CacheHitRef),
    /// Tile must be encoded and stored by `wprsc` into `slot_id`.
    Put(u16),
    /// Tile must be encoded without populating any cache slot (e.g. >1024
    /// unique tiles or >64 MiB pinned in a single frame).
    Uncached,
}

/// Output of [`ServerTileCache::plan_frame`].
pub(crate) struct FrameCachePlan {
    pub(crate) tile_rects: Vec<TileRect>,
    pub(crate) actions: Vec<TileCacheAction>,
    pub(crate) reset_tile_cache: bool,
    pub(crate) evict_slots: Vec<u16>,
}

enum DirtyCandidate {
    Single {
        rect: TileRect,
        hash: TileHash,
    },
    Splittable {
        rect: TileRect,
        parent_hash: TileHash,
        quads: [TileRect; 4],
        quad_hashes: [TileHash; 4],
        n_quads: usize,
    },
    PartialTile2D {
        rect: TileRect,
        parent_hash: Option<TileHash>,
        quads: [TileRect; 4],
        states: [SlotDirty; 4],
        quad_hashes: [TileHash; 4],
        n_quads: usize,
    },
}

enum Pass1Item {
    Hit {
        rect: TileRect,
        hit_ref: CacheHitRef,
    },
    Miss {
        rect: TileRect,
        hash: TileHash,
        sub_quads: Option<([TileRect; 4], [TileHash; 4], usize)>,
    },
    Uncached {
        rect: TileRect,
    },
}

/// Metadata for a slot allocated during a single-pass cold-cache full-frame
/// encode, committed to [`ServerTileCache`] after the worker pool joins.
pub(crate) struct ColdFillSlot {
    slot_id: u16,
    rect: TileRect,
    hash: TileHash,
    sub_quads: Option<([TileRect; 4], [TileHash; 4], usize)>,
}

/// Shared intra-frame dedup + slot allocator used by parallel workers on
/// single-pass full-frame encodes when [`ServerTileCache`] starts empty.
pub(crate) struct ColdFillState {
    intra_frame: HashMap<(TileHash, u32, u32), CacheHitRef, TileHashBuildHasher>,
    pending_slots: Vec<ColdFillSlot>,
    next_slot: u16,
    used_bytes: usize,
}

impl ColdFillState {
    /// Looks up `(hash, rect.w, rect.h)` in this frame's already-claimed slots
    /// (returning [`TileCacheAction::Hit`] on intra-frame duplicate) or claims
    /// the next free slot within [`MAX_TILE_CACHE_SLOTS`] and
    /// [`MAX_TILE_CACHE_BYTES`].
    pub(crate) fn claim_tile(
        &mut self,
        rect: TileRect,
        hash: TileHash,
        sub_quads: Option<([TileRect; 4], [TileHash; 4], usize)>,
    ) -> TileCacheAction {
        if let Some(&hit_ref) = self.intra_frame.get(&(hash, rect.w, rect.h)) {
            return TileCacheAction::Hit(hit_ref);
        }
        let need_bytes = rect.w as usize * rect.h as usize * 4;
        if (self.next_slot as usize) >= MAX_TILE_CACHE_SLOTS
            || self.used_bytes.saturating_add(need_bytes) > MAX_TILE_CACHE_BYTES
        {
            return TileCacheAction::Uncached;
        }
        let slot_id = self.next_slot;
        self.next_slot += 1;
        self.used_bytes += need_bytes;
        let full_ref = CacheHitRef {
            slot_id,
            region: CacheHitRegion::Full,
        };
        self.intra_frame.insert((hash, rect.w, rect.h), full_ref);
        if let Some((quads, quad_hashes, n_quads)) = sub_quads {
            for q in 0..n_quads {
                let qrect = quads[q];
                let region = quadrant_region(rect, qrect);
                self.intra_frame
                    .entry((quad_hashes[q], qrect.w, qrect.h))
                    .or_insert(CacheHitRef { slot_id, region });
            }
        }
        self.pending_slots.push(ColdFillSlot {
            slot_id,
            rect,
            hash,
            sub_quads,
        });
        TileCacheAction::Put(slot_id)
    }

    pub(crate) fn into_pending(self) -> Vec<ColdFillSlot> {
        self.pending_slots
    }
}

type ColdSubQuads = ([TileRect; 4], [TileHash; 4], usize);

#[derive(Copy, Clone)]
struct ColdTileHash {
    hash: TileHash,
    sub_quads: Option<ColdSubQuads>,
}

/// Lock-free / single-mutex deterministic frontier coordinating single-pass
/// cold-cache full-frame encoding across parallel workers.
///
/// Workers hash tiles in parallel (`publish_hash`) and advance a strict
/// raster-order planning frontier (`advance`) so every `slot_id`, intra-frame
/// dedup winner, and LRU order is bit-for-bit deterministic and identical to
/// two-pass [`ServerTileCache::plan_frame`], while encoding each planned tile
/// while its pixels remain hot in L2/L3 cache.
pub(crate) struct ColdFillFrontier {
    n_tiles: usize,
    hashed_ready: Vec<AtomicBool>,
    hash_results: Vec<UnsafeCell<MaybeUninit<ColdTileHash>>>,
    actions: Vec<UnsafeCell<MaybeUninit<TileCacheAction>>>,
    planned_up_to: AtomicUsize,
    next_encode: AtomicUsize,
    encode_claimed: Vec<AtomicBool>,
}

// SAFETY: `hash_results[i]` is written once by the exclusive owner of tile `i`
// before `hashed_ready[i].store(true, Release)` and read once under `cold_state`
// after `hashed_ready[i].load(Acquire)`. `actions[i]` is written once under
// `cold_state` before `planned_up_to.store(i + 1, Release)` and read once by
// the unique thread winning `encode_claimed[i].swap(true, AcqRel)` after
// observing `i < planned_up_to.load(Acquire)`.
unsafe impl Sync for ColdFillFrontier {}

impl ColdFillFrontier {
    pub(crate) fn new(n_tiles: usize) -> Self {
        Self {
            n_tiles,
            hashed_ready: (0..n_tiles).map(|_| AtomicBool::new(false)).collect(),
            hash_results: (0..n_tiles)
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect(),
            actions: (0..n_tiles)
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect(),
            planned_up_to: AtomicUsize::new(0),
            next_encode: AtomicUsize::new(0),
            encode_claimed: (0..n_tiles).map(|_| AtomicBool::new(false)).collect(),
        }
    }

    /// Publishes the completed hash (and optional sub-quadrant hashes) for
    /// `tile_idx`.
    ///
    /// # Safety
    /// Caller must be the exclusive writer for `tile_idx < self.n_tiles` and
    /// must call this at most once per `tile_idx`.
    #[inline]
    pub(crate) unsafe fn publish_hash(
        &self,
        tile_idx: usize,
        hash: TileHash,
        sub_quads: Option<ColdSubQuads>,
    ) {
        // SAFETY: Caller has exclusive write access to `hash_results[tile_idx]`
        // before setting `hashed_ready[tile_idx]` with `Release`.
        unsafe {
            (*self.hash_results[tile_idx].get()).write(ColdTileHash { hash, sub_quads });
        }
        self.hashed_ready[tile_idx].store(true, Ordering::Release);
    }

    /// Advances the deterministic raster-order planning frontier across all
    /// contiguous hashed tiles starting from `planned_up_to`.
    pub(crate) fn advance(&self, state: &Mutex<ColdFillState>, base_entries: &[BaseScanEntry]) {
        let mut guard = state.lock().unwrap();
        let mut p = self.planned_up_to.load(Ordering::Relaxed);
        while p < self.n_tiles && self.hashed_ready[p].load(Ordering::Acquire) {
            // SAFETY: `hashed_ready[p]` observed `true` with `Acquire`, so
            // `hash_results[p]` is initialized and immutable, and `guard`
            // grants exclusive write access to `actions[p]`.
            let h = unsafe { (*self.hash_results[p].get()).assume_init() };
            let action = guard.claim_tile(base_entries[p].rect(), h.hash, h.sub_quads);
            // SAFETY: `p >= planned_up_to` and `guard` serializes writers.
            unsafe { (*self.actions[p].get()).write(action) };
            p += 1;
            self.planned_up_to.store(p, Ordering::Release);
        }
    }

    /// Attempts to claim `tile_idx` for immediate L2-hot encoding if the
    /// planning frontier has already passed `tile_idx`.
    #[inline]
    pub(crate) fn try_claim_encode(&self, tile_idx: usize) -> Option<TileCacheAction> {
        if tile_idx < self.planned_up_to.load(Ordering::Acquire)
            && !self.encode_claimed[tile_idx].swap(true, Ordering::AcqRel)
        {
            // SAFETY: `tile_idx < planned_up_to` loaded with `Acquire` ensures
            // `actions[tile_idx]` was initialized before release.
            let action = unsafe { (*self.actions[tile_idx].get()).assume_init() };
            Some(action)
        } else {
            None
        }
    }

    /// Claims the next planned but unencoded tile in raster order so in-flight
    /// shadow bytes remain bounded in L3 cache.
    #[inline]
    pub(crate) fn try_claim_next_planned(&self) -> Option<(usize, TileCacheAction)> {
        loop {
            let planned = self.planned_up_to.load(Ordering::Acquire);
            let idx = self.next_encode.load(Ordering::Relaxed);
            if idx >= planned {
                return None;
            }
            if self
                .next_encode
                .compare_exchange_weak(idx, idx + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
                && !self.encode_claimed[idx].swap(true, Ordering::AcqRel)
            {
                // SAFETY: `idx < planned` loaded with `Acquire` ensures
                // `actions[idx]` was initialized before release.
                let action = unsafe { (*self.actions[idx].get()).assume_init() };
                return Some((idx, action));
            }
        }
    }

    /// Returns `true` once every tile has been planned and claimed for encoding.
    #[inline]
    pub(crate) fn is_done(&self) -> bool {
        self.planned_up_to.load(Ordering::Acquire) >= self.n_tiles
            && self.next_encode.load(Ordering::Relaxed) >= self.n_tiles
    }
}

/// Metadata-only connection-wide tile cache directory owned by `wprsd`.
///
/// Tracks which content hashes live in which `wprsc` cache slots using an O(1)
/// intrusive LRU list bounded by [`MAX_TILE_CACHE_SLOTS`] (`1024`) and
/// [`MAX_TILE_CACHE_BYTES`] (`64 MiB`), without storing duplicate pixels on
/// `wprsd`.
#[derive(Debug)]
pub struct ServerTileCache {
    slots: Vec<ServerSlot>,
    free_slots: Vec<u16>,
    index: HashMap<CacheKey, CacheHitRef, TileHashBuildHasher>,
    hash_key: TileHashKey,
    lru_head: u16,
    lru_tail: u16,
    used_bytes: usize,
    occupied_count: usize,
    frame_gen: u64,
    reset_pending: bool,
}

impl Default for ServerTileCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerTileCache {
    pub fn new() -> Self {
        let mut free_slots = Vec::with_capacity(MAX_TILE_CACHE_SLOTS);
        for id in (0..MAX_TILE_CACHE_SLOTS as u16).rev() {
            free_slots.push(id);
        }
        Self {
            slots: vec![ServerSlot::EMPTY; MAX_TILE_CACHE_SLOTS],
            free_slots,
            index: HashMap::with_capacity_and_hasher(
                MAX_TILE_CACHE_SLOTS * 2,
                TileHashBuildHasher::new(),
            ),
            hash_key: TileHashKey::new_random(),
            lru_head: NONE_SLOT,
            lru_tail: NONE_SLOT,
            used_bytes: 0,
            occupied_count: 0,
            frame_gen: 0,
            reset_pending: false,
        }
    }

    #[inline]
    pub(crate) fn hash_key(&self) -> &TileHashKey {
        &self.hash_key
    }

    /// Clears all directory entries (call on disconnect or new client connect).
    pub fn clear(&mut self) {
        if self.occupied_count == 0 {
            return;
        }
        self.slots.fill(ServerSlot::EMPTY);
        self.free_slots.clear();
        for id in (0..MAX_TILE_CACHE_SLOTS as u16).rev() {
            self.free_slots.push(id);
        }
        self.index.clear();
        self.lru_head = NONE_SLOT;
        self.lru_tail = NONE_SLOT;
        self.used_bytes = 0;
        self.occupied_count = 0;
    }

    /// Marks the next emitted [`crate::serialization::framing::RawBufferHeader`]
    /// to carry `reset_tile_cache = true` so `wprsc` clears its cache in band.
    pub fn mark_reset_pending(&mut self) {
        self.reset_pending = true;
    }

    /// Clears the directory and marks an in-band cache reset on the next frame
    /// (used on connect, `BufferResyncRequested`, and `FaultedAfterSend`).
    pub fn reset_for_peer(&mut self) {
        self.clear();
        self.reset_pending = true;
    }

    #[inline]
    pub fn occupied_slots(&self) -> usize {
        self.occupied_count
    }

    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    #[inline]
    pub(crate) fn take_reset_pending(&mut self) -> bool {
        std::mem::take(&mut self.reset_pending)
    }

    pub(crate) fn new_cold_fill_state(&self, n_tiles: usize, total_slots: usize) -> ColdFillState {
        debug_assert_eq!(self.occupied_count, 0);
        debug_assert_eq!(self.used_bytes, 0);
        let key_cap = (n_tiles + total_slots).min(MAX_TILE_CACHE_SLOTS * 5);
        ColdFillState {
            intra_frame: HashMap::with_capacity_and_hasher(key_cap, *self.index.hasher()),
            pending_slots: Vec::with_capacity(n_tiles.min(MAX_TILE_CACHE_SLOTS)),
            next_slot: 0,
            used_bytes: 0,
        }
    }

    /// Commits all slots claimed during a single-pass cold-cache full-frame
    /// encode into an empty [`ServerTileCache`], applying any Hybrid lossless
    /// promotions inline.
    pub(crate) fn commit_cold_fill(
        &mut self,
        pending_slots: Vec<ColdFillSlot>,
        has_alpha: bool,
        mode: CompressionMode,
        promotions: &[AtomicU8],
    ) {
        debug_assert_eq!(self.occupied_count, 0);
        debug_assert_eq!(self.used_bytes, 0);
        if pending_slots.is_empty() {
            return;
        }
        self.frame_gen = self.frame_gen.wrapping_add(1).max(1);
        let initial_quality = match mode {
            CompressionMode::Lossless => TileQuality::Lossless,
            CompressionMode::Hybrid { quality, subsamp } => TileQuality::Lossy { quality, subsamp },
        };
        let n_used = pending_slots.len();
        for entry in pending_slots {
            let slot_id = entry.slot_id;
            let idx = slot_id as usize;
            let promo = promotions.get(idx).map_or(0, |p| p.load(Ordering::Relaxed));
            let primary_quality = if (promo & PROMOTE_PRIMARY) != 0 {
                TileQuality::Lossless
            } else {
                initial_quality
            };
            let primary_key = CacheKey {
                hash: entry.hash,
                w: entry.rect.w,
                h: entry.rect.h,
                has_alpha,
                quality: primary_quality,
            };
            let mut sub_keys = [None; 4];
            let mut n_sub_keys = 0usize;
            if let Some((quads, quad_hashes, n_quads)) = entry.sub_quads {
                for q in 0..n_quads {
                    let qrect = quads[q];
                    let region = quadrant_region(entry.rect, qrect);
                    let q_quality = if (promo & region_promote_bit(region)) != 0 {
                        TileQuality::Lossless
                    } else {
                        initial_quality
                    };
                    let qkey = CacheKey {
                        hash: quad_hashes[q],
                        w: qrect.w,
                        h: qrect.h,
                        has_alpha,
                        quality: q_quality,
                    };
                    sub_keys[q] = Some(SubEntry { key: qkey, region });
                    self.index.insert(qkey, CacheHitRef { slot_id, region });
                }
                n_sub_keys = n_quads;
            }
            let need_bytes = entry.rect.w as usize * entry.rect.h as usize * 4;
            self.slots[idx] = ServerSlot {
                occupied: true,
                bytes: need_bytes,
                primary_key,
                sub_keys,
                n_sub_keys,
                lru_prev: NONE_SLOT,
                lru_next: NONE_SLOT,
                pinned_gen: self.frame_gen,
            };
            self.used_bytes += need_bytes;
            self.occupied_count += 1;
            self.lru_push_front(slot_id);
            self.index.insert(
                primary_key,
                CacheHitRef {
                    slot_id,
                    region: CacheHitRegion::Full,
                },
            );
        }
        self.free_slots.clear();
        for id in (n_used as u16..MAX_TILE_CACHE_SLOTS as u16).rev() {
            self.free_slots.push(id);
        }
    }

    #[inline]
    fn lru_unlink(&mut self, slot_id: u16) {
        let idx = slot_id as usize;
        let prev = self.slots[idx].lru_prev;
        let next = self.slots[idx].lru_next;
        if prev != NONE_SLOT {
            self.slots[prev as usize].lru_next = next;
        } else {
            self.lru_head = next;
        }
        if next != NONE_SLOT {
            self.slots[next as usize].lru_prev = prev;
        } else {
            self.lru_tail = prev;
        }
        self.slots[idx].lru_prev = NONE_SLOT;
        self.slots[idx].lru_next = NONE_SLOT;
    }

    #[inline]
    fn lru_push_front(&mut self, slot_id: u16) {
        let idx = slot_id as usize;
        let old_head = self.lru_head;
        self.slots[idx].lru_prev = NONE_SLOT;
        self.slots[idx].lru_next = old_head;
        if old_head != NONE_SLOT {
            self.slots[old_head as usize].lru_prev = slot_id;
        } else {
            self.lru_tail = slot_id;
        }
        self.lru_head = slot_id;
    }

    #[inline]
    fn touch_and_pin(&mut self, slot_id: u16) {
        let idx = slot_id as usize;
        debug_assert!(self.slots[idx].occupied);
        self.slots[idx].pinned_gen = self.frame_gen;
        if self.lru_head != slot_id {
            self.lru_unlink(slot_id);
            self.lru_push_front(slot_id);
        }
    }

    #[inline]
    fn remove_index_if_matches(&mut self, key: CacheKey, expected: CacheHitRef) {
        if self.index.get(&key) == Some(&expected) {
            self.index.remove(&key);
        }
    }

    fn evict_slot(&mut self, slot_id: u16) {
        let idx = slot_id as usize;
        let slot = self.slots[idx];
        debug_assert!(slot.occupied);
        self.lru_unlink(slot_id);
        self.remove_index_if_matches(
            slot.primary_key,
            CacheHitRef {
                slot_id,
                region: CacheHitRegion::Full,
            },
        );
        for sub in slot.sub_keys[..slot.n_sub_keys].iter().flatten() {
            self.remove_index_if_matches(
                sub.key,
                CacheHitRef {
                    slot_id,
                    region: sub.region,
                },
            );
        }
        self.used_bytes -= slot.bytes;
        self.occupied_count -= 1;
        self.slots[idx] = ServerSlot::EMPTY;
        self.free_slots.push(slot_id);
    }

    #[inline]
    fn lookup_lossless(
        &self,
        hash: TileHash,
        w: u32,
        h: u32,
        has_alpha: bool,
    ) -> Option<CacheHitRef> {
        let lossless_key = CacheKey {
            hash,
            w,
            h,
            has_alpha,
            quality: TileQuality::Lossless,
        };
        self.index.get(&lossless_key).copied()
    }

    #[inline]
    fn lookup(
        &self,
        hash: TileHash,
        w: u32,
        h: u32,
        has_alpha: bool,
        mode: CompressionMode,
    ) -> Option<CacheHitRef> {
        if let Some(hit_ref) = self.lookup_lossless(hash, w, h, has_alpha) {
            return Some(hit_ref);
        }
        if let CompressionMode::Hybrid { quality, subsamp } = mode {
            let lossy_key = CacheKey {
                hash,
                w,
                h,
                has_alpha,
                quality: TileQuality::Lossy { quality, subsamp },
            };
            if let Some(&hit_ref) = self.index.get(&lossy_key) {
                return Some(hit_ref);
            }
        }
        None
    }

    /// Allocates an unpinned slot with enough room for `need_bytes` within
    /// [`MAX_TILE_CACHE_SLOTS`] and [`MAX_TILE_CACHE_BYTES`], recording any
    /// evicted slots in `evicted_to_free`. Returns `None` without evicting
    /// anything if pinned slots make allocation impossible.
    fn allocate_unpinned_slot(
        &mut self,
        need_bytes: usize,
        evicted_to_free: &mut Vec<u16>,
    ) -> Option<u16> {
        if need_bytes > MAX_TILE_CACHE_BYTES {
            return None;
        }
        let mut need_slot = self.free_slots.is_empty();
        let mut bytes_to_free = self
            .used_bytes
            .saturating_add(need_bytes)
            .saturating_sub(MAX_TILE_CACHE_BYTES);

        if need_slot || bytes_to_free > 0 {
            // Pre-check from `lru_tail` so we never evict partially if pinned
            // slots prevent satisfying `need_bytes`.
            let mut freeable = 0usize;
            let mut found_slot = !need_slot;
            let mut cur = self.lru_tail;
            while cur != NONE_SLOT && (!found_slot || freeable < bytes_to_free) {
                let slot = self.slots[cur as usize];
                if slot.pinned_gen != self.frame_gen {
                    freeable += slot.bytes;
                    found_slot = true;
                }
                cur = slot.lru_prev;
            }
            if !found_slot || freeable < bytes_to_free {
                return None;
            }

            let mut cur = self.lru_tail;
            while cur != NONE_SLOT && (need_slot || bytes_to_free > 0) {
                let prev = self.slots[cur as usize].lru_prev;
                if self.slots[cur as usize].pinned_gen != self.frame_gen {
                    let freed = self.slots[cur as usize].bytes;
                    self.evict_slot(cur);
                    evicted_to_free.push(cur);
                    need_slot = false;
                    bytes_to_free = bytes_to_free.saturating_sub(freed);
                }
                cur = prev;
            }
        }

        self.free_slots.pop()
    }

    /// Attempts to satisfy a `> 128`px tile from already-cached parent or
    /// all-lossless quadrant entries (preferring lossless hits in Hybrid mode).
    /// Returns `true` and appends `Pass1Item::Hit` entries when fully satisfied.
    fn try_hit_splittable_parent(
        &mut self,
        rect: TileRect,
        parent_hash: TileHash,
        quads: &[TileRect; 4],
        quad_hashes: &[TileHash; 4],
        n_quads: usize,
        has_alpha: bool,
        mode: CompressionMode,
        pass1: &mut Vec<Pass1Item>,
    ) -> bool {
        if let Some(hit_ref) = self.lookup_lossless(parent_hash, rect.w, rect.h, has_alpha) {
            self.touch_and_pin(hit_ref.slot_id);
            pass1.push(Pass1Item::Hit { rect, hit_ref });
            return true;
        }
        // In Hybrid mode, if every quadrant already has a lossless cached entry
        // (e.g. after a Lossless refresh that un-coalesced a mixed tile), prefer
        // those bit-exact quadrants over a stale lossy parent slot.
        let mut all_quads_lossless = [None; 4];
        let all_lossless = matches!(mode, CompressionMode::Hybrid { .. })
            && (0..n_quads).all(|q| {
                let qrect = quads[q];
                if let Some(qhit) =
                    self.lookup_lossless(quad_hashes[q], qrect.w, qrect.h, has_alpha)
                {
                    all_quads_lossless[q] = Some(qhit);
                    true
                } else {
                    false
                }
            });
        if all_lossless {
            for q in 0..n_quads {
                let hit_ref = all_quads_lossless[q].unwrap();
                self.touch_and_pin(hit_ref.slot_id);
                pass1.push(Pass1Item::Hit {
                    rect: quads[q],
                    hit_ref,
                });
            }
            return true;
        }
        if let Some(hit_ref) = self.lookup(parent_hash, rect.w, rect.h, has_alpha, mode) {
            self.touch_and_pin(hit_ref.slot_id);
            pass1.push(Pass1Item::Hit { rect, hit_ref });
            return true;
        }
        false
    }

    /// Runs two-pass lookup/pinning (Pass 1) and intra-frame dedup + unpinned
    /// LRU slot allocation (Pass 2) over all dirty entries in deterministic
    /// raster order.
    #[instrument(skip_all, level = "debug")]
    pub(crate) fn plan_frame(
        &mut self,
        base_entries: &[BaseScanEntry],
        dirty_slots: Option<&[AtomicU32]>,
        slot_hashes: &[TileHash],
        has_alpha: bool,
        mode: CompressionMode,
    ) -> FrameCachePlan {
        self.frame_gen = self.frame_gen.wrapping_add(1).max(1);

        let mut candidates = Vec::with_capacity(base_entries.len());
        for entry in base_entries {
            match *entry {
                BaseScanEntry::Strip1D { rect, slot_idx } => {
                    let is_dirty = dirty_slots.is_none_or(|slots| {
                        slots[slot_idx].load(Ordering::Relaxed) != SlotDirty::CLEAN_RAW
                    });
                    if is_dirty {
                        candidates.push(DirtyCandidate::Single {
                            rect,
                            hash: slot_hashes[slot_idx],
                        });
                    }
                },
                BaseScanEntry::Tile2D {
                    rect,
                    quad_slots,
                    n_quads,
                } => {
                    let mut states = [SlotDirty::Clean; 4];
                    let mut full_count = 0usize;
                    let mut dirty_count = 0usize;
                    for (q, &slot) in quad_slots[..n_quads].iter().enumerate() {
                        let st = match dirty_slots {
                            None => SlotDirty::Full,
                            Some(slots) => SlotDirty::decode(slots[slot].load(Ordering::Relaxed)),
                        };
                        states[q] = st;
                        match st {
                            SlotDirty::Clean => {},
                            SlotDirty::Full => {
                                full_count += 1;
                                dirty_count += 1;
                            },
                            SlotDirty::Cropped { .. } => {
                                dirty_count += 1;
                            },
                        }
                    }
                    if dirty_count == 0 {
                        continue;
                    }
                    if full_count == n_quads {
                        if n_quads == 1 {
                            candidates.push(DirtyCandidate::Single {
                                rect,
                                hash: slot_hashes[quad_slots[0]],
                            });
                        } else {
                            let (quads, _) = tile_quadrants(rect);
                            let mut quad_hashes = [[0u8; 16]; 4];
                            for (q, &slot) in quad_slots[..n_quads].iter().enumerate() {
                                quad_hashes[q] = slot_hashes[slot];
                            }
                            let parent_hash = combine_quad_hashes(
                                &self.hash_key,
                                rect.w,
                                rect.h,
                                &quad_hashes[..n_quads],
                            );
                            candidates.push(DirtyCandidate::Splittable {
                                rect,
                                parent_hash,
                                quads,
                                quad_hashes,
                                n_quads,
                            });
                        }
                    } else {
                        let (quads, _) = tile_quadrants(rect);
                        let mut quad_hashes = [[0u8; 16]; 4];
                        let mut all_hashed = dirty_count == n_quads && n_quads > 1;
                        for (q, &slot) in quad_slots[..n_quads].iter().enumerate() {
                            if matches!(
                                states[q],
                                SlotDirty::Full
                                    | SlotDirty::Cropped {
                                        has_quad_hash: true,
                                        ..
                                    }
                            ) {
                                quad_hashes[q] = slot_hashes[slot];
                            } else {
                                all_hashed = false;
                            }
                        }
                        let parent_hash = all_hashed.then(|| {
                            combine_quad_hashes(
                                &self.hash_key,
                                rect.w,
                                rect.h,
                                &quad_hashes[..n_quads],
                            )
                        });
                        candidates.push(DirtyCandidate::PartialTile2D {
                            rect,
                            parent_hash,
                            quads,
                            states,
                            quad_hashes,
                            n_quads,
                        });
                    }
                },
            }
        }

        // Pass 1: resolve all inter-frame cache hits and pin their slots before
        // Pass 2 can allocate or evict any slot.
        let mut pass1 = Vec::with_capacity(candidates.len());
        for cand in candidates {
            match cand {
                DirtyCandidate::Single { rect, hash } => {
                    if let Some(hit_ref) = self.lookup(hash, rect.w, rect.h, has_alpha, mode) {
                        self.touch_and_pin(hit_ref.slot_id);
                        pass1.push(Pass1Item::Hit { rect, hit_ref });
                    } else {
                        pass1.push(Pass1Item::Miss {
                            rect,
                            hash,
                            sub_quads: None,
                        });
                    }
                },
                DirtyCandidate::Splittable {
                    rect,
                    parent_hash,
                    quads,
                    quad_hashes,
                    n_quads,
                } => {
                    if self.try_hit_splittable_parent(
                        rect,
                        parent_hash,
                        &quads,
                        &quad_hashes,
                        n_quads,
                        has_alpha,
                        mode,
                        &mut pass1,
                    ) {
                        continue;
                    }
                    let mut quad_hits = [None; 4];
                    let mut any_quad_hit = false;
                    for q in 0..n_quads {
                        let qrect = quads[q];
                        if let Some(qhit) =
                            self.lookup(quad_hashes[q], qrect.w, qrect.h, has_alpha, mode)
                        {
                            quad_hits[q] = Some(qhit);
                            any_quad_hit = true;
                        }
                    }
                    if any_quad_hit {
                        for q in 0..n_quads {
                            if let Some(hit_ref) = quad_hits[q] {
                                self.touch_and_pin(hit_ref.slot_id);
                                pass1.push(Pass1Item::Hit {
                                    rect: quads[q],
                                    hit_ref,
                                });
                            } else {
                                pass1.push(Pass1Item::Miss {
                                    rect: quads[q],
                                    hash: quad_hashes[q],
                                    sub_quads: None,
                                });
                            }
                        }
                    } else {
                        pass1.push(Pass1Item::Miss {
                            rect,
                            hash: parent_hash,
                            sub_quads: Some((quads, quad_hashes, n_quads)),
                        });
                    }
                },
                DirtyCandidate::PartialTile2D {
                    rect,
                    parent_hash,
                    quads,
                    states,
                    quad_hashes,
                    n_quads,
                } => {
                    // When every quadrant of a `> 128`px tile is dirty and
                    // carries a full-quadrant hash (mix of `Full` and lookup-
                    // hashed `Cropped`), prefer a single coalesced parent
                    // `CacheHit` if already cached (e.g. closing a popup whose
                    // border partially overlapped an edge quadrant) without
                    // ever polluting the cache on a miss.
                    if let Some(ph) = parent_hash
                        && self.try_hit_splittable_parent(
                            rect,
                            ph,
                            &quads,
                            &quad_hashes,
                            n_quads,
                            has_alpha,
                            mode,
                            &mut pass1,
                        )
                    {
                        continue;
                    }

                    let mut resolved_grid: [Option<Pass1Item>; 4] = [None, None, None, None];
                    let mut crop_miss_grid: [Option<TileRect>; 4] = [None; 4];
                    for q in 0..n_quads {
                        let qrect = quads[q];
                        let grid_slot = quad_grid_slot(rect, qrect);
                        match states[q] {
                            SlotDirty::Clean => {},
                            SlotDirty::Full => {
                                let hash = quad_hashes[q];
                                if let Some(hit_ref) =
                                    self.lookup(hash, qrect.w, qrect.h, has_alpha, mode)
                                {
                                    self.touch_and_pin(hit_ref.slot_id);
                                    resolved_grid[grid_slot] = Some(Pass1Item::Hit {
                                        rect: qrect,
                                        hit_ref,
                                    });
                                } else {
                                    resolved_grid[grid_slot] = Some(Pass1Item::Miss {
                                        rect: qrect,
                                        hash,
                                        sub_quads: None,
                                    });
                                }
                            },
                            SlotDirty::Cropped {
                                min_x,
                                max_x,
                                min_y,
                                max_y,
                                has_quad_hash,
                            } => {
                                if has_quad_hash
                                    && let Some(hit_ref) = self.lookup(
                                        quad_hashes[q],
                                        qrect.w,
                                        qrect.h,
                                        has_alpha,
                                        mode,
                                    )
                                {
                                    self.touch_and_pin(hit_ref.slot_id);
                                    resolved_grid[grid_slot] = Some(Pass1Item::Hit {
                                        rect: qrect,
                                        hit_ref,
                                    });
                                } else {
                                    crop_miss_grid[grid_slot] = Some(SlotDirty::cropped_rect(
                                        qrect, min_x, max_x, min_y, max_y,
                                    ));
                                }
                            },
                        }
                    }
                    merge_cropped_quad_slots(&mut crop_miss_grid);
                    for s in 0..4 {
                        if let Some(item) = resolved_grid[s].take() {
                            pass1.push(item);
                        } else if let Some(crect) = crop_miss_grid[s] {
                            pass1.push(Pass1Item::Uncached { rect: crect });
                        }
                    }
                },
            }
        }

        // Pass 2: deduplicate identical misses within this frame and allocate
        // unpinned LRU slots for unique misses.
        let initial_quality = match mode {
            CompressionMode::Lossless => TileQuality::Lossless,
            CompressionMode::Hybrid { quality, subsamp } => TileQuality::Lossy { quality, subsamp },
        };
        let mut intra_frame: HashMap<(TileHash, u32, u32), CacheHitRef, TileHashBuildHasher> =
            HashMap::with_hasher(*self.index.hasher());
        let mut evicted_to_free: Vec<u16> = Vec::new();
        let mut tile_rects = Vec::with_capacity(pass1.len());
        let mut actions = Vec::with_capacity(pass1.len());

        for item in pass1 {
            match item {
                Pass1Item::Hit { rect, hit_ref } => {
                    tile_rects.push(rect);
                    actions.push(TileCacheAction::Hit(hit_ref));
                },
                Pass1Item::Uncached { rect } => {
                    tile_rects.push(rect);
                    actions.push(TileCacheAction::Uncached);
                },
                Pass1Item::Miss {
                    rect,
                    hash,
                    sub_quads,
                } => {
                    if let Some(&hit_ref) = intra_frame.get(&(hash, rect.w, rect.h)) {
                        tile_rects.push(rect);
                        actions.push(TileCacheAction::Hit(hit_ref));
                        continue;
                    }
                    let need_bytes = rect.w as usize * rect.h as usize * 4;
                    if let Some(slot_id) =
                        self.allocate_unpinned_slot(need_bytes, &mut evicted_to_free)
                    {
                        let primary_key = CacheKey {
                            hash,
                            w: rect.w,
                            h: rect.h,
                            has_alpha,
                            quality: initial_quality,
                        };
                        let mut sub_keys = [None; 4];
                        let mut n_sub_keys = 0usize;
                        if let Some((quads, quad_hashes, n_quads)) = sub_quads {
                            for q in 0..n_quads {
                                let qrect = quads[q];
                                let region = quadrant_region(rect, qrect);
                                let qkey = CacheKey {
                                    hash: quad_hashes[q],
                                    w: qrect.w,
                                    h: qrect.h,
                                    has_alpha,
                                    quality: initial_quality,
                                };
                                sub_keys[q] = Some(SubEntry { key: qkey, region });
                                let qref = CacheHitRef { slot_id, region };
                                self.index.insert(qkey, qref);
                                intra_frame
                                    .entry((quad_hashes[q], qrect.w, qrect.h))
                                    .or_insert(qref);
                            }
                            n_sub_keys = n_quads;
                        }

                        let idx = slot_id as usize;
                        self.slots[idx] = ServerSlot {
                            occupied: true,
                            bytes: need_bytes,
                            primary_key,
                            sub_keys,
                            n_sub_keys,
                            lru_prev: NONE_SLOT,
                            lru_next: NONE_SLOT,
                            pinned_gen: self.frame_gen,
                        };
                        self.used_bytes += need_bytes;
                        self.occupied_count += 1;
                        self.lru_push_front(slot_id);

                        let full_ref = CacheHitRef {
                            slot_id,
                            region: CacheHitRegion::Full,
                        };
                        self.index.insert(primary_key, full_ref);
                        intra_frame.insert((hash, rect.w, rect.h), full_ref);

                        tile_rects.push(rect);
                        actions.push(TileCacheAction::Put(slot_id));
                    } else {
                        tile_rects.push(rect);
                        actions.push(TileCacheAction::Uncached);
                    }
                },
            }
        }

        // Only emit explicit evictions for slots that were evicted to free byte
        // budget and NOT subsequently reused by another `CachePut` in this frame.
        evicted_to_free.retain(|&slot_id| !self.slots[slot_id as usize].occupied);
        let reset_tile_cache = std::mem::take(&mut self.reset_pending);
        if reset_tile_cache {
            debug_assert!(evicted_to_free.is_empty());
        }

        FrameCachePlan {
            tile_rects,
            actions,
            reset_tile_cache,
            evict_slots: evicted_to_free,
        }
    }

    /// Promotes a single `CachePut` slot (and/or individual lossless quadrants
    /// of a `TilePayload::Split` tile) encoded under [`CompressionMode::Hybrid`]
    /// from [`TileQuality::Lossy`] to [`TileQuality::Lossless`] once Phase B
    /// confirms its actual encoded payload.
    pub(crate) fn promote_slot_lossless(&mut self, slot_id: u16, mask: u8) {
        if mask == 0 {
            return;
        }
        let idx = slot_id as usize;
        if idx >= MAX_TILE_CACHE_SLOTS || !self.slots[idx].occupied {
            return;
        }
        if (mask & PROMOTE_PRIMARY) != 0
            && self.slots[idx].primary_key.quality != TileQuality::Lossless
        {
            let old_key = self.slots[idx].primary_key;
            let full_ref = CacheHitRef {
                slot_id,
                region: CacheHitRegion::Full,
            };
            self.remove_index_if_matches(old_key, full_ref);
            let mut new_key = old_key;
            new_key.quality = TileQuality::Lossless;
            self.slots[idx].primary_key = new_key;
            self.index.insert(new_key, full_ref);
        }
        let n_sub = self.slots[idx].n_sub_keys;
        for q in 0..n_sub {
            if let Some(sub) = self.slots[idx].sub_keys[q]
                && (mask & region_promote_bit(sub.region)) != 0
                && sub.key.quality != TileQuality::Lossless
            {
                let sub_ref = CacheHitRef {
                    slot_id,
                    region: sub.region,
                };
                self.remove_index_if_matches(sub.key, sub_ref);
                let mut new_sub = sub;
                new_sub.key.quality = TileQuality::Lossless;
                self.slots[idx].sub_keys[q] = Some(new_sub);
                self.index.insert(new_sub.key, sub_ref);
            }
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct ClientCacheSlot {
    pub(crate) occupied: bool,
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) has_alpha: bool,
    pub(crate) pixels: Vec<Vec4u8>,
}

impl ClientCacheSlot {
    #[inline]
    fn byte_len(&self) -> usize {
        if self.occupied {
            self.w as usize * self.h as usize * 4
        } else {
            0
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.occupied = false;
        self.w = 0;
        self.h = 0;
        self.has_alpha = false;
        self.pixels = Vec::new();
    }
}

/// Raw pointer handle to [`ClientTileCache::slots`] enabling lock-free parallel
/// Step-1 `CachePut` writes to validated pairwise-disjoint slot indices.
pub(crate) struct SyncClientSlots {
    base: *mut ClientCacheSlot,
}

// SAFETY: Step 1 enforces via `put_claimed: [AtomicBool; MAX_TILE_CACHE_SLOTS]`
// that each `slot_id < MAX_TILE_CACHE_SLOTS` is accessed mutably by at most
// one worker thread, and `ClientTileCache` is not otherwise accessed until
// Step 1's `WorkerPool::execute` joins.
unsafe impl Send for SyncClientSlots {}
unsafe impl Sync for SyncClientSlots {}

impl SyncClientSlots {
    /// # Safety
    /// Caller must guarantee `slot_id < MAX_TILE_CACHE_SLOTS`.
    #[inline]
    pub(crate) unsafe fn slot_ptr(&self, slot_id: u16) -> *mut ClientCacheSlot {
        debug_assert!((slot_id as usize) < MAX_TILE_CACHE_SLOTS);
        // SAFETY: Caller guarantees `slot_id < MAX_TILE_CACHE_SLOTS`.
        unsafe { self.base.add(slot_id as usize) }
    }
}

/// Pre-validated Step-2 cache blit source descriptor.
#[derive(Debug, Copy, Clone)]
pub(crate) struct ValidatedCacheHit {
    pub(crate) tile_idx: usize,
    pub(crate) slot_id: u16,
    pub(crate) src_x: usize,
    pub(crate) src_y: usize,
}

/// Resolves `(src_x, src_y, region_w, region_h)` within a `slot_w x slot_h`
/// cached tile for `region`.
#[inline]
fn resolve_hit_region(
    slot_w: u32,
    slot_h: u32,
    region: CacheHitRegion,
) -> Result<(usize, usize, u32, u32)> {
    let sw = SUBTILE_W as u32;
    let sh = SUBTILE_H as u32;
    match region {
        CacheHitRegion::Full => Ok((0, 0, slot_w, slot_h)),
        CacheHitRegion::TopLeft => {
            if slot_w <= sw && slot_h <= sh {
                bail!("TopLeft region on non-splittable slot {slot_w}x{slot_h}");
            }
            Ok((0, 0, slot_w.min(sw), slot_h.min(sh)))
        },
        CacheHitRegion::TopRight => {
            if slot_w <= sw {
                bail!("TopRight region on narrow slot {slot_w}x{slot_h}");
            }
            Ok((SUBTILE_W, 0, slot_w - sw, slot_h.min(sh)))
        },
        CacheHitRegion::BottomLeft => {
            if slot_h <= sh {
                bail!("BottomLeft region on short slot {slot_w}x{slot_h}");
            }
            Ok((0, SUBTILE_H, slot_w.min(sw), slot_h - sh))
        },
        CacheHitRegion::BottomRight => {
            if slot_w <= sw || slot_h <= sh {
                bail!("BottomRight region on slot {slot_w}x{slot_h}");
            }
            Ok((SUBTILE_W, SUBTILE_H, slot_w - sw, slot_h - sh))
        },
    }
}

/// Bounded decoded-pixel array owned by `wprsc` for connection-wide inter-frame
/// tile caching (`<= MAX_TILE_CACHE_SLOTS` slots, `<= MAX_TILE_CACHE_BYTES`).
#[derive(Debug)]
pub struct ClientTileCache {
    slots: Vec<ClientCacheSlot>,
    used_bytes: usize,
    occupied_count: usize,
}

impl Default for ClientTileCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientTileCache {
    pub fn new() -> Self {
        Self {
            slots: (0..MAX_TILE_CACHE_SLOTS)
                .map(|_| ClientCacheSlot::default())
                .collect(),
            used_bytes: 0,
            occupied_count: 0,
        }
    }

    /// Unconditionally clears all cached tile slots and frees their backing memory.
    pub fn clear(&mut self) {
        if self.occupied_count == 0 && self.used_bytes == 0 {
            return;
        }
        for slot in &mut self.slots {
            slot.clear();
        }
        self.used_bytes = 0;
        self.occupied_count = 0;
    }

    /// Evicts `slot_id` and frees its backing pixel vector.
    pub fn evict(&mut self, slot_id: u16) -> Result<()> {
        let idx = slot_id as usize;
        if idx >= MAX_TILE_CACHE_SLOTS {
            bail!("evict slot_id {slot_id} out of range (max {MAX_TILE_CACHE_SLOTS})");
        }
        let slot = &mut self.slots[idx];
        if slot.occupied {
            self.used_bytes -= slot.byte_len();
            self.occupied_count -= 1;
            slot.clear();
        }
        Ok(())
    }

    #[inline]
    pub fn occupied_slots(&self) -> usize {
        self.occupied_count
    }

    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Applies header-level cache control operations (`reset_tile_cache` and
    /// `evict_slots`) before Step-1 decoding begins.
    pub(crate) fn apply_header_ops(
        &mut self,
        reset_tile_cache: bool,
        evict_slots: &[u16],
    ) -> Result<()> {
        if reset_tile_cache {
            self.clear();
        }
        for &slot_id in evict_slots {
            self.evict(slot_id).location(loc!())?;
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn sync_slots(&mut self) -> SyncClientSlots {
        SyncClientSlots {
            base: self.slots.as_mut_ptr(),
        }
    }

    /// Recomputes `used_bytes` and `occupied_count` after Step 1 joins,
    /// rejecting any frame whose cache updates violate [`MAX_TILE_CACHE_BYTES`].
    pub(crate) fn recompute_and_verify_bytes(&mut self) -> Result<()> {
        let mut total_bytes = 0usize;
        let mut occupied = 0usize;
        for slot in &self.slots {
            if slot.occupied {
                total_bytes = total_bytes.saturating_add(slot.byte_len());
                occupied += 1;
            }
        }
        self.used_bytes = total_bytes;
        self.occupied_count = occupied;
        if total_bytes > MAX_TILE_CACHE_BYTES {
            self.clear();
            bail!(
                "tile cache byte usage {total_bytes} exceeds MAX_TILE_CACHE_BYTES ({MAX_TILE_CACHE_BYTES})"
            );
        }
        Ok(())
    }

    /// Reserves uninitialized capacity for `rect.w * rect.h` pixels in
    /// `slots[slot_id]` (reusing existing capacity when already sized) and
    /// returns a tightly packed [`FrameRows`] (`stride_px = rect.w`) over its
    /// spare capacity so Step-1 decoding writes directly into L2-hot slot memory
    /// without zero-filling or polluting `wl_shm` cache lines.
    ///
    /// Caller MUST call [`Self::commit_slot`] after initializing every pixel in
    /// `0..rect.w * rect.h`, or [`Self::invalidate_slot`] on error.
    ///
    /// # Safety
    /// Caller must hold exclusive ownership of `slot_id < MAX_TILE_CACHE_SLOTS`
    /// and drop the returned [`FrameRows`] before releasing the slot.
    #[inline]
    pub(crate) unsafe fn prepare_slot_target_uninit(
        slots: &SyncClientSlots,
        slot_id: u16,
        rect: TileRect,
    ) -> FrameRows {
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let n_px = tw * th;

        // SAFETY: Caller holds exclusive ownership of `slot_id`.
        let slot = unsafe { &mut *slots.slot_ptr(slot_id) };
        slot.occupied = false;
        slot.pixels.clear();
        if slot.pixels.capacity() < n_px {
            slot.pixels.reserve_exact(n_px);
        } else if slot.pixels.capacity() > n_px.saturating_mul(2) {
            slot.pixels.shrink_to(n_px);
        }
        FrameRows {
            base: slot.pixels.as_mut_ptr().cast(),
            stride_px: tw,
        }
    }

    /// Finalizes `slots[slot_id]` after every pixel in `0..rect.w * rect.h`
    /// has been initialized via [`Self::prepare_slot_target_uninit`], returning
    /// an immutable slice of the freshly decoded L2-hot pixels for immediate
    /// streaming blit into the surface canvas.
    ///
    /// # Safety
    /// Caller must hold exclusive ownership of `slot_id < MAX_TILE_CACHE_SLOTS`
    /// and must have initialized all `rect.w * rect.h` elements in
    /// `slots[slot_id].pixels`.
    #[inline]
    pub(crate) unsafe fn commit_slot(
        slots: &SyncClientSlots,
        slot_id: u16,
        rect: TileRect,
        has_alpha: bool,
    ) -> &[Vec4u8] {
        let n_px = rect.w as usize * rect.h as usize;
        // SAFETY: Caller holds exclusive ownership of `slot_id` and initialized
        // all `n_px` pixels.
        let slot = unsafe { &mut *slots.slot_ptr(slot_id) };
        unsafe { slot.pixels.set_len(n_px) };
        slot.w = rect.w;
        slot.h = rect.h;
        slot.has_alpha = has_alpha;
        slot.occupied = true;
        &slot.pixels
    }

    /// Marks `slot_id` empty after a failed `CachePut` decode so stale pixels
    /// can never survive in the slot.
    ///
    /// # Safety
    /// Caller must hold exclusive ownership of `slot_id < MAX_TILE_CACHE_SLOTS`.
    #[inline]
    pub(crate) unsafe fn invalidate_slot(slots: &SyncClientSlots, slot_id: u16) {
        // SAFETY: Caller holds exclusive ownership of `slot_id`.
        unsafe { &mut *slots.slot_ptr(slot_id) }.clear();
    }

    /// Validates that `hit_ref` references a live slot matching `rect`'s
    /// dimensions and `has_alpha` format.
    #[inline]
    pub(crate) fn validate_hit(
        &self,
        tile_idx: usize,
        hit_ref: CacheHitRef,
        rect: TileRect,
        has_alpha: bool,
    ) -> Result<ValidatedCacheHit> {
        let idx = hit_ref.slot_id as usize;
        let Some(slot) = self.slots.get(idx) else {
            bail!(
                "CacheHit slot_id {} out of range (max {MAX_TILE_CACHE_SLOTS})",
                hit_ref.slot_id
            );
        };
        if !slot.occupied {
            bail!("CacheHit references empty slot {}", hit_ref.slot_id);
        }
        if slot.has_alpha != has_alpha {
            bail!(
                "CacheHit slot {} has_alpha={} does not match frame has_alpha={has_alpha}",
                hit_ref.slot_id,
                slot.has_alpha
            );
        }
        let (src_x, src_y, exp_w, exp_h) =
            resolve_hit_region(slot.w, slot.h, hit_ref.region).location(loc!())?;
        if exp_w != rect.w || exp_h != rect.h {
            bail!(
                "CacheHit slot {} ({:?}) dimensions {exp_w}x{exp_h} != tile rect {}x{}",
                hit_ref.slot_id,
                hit_ref.region,
                rect.w,
                rect.h
            );
        }
        Ok(ValidatedCacheHit {
            tile_idx,
            slot_id: hit_ref.slot_id,
            src_x,
            src_y,
        })
    }

    /// Returns the source pixel slice starting at `(hit.src_x, hit.src_y)` and
    /// the slot's pixel row stride (`slot.w`) for a pre-validated
    /// [`ValidatedCacheHit`].
    #[inline]
    pub(crate) fn slot_hit_src(&self, hit: ValidatedCacheHit) -> (&[Vec4u8], usize) {
        let slot = &self.slots[hit.slot_id as usize];
        let slot_w = slot.w as usize;
        let start = hit.src_y * slot_w + hit.src_x;
        (&slot.pixels[start..], slot_w)
    }
}
