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

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::collections::hash_map::Entry;
use std::mem;
use std::num::NonZeroUsize;
use std::os::fd::OwnedFd;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use smithay::input::Seat;
use smithay::input::SeatState;
use smithay::input::pointer::CursorImageStatus as SmithayCursorImageStatus;
use smithay::output::Output;
use smithay::reexports::calloop::LoopHandle;
use smithay::reexports::calloop::timer::TimeoutAction;
use smithay::reexports::calloop::timer::Timer;
use smithay::reexports::wayland_protocols_misc::server_decoration::server::org_kde_kwin_server_decoration_manager::Mode as KdeDecorationMode;
use smithay::reexports::wayland_server::DisplayHandle;
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::backend::GlobalId;
use smithay::reexports::wayland_server::backend::ObjectId;
use smithay::reexports::wayland_server::protocol::wl_callback::WlCallback;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::wayland::compositor;
use smithay::wayland::compositor::CompositorState;
use smithay::wayland::output::OutputManagerState;
use smithay::wayland::selection::primary_selection::PrimarySelectionState;
use smithay::wayland::shell::kde::decoration::KdeDecorationState;
use smithay::wayland::shell::xdg::XdgShellState;
use smithay::wayland::shell::xdg::decoration::XdgDecorationState;
use smithay::wayland::shm::ShmState;
use smithay::wayland::viewporter::ViewporterState;

use crate::constants;
use crate::filtering::ServerTileCache;
use crate::filtering::ShadowBuffer;
use crate::prelude::*;
use crate::serialization::CompressionMode;
use crate::serialization::Event;
use crate::serialization::Request;
use crate::serialization::SendType;
use crate::serialization::Serializer;
use crate::serialization::geometry::Point;
use crate::serialization::geometry::Rectangle;
use crate::serialization::wayland::BufferMetadata;
use crate::serialization::wayland::SurfaceRequest;
use crate::serialization::wayland::SurfaceRequestPayload;
use crate::serialization::wayland::SurfaceState;
use crate::serialization::wayland::WlSurfaceId;
use crate::sharding_compression::ShardingCompressor;
use crate::utils::SerialMap;

pub mod client_handlers;
pub mod data_device;
pub mod smithay_handlers;

use data_device::WprsDataDeviceState;

/// Buffer-space damage accumulated across deferred buffer commits while a
/// surface is waiting for a frame credit.
#[derive(Debug, Default)]
pub(crate) enum PendingDamage {
    #[default]
    None,
    Rects(Vec<Rectangle<i32>>),
    Full,
}

impl PendingDamage {
    pub(crate) fn mark_full(&mut self) {
        *self = Self::Full;
    }

    pub(crate) fn merge(&mut self, rects: Vec<Rectangle<i32>>) {
        if rects.is_empty() {
            return;
        }
        match self {
            Self::Full => {},
            Self::None => {
                if rects.len() >= constants::SENT_DAMAGE_LIMIT {
                    *self = Self::Full;
                } else {
                    *self = Self::Rects(rects);
                }
            },
            Self::Rects(existing) => {
                if existing.len().saturating_add(rects.len()) >= constants::SENT_DAMAGE_LIMIT {
                    *self = Self::Full;
                } else {
                    existing.extend(rects);
                }
            },
        }
    }

    pub(crate) fn take_rects(&mut self) -> Option<Vec<Rectangle<i32>>> {
        match mem::take(self) {
            Self::Rects(rects) if !rects.is_empty() => Some(rects),
            Self::None | Self::Rects(_) | Self::Full => None,
        }
    }
}

pub(crate) struct TrackedSurfaceState {
    pub(crate) state: SurfaceState,
    /// Whether this surface has received at least one `wl_surface.commit` for
    /// its current role lifecycle (reset when `xdg_toplevel`/`xdg_popup` is
    /// created or destroyed so uncommitted shell roles are not synced early).
    pub(crate) has_committed: bool,
    /// Whether `state` has pending out-of-commit mutations (role assignment,
    /// pre-commit toplevel/decoration requests, or an `xdg_shell` unmap
    /// awaiting a bufferless remap commit) that must not be deduplicated away
    /// on the next bufferless commit.
    pub(crate) dirty: bool,
    /// Accumulated per-commit `wl_surface.offset` / `attach(dx, dy)` delta
    /// waiting to be sent with the next forwarded `Commit`.
    pub(crate) pending_buffer_delta: Option<Point<i32>>,
    /// Buffer-space damage accumulated while this surface is deferred.
    pub(crate) pending_damage: PendingDamage,
}

impl TrackedSurfaceState {
    pub(crate) fn add_buffer_delta(&mut self, delta: Option<Point<i32>>) {
        let Some(delta) = delta else {
            return;
        };
        if delta.x == 0 && delta.y == 0 {
            return;
        }
        match &mut self.pending_buffer_delta {
            Some(pending) => {
                pending.x = pending.x.saturating_add(delta.x);
                pending.y = pending.y.saturating_add(delta.y);
                if pending.x == 0 && pending.y == 0 {
                    self.pending_buffer_delta = None;
                }
            },
            None => {
                self.pending_buffer_delta = Some(delta);
            },
        }
    }
}

struct LockedSurfaceState(Mutex<TrackedSurfaceState>);

/// End-to-end in-flight `RawBuffer` credit cap and fair FIFO deferred surface
/// queue for the active `wprsc` connection.
#[derive(Debug)]
pub(crate) struct FrameBackpressure {
    next_frame_seq: u64,
    max_in_flight: usize,
    in_flight: HashSet<u64>,
    deferred_queue: VecDeque<WlSurfaceId>,
    deferred_set: HashSet<WlSurfaceId>,
}

impl FrameBackpressure {
    pub(crate) fn new(max_in_flight: usize) -> Self {
        Self {
            next_frame_seq: 0,
            max_in_flight,
            in_flight: HashSet::with_capacity(max_in_flight),
            deferred_queue: VecDeque::new(),
            deferred_set: HashSet::new(),
        }
    }

    pub(crate) fn has_credit(&self) -> bool {
        self.in_flight.len() < self.max_in_flight
    }

    #[cfg(test)]
    pub(crate) fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    pub(crate) fn allocate_seq(&mut self) -> u64 {
        self.next_frame_seq += 1;
        let seq = self.next_frame_seq;
        self.in_flight.insert(seq);
        seq
    }

    pub(crate) fn ack(&mut self, seq: u64) -> bool {
        self.in_flight.remove(&seq)
    }

    pub(crate) fn defer(&mut self, surface_id: WlSurfaceId) {
        if self.deferred_set.insert(surface_id) {
            self.deferred_queue.push_back(surface_id);
        }
    }

    pub(crate) fn is_deferred(&self, surface_id: WlSurfaceId) -> bool {
        self.deferred_set.contains(&surface_id)
    }

    pub(crate) fn remove_deferred(&mut self, surface_id: WlSurfaceId) -> bool {
        if self.deferred_set.remove(&surface_id) {
            self.deferred_queue.retain(|id| *id != surface_id);
            true
        } else {
            false
        }
    }

    pub(crate) fn pop_deferred(&mut self) -> Option<WlSurfaceId> {
        while let Some(id) = self.deferred_queue.pop_front() {
            if self.deferred_set.remove(&id) {
                return Some(id);
            }
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn deferred_len(&self) -> usize {
        self.deferred_set.len()
    }

    pub(crate) fn reset(&mut self) {
        self.in_flight.clear();
        self.deferred_queue.clear();
        self.deferred_set.clear();
    }
}

pub(crate) struct HeldFrameCallbacks {
    pub(crate) surface: WlSurface,
    pub(crate) ready_at: Instant,
    pub(crate) callbacks: Vec<WlCallback>,
}

/// Tracks client activity timestamps on `wprsd` to detect silently stalled
/// `wprsc` peers.
#[derive(Debug)]
pub(crate) struct ServerHeartbeatState {
    stall_timeout: Duration,
    last_client_activity: Option<Instant>,
}

impl ServerHeartbeatState {
    pub(crate) fn new(stall_timeout: Duration) -> Self {
        Self {
            stall_timeout,
            last_client_activity: None,
        }
    }

    pub(crate) fn on_connected(&mut self, now: Instant) {
        self.last_client_activity = Some(now);
    }

    pub(crate) fn on_activity(&mut self, now: Instant) {
        self.last_client_activity = Some(now);
    }

    pub(crate) fn on_disconnected(&mut self) {
        self.last_client_activity = None;
    }

    pub(crate) fn is_stalled(&self, now: Instant) -> bool {
        self.last_client_activity
            .is_some_and(|last| now.saturating_duration_since(last) >= self.stall_timeout)
    }

    pub(crate) fn stall_timeout(&self) -> Duration {
        self.stall_timeout
    }
}

/// Last buffer transmitted to the currently connected client for a surface,
/// retaining a local shadow copy of its pixels (plus per-quadrant hashes) so
/// subsequent commits skip unchanged `128x128` quadrants and identical frames.
pub struct SyncedSurfaceBuffer {
    pub metadata: BufferMetadata,
    pub shadow: ShadowBuffer,
    pub last_active: Instant,
}

/// Advises glibc to return freed `brk` heap pages (such as dropped dormant
/// shadow buffers below dynamic `M_MMAP_THRESHOLD`) back to the OS.
fn release_freed_heap_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu", not(miri)))]
    // SAFETY: `malloc_trim(0)` is a thread-safe glibc heap-trimming hint.
    unsafe {
        let _ = nix::libc::malloc_trim(0);
    }
}

fn surface_destruction_callback(state: &mut WprsServerState, surface: &WlSurface) {
    let was_deferred = compositor::with_states(surface, |surface_data| {
        let surface_state = &surface_data
            .data_map
            .get::<LockedSurfaceState>()
            .unwrap()
            .0
            .lock()
            .unwrap()
            .state;

        let writer = state.serializer.writer();

        writer.send(SendType::Object(Request::Surface(SurfaceRequest {
            client: surface_state.client,
            surface: surface_state.id,
            payload: SurfaceRequestPayload::Destroyed,
        })));

        state.object_map.remove(&surface_state.id);
        state.synced_buffers.remove(&surface_state.id);
        state.dirty_sync_surfaces.remove(&surface_state.id);
        state.committed_surfaces.remove(&surface_state.id);
        let was_deferred = state.frame_backpressure.remove_deferred(surface_state.id);
        state.held_frame_callbacks.remove(&surface_state.id);
        state.deferred_sync_ancestors.remove(&surface_state.id);
        if matches!(
            &state.cursor_status,
            Some(SmithayCursorImageStatus::Surface(cursor_surface)) if cursor_surface == surface
        ) {
            state.cursor_status = None;
        }
        was_deferred
    });
    if was_deferred {
        state.flush_ready_deferred_sync_ancestors();
    }
    state.on_dnd_surface_destroyed(surface);
}

pub struct WprsServerState {
    pub dh: DisplayHandle,
    pub lh: LoopHandle<'static, Self>,
    pub compositor_state: CompositorState,
    pub start_time: Instant,
    pub frame_interval: Duration,
    pub xwayland_enabled: bool,
    pub xdg_shell_state: XdgShellState,
    pub xdg_decoration_state: XdgDecorationState,
    // TODO(https://gitlab.gnome.org/GNOME/gtk/-/merge_requests/6398): rip this
    // out once GTK switches to xdg-decoration-protocol and applications/distros
    // move to GTK4.
    pub kde_decoration_state: KdeDecorationState,
    pub shm_state: ShmState,
    pub seat_state: SeatState<Self>,
    pub output_manager_state: OutputManagerState,
    pub data_device_state: WprsDataDeviceState,
    pub primary_selection_state: PrimarySelectionState,
    pub viewporter_state: ViewporterState,

    pub seat: Seat<Self>,

    pub serializer: Serializer<Request, Event>,
    pub compressor: ShardingCompressor,
    pub compression_mode: CompressionMode,
    /// Metadata-only connection-wide inter-frame tile cache directory tracking
    /// which content hashes live in `wprsc`'s bounded decoded pixel slots.
    pub tile_cache: ServerTileCache,
    /// Reverse map from WlSurfaceId, which is the hash of ObjectId, back to its
    /// source ObjectId. We can't put this in SurfaceState because is
    /// serializable, while this only has meaning locally. We need this for
    /// finding the local surface associated with a remote pointer/keyboard
    /// event. Left is the serialized surface id and right is the local, native
    /// surface id.
    // object_map:
    // left: serialized surface id, right: local native surface id
    pub object_map: HashMap<WlSurfaceId, ObjectId>,
    /// Tracks the last synced buffer metadata and shadow pixels sent to the
    /// currently connected client for each surface so `commit_impl` can diff
    /// incremental updates and skip unchanged quadrants/frames.
    pub synced_buffers: HashMap<WlSurfaceId, SyncedSurfaceBuffer>,
    /// Tracks synchronized subsurfaces that sent a `Commit` to `wprsc` since
    /// their desynchronized ancestor last committed, so bufferless parent sync
    /// commits still forward `Commit(parent)` to flush them on `wprsc`.
    pub dirty_sync_surfaces: HashSet<WlSurfaceId>,
    /// Tracks surfaces for which the currently connected `wprsc` has received
    /// at least one `Commit` under the surface's active shell role, so
    /// pre-commit toplevel/decoration/reposition requests are buffered into
    /// `SurfaceState` instead of targeting an unknown surface on `wprsc`.
    pub committed_surfaces: HashSet<WlSurfaceId>,
    pub(crate) active_conn_id: Option<u64>,
    pub(crate) frame_backpressure: FrameBackpressure,
    pub(crate) held_frame_callbacks: HashMap<WlSurfaceId, HeldFrameCallbacks>,
    /// Desynchronized ancestors that already committed while one or more of
    /// their effectively-synchronized descendants were deferred, so a
    /// bufferless parent `Commit` is emitted once those descendants flush.
    pub(crate) deferred_sync_ancestors: HashSet<WlSurfaceId>,
    pub(crate) heartbeat: ServerHeartbeatState,
    pub cursor_status: Option<SmithayCursorImageStatus>,
    pub outputs: HashMap<u32, (Output, GlobalId)>,
    serial_map: SerialMap,
    pressed_keys: HashSet<u32>,
    pressed_buttons: HashSet<u32>,

    next_transfer_id: u64,
    pending_receive_pipes: HashMap<u64, OwnedFd>,
}

impl WprsServerState {
    pub fn new(
        dh: DisplayHandle,
        lh: LoopHandle<'static, Self>,
        serializer: Serializer<Request, Event>,
        xwayland_enabled: bool,
        frame_interval: Duration,
        kde_server_side_decorations: bool,
    ) -> Self {
        let mut seat_state = SeatState::new();
        let seat = seat_state.new_wl_seat(&dh, "wprs");
        let kde_default_decoration_mode = if kde_server_side_decorations {
            KdeDecorationMode::Server
        } else {
            KdeDecorationMode::Client
        };

        lh.insert_source(
            Timer::from_duration(constants::HEARTBEAT_INTERVAL),
            |_, _, state| {
                state.check_client_stall(Instant::now());
                TimeoutAction::ToDuration(constants::HEARTBEAT_INTERVAL)
            },
        )
        .expect("timer registration should never fail");

        lh.insert_source(
            Timer::from_duration(constants::DORMANT_SHADOW_SWEEP_INTERVAL),
            |_, _, state| {
                state.compress_dormant_shadows(Instant::now());
                TimeoutAction::ToDuration(constants::DORMANT_SHADOW_SWEEP_INTERVAL)
            },
        )
        .expect("timer registration should never fail");

        Self {
            dh: dh.clone(),
            lh,
            compositor_state: CompositorState::new::<Self>(&dh),
            start_time: Instant::now(),
            frame_interval,
            xwayland_enabled,
            xdg_shell_state: XdgShellState::new::<Self>(&dh),
            xdg_decoration_state: XdgDecorationState::new::<Self>(&dh),
            kde_decoration_state: KdeDecorationState::new::<Self>(&dh, kde_default_decoration_mode),
            shm_state: ShmState::new::<Self>(&dh, Vec::new()),
            seat_state,
            output_manager_state: OutputManagerState::new_with_xdg_output::<Self>(&dh),
            data_device_state: WprsDataDeviceState::new(&dh),
            primary_selection_state: PrimarySelectionState::new::<Self>(&dh),
            viewporter_state: ViewporterState::new::<Self>(&dh),
            seat,
            serializer,
            // TODO: try tuning this based on the number of cpus the machine has.
            compressor: ShardingCompressor::new(NonZeroUsize::new(16).unwrap(), 1).unwrap(),
            compression_mode: CompressionMode::Lossless,
            tile_cache: ServerTileCache::new(),
            object_map: HashMap::new(),
            synced_buffers: HashMap::new(),
            dirty_sync_surfaces: HashSet::new(),
            committed_surfaces: HashSet::new(),
            active_conn_id: None,
            frame_backpressure: FrameBackpressure::new(constants::MAX_IN_FLIGHT_FRAMES),
            held_frame_callbacks: HashMap::new(),
            deferred_sync_ancestors: HashSet::new(),
            heartbeat: ServerHeartbeatState::new(constants::STALL_TIMEOUT),
            cursor_status: None,
            outputs: HashMap::new(),
            serial_map: SerialMap::new(),
            pressed_keys: HashSet::new(),
            pressed_buttons: HashSet::new(),
            next_transfer_id: 1,
            pending_receive_pipes: HashMap::new(),
        }
    }

    pub(crate) fn frame_callback_ready_at(&self, now: Instant) -> Instant {
        // "The server should give some time for the client to draw and commit
        // after sending the frame callback events to let it hit the next output
        // refresh."
        now + self.frame_interval.saturating_sub(Duration::from_millis(2))
    }

    pub(crate) fn fire_frame_callbacks(&self, surface: &WlSurface, callbacks: Vec<WlCallback>) {
        if !surface.is_alive() {
            return;
        }
        let now_ms = self.start_time.elapsed().as_millis() as u32;
        for callback in callbacks {
            debug!(
                "Sending callback for surface {:?}: {:?}",
                surface.id(),
                callback.id()
            );
            callback.done(now_ms);
        }
    }

    pub(crate) fn hold_frame_callbacks(
        &mut self,
        surface: &WlSurface,
        surface_id: WlSurfaceId,
        callbacks: Vec<WlCallback>,
        ready_at: Instant,
    ) {
        if callbacks.is_empty() {
            return;
        }
        match self.held_frame_callbacks.entry(surface_id) {
            Entry::Occupied(mut entry) => {
                let held = entry.get_mut();
                held.callbacks.extend(callbacks);
                // Keep the earliest deadline so coalesced commits do not push
                // back the callback deadline indefinitely.
                held.ready_at = held.ready_at.min(ready_at);
            },
            Entry::Vacant(entry) => {
                entry.insert(HeldFrameCallbacks {
                    surface: surface.clone(),
                    ready_at,
                    callbacks,
                });
            },
        }
    }

    pub(crate) fn schedule_or_fire_frame_callbacks(
        &mut self,
        surface: WlSurface,
        surface_id: WlSurfaceId,
        mut callbacks: Vec<WlCallback>,
        ready_at: Instant,
    ) {
        if callbacks.is_empty() || !surface.is_alive() {
            return;
        }
        let delay = ready_at.saturating_duration_since(Instant::now());
        if delay.is_zero() {
            self.fire_frame_callbacks(&surface, callbacks);
            return;
        }
        self.lh
            .insert_source(Timer::from_duration(delay), move |_, _, state| {
                if !surface.is_alive() {
                    return TimeoutAction::Drop;
                }
                if state.serializer.other_end_connected()
                    && state.frame_backpressure.is_deferred(surface_id)
                {
                    state.hold_frame_callbacks(
                        &surface,
                        surface_id,
                        mem::take(&mut callbacks),
                        Instant::now(),
                    );
                    return TimeoutAction::Drop;
                }
                state.fire_frame_callbacks(&surface, mem::take(&mut callbacks));
                TimeoutAction::Drop
            })
            .expect("timer registration should never fail");
    }

    pub(crate) fn release_held_frame_callbacks(&mut self, surface_id: WlSurfaceId) {
        if let Some(held) = self.held_frame_callbacks.remove(&surface_id) {
            self.schedule_or_fire_frame_callbacks(
                held.surface,
                surface_id,
                held.callbacks,
                held.ready_at,
            );
        }
    }

    pub(crate) fn release_unthrottled_held_frame_callbacks(&mut self) {
        let ready_ids: Vec<WlSurfaceId> = self
            .held_frame_callbacks
            .keys()
            .copied()
            .filter(|id| !self.frame_backpressure.is_deferred(*id))
            .collect();
        for surface_id in ready_ids {
            self.release_held_frame_callbacks(surface_id);
        }
    }

    /// Compresses up to [`constants::MAX_DORMANT_COMPRESS_PER_TICK`] eligible
    /// idle surface shadow buffers in place, returning the number of surfaces
    /// compressed on this tick.
    #[instrument(skip(self), level = "debug")]
    pub(crate) fn compress_dormant_shadows(&mut self, now: Instant) -> usize {
        let mut candidates: Vec<(Instant, WlSurfaceId)> = self
            .synced_buffers
            .iter()
            .filter_map(|(&id, synced)| {
                (synced.shadow.is_dormant_compressible()
                    && now.saturating_duration_since(synced.last_active)
                        >= constants::DORMANT_SHADOW_IDLE_TIMEOUT)
                    .then_some((synced.last_active, id))
            })
            .collect();
        if candidates.is_empty() {
            return 0;
        }
        // Compress the oldest dormant surfaces first; break ties deterministically.
        candidates.sort_unstable_by_key(|&(last_active, id)| (last_active, id.0));

        let mut attempts = 0usize;
        let mut compressed = 0usize;
        for (_, id) in candidates {
            if attempts >= constants::MAX_DORMANT_COMPRESS_PER_TICK {
                break;
            }
            if let Some(synced) = self.synced_buffers.get_mut(&id) {
                attempts += 1;
                if synced.shadow.compress_dormant(&mut self.compressor) {
                    compressed += 1;
                }
            }
        }
        if compressed > 0 {
            release_freed_heap_memory();
        }
        compressed
    }

    /// Inserts `surface` into `object_map` and initializes its
    /// `LockedSurfaceState` (every surface in `object_map` has
    /// `LockedSurfaceState`).
    #[instrument(skip(self), level = "debug")]
    pub fn insert_surface(&mut self, surface: &WlSurface) -> Result<()> {
        let initial_state = SurfaceState::new(surface, None).location(loc!())?;
        self.object_map
            .insert(WlSurfaceId::new(surface), surface.id());

        let newly_inserted = compositor::with_states(surface, |surface_data| {
            surface_data.data_map.insert_if_missing_threadsafe(|| {
                LockedSurfaceState(Mutex::new(TrackedSurfaceState {
                    state: initial_state,
                    has_committed: false,
                    dirty: true,
                    pending_buffer_delta: None,
                    pending_damage: PendingDamage::None,
                }))
            })
        });
        // TODO: https://github.com/Smithay/smithay/issues/538 - move into block
        // above.
        if newly_inserted {
            compositor::add_destruction_hook(surface, surface_destruction_callback);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_backpressure_credits_fifo_dedup_and_reset() {
        let mut bp = FrameBackpressure::new(3);
        let s1 = WlSurfaceId(1);
        let s2 = WlSurfaceId(2);
        let s3 = WlSurfaceId(3);

        assert!(bp.has_credit());
        assert_eq!(bp.in_flight_count(), 0);

        let seq1 = bp.allocate_seq();
        let seq2 = bp.allocate_seq();
        let seq3 = bp.allocate_seq();
        assert_eq!((seq1, seq2, seq3), (1, 2, 3));
        assert!(!bp.has_credit());
        assert_eq!(bp.in_flight_count(), 3);

        // Stale/unknown ack is ignored.
        assert!(!bp.ack(999));
        assert!(!bp.has_credit());

        // Deferring deduplicates while preserving fair FIFO arrival order.
        bp.defer(s1);
        bp.defer(s2);
        bp.defer(s1);
        bp.defer(s3);
        assert_eq!(bp.deferred_len(), 3);
        assert!(bp.is_deferred(s1));
        assert!(bp.remove_deferred(s2));
        assert!(!bp.remove_deferred(s2));
        assert_eq!(bp.deferred_len(), 2);
        assert_eq!(bp.pop_deferred(), Some(s1));
        assert_eq!(bp.pop_deferred(), Some(s3));
        assert_eq!(bp.pop_deferred(), None);

        // Valid ack frees credit once; duplicate ack is ignored.
        assert!(bp.ack(seq2));
        assert!(bp.has_credit());
        assert_eq!(bp.in_flight_count(), 2);
        assert!(!bp.ack(seq2));

        // Reset clears in-flight and deferred state without reusing old seqs.
        bp.defer(s1);
        bp.reset();
        assert_eq!(bp.in_flight_count(), 0);
        assert_eq!(bp.deferred_len(), 0);
        assert!(!bp.ack(seq1));
        assert!(bp.allocate_seq() > seq3);
    }

    #[test]
    fn test_pending_damage_merge_cap_and_full() {
        let mut damage = PendingDamage::default();
        assert_eq!(damage.take_rects(), None);

        let r1 = Rectangle::new(0, 0, 8, 8);
        let r2 = Rectangle::new(8, 8, 8, 8);
        damage.merge(vec![r1]);
        damage.merge(vec![r2]);
        assert_eq!(damage.take_rects(), Some(vec![r1, r2]));
        assert_eq!(damage.take_rects(), None);

        // Reaching `SENT_DAMAGE_LIMIT` promotes to `Full` (`None`).
        damage.merge(vec![r1; constants::SENT_DAMAGE_LIMIT]);
        assert_eq!(damage.take_rects(), None);

        // Explicit `mark_full` stays full even after further merges.
        damage.merge(vec![r1]);
        damage.mark_full();
        damage.merge(vec![r2]);
        assert_eq!(damage.take_rects(), None);
    }

    #[test]
    fn test_server_heartbeat_state() {
        let timeout = Duration::from_secs(20);
        let t0 = Instant::now();
        let mut hb = ServerHeartbeatState::new(timeout);
        assert!(!hb.is_stalled(t0 + timeout));

        hb.on_connected(t0);
        assert!(!hb.is_stalled(t0 + timeout - Duration::from_millis(1)));
        assert!(hb.is_stalled(t0 + timeout));

        let t1 = t0 + Duration::from_secs(12);
        hb.on_activity(t1);
        assert!(!hb.is_stalled(t0 + timeout));
        assert!(hb.is_stalled(t1 + timeout));

        hb.on_disconnected();
        assert!(!hb.is_stalled(t1 + timeout + Duration::from_secs(60)));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compress_dormant_shadows_timeout_ordering_batch_cap_and_incompressible() {
        use smithay::reexports::calloop::EventLoop;
        use smithay::reexports::wayland_server::Display;

        use crate::filtering::MIN_DORMANT_COMPRESS_BYTES;
        use crate::serialization::wayland::BufferFormat;

        let display: Display<WprsServerState> = Display::new().unwrap();
        let event_loop: EventLoop<WprsServerState> = EventLoop::try_new().unwrap();
        let serializer: Serializer<Request, Event> = Serializer::new_test_server();
        let mut state = WprsServerState::new(
            display.handle(),
            event_loop.handle(),
            serializer,
            false,
            Duration::from_millis(16),
            false,
        );

        let big_meta = BufferMetadata {
            width: 512,
            height: 512,
            stride: 512 * 4,
            format: BufferFormat::Xrgb8888,
        };
        let tiny_meta = BufferMetadata {
            width: 64,
            height: 64,
            stride: 64 * 4,
            format: BufferFormat::Xrgb8888,
        };
        let make_compressible = |seed: u8| {
            let mut v = vec![0u8; MIN_DORMANT_COMPRESS_BYTES];
            for (i, b) in v.iter_mut().enumerate() {
                *b = seed.wrapping_add(((i / 256) & 0xFF) as u8);
            }
            ShadowBuffer::from_pixels(v)
        };
        let make_noise = || {
            let mut v = vec![0u8; MIN_DORMANT_COMPRESS_BYTES];
            let mut prng = 0x9E37_79B9_7F4A_7C15u64;
            for chunk in v.as_chunks_mut::<8>().0 {
                prng = prng.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = prng;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                chunk.copy_from_slice(&z.to_le_bytes());
            }
            ShadowBuffer::from_pixels(v)
        };

        let t0 = Instant::now();
        let s_noise = WlSurfaceId(1); // oldest (t0) — incompressible
        let s_old = WlSurfaceId(2); // second oldest (t0 + 1s)
        let s_mid = WlSurfaceId(3); // third oldest (t0 + 2s)
        let s_fresh = WlSurfaceId(4); // active (t0 + 25s)
        let s_tiny = WlSurfaceId(5); // tiny (< 1 MiB, oldest)

        state.synced_buffers.insert(
            s_noise,
            SyncedSurfaceBuffer {
                metadata: big_meta,
                shadow: make_noise(),
                last_active: t0,
            },
        );
        state.synced_buffers.insert(
            s_old,
            SyncedSurfaceBuffer {
                metadata: big_meta,
                shadow: make_compressible(1),
                last_active: t0 + Duration::from_secs(1),
            },
        );
        state.synced_buffers.insert(
            s_mid,
            SyncedSurfaceBuffer {
                metadata: big_meta,
                shadow: make_compressible(2),
                last_active: t0 + Duration::from_secs(2),
            },
        );
        state.synced_buffers.insert(
            s_fresh,
            SyncedSurfaceBuffer {
                metadata: big_meta,
                shadow: make_compressible(3),
                last_active: t0 + Duration::from_secs(25),
            },
        );
        state.synced_buffers.insert(
            s_tiny,
            SyncedSurfaceBuffer {
                metadata: tiny_meta,
                shadow: ShadowBuffer::from_pixels(vec![0x42u8; 64 * 64 * 4]),
                last_active: t0,
            },
        );

        // Before idle timeout: nothing compressed.
        let t_early = t0 + constants::DORMANT_SHADOW_IDLE_TIMEOUT - Duration::from_secs(1);
        assert_eq!(state.compress_dormant_shadows(t_early), 0);

        // Tick 1 at `t0 + 35s`: oldest 2 eligible surfaces (`s_noise`, `s_old`) are
        // attempted (`MAX_DORMANT_COMPRESS_PER_TICK == 2`). `s_noise` is marked
        // incompressible and stays raw; `s_old` compresses; `s_mid` waits for tick 2;
        // `s_fresh` is not yet idle; `s_tiny` is always skipped.
        let t1 = t0 + constants::DORMANT_SHADOW_IDLE_TIMEOUT + Duration::from_secs(5);
        assert_eq!(state.compress_dormant_shadows(t1), 1);
        assert!(!state.synced_buffers[&s_noise].shadow.is_compressed());
        assert!(
            !state.synced_buffers[&s_noise]
                .shadow
                .is_dormant_compressible()
        );
        assert!(state.synced_buffers[&s_old].shadow.is_compressed());
        assert!(!state.synced_buffers[&s_mid].shadow.is_compressed());
        assert!(!state.synced_buffers[&s_fresh].shadow.is_compressed());
        assert!(!state.synced_buffers[&s_tiny].shadow.is_compressed());

        // Tick 2 at `t1`: `s_noise` is skipped (already marked incompressible),
        // so `s_mid` compresses cleanly.
        assert_eq!(state.compress_dormant_shadows(t1), 1);
        assert!(state.synced_buffers[&s_mid].shadow.is_compressed());
        assert!(!state.synced_buffers[&s_fresh].shadow.is_compressed());

        // Tick 3 at `t1`: no eligible surfaces left until `s_fresh` ages out.
        assert_eq!(state.compress_dormant_shadows(t1), 0);

        // Once `s_fresh` reaches `DORMANT_SHADOW_IDLE_TIMEOUT`, it compresses too;
        // `s_tiny` stays raw forever.
        let t2 = t0 + Duration::from_secs(25) + constants::DORMANT_SHADOW_IDLE_TIMEOUT;
        assert_eq!(state.compress_dormant_shadows(t2), 1);
        assert!(state.synced_buffers[&s_fresh].shadow.is_compressed());
        assert!(!state.synced_buffers[&s_tiny].shadow.is_compressed());
    }
}
