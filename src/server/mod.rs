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
use smithay::reexports::wayland_server::backend::GlobalId;
use smithay::reexports::wayland_server::backend::ObjectId;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::DisplayHandle;
use smithay::reexports::wayland_server::Resource;
use smithay::wayland::compositor;
use smithay::wayland::compositor::CompositorState;
use smithay::wayland::output::OutputManagerState;
use smithay::wayland::selection::primary_selection::PrimarySelectionState;
use smithay::wayland::shell::kde::decoration::KdeDecorationState;
use smithay::wayland::shell::xdg::XdgShellState;
use smithay::wayland::shell::xdg::decoration::XdgDecorationState;
use smithay::wayland::shm::ShmState;
use smithay::reexports::wayland_protocols_misc::server_decoration::server::org_kde_kwin_server_decoration_manager::Mode as KdeDecorationMode;
use smithay::wayland::viewporter::ViewporterState;

use crate::prelude::*;
use crate::serialization::wayland::BufferMetadata;
use crate::serialization::wayland::SurfaceRequest;
use crate::serialization::wayland::SurfaceRequestPayload;
use crate::serialization::wayland::SurfaceState;
use crate::serialization::wayland::WlSurfaceId;
use crate::serialization::CompressionMode;
use crate::serialization::Event;
use crate::serialization::Request;
use crate::serialization::SendType;
use crate::serialization::Serializer;
use crate::sharding_compression::ShardingCompressor;
use crate::utils::SerialMap;

pub mod client_handlers;
pub mod data_device;
pub mod smithay_handlers;

use data_device::WprsDataDeviceState;

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
}

struct LockedSurfaceState(Mutex<TrackedSurfaceState>);

/// Last buffer transmitted to the currently connected client for a surface,
/// retaining a local shadow copy of its pixels so subsequent commits skip
/// unchanged `128x128` quadrants and identical frames.
pub struct SyncedSurfaceBuffer {
    pub metadata: BufferMetadata,
    pub pixels: Vec<u8>,
}

fn surface_destruction_callback(state: &mut WprsServerState, surface: &WlSurface) {
    compositor::with_states(surface, |surface_data| {
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
        if matches!(
            &state.cursor_status,
            Some(SmithayCursorImageStatus::Surface(cursor_surface)) if cursor_surface == surface
        ) {
            state.cursor_status = None;
        }
    });
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

        Self {
            dh: dh.clone(),
            lh,
            compositor_state: CompositorState::new::<Self>(&dh),
            start_time: Instant::now(),
            xwayland_enabled,
            frame_interval,
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
            object_map: HashMap::new(),
            synced_buffers: HashMap::new(),
            dirty_sync_surfaces: HashSet::new(),
            committed_surfaces: HashSet::new(),
            cursor_status: None,
            outputs: HashMap::new(),
            serial_map: SerialMap::new(),
            pressed_keys: HashSet::new(),
            pressed_buttons: HashSet::new(),
            next_transfer_id: 1,
            pending_receive_pipes: HashMap::new(),
        }
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
