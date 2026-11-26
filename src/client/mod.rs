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
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::time::Duration;

use bimap::BiMap;
use enum_as_inner::EnumAsInner;
use smithay::reexports::wayland_protocols::wp::viewporter::client::wp_viewport::WpViewport;
use smithay::reexports::wayland_protocols::wp::viewporter::client::wp_viewporter::WpViewporter;
use smithay_client_toolkit::compositor::CompositorState;
use smithay_client_toolkit::compositor::FrameCallbackData;
use smithay_client_toolkit::compositor::Surface;
use smithay_client_toolkit::data_device_manager::DataDeviceManagerState;
use smithay_client_toolkit::data_device_manager::WritePipe;
use smithay_client_toolkit::data_device_manager::data_offer::DragOffer;
use smithay_client_toolkit::data_device_manager::data_offer::SelectionOffer;
use smithay_client_toolkit::data_device_manager::data_source::CopyPasteSource;
use smithay_client_toolkit::data_device_manager::data_source::DragSource;
use smithay_client_toolkit::output::OutputState;
use smithay_client_toolkit::primary_selection::PrimarySelectionManagerState;
use smithay_client_toolkit::primary_selection::offer::PrimarySelectionOffer;
use smithay_client_toolkit::primary_selection::selection::PrimarySelectionSource;
use smithay_client_toolkit::reexports::client::Connection;
use smithay_client_toolkit::reexports::client::Proxy;
use smithay_client_toolkit::reexports::client::QueueHandle;
use smithay_client_toolkit::reexports::client::backend::ObjectId as SctkObjectId;
use smithay_client_toolkit::reexports::client::globals::GlobalList;
use smithay_client_toolkit::reexports::client::protocol::wl_output::Transform;
use smithay_client_toolkit::reexports::client::protocol::wl_subcompositor::WlSubcompositor;
use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface;
use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_surface;
use smithay_client_toolkit::registry::RegistryState;
use smithay_client_toolkit::registry::SimpleGlobal;
use smithay_client_toolkit::seat::SeatState;
use smithay_client_toolkit::seat::pointer::ThemedPointer;
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shell::xdg::XdgShell;
use smithay_client_toolkit::shell::xdg::XdgSurface;
use smithay_client_toolkit::shm::Shm;
use smithay_client_toolkit::shm::slot::Buffer as SlotBuffer;
use smithay_client_toolkit::shm::slot::Slot;
use smithay_client_toolkit::shm::slot::SlotPool;

use crate::client_utils::FrameDamage;
use crate::client_utils::SeatObject;
use crate::client_utils::SwapchainStep;
use crate::client_utils::advance_two_slot_swapchain;
use crate::client_utils::combine_rect_history;
use crate::client_utils::rect_covered_by;
use crate::client_utils::snapshot_rect_history;
use crate::prelude::*;
use crate::serialization::Capabilities;
use crate::serialization::ClientId;
use crate::serialization::Event;
use crate::serialization::Request;
use crate::serialization::Serializer;
use crate::serialization::framing::TILE_PIXELS;
use crate::serialization::framing::TileRect;
use crate::serialization::wayland::BufferAssignment;
use crate::serialization::wayland::BufferMetadata;
use crate::serialization::wayland::CursorImage;
use crate::serialization::wayland::Region;
use crate::serialization::wayland::SubsurfacePosition;
use crate::serialization::wayland::ViewportState;
use crate::serialization::wayland::WlSurfaceId;
use crate::sharding_compression::ShardingDecompressor;

pub mod server_handlers;
pub mod smithay_handlers;
mod subsurface;
mod xdg_shell;

use smithay_handlers::SubCompositorData;
use subsurface::RemoteSubSurface;
use xdg_shell::RemoteXdgPopup;
use xdg_shell::RemoteXdgToplevel;

type ObjectBimap = BiMap<(ClientId, WlSurfaceId), SctkObjectId>;

pub struct ClientOptions {
    pub title_prefix: String,
}

#[derive(Debug, Default)]
pub struct CapabilitiesLatch {
    state: Mutex<Option<Capabilities>>,
    ready: Condvar,
}

impl CapabilitiesLatch {
    pub fn set(&self, caps: Capabilities) -> Result<()> {
        let mut guard = self.state.lock().unwrap();
        if guard.is_some() {
            bail!("attempted to set capabilities more than once");
        }
        *guard = Some(caps);
        self.ready.notify_all();
        Ok(())
    }

    pub fn get(&self, timeout: Duration) -> Option<Capabilities> {
        let guard = self.state.lock().unwrap();
        let (guard, _) = self
            .ready
            .wait_timeout_while(guard, timeout, |caps| caps.is_none())
            .unwrap();
        guard.clone()
    }
}

pub struct WprsClientState {
    qh: QueueHandle<Self>,
    conn: Connection,
    pub capabilities: Arc<CapabilitiesLatch>,

    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    compositor_state: CompositorState,
    subcompositor: WlSubcompositor,
    shm_state: Shm,
    xdg_shell_state: XdgShell,
    wp_viewporter: Option<SimpleGlobal<WpViewporter, 1>>,

    data_device_manager_state: DataDeviceManagerState,
    primary_selection_manager_state: Option<PrimarySelectionManagerState>,

    pool: SlotPool,

    seat_objects: Vec<SeatObject<ThemedPointer>>,
    selection_source: Option<CopyPasteSource>,
    selection_offer: Option<SelectionOffer>,
    selection_pipe: Option<WritePipe>,
    dnd_source: Option<DragSource>,
    dnd_offer: Option<DragOffer>,
    dnd_pipe: Option<WritePipe>,
    dnd_accept_counter: u32,
    primary_selection_source: Option<PrimarySelectionSource>,
    primary_selection_pipe: Option<WritePipe>,
    primary_selection_offer: Option<PrimarySelectionOffer>,

    serializer: Serializer<Event, Request>,
    remote_display: RemoteDisplay,
    // left: remote object IDs, right: local "native" object IDs
    pub object_bimap: ObjectBimap,

    last_enter_serial: u32,
    last_implicit_grab_serial: Option<u32>,
    last_mouse_down_serial: Option<u32>,
    current_focus: Option<WlSurface>,
    current_cursor_image: Option<CursorImage>,

    title_prefix: String,

    decompressor: ShardingDecompressor,
    slot_copy_scratch: Vec<u8>,
}

impl WprsClientState {
    pub fn new(
        qh: QueueHandle<Self>,
        globals: GlobalList,
        conn: Connection,
        serializer: Serializer<Event, Request>,
        options: ClientOptions,
    ) -> Result<Self> {
        let shm_state = Shm::bind(&globals, &qh).context(loc!(), "wl_shm is not available")?;

        // size doesn't really matter, the pool will be automatically grown as
        // necessary.
        let pool =
            SlotPool::new(3840 * 2160, &shm_state).context(loc!(), "failed to create pool")?;

        Ok(Self {
            qh: qh.clone(),
            conn,
            capabilities: Arc::default(),
            registry_state: RegistryState::new(&globals),
            seat_state: SeatState::new(&globals, &qh),
            output_state: OutputState::new(&globals, &qh),
            compositor_state: CompositorState::bind(&globals, &qh)
                .context(loc!(), "wl_compositor is not available")?,
            subcompositor: globals
                .bind(&qh, 1..=1, SubCompositorData)
                .context(loc!(), "wl_subcompositor is not available")?,
            shm_state,
            xdg_shell_state: XdgShell::bind(&globals, &qh)
                .context(loc!(), "xdg shell is not available")?,
            wp_viewporter: SimpleGlobal::<WpViewporter, 1>::bind(&globals, &qh)
                .context(loc!(), "wp_viewporter is not available")
                .warn(loc!())
                .ok(),
            data_device_manager_state: DataDeviceManagerState::bind(&globals, &qh)
                .context(loc!(), "data device manager is not available")?,
            primary_selection_manager_state: PrimarySelectionManagerState::bind(&globals, &qh)
                .context(loc!(), "primary selection manager is not available")
                .warn(loc!())
                .ok(),

            pool,

            seat_objects: Vec::new(),
            selection_source: None,
            selection_offer: None,
            selection_pipe: None,
            dnd_source: None,
            dnd_offer: None,
            dnd_pipe: None,
            dnd_accept_counter: 0,
            primary_selection_source: None,
            primary_selection_offer: None,
            primary_selection_pipe: None,

            serializer,
            remote_display: RemoteDisplay::new(),
            object_bimap: BiMap::new(),

            last_enter_serial: 0,
            last_implicit_grab_serial: None,
            last_mouse_down_serial: None,
            current_focus: None,
            current_cursor_image: None,
            title_prefix: options.title_prefix,
            decompressor: ShardingDecompressor::new(NonZeroUsize::new(8).unwrap())
                .location(loc!())?,
            slot_copy_scratch: Vec::new(),
        })
    }
}

const SLOT_COPY_CHUNK: usize = TILE_PIXELS * 4;

enum SlotCopyRegion<'a> {
    Full,
    Rects(&'a [TileRect]),
}

fn copy_slot_region_with<P, S>(
    pool: &mut P,
    src_slot: &S,
    dst_slot: &S,
    byte_len: usize,
    stride: usize,
    region: SlotCopyRegion<'_>,
    scratch: &mut Vec<u8>,
    mut slot_data: impl for<'a> FnMut(&'a mut P, &S) -> &'a mut [u8],
) -> usize {
    let mut bytes_copied = 0;
    match region {
        SlotCopyRegion::Full => {
            let mut offset = 0;
            while offset < byte_len {
                let chunk = (byte_len - offset).min(SLOT_COPY_CHUNK);
                if scratch.len() < chunk {
                    scratch.resize(SLOT_COPY_CHUNK, 0);
                }
                scratch[..chunk]
                    .copy_from_slice(&slot_data(pool, src_slot)[offset..offset + chunk]);
                slot_data(pool, dst_slot)[offset..offset + chunk]
                    .copy_from_slice(&scratch[..chunk]);
                bytes_copied += chunk;
                offset += chunk;
            }
        },
        SlotCopyRegion::Rects(rects) => {
            for &rect in rects {
                let x_bytes = rect.x as usize * 4;
                let row_bytes = rect.w as usize * 4;
                let rows_per_batch = (SLOT_COPY_CHUNK / row_bytes).max(1);
                let h = rect.h as usize;
                let mut ry = 0;
                while ry < h {
                    let batch_rows = (h - ry).min(rows_per_batch);
                    let batch_bytes = batch_rows * row_bytes;
                    if scratch.len() < batch_bytes {
                        scratch.resize(batch_bytes.max(SLOT_COPY_CHUNK), 0);
                    }
                    {
                        let src = slot_data(pool, src_slot);
                        for i in 0..batch_rows {
                            let src_off = (rect.y as usize + ry + i) * stride + x_bytes;
                            scratch[i * row_bytes..(i + 1) * row_bytes]
                                .copy_from_slice(&src[src_off..src_off + row_bytes]);
                        }
                    }
                    {
                        let dst = slot_data(pool, dst_slot);
                        for i in 0..batch_rows {
                            let dst_off = (rect.y as usize + ry + i) * stride + x_bytes;
                            dst[dst_off..dst_off + row_bytes]
                                .copy_from_slice(&scratch[i * row_bytes..(i + 1) * row_bytes]);
                        }
                    }
                    bytes_copied += batch_bytes;
                    ry += batch_rows;
                }
            }
        },
    }
    bytes_copied
}

fn copy_slot_region(
    pool: &mut SlotPool,
    src_slot: &Slot,
    dst_slot: &Slot,
    byte_len: usize,
    stride: usize,
    region: SlotCopyRegion<'_>,
    scratch: &mut Vec<u8>,
) {
    copy_slot_region_with(
        pool,
        src_slot,
        dst_slot,
        byte_len,
        stride,
        region,
        scratch,
        |pool, slot| pool.raw_data_mut(slot),
    );
}

fn prepare_remote_swapchain<P, B>(
    pool: &mut P,
    active_buffer: &mut B,
    back_buffer: &mut Option<(B, Option<Vec<TileRect>>)>,
    cur_rects: Option<&[TileRect]>,
    is_released: impl FnMut(&mut P, &B) -> bool,
    alloc_buffer: impl FnOnce(&mut P, bool) -> Result<B>,
    copy_region: impl FnOnce(&mut P, &B, &B, SlotCopyRegion<'_>),
) -> Result<()> {
    let step = advance_two_slot_swapchain(
        pool,
        active_buffer,
        back_buffer,
        cur_rects,
        is_released,
        combine_rect_history,
        snapshot_rect_history,
        alloc_buffer,
    )?;
    if let Some(cur) = cur_rects {
        let prev_rects = match step {
            SwapchainStep::ReusedActive => return Ok(()),
            SwapchainStep::Swapped { prev_history } => prev_history,
            SwapchainStep::Allocated => None,
        };
        let old_active = &back_buffer.as_ref().unwrap().0;
        match prev_rects {
            Some(prev) => {
                let to_copy: Vec<TileRect> = prev
                    .into_iter()
                    .filter(|&r| !rect_covered_by(r, cur))
                    .collect();
                if !to_copy.is_empty() {
                    copy_region(
                        pool,
                        old_active,
                        active_buffer,
                        SlotCopyRegion::Rects(&to_copy),
                    );
                }
            },
            None => copy_region(pool, old_active, active_buffer, SlotCopyRegion::Full),
        }
    }
    Ok(())
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RemoteBufferState {
    /// Decoded from `RawBuffer` and waiting for its matching `Commit`.
    Prepared,
    /// Promoted by `Commit` and waiting to be attached to the local `WlSurface`.
    Dirty,
    /// Attached and committed to the local `WlSurface`.
    Clean,
}

#[derive(Debug)]
pub struct RemoteBuffer {
    pub metadata: BufferMetadata,
    pub active_buffer: SlotBuffer,
    /// Previous swapchain slot and the 2D tile rects written to `active_buffer`
    /// since this slot was last current (`None` requires a full copy from
    /// `active_buffer` on next incremental reuse).
    back_buffer: Option<(SlotBuffer, Option<Vec<TileRect>>)>,
    pub state: RemoteBufferState,
}

impl RemoteBuffer {
    pub fn new(metadata: BufferMetadata, pool: &mut SlotPool) -> Result<Self> {
        let active_buffer = pool
            .create_buffer(
                metadata.width,
                metadata.height,
                metadata.stride,
                metadata.format.into(),
            )
            .location(loc!())?
            .0;

        Ok(Self {
            metadata,
            active_buffer,
            back_buffer: None,
            state: RemoteBufferState::Prepared,
        })
    }

    fn canvas<'a>(
        &mut self,
        incremental_rects: Option<&[TileRect]>,
        pool: &'a mut SlotPool,
        copy_scratch: &mut Vec<u8>,
    ) -> Result<&'a mut [u8]> {
        let metadata = self.metadata;
        let stride = metadata.stride as usize;
        let byte_len = metadata.height as usize * stride;
        prepare_remote_swapchain(
            pool,
            &mut self.active_buffer,
            &mut self.back_buffer,
            incremental_rects,
            |pool, buf| pool.canvas(buf).is_some(),
            |pool, back_buffer_held| {
                debug!(
                    back_buffer_held,
                    incremental = incremental_rects.is_some(),
                    tiles = incremental_rects.map_or(0, <[_]>::len),
                    width = metadata.width,
                    height = metadata.height,
                    "creating new buffer",
                );
                Ok(pool
                    .create_buffer(
                        metadata.width,
                        metadata.height,
                        metadata.stride,
                        metadata.format.into(),
                    )
                    .location(loc!())?
                    .0)
            },
            |pool, src_buf, dst_buf, region| {
                copy_slot_region(
                    pool,
                    &src_buf.slot(),
                    &dst_buf.slot(),
                    byte_len,
                    stride,
                    region,
                    copy_scratch,
                );
            },
        )?;
        pool.canvas(&self.active_buffer).location(loc!())
    }
}

#[derive(Debug, EnumAsInner)]
pub enum Role {
    Cursor,
    SubSurface(RemoteSubSurface),
    XdgToplevel(RemoteXdgToplevel),
    XdgPopup(RemoteXdgPopup),
}

impl WaylandSurface for RemoteSurface {
    fn wl_surface(&self) -> &WlSurface {
        match &self.role {
            None | Some(Role::Cursor) => self.local_surface.as_ref().unwrap().wl_surface(),
            Some(Role::SubSurface(remote_subsurface)) => {
                remote_subsurface.local_surface.wl_surface()
            },
            Some(Role::XdgToplevel(remote_xdg_toplevel)) => {
                remote_xdg_toplevel.local_window.wl_surface()
            },
            Some(Role::XdgPopup(remote_xdg_popup)) => remote_xdg_popup.local_popup.wl_surface(),
        }
    }
}

impl WaylandSurface for RemoteXdgToplevel {
    fn wl_surface(&self) -> &WlSurface {
        self.local_window.wl_surface()
    }
}

impl XdgSurface for RemoteXdgToplevel {
    fn xdg_surface(&self) -> &xdg_surface::XdgSurface {
        self.local_window.xdg_surface()
    }
}

impl WaylandSurface for RemoteXdgPopup {
    fn wl_surface(&self) -> &WlSurface {
        self.local_popup.wl_surface()
    }
}

impl XdgSurface for RemoteXdgPopup {
    fn xdg_surface(&self) -> &xdg_surface::XdgSurface {
        self.local_popup.xdg_surface()
    }
}

#[derive(Debug)]
pub struct RemoteSurface {
    pub client: ClientId,
    pub id: WlSurfaceId,
    pub buffer: Option<RemoteBuffer>,
    // None when the surface is owned by a role object (e.g., a Window).
    pub local_surface: Option<Surface>,
    pub role: Option<Role>,
    pub opaque_region: Option<Region>,
    pub input_region: Option<Region>,
    pub z_ordered_children: Vec<SubsurfacePosition>,
    pub frame_callback_completed: bool,
    pub frame_damage: FrameDamage,
    pub viewport: Option<WpViewport>,
    pub current_viewport_state: Option<ViewportState>,
}

impl RemoteSurface {
    pub fn new(
        client_id: ClientId,
        id: WlSurfaceId,
        compositor_state: &CompositorState,
        qh: &QueueHandle<WprsClientState>,
        object_bimap: &mut ObjectBimap,
    ) -> Result<Self> {
        let local_surface = Some(Surface::new(compositor_state, qh).location(loc!())?);

        object_bimap.insert(
            (client_id, id),
            local_surface.as_ref().location(loc!())?.wl_surface().id(),
        );

        Ok(Self {
            client: client_id,
            id,
            buffer: None,
            local_surface,
            role: None,
            opaque_region: None,
            input_region: None,
            z_ordered_children: vec![SubsurfacePosition {
                id,
                position: (0, 0).into(),
            }],
            frame_callback_completed: true,
            frame_damage: FrameDamage::Empty,
            viewport: None,
            current_viewport_state: None,
        })
    }

    fn reorder_children(
        &mut self,
        new_order: &[SubsurfacePosition],
    ) -> Vec<(WlSurfaceId, WlSurfaceId)> {
        let mut moves = Vec::new();

        let mut new_order = new_order.to_vec();
        new_order.reverse();

        debug!(
            "REORDER_CHILDREN, {:?}, {:?}, {:?}",
            &self.id, &self.z_ordered_children, &new_order
        );

        let z_ordered_children_set: HashSet<WlSurfaceId> =
            self.z_ordered_children.iter().map(|c| c.id).collect();

        let new_order: Vec<SubsurfacePosition> = new_order
            .iter()
            .filter(|elem| z_ordered_children_set.contains(&elem.id))
            .cloned()
            .collect();

        for (idx, elem) in new_order.iter().enumerate() {
            let current_elem = self.z_ordered_children[idx];
            if current_elem.id == elem.id {
                self.z_ordered_children[idx] = *elem; // position may have changed
                continue;
            }

            let current_idx = self
                .z_ordered_children
                .iter()
                .position(|x| x.id == elem.id)
                .unwrap(); // we already filtered out non-present elements

            self.z_ordered_children.remove(current_idx);
            // Insert elem instead of child because position may have changed.
            self.z_ordered_children.insert(idx, *elem);
            moves.push((elem.id, current_elem.id));
        }

        moves
    }

    fn has_in_flight_buffer(&self) -> bool {
        matches!(
            &self.buffer,
            Some(buffer) if buffer.state == RemoteBufferState::Prepared
        )
    }

    #[instrument(skip(self, pool, copy_scratch), level = "debug")]
    pub(crate) fn prepare_buffer_canvas<'a>(
        &mut self,
        metadata: BufferMetadata,
        incremental_rects: Option<&[TileRect]>,
        pool: &'a mut SlotPool,
        copy_scratch: &mut Vec<u8>,
    ) -> Result<&'a mut [u8]> {
        let buffer = if let Some(buffer) = &mut self.buffer
            && buffer.metadata == metadata
        {
            buffer.state = RemoteBufferState::Prepared;
            buffer
        } else {
            if incremental_rects.is_some() {
                bail!(
                    "received incremental tile update without a matching buffer ({:?} -> {metadata:?})",
                    self.buffer.as_ref().map(|b| b.metadata)
                );
            }
            self.frame_damage = FrameDamage::Full;
            self.buffer
                .insert(RemoteBuffer::new(metadata, pool).location(loc!())?)
        };
        buffer
            .canvas(incremental_rects, pool, copy_scratch)
            .location(loc!())
    }

    pub(crate) fn drop_buffer(&mut self) {
        self.buffer = None;
        self.frame_damage = FrameDamage::Empty;
    }

    #[instrument(skip(self), level = "debug")]
    fn clear_buffer(&mut self) {
        let wl_surface = self.wl_surface().clone();
        self.drop_buffer();
        wl_surface.attach(None, 0, 0);
    }

    #[instrument(skip(self), level = "debug")]
    pub fn apply_buffer(&mut self, new_buffer: Option<BufferAssignment>) -> Result<()> {
        match new_buffer {
            Some(BufferAssignment::New(new_metadata)) => {
                let Some(buffer) = &mut self.buffer else {
                    return Err(anyhow!(
                        "Received buffer commit with empty data. This can happen if wprsc reattaches between wprsd sending a buffer message and a commit message."
                    ));
                };
                if buffer.state != RemoteBufferState::Prepared || buffer.metadata != new_metadata {
                    bail!(
                        "Prepared buffer ({:?}, {:?}) did not match commit ({new_metadata:?})",
                        buffer.state,
                        buffer.metadata,
                    );
                }
                buffer.state = RemoteBufferState::Dirty;
            },
            Some(BufferAssignment::Removed) => {
                self.clear_buffer();
            },
            None => {
                if self.has_in_flight_buffer() {
                    bail!(
                        "received non-buffer Commit while a prepared RawBuffer was awaiting Commit"
                    );
                }
            },
        }
        Ok(())
    }

    fn draw_buffer_impl(&mut self, frame_qh: Option<&QueueHandle<WprsClientState>>) -> Result<()> {
        // Don't commit while a decoded `Prepared` buffer awaits its
        // `Request::Surface(Commit)`: a bare commit would apply surface state
        // left pending by an earlier deferred `handle_commit` on top of the
        // previously attached buffer; `apply_buffer` promotes `Prepared -> Dirty`
        // and the next draw commits both together.
        if self.has_in_flight_buffer() {
            return Ok(());
        }
        let wl_surface = &self.wl_surface().clone();
        let attached = if let Some(buffer) = &mut self.buffer
            && buffer.state == RemoteBufferState::Dirty
        {
            buffer.active_buffer.attach_to(wl_surface).context(
                loc!(),
                "attaching a buffer failed, this probably means we're leaking buffers",
            )?;
            self.frame_damage.apply_and_clear(wl_surface);
            buffer.state = RemoteBufferState::Clean;
            true
        } else {
            false
        };
        if attached && let Some(qh) = frame_qh {
            self.frame(qh);
            self.frame_callback_completed = false;
        }
        self.commit();
        Ok(())
    }

    pub fn draw_buffer(&mut self) -> Result<()> {
        self.draw_buffer_impl(None)
    }

    pub fn draw_buffer_send_frame(&mut self, qh: &QueueHandle<WprsClientState>) -> Result<()> {
        self.draw_buffer_impl(Some(qh))
    }

    pub fn set_transformation(&mut self, scale: i32, transform: Option<Transform>) {
        self.wl_surface().set_buffer_scale(scale);
        if let Some(transform) = transform {
            self.wl_surface().set_buffer_transform(transform);
        }
    }

    pub fn set_viewport_state(
        &mut self,
        viewport_state: Option<ViewportState>,
        wp_viewporter: &Option<SimpleGlobal<WpViewporter, 1>>,
        qh: &QueueHandle<WprsClientState>,
    ) {
        let Some(wp_viewporter) = wp_viewporter else {
            return;
        };
        let Ok(wp_viewporter) = wp_viewporter.get() else {
            return;
        };

        // None still has to clear a viewport we set earlier.
        let viewport_state = viewport_state.unwrap_or(ViewportState {
            src: None,
            dst: None,
        });

        // skip if the viewport state hasn't changed
        if self.current_viewport_state == Some(viewport_state) {
            return;
        }

        // Don't create a viewport object just to unset a viewport we never set.
        if self.viewport.is_none() && viewport_state.src.is_none() && viewport_state.dst.is_none() {
            self.current_viewport_state = Some(viewport_state);
            return;
        }

        let wl_surface = self.wl_surface().clone();
        let viewport = self
            .viewport
            .get_or_insert_with(|| wp_viewporter.get_viewport(&wl_surface, qh, ()));

        // wp_viewport keeps the previous source/destination until explicitly
        // unset with the -1 sentinels.
        match viewport_state.src {
            Some(src) => viewport.set_source(src.loc.x, src.loc.y, src.size.w, src.size.h),
            None => viewport.set_source(-1.0, -1.0, -1.0, -1.0),
        }
        match viewport_state.dst {
            Some(dst) => viewport.set_destination(dst.w, dst.h),
            None => viewport.set_destination(-1, -1),
        }
        self.current_viewport_state = Some(viewport_state);
    }

    pub fn set_input_region(
        &mut self,
        region: Option<Region>,
        compositor_state: &CompositorState,
    ) -> Result<()> {
        if self.input_region == region {
            return Ok(());
        }

        self.input_region = region;

        if let Some(region) = &self.input_region {
            self.wl_surface().set_input_region(Some(
                region
                    .create_compositor_region(compositor_state)
                    .location(loc!())?
                    .wl_region(),
            ));
        } else {
            self.wl_surface().set_input_region(None);
        }
        Ok(())
    }

    pub fn set_opaque_region(
        &mut self,
        region: Option<Region>,
        compositor_state: &CompositorState,
    ) -> Result<()> {
        if self.opaque_region == region {
            return Ok(());
        }

        self.opaque_region = region;

        if let Some(region) = &self.opaque_region {
            self.wl_surface().set_opaque_region(Some(
                region
                    .create_compositor_region(compositor_state)
                    .location(loc!())?
                    .wl_region(),
            ));
        } else {
            self.wl_surface().set_opaque_region(None);
        }
        Ok(())
    }

    pub fn commit(&mut self) {
        self.wl_surface().commit();
    }

    pub fn frame(&self, qh: &QueueHandle<WprsClientState>) {
        self.wl_surface()
            .frame(qh, FrameCallbackData(self.wl_surface().clone()));
    }

    pub fn get_role(&self) -> Result<&Role> {
        self.role.as_ref().context(loc!(), "Role was None.")
    }

    pub fn get_mut_role(&mut self) -> Result<&mut Role> {
        self.role.as_mut().context(loc!(), "Role was None.")
    }

    pub fn xdg_surface(&self) -> Option<xdg_surface::XdgSurface> {
        match &self.role {
            Some(Role::XdgToplevel(toplevel)) => Some(toplevel.xdg_surface().clone()),
            Some(Role::XdgPopup(popup)) => Some(popup.xdg_surface().clone()),
            _ => None,
        }
    }

    pub fn xdg_toplevel(&self) -> Result<&RemoteXdgToplevel> {
        self.get_role()
            .location(loc!())?
            .as_xdg_toplevel()
            .context(loc!(), "Role was not XdgToplevel.")
    }

    pub fn xdg_popup(&self) -> Result<&RemoteXdgPopup> {
        self.get_role()
            .location(loc!())?
            .as_xdg_popup()
            .context(loc!(), "Role was not XdgPopup.")
    }

    pub fn xdg_popup_mut(&mut self) -> Result<&mut RemoteXdgPopup> {
        self.get_mut_role()
            .location(loc!())?
            .as_xdg_popup_mut()
            .context(loc!(), "Role was not XdgPopup.")
    }
}

impl Drop for RemoteSurface {
    fn drop(&mut self) {
        if let Some(viewport) = &self.viewport {
            viewport.destroy();
        }
    }
}

#[derive(Debug)]
pub struct RemoteClient {
    pub id: ClientId,
    pub surfaces: HashMap<WlSurfaceId, RemoteSurface>,
}

impl RemoteClient {
    pub fn new(id: ClientId) -> Self {
        Self {
            id,
            surfaces: HashMap::new(),
        }
    }

    pub fn surface(&mut self, id: &WlSurfaceId) -> Result<&mut RemoteSurface> {
        self.surfaces
            .get_mut(id)
            .with_context(loc!(), || format!("Unknown surface id: {id:?}"))
    }
}

#[derive(Debug)]
pub struct RemoteDisplay {
    pub clients: HashMap<ClientId, RemoteClient>,
}

impl RemoteDisplay {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
        }
    }

    pub fn client(&mut self, id: &ClientId) -> &mut RemoteClient {
        self.clients.entry(*id).or_insert(RemoteClient::new(*id))
    }
}

impl Default for RemoteDisplay {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tr(x: u32, y: u32, w: u32, h: u32) -> TileRect {
        TileRect { x, y, w, h }
    }

    fn patch_rect(dst: &mut [u8], src: &[u8], stride: usize, rect: TileRect) {
        let x_bytes = rect.x as usize * 4;
        let row_bytes = rect.w as usize * 4;
        for ry in 0..rect.h as usize {
            let off = (rect.y as usize + ry) * stride + x_bytes;
            dst[off..off + row_bytes].copy_from_slice(&src[off..off + row_bytes]);
        }
    }

    fn assert_visible_eq(got: &[u8], expected: &[u8], width: usize, height: usize, stride: usize) {
        let vis = width * 4;
        for y in 0..height {
            let off = y * stride;
            assert_eq!(
                &got[off..off + vis],
                &expected[off..off + vis],
                "visible pixel mismatch on row {y}"
            );
        }
    }

    fn apply_remote_frame(
        active: &mut usize,
        back: &mut Option<(usize, Option<Vec<TileRect>>)>,
        slots: &mut Vec<Vec<u8>>,
        scratch: &mut Vec<u8>,
        released: &[bool],
        frame: &[u8],
        incremental_rects: Option<&[TileRect]>,
        stride: usize,
    ) -> (usize, Option<bool>) {
        let byte_len = frame.len();
        let mut bytes_copied = 0usize;
        let mut alloc_back_held = None;

        prepare_remote_swapchain(
            slots,
            active,
            back,
            incremental_rects,
            |_, &idx| released[idx],
            |slots, back_buffer_held| {
                alloc_back_held = Some(back_buffer_held);
                let idx = slots.len();
                // Poison new slots so any uncopied pixel fails verification.
                slots.push(vec![0xAAu8; byte_len]);
                Ok(idx)
            },
            |slots, &src_idx, &dst_idx, region| {
                bytes_copied += copy_slot_region_with(
                    slots,
                    &src_idx,
                    &dst_idx,
                    byte_len,
                    stride,
                    region,
                    scratch,
                    |slots, &idx| &mut slots[idx],
                );
            },
        )
        .unwrap();

        // Simulate `decompress_and_unfilter_streaming` writing only `incremental_rects`
        // (or the full frame when `None`).
        let canvas = &mut slots[*active];
        match incremental_rects {
            Some(rects) => {
                for &rect in rects {
                    patch_rect(canvas, frame, stride, rect);
                }
            },
            None => canvas.copy_from_slice(frame),
        }

        (bytes_copied, alloc_back_held)
    }

    #[test]
    fn test_copy_slot_region_multi_chunk_full_and_rect() {
        // Multi-chunk full copy (`> SLOT_COPY_CHUNK`) + single-rect copy with padded stride.
        let width = 300usize;
        let height = 260usize;
        let stride = width * 4 + 16;
        let byte_len = height * stride;
        let src: Vec<u8> = (0..byte_len).map(|i| ((i * 29 + 7) & 0xFF) as u8).collect();
        let mut slots = vec![src.clone(), vec![0xCCu8; byte_len]];
        let mut scratch = Vec::new();

        let rect = tr(16, 10, 256, 200);
        let copied = copy_slot_region_with(
            &mut slots,
            &0,
            &1,
            byte_len,
            stride,
            SlotCopyRegion::Rects(&[rect]),
            &mut scratch,
            |s, &idx| &mut s[idx],
        );
        assert_eq!(copied, 256 * 200 * 4);
        for y in 0..height {
            for x in 0..width {
                let off = y * stride + x * 4;
                let inside = (16..272).contains(&x) && (10..210).contains(&y);
                let exp = if inside {
                    &src[off..off + 4]
                } else {
                    &[0xCC; 4]
                };
                assert_eq!(&slots[1][off..off + 4], exp, "px ({x},{y})");
            }
            assert_eq!(
                &slots[1][y * stride + width * 4..(y + 1) * stride],
                &[0xCC; 16],
                "stride padding touched on row {y}"
            );
        }

        slots[1].fill(0x55);
        let full_copied = copy_slot_region_with(
            &mut slots,
            &0,
            &1,
            byte_len,
            stride,
            SlotCopyRegion::Full,
            &mut scratch,
            |s, &idx| &mut s[idx],
        );
        assert_eq!(full_copied, byte_len);
        assert_eq!(slots[1], src);
    }

    #[test]
    fn test_remote_buffer_swapchain_2d_rect_sync() {
        let width = 512usize;
        let height = 384usize;
        let stride = width * 4 + 16; // padded stride
        let byte_len = height * stride;

        let make_frame = |seed: u8| -> Vec<u8> {
            (0..byte_len)
                .map(|i| seed.wrapping_mul(37).wrapping_add((i % 251) as u8))
                .collect()
        };
        let patch = |frame: &mut [u8], seed: u8, rects: &[TileRect]| {
            let src = make_frame(seed);
            for &rect in rects {
                patch_rect(frame, &src, stride, rect);
            }
        };

        let q_left = tr(0, 0, 128, 128);
        let q_right_same_band = tr(256, 0, 128, 128);
        let q_parent_left = tr(0, 0, 256, 256);
        let q_bottom = tr(128, 256, 128, 128);

        let mut frame = make_frame(1);
        let mut slots = vec![frame.clone()];
        let mut scratch = Vec::new();
        let mut active = 0usize;
        let mut back: Option<(usize, Option<Vec<TileRect>>)> = None;

        // 1. In-place reuse keeps `back_buffer == None` (zero extra slot or copy).
        patch(&mut frame, 2, &[q_left]);
        let (copied, alloc) = apply_remote_frame(
            &mut active,
            &mut back,
            &mut slots,
            &mut scratch,
            &[true],
            &frame,
            Some(&[q_left]),
            stride,
        );
        assert_eq!((active, slots.len(), copied, alloc), (0, 1, 0, None));
        assert_eq!(back, None);
        assert_visible_eq(&slots[active], &frame, width, height, stride);

        // 2. First miss allocates Slot B (`back_buffer_held == false`), copies
        // full Slot A -> Slot B, then applies `q_left` to Slot B. Slot A is now
        // stale at `[q_left]`.
        patch(&mut frame, 3, &[q_left]);
        let (copied, alloc) = apply_remote_frame(
            &mut active,
            &mut back,
            &mut slots,
            &mut scratch,
            &[false],
            &frame,
            Some(&[q_left]),
            stride,
        );
        assert_eq!(
            (active, slots.len(), copied, alloc),
            (1, 2, byte_len, Some(false))
        );
        assert_eq!(back, Some((0, Some(vec![q_left]))));
        assert_visible_eq(&slots[active], &frame, width, height, stride);

        // 3. CRITICAL 2D regression test: Frame 1 modified `(0, 0, 128, 128)` on
        // active Slot B; Frame 2 swaps to Slot A and modifies `(256, 0, 128, 128)`
        // in the SAME row band (`y = 0..128`) -> must copy `(0, 0, 128, 128)`
        // from Slot B to Slot A so Slot A has bit-exact pixels at BOTH rects!
        patch(&mut frame, 4, &[q_right_same_band]);
        let (copied, alloc) = apply_remote_frame(
            &mut active,
            &mut back,
            &mut slots,
            &mut scratch,
            &[true, false],
            &frame,
            Some(&[q_right_same_band]),
            stride,
        );
        assert_eq!(
            (active, slots.len(), copied, alloc),
            (0, 2, 128 * 128 * 4, None)
        );
        assert_eq!(back, Some((1, Some(vec![q_right_same_band]))));
        assert_visible_eq(&slots[active], &frame, width, height, stride);

        // 4. Cursor-blink alternating swap (`prev == cur`) copies 0 bytes!
        patch(&mut frame, 5, &[q_right_same_band]);
        let (copied, alloc) = apply_remote_frame(
            &mut active,
            &mut back,
            &mut slots,
            &mut scratch,
            &[false, true],
            &frame,
            Some(&[q_right_same_band]),
            stride,
        );
        assert_eq!((active, slots.len(), copied, alloc), (1, 2, 0, None));
        assert_eq!(back, Some((0, Some(vec![q_right_same_band]))));
        assert_visible_eq(&slots[active], &frame, width, height, stride);

        // 5. In-place reuse accumulates `q_left`; subsequent swap writing
        // `q_parent_left` covers `q_left` so only `q_right_same_band` is copied!
        patch(&mut frame, 6, &[q_left]);
        let (copied, alloc) = apply_remote_frame(
            &mut active,
            &mut back,
            &mut slots,
            &mut scratch,
            &[false, true],
            &frame,
            Some(&[q_left]),
            stride,
        );
        assert_eq!((active, slots.len(), copied, alloc), (1, 2, 0, None));
        assert_eq!(back, Some((0, Some(vec![q_right_same_band, q_left]))));
        assert_visible_eq(&slots[active], &frame, width, height, stride);

        patch(&mut frame, 7, &[q_parent_left]);
        let (copied, alloc) = apply_remote_frame(
            &mut active,
            &mut back,
            &mut slots,
            &mut scratch,
            &[true, false],
            &frame,
            Some(&[q_parent_left]),
            stride,
        );
        assert_eq!(
            (active, slots.len(), copied, alloc),
            (0, 2, 128 * 128 * 4, None)
        );
        assert_eq!(back, Some((1, Some(vec![q_parent_left]))));
        assert_visible_eq(&slots[active], &frame, width, height, stride);

        // 6. Full-frame update (`None`) on active invalidates `back` to `None`;
        // next incremental swap copies full `old_active` without a 3rd alloc.
        frame = make_frame(8);
        let (copied, alloc) = apply_remote_frame(
            &mut active,
            &mut back,
            &mut slots,
            &mut scratch,
            &[true, false],
            &frame,
            None,
            stride,
        );
        assert_eq!((active, slots.len(), copied, alloc), (0, 2, 0, None));
        assert_eq!(back, Some((1, None)));
        assert_visible_eq(&slots[active], &frame, width, height, stride);

        patch(&mut frame, 9, &[q_bottom]);
        let (copied, alloc) = apply_remote_frame(
            &mut active,
            &mut back,
            &mut slots,
            &mut scratch,
            &[false, true],
            &frame,
            Some(&[q_bottom]),
            stride,
        );
        assert_eq!((active, slots.len(), copied, alloc), (1, 2, byte_len, None));
        assert_eq!(back, Some((0, Some(vec![q_bottom]))));
        assert_visible_eq(&slots[active], &frame, width, height, stride);

        // 7. Full-frame update on swap copies 0 bytes and marks `back = None`.
        frame = make_frame(10);
        let (copied, alloc) = apply_remote_frame(
            &mut active,
            &mut back,
            &mut slots,
            &mut scratch,
            &[true, false],
            &frame,
            None,
            stride,
        );
        assert_eq!((active, slots.len(), copied, alloc), (0, 2, 0, None));
        assert_eq!(back, Some((1, None)));
        assert_visible_eq(&slots[active], &frame, width, height, stride);

        // 8. Both slots held -> triple-buffer fallback (`back_buffer_held == true`)
        // allocates a 3rd slot and copies full `old_active` on incremental.
        patch(&mut frame, 11, &[q_left]);
        let (copied, alloc) = apply_remote_frame(
            &mut active,
            &mut back,
            &mut slots,
            &mut scratch,
            &[false, false],
            &frame,
            Some(&[q_left]),
            stride,
        );
        assert_eq!(
            (active, slots.len(), copied, alloc),
            (2, 3, byte_len, Some(true))
        );
        assert_eq!(back, Some((0, Some(vec![q_left]))));
        assert_visible_eq(&slots[active], &frame, width, height, stride);

        // 9. Both slots held on a full-frame update allocates with 0-byte copy.
        frame = make_frame(12);
        let (copied, alloc) = apply_remote_frame(
            &mut active,
            &mut back,
            &mut slots,
            &mut scratch,
            &[false, false, false],
            &frame,
            None,
            stride,
        );
        assert_eq!((active, slots.len(), copied, alloc), (3, 4, 0, Some(true)));
        assert_eq!(back, Some((2, None)));
        assert_visible_eq(&slots[active], &frame, width, height, stride);
    }

    #[test]
    fn test_capabilities_latch_single_assignment() {
        let latch = CapabilitiesLatch::default();
        assert_eq!(latch.get(Duration::ZERO), None);

        latch.set(Capabilities { xwayland: true }).unwrap();
        assert_eq!(
            latch.get(Duration::ZERO),
            Some(Capabilities { xwayland: true })
        );

        assert!(latch.set(Capabilities { xwayland: false }).is_err());
        assert_eq!(
            latch.get(Duration::ZERO),
            Some(Capabilities { xwayland: true })
        );
    }
}
