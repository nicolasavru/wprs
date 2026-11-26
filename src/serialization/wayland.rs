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

use std::fmt;
use std::fmt::Debug;
use std::num::NonZeroU32;

use anyhow::Error;
use enum_as_inner::EnumAsInner;
use rkyv::Archive;
use rkyv::Deserialize;
use rkyv::Serialize;
use smithay::backend::input::AxisRelativeDirection as SmithayAxisRelativeDirection;
use smithay::backend::input::AxisSource as SmithayAxisSource;
use smithay::output::Subpixel as SmithaySubpixel;
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::backend;
use smithay::reexports::wayland_server::protocol::wl_output::Transform as SmithayWlTransform;
use smithay::reexports::wayland_server::protocol::wl_shm::Format as SmithayBufferFormat;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::Transform as SmithayTransform;
use smithay::wayland::compositor::RectangleKind as SmithayRectangleKind;
use smithay::wayland::compositor::RegionAttributes;
use smithay::wayland::shm::BufferData as SmithayBufferData;
use smithay::wayland::viewporter::ViewportCachedState;
use smithay_client_toolkit::compositor::CompositorState;
use smithay_client_toolkit::compositor::Region as SctkRegion;
use smithay_client_toolkit::output::Mode as SctkMode;
use smithay_client_toolkit::output::OutputInfo as SctkOutputInfo;
use smithay_client_toolkit::reexports::client::protocol::wl_output::Subpixel as SctkSubpixel;
use smithay_client_toolkit::reexports::client::protocol::wl_output::Transform as SctkTransform;
use smithay_client_toolkit::reexports::client::protocol::wl_pointer::AxisRelativeDirection as SctkAxisRelativeDirection;
use smithay_client_toolkit::reexports::client::protocol::wl_pointer::AxisSource as SctkAxisSource;
use smithay_client_toolkit::reexports::client::protocol::wl_shm::Format as SctkBufferFormat;
use smithay_client_toolkit::seat::keyboard::Modifiers as SmithayModifiers;
use smithay_client_toolkit::seat::keyboard::RepeatInfo as SctkRepeatInfo;
use smithay_client_toolkit::seat::pointer::AxisScroll as SctkAxisScroll;
use smithay_client_toolkit::seat::pointer::PointerEvent as SctkPointerEvent;
use smithay_client_toolkit::seat::pointer::PointerEventKind as SctkPointerEventKind;

use super::tuple::Tuple2;
use crate::args;
use crate::prelude::*;
use crate::serialization;
use crate::serialization::ClientId;
use crate::serialization::geometry::Point;
use crate::serialization::geometry::Rectangle;
use crate::serialization::geometry::Size;
use crate::serialization::xdg_shell;

#[derive(Archive, Deserialize, Serialize, Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct WlSurfaceId(pub u64);

impl WlSurfaceId {
    pub fn new(wl_surface: &WlSurface) -> Self {
        Self(serialization::hash(&wl_surface.id()))
    }
}

impl From<&backend::ObjectId> for WlSurfaceId {
    fn from(object_id: &backend::ObjectId) -> Self {
        Self(serialization::hash(object_id))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct ClientSurface {
    pub client: ClientId,
    pub surface: WlSurfaceId,
}

impl ClientSurface {
    pub fn new(wl_surface: &WlSurface) -> Result<Self> {
        Ok(Self {
            client: ClientId::new(&wl_surface.client().location(loc!())?),
            surface: WlSurfaceId::new(wl_surface),
        })
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct SubSurfaceId(pub u64);

impl SubSurfaceId {
    pub fn new(subsurface_id: &WlSurfaceId) -> Self {
        Self(serialization::hash(&subsurface_id))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, EnumAsInner, Archive, Deserialize, Serialize)]
pub enum BufferFormat {
    Argb8888,
    Xrgb8888,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct BufferMetadata {
    pub width: i32,
    pub height: i32,
    pub stride: i32,
    pub format: BufferFormat,
}

impl TryFrom<SmithayBufferFormat> for BufferFormat {
    type Error = Error;
    fn try_from(format: SmithayBufferFormat) -> Result<Self> {
        match format {
            SmithayBufferFormat::Argb8888 => Ok(Self::Argb8888),
            SmithayBufferFormat::Xrgb8888 => Ok(Self::Xrgb8888),
            _ => bail!("invalid buffer format {format:?}"),
        }
    }
}

impl TryFrom<SctkBufferFormat> for BufferFormat {
    type Error = Error;
    fn try_from(format: SctkBufferFormat) -> Result<Self> {
        match format {
            SctkBufferFormat::Argb8888 => Ok(Self::Argb8888),
            SctkBufferFormat::Xrgb8888 => Ok(Self::Xrgb8888),
            _ => bail!("invalid buffer format {format:?}"),
        }
    }
}

impl From<BufferFormat> for SctkBufferFormat {
    fn from(format: BufferFormat) -> Self {
        match format {
            BufferFormat::Argb8888 => Self::Argb8888,
            BufferFormat::Xrgb8888 => Self::Xrgb8888,
        }
    }
}

impl BufferMetadata {
    // TODO: replace with impl From
    pub fn from_buffer_data(spec: &SmithayBufferData) -> Result<Self> {
        let metadata = Self {
            width: spec.width,
            height: spec.height,
            stride: spec.stride,
            format: spec.format.try_into().location(loc!())?,
        };
        metadata.validate().location(loc!())?;
        Ok(metadata)
    }

    /// Validates positive dimensions, 4-byte-aligned stride `>= width * 4`, and
    /// a single buffer's signed 32-bit `wl_shm` byte size (leaving 63 bytes of
    /// headroom so rounding that single buffer up to 64-byte slot alignment fits
    /// in `i32`; does not bound the shared `SlotPool` across multiple buffers),
    /// returning total buffer pixels (`height * stride / 4`).
    pub fn validate(&self) -> Result<usize> {
        const MAX_SHM_BUFFER_BYTES: i32 = i32::MAX - 63;

        if self.width <= 0 || self.height <= 0 {
            bail!("invalid buffer dimensions: {}x{}", self.width, self.height);
        }
        if self.stride % 4 != 0 {
            bail!("buffer stride {} is not a multiple of 4", self.stride);
        }
        let min_stride = self
            .width
            .checked_mul(4)
            .context(loc!(), "buffer width * 4 overflow")?;
        if self.stride < min_stride {
            bail!("buffer stride {} < width * 4 ({min_stride})", self.stride);
        }
        let byte_len = self
            .height
            .checked_mul(self.stride)
            .filter(|&len| len <= MAX_SHM_BUFFER_BYTES)
            .with_context(loc!(), || {
                format!(
                    "buffer byte size ({} * {}) exceeds wl_shm limit ({MAX_SHM_BUFFER_BYTES})",
                    self.height, self.stride
                )
            })? as usize;
        Ok(byte_len / 4)
    }
}

// TODO: consider splitting SurfaceState, this only really makes sense for the
// surface state we're sending, not the one we're storing.
#[derive(Debug, Clone, Eq, PartialEq, EnumAsInner, Archive, Deserialize, Serialize)]
pub enum BufferAssignment {
    New(BufferMetadata),
    Removed,
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum CursorImageStatus {
    Hidden,
    Named(String),
    Surface {
        client_surface: ClientSurface,
        hotspot: Point<i32>,
    },
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct CursorImage {
    pub serial: u32,
    pub status: CursorImageStatus,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum KeyState {
    Released,
    Pressed,
    Repeated,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum RepeatInfo {
    Repeat { rate: NonZeroU32, delay: u32 },
    Disable,
}

impl From<SctkRepeatInfo> for RepeatInfo {
    fn from(info: SctkRepeatInfo) -> Self {
        match info {
            SctkRepeatInfo::Repeat { rate, delay } => Self::Repeat { rate, delay },
            SctkRepeatInfo::Disable => Self::Disable,
        }
    }
}

// Make this a separate struct so we can override debug just for this variant instead of the entire enum.
#[derive(Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct KeyInner {
    pub serial: u32,
    pub raw_code: u32,
    pub state: KeyState,
}

impl fmt::Debug for KeyInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Key")
            .field("serial", &self.serial)
            .field(
                "raw_code",
                if args::get_log_priv_data() {
                    &self.raw_code
                } else {
                    &"<redacted>"
                },
            )
            .field("state", &self.state)
            .finish()
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct ModifierState {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    pub caps_lock: bool,
    pub logo: bool,
    pub num_lock: bool,
}

impl From<SmithayModifiers> for ModifierState {
    fn from(modifiers: SmithayModifiers) -> Self {
        Self {
            ctrl: modifiers.ctrl,
            alt: modifiers.alt,
            shift: modifiers.shift,
            caps_lock: modifiers.caps_lock,
            logo: modifiers.logo,
            num_lock: modifiers.num_lock,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum KeyboardEvent {
    Enter {
        serial: u32,
        surface_id: WlSurfaceId,
        keycodes: Vec<u32>,
        keysyms: Vec<u32>,
    },
    Leave {
        serial: u32,
    },
    Key(KeyInner),
    RepeatInfo(RepeatInfo),
    Keymap(String),
    Modifiers {
        modifier_state: ModifierState,
        layout_index: u32,
    },
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum AxisRelativeDirection {
    #[default]
    Identical,
    Inverted,
}

impl From<SctkAxisRelativeDirection> for AxisRelativeDirection {
    fn from(direction: SctkAxisRelativeDirection) -> Self {
        match direction {
            SctkAxisRelativeDirection::Identical => Self::Identical,
            SctkAxisRelativeDirection::Inverted => Self::Inverted,
            _ => {
                warn!("Unknown axis relative direction {direction:?}, using Identical instead.");
                Self::Identical
            },
        }
    }
}

impl From<AxisRelativeDirection> for SmithayAxisRelativeDirection {
    fn from(direction: AxisRelativeDirection) -> Self {
        match direction {
            AxisRelativeDirection::Identical => Self::Identical,
            AxisRelativeDirection::Inverted => Self::Inverted,
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub struct AxisScroll {
    pub absolute: f64,
    /// High-resolution wheel scroll steps in units of 1/120th of a detent
    /// (from `wl_pointer::axis_value120` on v8+ or `axis_discrete * 120` on
    /// v5-v7; zero for continuous scrolling).
    pub v120: i32,
    pub relative_direction: AxisRelativeDirection,
    pub stop: bool,
}

impl From<SctkAxisScroll> for AxisScroll {
    fn from(axis_scroll: SctkAxisScroll) -> Self {
        let v120 = if axis_scroll.value120 != 0 {
            axis_scroll.value120
        } else {
            axis_scroll.discrete.saturating_mul(120)
        };
        Self {
            absolute: axis_scroll.absolute,
            v120,
            relative_direction: axis_scroll
                .relative_direction
                .map(Into::into)
                .unwrap_or_default(),
            stop: axis_scroll.stop,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum AxisSource {
    Finger,
    Continuous,
    Wheel,
    WheelTilt,
}

impl From<SctkAxisSource> for AxisSource {
    fn from(axis_source: SctkAxisSource) -> Self {
        match axis_source {
            SctkAxisSource::Wheel => Self::Wheel,
            SctkAxisSource::Finger => Self::Finger,
            SctkAxisSource::Continuous => Self::Continuous,
            SctkAxisSource::WheelTilt => Self::WheelTilt,
            _ => unreachable!("got unknown AxisSource {axis_source:?}"),
        }
    }
}

impl From<AxisSource> for SmithayAxisSource {
    fn from(axis_source: AxisSource) -> Self {
        match axis_source {
            AxisSource::Wheel => Self::Wheel,
            AxisSource::Finger => Self::Finger,
            AxisSource::Continuous => Self::Continuous,
            AxisSource::WheelTilt => Self::WheelTilt,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub enum PointerEventKind {
    Enter {
        serial: u32,
    },
    Leave {
        serial: u32,
    },
    Motion,
    Press {
        button: u32,
        serial: u32,
    },
    Release {
        button: u32,
        serial: u32,
    },
    Axis {
        horizontal: AxisScroll,
        vertical: AxisScroll,
        source: Option<AxisSource>,
    },
}

impl From<SctkPointerEventKind> for PointerEventKind {
    fn from(event: SctkPointerEventKind) -> Self {
        match event {
            SctkPointerEventKind::Enter { serial } => Self::Enter { serial },
            SctkPointerEventKind::Leave { serial } => Self::Leave { serial },
            SctkPointerEventKind::Motion { time: _ } => Self::Motion,
            SctkPointerEventKind::Press {
                time: _,
                button,
                serial,
            } => Self::Press { button, serial },
            SctkPointerEventKind::Release {
                time: _,
                button,
                serial,
            } => Self::Release { button, serial },
            SctkPointerEventKind::Axis {
                time: _,
                horizontal,
                vertical,
                source,
            } => Self::Axis {
                horizontal: horizontal.into(),
                vertical: vertical.into(),
                source: source.map(Into::into),
            },
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub struct PointerEvent {
    pub surface_id: WlSurfaceId,
    pub position: Point<f64>,
    pub kind: PointerEventKind,
}

impl PointerEvent {
    pub fn from_smithay(surface_id: &WlSurfaceId, event: &SctkPointerEvent) -> Self {
        Self {
            surface_id: *surface_id,
            position: event.position.into(),
            kind: event.kind.clone().into(),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct SubSurfaceState {
    pub parent: WlSurfaceId,
    pub location: Point<i32>,
    pub sync: bool,
}

impl SubSurfaceState {
    pub fn new(parent: &WlSurface) -> Self {
        Self {
            parent: WlSurfaceId::new(parent),
            location: (0, 0).into(),
            sync: true,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, EnumAsInner, Archive, Deserialize, Serialize)]
pub enum Role {
    Cursor(Point<i32>),
    SubSurface(SubSurfaceState),
    XdgToplevel(xdg_shell::XdgToplevelState),
    XdgPopup(xdg_shell::XdgPopupState),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum RectangleKind {
    Add,
    Subtract,
}

impl From<&SmithayRectangleKind> for RectangleKind {
    fn from(kind: &SmithayRectangleKind) -> Self {
        match kind {
            SmithayRectangleKind::Add => Self::Add,
            SmithayRectangleKind::Subtract => Self::Subtract,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct Region {
    rects: Vec<Tuple2<RectangleKind, Rectangle<i32>>>,
}

impl From<&RegionAttributes> for Region {
    fn from(region: &RegionAttributes) -> Self {
        Self {
            rects: region
                .rects
                .iter()
                .map(|(kind, rect)| (kind.into(), (*rect).into()).into())
                .collect(),
        }
    }
}

impl Region {
    pub fn new() -> Self {
        Self { rects: Vec::new() }
    }

    pub fn create_compositor_region(
        &self,
        compositor_state: &CompositorState,
    ) -> Result<SctkRegion> {
        let region = SctkRegion::new(compositor_state).location(loc!())?;
        self.rects.iter().for_each(|rect| match rect.0 {
            RectangleKind::Add => {
                region.add(rect.1.loc.x, rect.1.loc.y, rect.1.size.w, rect.1.size.h);
            },
            RectangleKind::Subtract => {
                region.subtract(rect.1.loc.x, rect.1.loc.y, rect.1.size.w, rect.1.size.h);
            },
        });
        Ok(region)
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum Transform {
    Normal,
    _90,
    _180,
    _270,
    Flipped,
    Flipped90,
    Flipped180,
    Flipped270,
}

impl From<SctkTransform> for Transform {
    fn from(transform: SctkTransform) -> Self {
        match transform {
            SctkTransform::Normal => Self::Normal,
            SctkTransform::_90 => Self::_90,
            SctkTransform::_180 => Self::_180,
            SctkTransform::_270 => Self::_270,
            SctkTransform::Flipped => Self::Flipped,
            SctkTransform::Flipped90 => Self::Flipped90,
            SctkTransform::Flipped180 => Self::Flipped180,
            SctkTransform::Flipped270 => Self::Flipped270,
            _ => {
                warn!("Unknown transformation {transform:?}, using Normal instead.");
                Self::Normal
            },
        }
    }
}

impl From<Transform> for SctkTransform {
    fn from(transform: Transform) -> Self {
        match transform {
            Transform::Normal => Self::Normal,
            Transform::_90 => Self::_90,
            Transform::_180 => Self::_180,
            Transform::_270 => Self::_270,
            Transform::Flipped => Self::Flipped,
            Transform::Flipped90 => Self::Flipped90,
            Transform::Flipped180 => Self::Flipped180,
            Transform::Flipped270 => Self::Flipped270,
        }
    }
}

impl From<SmithayTransform> for Transform {
    fn from(transform: SmithayTransform) -> Self {
        match transform {
            SmithayTransform::Normal => Self::Normal,
            SmithayTransform::_90 => Self::_90,
            SmithayTransform::_180 => Self::_180,
            SmithayTransform::_270 => Self::_270,
            SmithayTransform::Flipped => Self::Flipped,
            SmithayTransform::Flipped90 => Self::Flipped90,
            SmithayTransform::Flipped180 => Self::Flipped180,
            SmithayTransform::Flipped270 => Self::Flipped270,
        }
    }
}

impl From<SmithayWlTransform> for Transform {
    fn from(transform: SmithayWlTransform) -> Self {
        match transform {
            SmithayWlTransform::Normal => Self::Normal,
            SmithayWlTransform::_90 => Self::_90,
            SmithayWlTransform::_180 => Self::_180,
            SmithayWlTransform::_270 => Self::_270,
            SmithayWlTransform::Flipped => Self::Flipped,
            SmithayWlTransform::Flipped90 => Self::Flipped90,
            SmithayWlTransform::Flipped180 => Self::Flipped180,
            SmithayWlTransform::Flipped270 => Self::Flipped270,
            _ => {
                warn!("Unknown transformation {transform:?}, using Normal instead.");
                Self::Normal
            },
        }
    }
}

impl From<Transform> for SmithayTransform {
    fn from(transform: Transform) -> Self {
        match transform {
            Transform::Normal => Self::Normal,
            Transform::_90 => Self::_90,
            Transform::_180 => Self::_180,
            Transform::_270 => Self::_270,
            Transform::Flipped => Self::Flipped,
            Transform::Flipped90 => Self::Flipped90,
            Transform::Flipped180 => Self::Flipped180,
            Transform::Flipped270 => Self::Flipped270,
        }
    }
}

/// An entry for a vector of child surfaces. The (x, y) position is stored
/// explicitly, the z position (stacking order) is stored implicitly based on
/// the index of the item in the vector.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct SubsurfacePosition {
    pub id: WlSurfaceId,
    pub position: Point<i32>,
}

#[derive(Debug, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub struct SurfaceState {
    pub client: ClientId,
    pub id: WlSurfaceId,
    pub buffer: Option<BufferAssignment>,
    pub role: Option<Role>,
    // TODO: include buffer_delta, transform from SurfaceAttributes
    pub buffer_scale: i32,
    pub buffer_transform: Option<Transform>,
    pub opaque_region: Option<Region>,
    pub input_region: Option<Region>,
    pub z_ordered_children: Vec<SubsurfacePosition>,
    /// Buffer-space damage rects; `None` or empty damage with
    /// `BufferAssignment::New` denotes full-buffer damage.
    pub damage: Option<Vec<Rectangle<i32>>>,
    // server-side only
    pub output_ids: Vec<u32>,
    pub viewport_state: Option<ViewportState>,

    // Interfaces
    pub xdg_surface_state: Option<xdg_shell::XdgSurfaceState>,
}

impl SurfaceState {
    pub fn new(surface: &WlSurface, buffer: Option<BufferAssignment>) -> Result<Self> {
        Ok(Self {
            client: ClientId::new(&surface.client().location(loc!())?),
            id: WlSurfaceId::new(surface),
            buffer,
            role: None,
            buffer_scale: 1,
            buffer_transform: None,
            opaque_region: None,
            input_region: None,
            // TODO: insert own id into z_ordered_children after figuring out
            // client isolation.
            z_ordered_children: Vec::new(),
            damage: None,
            output_ids: Vec::new(),
            viewport_state: None,
            xdg_surface_state: None,
        })
    }

    pub fn set_buffer(&mut self, metadata: &SmithayBufferData) -> Result<BufferMetadata> {
        let metadata = BufferMetadata::from_buffer_data(metadata).location(loc!())?;
        self.buffer = Some(BufferAssignment::New(metadata));
        Ok(metadata)
    }

    #[instrument(skip_all, level = "debug")]
    pub fn clone_without_buffer(&self) -> Self {
        let mut clone = self.clone();
        clone.buffer = None;
        clone
    }

    pub fn get_role(&self) -> Result<&Role> {
        self.role.as_ref().ok_or(anyhow!("Role was None."))
    }

    pub fn get_role_mut(&mut self) -> Result<&mut Role> {
        self.role.as_mut().ok_or(anyhow!("Role was None."))
    }

    pub fn xdg_toplevel(&self) -> Result<&xdg_shell::XdgToplevelState> {
        self.get_role()
            .location(loc!())?
            .as_xdg_toplevel()
            .ok_or(anyhow!("Role was not XdgToplevel."))
    }

    pub fn xdg_popup(&self) -> Result<&xdg_shell::XdgPopupState> {
        self.get_role()
            .location(loc!())?
            .as_xdg_popup()
            .ok_or(anyhow!("Role was not XdgPopup."))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum Subpixel {
    Unknown,
    None,
    HorizontalRgb,
    HorizontalBgr,
    VerticalRgb,
    VerticalBgr,
}

impl From<SctkSubpixel> for Subpixel {
    fn from(subpixel: SctkSubpixel) -> Self {
        match subpixel {
            SctkSubpixel::Unknown => Self::Unknown,
            SctkSubpixel::None => Self::None,
            SctkSubpixel::HorizontalRgb => Self::HorizontalRgb,
            SctkSubpixel::HorizontalBgr => Self::HorizontalBgr,
            SctkSubpixel::VerticalRgb => Self::VerticalRgb,
            SctkSubpixel::VerticalBgr => Self::VerticalBgr,
            _ => Self::Unknown,
        }
    }
}

impl From<Subpixel> for SmithaySubpixel {
    fn from(subpixel: Subpixel) -> Self {
        match subpixel {
            Subpixel::Unknown => Self::Unknown,
            Subpixel::None => Self::None,
            Subpixel::HorizontalRgb => Self::HorizontalRgb,
            Subpixel::HorizontalBgr => Self::HorizontalBgr,
            Subpixel::VerticalRgb => Self::VerticalRgb,
            Subpixel::VerticalBgr => Self::VerticalBgr,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct Mode {
    pub dimensions: Size<i32>,
    pub refresh_rate: i32,
    pub current: bool,
    pub preferred: bool,
}

impl From<&SctkMode> for Mode {
    fn from(mode: &SctkMode) -> Self {
        Self {
            dimensions: mode.dimensions.into(),
            refresh_rate: mode.refresh_rate,
            current: mode.current,
            preferred: mode.preferred,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct OutputInfo {
    pub id: u32,
    pub model: String,
    pub make: String,
    pub location: Point<i32>,
    pub physical_size: Size<i32>,
    pub subpixel: Subpixel,
    pub transform: Transform,
    pub scale_factor: i32,
    pub mode: Mode,
    pub logical_position: Option<Point<i32>>,
    pub logical_size: Option<Size<i32>>,
    pub name: Option<String>,
    pub description: Option<String>,
}

impl From<SctkOutputInfo> for OutputInfo {
    fn from(output: SctkOutputInfo) -> Self {
        Self {
            id: output.id,
            model: output.model.clone(),
            make: output.make.clone(),
            location: output.location.into(),
            physical_size: output.physical_size.into(),
            subpixel: output.subpixel.into(),
            transform: output.transform.into(),
            scale_factor: output.scale_factor,
            mode: output
                .modes
                .iter()
                .rfind(|mode| mode.current)
                .unwrap()
                .into(),
            logical_position: output.logical_position.map(Into::into),
            logical_size: output.logical_size.map(Into::into),
            name: output.name.clone(),
            description: output.description.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub enum SurfaceRequestPayload {
    Commit(SurfaceState),
    Destroyed,
}

#[derive(Debug, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub struct SurfaceRequest {
    pub client: ClientId,
    pub surface: WlSurfaceId,
    pub payload: SurfaceRequestPayload,
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct SourceMetadata {
    pub mime_types: Vec<String>,
    pub dnd_actions: u32,
}

impl SourceMetadata {
    pub fn from_mime_types(mime_types: Vec<String>) -> Self {
        Self {
            mime_types,
            dnd_actions: 0,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum DataSource {
    Selection,
    Primary,
}

/// Identifies the source or offer of a data transfer request: clipboard
/// selection, primary selection, or a specific DnD offer/source by ID.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum DataTransferTarget {
    Selection,
    Primary,
    DnD(u64),
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct StartDrag {
    pub source_id: u64,
    pub source: SourceMetadata,
    pub origin: ClientSurface,
    pub icon: Option<ClientSurface>,
    pub serial: u32,
}

#[derive(Debug, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub struct DragEnter {
    pub offer_id: u64,
    pub serial: u32,
    pub surface: WlSurfaceId,
    pub loc: Point<f64>,
    pub source_actions: u32,
    pub selected_action: u32,
    pub mime_types: Vec<String>,
}

#[derive(Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct DataToTransfer(pub Vec<u8>);

impl fmt::Debug for DataToTransfer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataToTransfer")
            .field(
                "0",
                if args::get_log_priv_data() {
                    &self.0
                } else {
                    &"<redacted>"
                },
            )
            .finish()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum DataSourceRequest {
    // wl_data_device requests
    StartDrag(StartDrag),
    CancelDrag(u64),
    SetSelection(DataSource, SourceMetadata),
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum DataSourceEvent {
    // wl_data_source events
    DnDMimeTypeAcceptedByDestination {
        source_id: u64,
        mime_type: Option<String>,
    },
    MimeTypeSendRequestedByDestination {
        target: DataTransferTarget,
        mime_type: String,
        transfer_id: u64,
    },
    DnDActionSelected {
        source_id: u64,
        action: u32,
    },
    DnDDropPerformed {
        source_id: u64,
    },
    DnDCancelled {
        source_id: u64,
    },
    DnDFinished {
        source_id: u64,
    },
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum DataDestinationRequest {
    // wl_data_offer requests
    DnDAcceptMimeType {
        offer_id: u64,
        mime_type: Option<String>,
    },
    RequestDataTransfer {
        target: DataTransferTarget,
        mime_type: String,
        transfer_id: u64,
    },
    DnDFinish {
        offer_id: u64,
    },
    DnDCancel {
        offer_id: u64,
    },
    DnDSetDestinationActions {
        offer_id: u64,
        dnd_actions: u32,
        preferred_action: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub enum DataDestinationEvent {
    // wl_data_offer events
    DnDSourceActions { offer_id: u64, actions: u32 },
    DnDActionSelected { offer_id: u64, action: u32 },

    // wl_data_device events
    DnDEnter(DragEnter),
    DnDLeave { offer_id: u64 },
    DnDMotion { offer_id: u64, loc: Point<f64> },
    DnDDrop { offer_id: u64 },
    SelectionSet(DataSource, SourceMetadata),
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum DataRequest {
    // Requests from remote source application to host compositor.
    SourceRequest(DataSourceRequest),
    // Requests from remote destination application to host compositor.
    DestinationRequest(DataDestinationRequest),
    TransferData(u64, DataToTransfer),
}

#[derive(Debug, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub enum DataEvent {
    // Feedback from host compositor to remote source application.
    SourceEvent(DataSourceEvent),
    // Events from host compositor for remote data destination.
    DestinationEvent(DataDestinationEvent),
    TransferData(u64, DataToTransfer),
}

#[derive(Debug, Clone, PartialEq, Eq, Archive, Deserialize, Serialize)]
pub enum OutputEvent {
    New(OutputInfo),
    Update(OutputInfo),
    Destroy(OutputInfo),
}

#[derive(Debug, Clone, PartialEq, Eq, Archive, Deserialize, Serialize)]
pub struct Output {
    pub id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Archive, Deserialize, Serialize)]
pub enum SurfaceEventPayload {
    OutputsChanged(Vec<Output>),
    BufferResyncRequested,
}

#[derive(Debug, Clone, PartialEq, Eq, Archive, Deserialize, Serialize)]
pub struct SurfaceEvent {
    pub surface_id: WlSurfaceId,
    pub payload: SurfaceEventPayload,
}

#[derive(Debug, Copy, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub struct ViewportState {
    pub src: Option<Rectangle<f64>>,
    pub dst: Option<Size<i32>>,
}

impl From<&ViewportCachedState> for ViewportState {
    fn from(viewport_state: &ViewportCachedState) -> Self {
        Self {
            src: viewport_state.src.map(Into::into),
            dst: viewport_state.dst.map(Into::into),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_metadata_validate() {
        let min_valid = BufferMetadata {
            width: 1,
            height: 1,
            stride: 4,
            format: BufferFormat::Argb8888,
        };
        assert_eq!(min_valid.validate().unwrap(), 1);

        let valid = BufferMetadata {
            width: 640,
            height: 480,
            stride: 640 * 4 + 16,
            format: BufferFormat::Argb8888,
        };
        assert_eq!(valid.validate().unwrap(), 480 * (640 + 4));

        // Exact `MAX_SHM_BUFFER_BYTES` (`i32::MAX - 63 == 0x7FFF_FFC0`) boundary
        // with tight unpadded `stride == width * 4`.
        let max_valid = BufferMetadata {
            width: (i32::MAX - 63) / 4,
            height: 1,
            stride: i32::MAX - 63,
            format: BufferFormat::Xrgb8888,
        };
        assert_eq!(
            max_valid.validate().unwrap(),
            ((i32::MAX - 63) as usize) / 4
        );

        let invalid_cases = [
            // Non-positive dimensions.
            (0, 480, 640 * 4),
            (-1, 480, 640 * 4),
            (640, 0, 640 * 4),
            (640, -1, 640 * 4),
            // Unaligned stride (`stride % 4 != 0`).
            (10, 10, 42),
            // Stride smaller than `width * 4` (including `<= 0`) or `width * 4` overflow.
            (640, 480, 0),
            (640, 480, -4),
            (10, 10, 36),
            (i32::MAX / 2 + 1, 1, i32::MAX - 63),
            // Exceeds `i32::MAX - 63` (`0x7FFF_FFC0`) or overflows `height * stride`.
            (1, 1, i32::MAX - 59),
            (16384, 65536, 65536),
        ];
        for (width, height, stride) in invalid_cases {
            let meta = BufferMetadata {
                width,
                height,
                stride,
                format: BufferFormat::Argb8888,
            };
            assert!(
                meta.validate().is_err(),
                "expected validate() error for ({width}, {height}, {stride})"
            );
        }
    }

    #[test]
    fn test_axis_scroll_from_sctk() {
        // `wl_pointer` v8+: `value120` populated, deprecated `discrete == 0`.
        for (value120, absolute, rel_dir, expected_dir) in [
            (
                120,
                15.0,
                Some(SctkAxisRelativeDirection::Identical),
                AxisRelativeDirection::Identical,
            ),
            (
                -120,
                -15.0,
                Some(SctkAxisRelativeDirection::Inverted),
                AxisRelativeDirection::Inverted,
            ),
            (60, 7.5, None, AxisRelativeDirection::Identical),
        ] {
            let converted = AxisScroll::from(SctkAxisScroll {
                absolute,
                discrete: 0,
                value120,
                relative_direction: rel_dir,
                stop: false,
            });
            assert_eq!(
                converted,
                AxisScroll {
                    absolute,
                    v120: value120,
                    relative_direction: expected_dir,
                    stop: false,
                }
            );
        }

        // `wl_pointer` v5-v7: `discrete` populated, `value120 == 0`.
        for (discrete, expected_v120) in [
            (1, 120),
            (-2, -240),
            (i32::MAX, i32::MAX),
            (i32::MIN, i32::MIN),
        ] {
            let converted = AxisScroll::from(SctkAxisScroll {
                absolute: 15.0,
                discrete,
                value120: 0,
                relative_direction: None,
                stop: false,
            });
            assert_eq!(
                converted,
                AxisScroll {
                    absolute: 15.0,
                    v120: expected_v120,
                    relative_direction: AxisRelativeDirection::Identical,
                    stop: false,
                }
            );
        }

        // Continuous touchpad scroll: `absolute != 0.0`, `discrete == 0`, `value120 == 0`.
        let continuous = AxisScroll::from(SctkAxisScroll {
            absolute: 12.25,
            discrete: 0,
            value120: 0,
            relative_direction: Some(SctkAxisRelativeDirection::Inverted),
            stop: true,
        });
        assert_eq!(
            continuous,
            AxisScroll {
                absolute: 12.25,
                v120: 0,
                relative_direction: AxisRelativeDirection::Inverted,
                stop: true,
            }
        );

        // If both `value120` and `discrete` are non-zero, `value120` takes precedence.
        let both = AxisScroll::from(SctkAxisScroll {
            absolute: 7.5,
            discrete: 1,
            value120: 60,
            relative_direction: None,
            stop: false,
        });
        assert_eq!(both.v120, 60);

        // Wire round-trip via `rkyv`.
        let event = PointerEventKind::Axis {
            horizontal: continuous,
            vertical: AxisScroll {
                absolute: -15.0,
                v120: -120,
                relative_direction: AxisRelativeDirection::Identical,
                stop: false,
            },
            source: Some(AxisSource::Wheel),
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&event).unwrap();
        let decoded = rkyv::from_bytes::<PointerEventKind, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(decoded, event);
    }

    #[test]
    fn test_dnd_wire_roundtrip() {
        let requests = vec![
            DataRequest::SourceRequest(DataSourceRequest::StartDrag(StartDrag {
                source_id: 7,
                source: SourceMetadata {
                    mime_types: vec!["text/plain".to_string(), "text/uri-list".to_string()],
                    dnd_actions: 0b111,
                },
                origin: ClientSurface {
                    client: ClientId(1),
                    surface: WlSurfaceId(2),
                },
                icon: Some(ClientSurface {
                    client: ClientId(1),
                    surface: WlSurfaceId(3),
                }),
                serial: 42,
            })),
            DataRequest::SourceRequest(DataSourceRequest::CancelDrag(7)),
            DataRequest::SourceRequest(DataSourceRequest::SetSelection(
                DataSource::Selection,
                SourceMetadata::from_mime_types(vec!["text/plain".to_string()]),
            )),
            DataRequest::DestinationRequest(DataDestinationRequest::DnDAcceptMimeType {
                offer_id: 9,
                mime_type: Some("text/plain".to_string()),
            }),
            DataRequest::DestinationRequest(DataDestinationRequest::RequestDataTransfer {
                target: DataTransferTarget::DnD(9),
                mime_type: "text/plain".to_string(),
                transfer_id: 11,
            }),
            DataRequest::DestinationRequest(DataDestinationRequest::DnDSetDestinationActions {
                offer_id: 9,
                dnd_actions: 0b11,
                preferred_action: 0b10,
            }),
            DataRequest::DestinationRequest(DataDestinationRequest::DnDFinish { offer_id: 9 }),
            DataRequest::DestinationRequest(DataDestinationRequest::DnDCancel { offer_id: 9 }),
            DataRequest::TransferData(11, DataToTransfer(b"hello".to_vec())),
        ];
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&requests).unwrap();
        let decoded = rkyv::from_bytes::<Vec<DataRequest>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(decoded, requests);

        let events = vec![
            DataEvent::SourceEvent(DataSourceEvent::DnDMimeTypeAcceptedByDestination {
                source_id: 7,
                mime_type: Some("text/plain".to_string()),
            }),
            DataEvent::SourceEvent(DataSourceEvent::MimeTypeSendRequestedByDestination {
                target: DataTransferTarget::DnD(7),
                mime_type: "text/plain".to_string(),
                transfer_id: 12,
            }),
            DataEvent::SourceEvent(DataSourceEvent::DnDActionSelected {
                source_id: 7,
                action: 2,
            }),
            DataEvent::SourceEvent(DataSourceEvent::DnDDropPerformed { source_id: 7 }),
            DataEvent::SourceEvent(DataSourceEvent::DnDFinished { source_id: 7 }),
            DataEvent::SourceEvent(DataSourceEvent::DnDCancelled { source_id: 7 }),
            DataEvent::DestinationEvent(DataDestinationEvent::DnDEnter(DragEnter {
                offer_id: 9,
                serial: 99,
                surface: WlSurfaceId(4),
                loc: (12.5, 34.25).into(),
                source_actions: 0b111,
                selected_action: 0b100,
                mime_types: vec!["text/uri-list".to_string()],
            })),
            DataEvent::DestinationEvent(DataDestinationEvent::DnDMotion {
                offer_id: 9,
                loc: (20.0, 40.0).into(),
            }),
            DataEvent::DestinationEvent(DataDestinationEvent::DnDSourceActions {
                offer_id: 9,
                actions: 0b11,
            }),
            DataEvent::DestinationEvent(DataDestinationEvent::DnDActionSelected {
                offer_id: 9,
                action: 0b10,
            }),
            DataEvent::DestinationEvent(DataDestinationEvent::DnDDrop { offer_id: 9 }),
            DataEvent::DestinationEvent(DataDestinationEvent::DnDLeave { offer_id: 9 }),
            DataEvent::DestinationEvent(DataDestinationEvent::SelectionSet(
                DataSource::Primary,
                SourceMetadata::from_mime_types(vec!["text/plain".to_string()]),
            )),
            DataEvent::TransferData(12, DataToTransfer(b"world".to_vec())),
        ];
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&events).unwrap();
        let decoded = rkyv::from_bytes::<Vec<DataEvent>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(decoded, events);
    }

    #[test]
    fn test_output_info_wire_roundtrip() {
        let with_xdg = OutputInfo {
            id: 42,
            model: "U2720Q".to_string(),
            make: "Dell".to_string(),
            location: (0, 0).into(),
            physical_size: (600, 340).into(),
            subpixel: Subpixel::HorizontalRgb,
            transform: Transform::Normal,
            scale_factor: 2,
            mode: Mode {
                dimensions: (3840, 2160).into(),
                refresh_rate: 60000,
                current: true,
                preferred: true,
            },
            logical_position: Some((1920, 0).into()),
            logical_size: Some((2560, 1440).into()),
            name: Some("DP-1".to_string()),
            description: Some("Dell U2720Q".to_string()),
        };
        let without_xdg = OutputInfo {
            id: 43,
            logical_position: None,
            logical_size: None,
            ..with_xdg.clone()
        };

        let events = vec![
            OutputEvent::New(with_xdg),
            OutputEvent::Update(without_xdg.clone()),
            OutputEvent::Destroy(without_xdg),
        ];
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&events).unwrap();
        let decoded = rkyv::from_bytes::<Vec<OutputEvent>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(decoded, events);
    }
}
