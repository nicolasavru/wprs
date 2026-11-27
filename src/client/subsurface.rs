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

use smithay_client_toolkit::compositor::CompositorState;
use smithay_client_toolkit::compositor::Surface;
use smithay_client_toolkit::reexports::client::QueueHandle;
use smithay_client_toolkit::reexports::client::protocol::wl_subcompositor::WlSubcompositor;
use smithay_client_toolkit::reexports::client::protocol::wl_subsurface::WlSubsurface;
use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface;
use smithay_client_toolkit::shell::WaylandSurface;

use crate::client::ObjectBimap;
use crate::client::RemoteSurface;
use crate::client::Role;
use crate::client::WprsClientState;
use crate::client::ZOrderMove;
use crate::client::smithay_handlers::SubSurfaceData;
use crate::fallible_entry::FallibleEntryExt;
use crate::prelude::*;
use crate::serialization::ClientId;
use crate::serialization::wayland::SubSurfaceState;
use crate::serialization::wayland::SubsurfacePosition;
use crate::serialization::wayland::SurfaceState;
use crate::serialization::wayland::WlSurfaceId;

trait SubSurfaceNode {
    fn subsurface_parent(&self) -> Option<WlSurfaceId>;
    fn has_non_subsurface_role(&self) -> bool;
    fn z_ordered_children(&self) -> &[SubsurfacePosition];
    fn z_ordered_children_mut(&mut self) -> &mut Vec<SubsurfacePosition>;
    /// Destroys the active `wl_subsurface` while keeping the underlying
    /// `wl_surface` alive. Returns the previous parent ID if detached.
    fn detach_local_subsurface(&mut self) -> Option<WlSurfaceId>;
}

impl SubSurfaceNode for RemoteSurface {
    fn subsurface_parent(&self) -> Option<WlSurfaceId> {
        self.role
            .as_ref()
            .and_then(Role::as_sub_surface)
            .map(|sub| sub.parent)
    }

    fn has_non_subsurface_role(&self) -> bool {
        self.role
            .as_ref()
            .is_some_and(|role| !matches!(role, Role::SubSurface(_)))
    }

    fn z_ordered_children(&self) -> &[SubsurfacePosition] {
        &self.z_ordered_children
    }

    fn z_ordered_children_mut(&mut self) -> &mut Vec<SubsurfacePosition> {
        &mut self.z_ordered_children
    }

    fn detach_local_subsurface(&mut self) -> Option<WlSurfaceId> {
        if !matches!(self.role, Some(Role::SubSurface(_))) {
            return None;
        }
        let Some(Role::SubSurface(subsurface)) = self.role.take() else {
            unreachable!();
        };
        let parent_id = subsurface.parent;
        self.local_surface = Some(subsurface.into_surface());
        Some(parent_id)
    }
}

fn is_subsurface_ancestor<S: SubSurfaceNode>(
    ancestor: WlSurfaceId,
    mut node: WlSurfaceId,
    surfaces: &HashMap<WlSurfaceId, S>,
) -> bool {
    for _ in 0..=surfaces.len() {
        if node == ancestor {
            return true;
        }
        let Some(parent) = surfaces
            .get(&node)
            .and_then(SubSurfaceNode::subsurface_parent)
        else {
            return false;
        };
        node = parent;
    }
    false
}

fn detach_subsurface_in<S: SubSurfaceNode>(
    surface_id: WlSurfaceId,
    surfaces: &mut HashMap<WlSurfaceId, S>,
) {
    let Some(old_parent_id) = surfaces
        .get_mut(&surface_id)
        .and_then(SubSurfaceNode::detach_local_subsurface)
    else {
        return;
    };
    if let Some(parent) = surfaces.get_mut(&old_parent_id) {
        parent
            .z_ordered_children_mut()
            .retain(|child| child.id != surface_id);
    }
}

fn detach_children_of_in<S: SubSurfaceNode>(
    parent_id: WlSurfaceId,
    surfaces: &mut HashMap<WlSurfaceId, S>,
) {
    let child_ids: Vec<WlSurfaceId> = surfaces
        .iter()
        .filter_map(|(&id, surface)| (surface.subsurface_parent() == Some(parent_id)).then_some(id))
        .collect();
    for child_id in child_ids {
        detach_subsurface_in(child_id, surfaces);
    }
}

fn detach_all_subsurfaces_in<S: SubSurfaceNode>(surfaces: &mut HashMap<WlSurfaceId, S>) {
    let subsurface_ids: Vec<WlSurfaceId> = surfaces
        .iter()
        .filter_map(|(&id, surface)| surface.subsurface_parent().map(|_| id))
        .collect();
    for surface_id in subsurface_ids {
        detach_subsurface_in(surface_id, surfaces);
    }
}

fn destroy_surface_in<S: SubSurfaceNode>(
    surface_id: WlSurfaceId,
    surfaces: &mut HashMap<WlSurfaceId, S>,
) {
    let Some(surface) = surfaces.remove(&surface_id) else {
        return;
    };
    // Detach child subsurfaces while `surface` (and its `wl_surface`) is still
    // alive so every child `wl_subsurface.destroy()` precedes parent
    // `wl_surface.destroy()`.
    detach_children_of_in(surface_id, surfaces);
    if let Some(parent_id) = surface.subsurface_parent()
        && let Some(parent) = surfaces.get_mut(&parent_id)
    {
        parent
            .z_ordered_children_mut()
            .retain(|child| child.id != surface_id);
    }
    drop(surface);
}

fn set_subsurface_role_in<S: SubSurfaceNode>(
    parent_id: WlSurfaceId,
    surface_id: WlSurfaceId,
    surfaces: &mut HashMap<WlSurfaceId, S>,
    create_surface: &mut impl FnMut(WlSurfaceId) -> Result<S>,
    attach_subsurface: &mut impl FnMut(
        &mut HashMap<WlSurfaceId, S>,
        WlSurfaceId,
        WlSurfaceId,
    ) -> Result<()>,
) -> Result<()> {
    if parent_id == surface_id || is_subsurface_ancestor(surface_id, parent_id, surfaces) {
        bail!("refusing to create cyclic subsurface {surface_id:?} -> {parent_id:?}");
    }

    let (current_parent, has_non_subsurface_role) = {
        let surface = surfaces.get(&surface_id).location(loc!())?;
        (
            surface.subsurface_parent(),
            surface.has_non_subsurface_role(),
        )
    };
    if has_non_subsurface_role {
        bail!("surface {surface_id:?} already has a non-subsurface role");
    }
    if current_parent == Some(parent_id) {
        if let Some(parent) = surfaces.get_mut(&parent_id)
            && !parent
                .z_ordered_children()
                .iter()
                .any(|child| child.id == surface_id)
        {
            parent.z_ordered_children_mut().push(SubsurfacePosition {
                id: surface_id,
                position: (0, 0).into(),
            });
        }
        return Ok(());
    }
    if current_parent.is_some() {
        // Destroy the old `wl_subsurface` before creating a new one for the new
        // parent; Wayland forbids calling `get_subsurface` while an existing
        // `wl_subsurface` object is still attached.
        detach_subsurface_in(surface_id, surfaces);
    }

    // It's possible that on initial sync or before the parent commits, the
    // server sent over a subsurface before the parent surface, so insert a
    // placeholder entry for the parent here if we need to.
    let remote_parent_surface = surfaces
        .entry(parent_id)
        .or_insert_with_result(|| create_surface(parent_id))
        .location(loc!())?;
    if !remote_parent_surface
        .z_ordered_children()
        .iter()
        .any(|child| child.id == surface_id)
    {
        remote_parent_surface
            .z_ordered_children_mut()
            .push(SubsurfacePosition {
                id: surface_id,
                position: (0, 0).into(),
            });
    }

    attach_subsurface(surfaces, parent_id, surface_id).location(loc!())
}

fn populate_subsurfaces_in<S: SubSurfaceNode>(
    parent_id: WlSurfaceId,
    new_children: &[SubsurfacePosition],
    surfaces: &mut HashMap<WlSurfaceId, S>,
    create_surface: &mut impl FnMut(WlSurfaceId) -> Result<S>,
    attach_subsurface: &mut impl FnMut(
        &mut HashMap<WlSurfaceId, S>,
        WlSurfaceId,
        WlSurfaceId,
    ) -> Result<()>,
) -> Result<()> {
    let new_child_ids: HashSet<WlSurfaceId> = new_children
        .iter()
        .filter(|c| c.id != parent_id)
        .map(|c| c.id)
        .collect();

    let stale_child_ids: Vec<WlSurfaceId> = {
        let parent = surfaces.get_mut(&parent_id).location(loc!())?;
        let stale: Vec<WlSurfaceId> = parent
            .z_ordered_children()
            .iter()
            .filter(|c| c.id != parent_id && !new_child_ids.contains(&c.id))
            .map(|c| c.id)
            .collect();
        parent
            .z_ordered_children_mut()
            .retain(|c| c.id == parent_id || new_child_ids.contains(&c.id));
        stale
    };
    for stale_id in stale_child_ids {
        if surfaces
            .get(&stale_id)
            .and_then(SubSurfaceNode::subsurface_parent)
            == Some(parent_id)
        {
            detach_subsurface_in(stale_id, surfaces);
        }
    }

    for child in new_children.iter().filter(|c| c.id != parent_id) {
        surfaces
            .entry(child.id)
            .or_insert_with_result(|| create_surface(child.id))
            .location(loc!())?;

        set_subsurface_role_in(
            parent_id,
            child.id,
            surfaces,
            create_surface,
            attach_subsurface,
        )
        .location(loc!())?;
    }
    Ok(())
}

pub(crate) fn detach_subsurface(
    surface_id: WlSurfaceId,
    surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
) {
    detach_subsurface_in(surface_id, surfaces);
}

pub(crate) fn detach_all_subsurfaces(surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>) {
    detach_all_subsurfaces_in(surfaces);
}

pub(crate) fn destroy_surface(
    surface_id: WlSurfaceId,
    surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
) {
    destroy_surface_in(surface_id, surfaces);
}

pub(crate) fn populate_subsurfaces(
    client_id: ClientId,
    parent_id: WlSurfaceId,
    new_children: &[SubsurfacePosition],
    surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
    compositor: &CompositorState,
    subcompositor: &WlSubcompositor,
    qh: &QueueHandle<WprsClientState>,
    object_bimap: &mut ObjectBimap,
) -> Result<()> {
    populate_subsurfaces_in(
        parent_id,
        new_children,
        surfaces,
        &mut |id| RemoteSurface::new(client_id, id, compositor, qh, object_bimap),
        &mut |surfaces, parent_id, surface_id| {
            RemoteSubSurface::attach(parent_id, surface_id, surfaces, subcompositor, qh)
        },
    )
}

fn commit_sync_children_impl(
    surface_id: WlSurfaceId,
    surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
    parent_is_sync: bool,
) -> Result<()> {
    let remote_surface = surfaces.get_mut(&surface_id).location(loc!())?;
    let surface_is_sync = remote_surface
        .role
        .as_ref()
        .location(loc!())?
        .as_sub_surface()
        .location(loc!())?
        .sync;
    let is_sync = parent_is_sync | surface_is_sync;
    if is_sync {
        remote_surface.draw_buffer()?;
    }

    let children = surfaces
        .get(&surface_id)
        .location(loc!())?
        .z_ordered_children
        .clone();
    for child in children.into_iter().filter(|c| c.id != surface_id) {
        commit_sync_children_impl(child.id, surfaces, is_sync).location(loc!())?;
    }
    Ok(())
}

pub(crate) fn commit_sync_children(
    surface_id: WlSurfaceId,
    surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
) -> Result<()> {
    let Some(surface) = surfaces.get(&surface_id) else {
        // TODO: should this be an error?
        return Ok(());
    };

    // If the current surface is a sync subsurface, then don't process its
    // children and let them be processed when the closest desync ancestor is
    // being processed.
    if let Some(Role::SubSurface(subsurface)) = &surface.role
        && subsurface.sync
    {
        return Ok(());
    }

    let children = surface.z_ordered_children.clone();
    for child in children.into_iter().filter(|c| c.id != surface_id) {
        commit_sync_children_impl(child.id, surfaces, false)?;
    }
    Ok(())
}

pub(crate) fn reorder_subsurfaces(
    surface_id: WlSurfaceId,
    surface_state: &SurfaceState,
    surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
) -> Result<()> {
    enum ResolvedZOrderMove<'a> {
        PlaceAbove(&'a WlSubsurface, &'a WlSurface),
        PlaceBelow(&'a WlSubsurface, &'a WlSurface),
    }

    let moves = {
        let remote_surface = surfaces.get_mut(&surface_id).location(loc!())?;
        remote_surface.reorder_children(&surface_state.z_ordered_children)
    };

    let moves: Result<Vec<_>> = moves
        .iter()
        .map(|&mv| -> Result<ResolvedZOrderMove<'_>> {
            let (surf_to_move, sibling, above) = match mv {
                ZOrderMove::PlaceAbove { surface, sibling } => (surface, sibling, true),
                ZOrderMove::PlaceBelow { surface, sibling } => (surface, sibling, false),
            };
            let sub = surfaces
                .get(&surf_to_move)
                .location(loc!())?
                .get_role()
                .location(loc!())?
                .as_sub_surface()
                .location(loc!())?;
            if sub.parent != surface_id {
                bail!(
                    "subsurface {surf_to_move:?} has parent {:?}, expected {surface_id:?}",
                    sub.parent
                );
            }
            let sibling_surface = surfaces.get(&sibling).location(loc!())?.wl_surface();
            Ok(if above {
                ResolvedZOrderMove::PlaceAbove(&sub.local_subsurface.0, sibling_surface)
            } else {
                ResolvedZOrderMove::PlaceBelow(&sub.local_subsurface.0, sibling_surface)
            })
        })
        .collect();

    for mv in moves.location(loc!())? {
        match mv {
            ResolvedZOrderMove::PlaceAbove(sub, sibling) => sub.place_above(sibling),
            ResolvedZOrderMove::PlaceBelow(sub, sibling) => sub.place_below(sibling),
        }
    }

    let children = surfaces
        .get(&surface_id)
        .location(loc!())?
        .z_ordered_children
        .clone();

    for child in children.into_iter().filter(|c| c.id != surface_id) {
        debug!(
            "Setting subsurface {:?} position to {:?}.",
            child.id,
            (child.position.x, child.position.y)
        );
        let sub = surfaces
            .get_mut(&child.id)
            .location(loc!())?
            .role
            .as_mut()
            .location(loc!())?
            .as_sub_surface_mut()
            .location(loc!())?;
        if sub.parent != surface_id {
            bail!(
                "subsurface {:?} has parent {:?}, expected {surface_id:?}",
                child.id,
                sub.parent
            );
        }
        sub.local_subsurface
            .0
            .set_position(child.position.x, child.position.y);
    }

    Ok(())
}

#[derive(Debug)]
struct OwnedSubSurface(WlSubsurface);

impl Drop for OwnedSubSurface {
    fn drop(&mut self) {
        self.0.destroy();
    }
}

#[derive(Debug)]
pub struct RemoteSubSurface {
    pub(crate) parent: WlSurfaceId,
    pub(crate) sync: bool,
    // Declared before `local_surface` so `wl_subsurface` is destroyed before
    // `wl_surface` on drop.
    local_subsurface: OwnedSubSurface,
    pub(crate) local_surface: Surface,
}

impl RemoteSubSurface {
    pub(crate) fn into_surface(self) -> Surface {
        drop(self.local_subsurface);
        self.local_surface
    }

    fn attach(
        parent_id: WlSurfaceId,
        surface_id: WlSurfaceId,
        surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
        subcompositor: &WlSubcompositor,
        qh: &QueueHandle<WprsClientState>,
    ) -> Result<()> {
        let parent = surfaces
            .get(&parent_id)
            .location(loc!())?
            .wl_surface()
            .clone();
        let surface = surfaces.get_mut(&surface_id).location(loc!())?;
        let local_surface = surface.local_surface.take().location(loc!())?;
        let local_subsurface =
            subcompositor.get_subsurface(local_surface.wl_surface(), &parent, qh, SubSurfaceData);
        surface.role = Some(Role::SubSurface(Self {
            parent: parent_id,
            sync: true,
            local_subsurface: OwnedSubSurface(local_subsurface),
            local_surface,
        }));
        Ok(())
    }

    pub fn set_role(
        client_id: ClientId,
        parent_id: WlSurfaceId,
        surface_id: WlSurfaceId,
        surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
        compositor: &CompositorState,
        subcompositor: &WlSubcompositor,
        qh: &QueueHandle<WprsClientState>,
        object_bimap: &mut ObjectBimap,
    ) -> Result<()> {
        set_subsurface_role_in(
            parent_id,
            surface_id,
            surfaces,
            &mut |id| RemoteSurface::new(client_id, id, compositor, qh, object_bimap),
            &mut |surfaces, parent_id, surface_id| {
                Self::attach(parent_id, surface_id, surfaces, subcompositor, qh)
            },
        )
    }

    pub fn update(subsurface_state: &SubSurfaceState, surface: &mut RemoteSurface) -> Result<()> {
        let role = &mut surface
            .role
            .as_mut()
            .location(loc!())?
            .as_sub_surface_mut()
            .location(loc!())?;

        role.local_subsurface
            .0
            .set_position(subsurface_state.location.x, subsurface_state.location.y);

        if role.sync == subsurface_state.sync {
            return Ok(());
        }

        role.sync = subsurface_state.sync;

        if role.sync {
            role.local_subsurface.0.set_sync();
        } else {
            role.local_subsurface.0.set_desync();
        }

        Ok(())
    }

    pub fn apply(
        client_id: ClientId,
        surface_state: SurfaceState,
        surface_id: WlSurfaceId,
        surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
        compositor: &CompositorState,
        subcompositor: &WlSubcompositor,
        qh: &QueueHandle<WprsClientState>,
        object_bimap: &mut ObjectBimap,
    ) -> Result<()> {
        let subsurface_state = surface_state
            .role
            .as_ref()
            .location(loc!())?
            .as_sub_surface()
            .location(loc!())?;
        Self::set_role(
            client_id,
            subsurface_state.parent,
            surface_id,
            surfaces,
            compositor,
            subcompositor,
            qh,
            object_bimap,
        )
        .location(loc!())?;
        let surface = surfaces.get_mut(&surface_id).location(loc!())?;
        Self::update(subsurface_state, surface).location(loc!())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;

    #[derive(Debug, Eq, PartialEq)]
    enum Op {
        CreateSurface(WlSurfaceId),
        GetSubsurface {
            child: WlSurfaceId,
            parent: WlSurfaceId,
        },
        DestroySubsurface {
            child: WlSurfaceId,
            parent: WlSurfaceId,
        },
        DestroySurface(WlSurfaceId),
    }

    struct FakeWlSurface {
        id: WlSurfaceId,
        log: Rc<RefCell<Vec<Op>>>,
    }

    impl Drop for FakeWlSurface {
        fn drop(&mut self) {
            self.log.borrow_mut().push(Op::DestroySurface(self.id));
        }
    }

    struct FakeWlSubsurface {
        child: WlSurfaceId,
        parent: WlSurfaceId,
        log: Rc<RefCell<Vec<Op>>>,
    }

    impl Drop for FakeWlSubsurface {
        fn drop(&mut self) {
            self.log.borrow_mut().push(Op::DestroySubsurface {
                child: self.child,
                parent: self.parent,
            });
        }
    }

    struct FakeSubSurface {
        parent: WlSurfaceId,
        // Mirrors `RemoteSubSurface` field order: subsurface drops before surface.
        local_subsurface: FakeWlSubsurface,
        local_surface: FakeWlSurface,
    }

    impl FakeSubSurface {
        fn into_surface(self) -> FakeWlSurface {
            drop(self.local_subsurface);
            self.local_surface
        }
    }

    enum FakeRole {
        Toplevel(#[expect(dead_code)] FakeWlSurface),
        SubSurface(FakeSubSurface),
    }

    struct FakeSurface {
        local_surface: Option<FakeWlSurface>,
        role: Option<FakeRole>,
        z_ordered_children: Vec<SubsurfacePosition>,
    }

    impl FakeSurface {
        fn new(id: WlSurfaceId, log: &Rc<RefCell<Vec<Op>>>) -> Self {
            log.borrow_mut().push(Op::CreateSurface(id));
            Self {
                local_surface: Some(FakeWlSurface {
                    id,
                    log: Rc::clone(log),
                }),
                role: None,
                z_ordered_children: vec![pos(id)],
            }
        }

        fn set_toplevel(&mut self) {
            let surf = self.local_surface.take().unwrap();
            self.role = Some(FakeRole::Toplevel(surf));
        }
    }

    impl SubSurfaceNode for FakeSurface {
        fn subsurface_parent(&self) -> Option<WlSurfaceId> {
            match &self.role {
                Some(FakeRole::SubSurface(sub)) => Some(sub.parent),
                _ => None,
            }
        }

        fn has_non_subsurface_role(&self) -> bool {
            matches!(self.role, Some(FakeRole::Toplevel(_)))
        }

        fn z_ordered_children(&self) -> &[SubsurfacePosition] {
            &self.z_ordered_children
        }

        fn z_ordered_children_mut(&mut self) -> &mut Vec<SubsurfacePosition> {
            &mut self.z_ordered_children
        }

        fn detach_local_subsurface(&mut self) -> Option<WlSurfaceId> {
            if !matches!(self.role, Some(FakeRole::SubSurface(_))) {
                return None;
            }
            let Some(FakeRole::SubSurface(sub)) = self.role.take() else {
                unreachable!();
            };
            let parent = sub.parent;
            self.local_surface = Some(sub.into_surface());
            Some(parent)
        }
    }

    fn pos(id: WlSurfaceId) -> SubsurfacePosition {
        SubsurfacePosition {
            id,
            position: (0, 0).into(),
        }
    }

    fn attach_fake(
        log: &Rc<RefCell<Vec<Op>>>,
        surfaces: &mut HashMap<WlSurfaceId, FakeSurface>,
        parent_id: WlSurfaceId,
        child_id: WlSurfaceId,
    ) -> Result<()> {
        assert!(surfaces.contains_key(&parent_id));
        let surface = surfaces.get_mut(&child_id).location(loc!())?;
        let local_surface = surface.local_surface.take().location(loc!())?;
        log.borrow_mut().push(Op::GetSubsurface {
            child: child_id,
            parent: parent_id,
        });
        surface.role = Some(FakeRole::SubSurface(FakeSubSurface {
            parent: parent_id,
            local_subsurface: FakeWlSubsurface {
                child: child_id,
                parent: parent_id,
                log: Rc::clone(log),
            },
            local_surface,
        }));
        Ok(())
    }

    #[test]
    fn test_parent_destroy_detaches_child_before_parent_drop_and_allows_reparent() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut surfaces = HashMap::new();
        let p1 = WlSurfaceId(1);
        let p2 = WlSurfaceId(2);
        let child = WlSurfaceId(35);

        let mut parent1 = FakeSurface::new(p1, &log);
        parent1.set_toplevel();
        surfaces.insert(p1, parent1);
        surfaces.insert(child, FakeSurface::new(child, &log));

        set_subsurface_role_in(
            p1,
            child,
            &mut surfaces,
            &mut |id| Ok(FakeSurface::new(id, &log)),
            &mut |s, p, c| attach_fake(&log, s, p, c),
        )
        .unwrap();
        assert_eq!(surfaces[&child].subsurface_parent(), Some(p1));
        log.borrow_mut().clear();

        // Destroying `p1` (e.g. tooltip popup closed) must destroy child's
        // `wl_subsurface` BEFORE `p1`'s `wl_surface`, while keeping `child`'s
        // `wl_surface` alive for tooltip reuse.
        destroy_surface_in(p1, &mut surfaces);
        assert_eq!(
            *log.borrow(),
            vec![
                Op::DestroySubsurface { child, parent: p1 },
                Op::DestroySurface(p1),
            ]
        );
        assert!(!surfaces.contains_key(&p1));
        assert!(surfaces[&child].role.is_none());
        assert!(surfaces[&child].local_surface.is_some());

        // Reattaching the surviving `child` surface to a new parent `p2`
        // (even before `p2` has committed) creates a placeholder `p2` and
        // attaches a fresh `wl_subsurface`.
        log.borrow_mut().clear();
        set_subsurface_role_in(
            p2,
            child,
            &mut surfaces,
            &mut |id| Ok(FakeSurface::new(id, &log)),
            &mut |s, p, c| attach_fake(&log, s, p, c),
        )
        .unwrap();
        assert_eq!(
            *log.borrow(),
            vec![
                Op::CreateSurface(p2),
                Op::GetSubsurface { child, parent: p2 },
            ]
        );
        assert_eq!(surfaces[&child].subsurface_parent(), Some(p2));
        assert_eq!(surfaces[&p2].z_ordered_children, vec![pos(p2), pos(child)]);
    }

    #[test]
    fn test_reparent_while_old_parent_alive_and_populate_subsurfaces_lifecycle() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut surfaces = HashMap::new();
        let p1 = WlSurfaceId(10);
        let p2 = WlSurfaceId(20);
        let c1 = WlSurfaceId(30);
        let c2 = WlSurfaceId(40);

        surfaces.insert(p1, FakeSurface::new(p1, &log));
        surfaces.insert(p2, FakeSurface::new(p2, &log));
        log.borrow_mut().clear();

        // `populate_subsurfaces` uses `surface_state.z_ordered_children` to
        // create placeholder uncommitted children `c1` and `c2`.
        populate_subsurfaces_in(
            p1,
            &[pos(c1), pos(c2), pos(p1)],
            &mut surfaces,
            &mut |id| Ok(FakeSurface::new(id, &log)),
            &mut |s, p, c| attach_fake(&log, s, p, c),
        )
        .unwrap();
        assert_eq!(surfaces[&c1].subsurface_parent(), Some(p1));
        assert_eq!(surfaces[&c2].subsurface_parent(), Some(p1));
        assert_eq!(
            surfaces[&p1].z_ordered_children,
            vec![pos(p1), pos(c1), pos(c2)]
        );

        // Reparenting `c1` to `p2` via `p2`'s commit while `p1` is still alive
        // destroys the old `(c1, p1)` subsurface BEFORE `get_subsurface(c1, p2)`
        // and removes `c1` from `p1.z_ordered_children`.
        log.borrow_mut().clear();
        populate_subsurfaces_in(
            p2,
            &[pos(c1), pos(p2)],
            &mut surfaces,
            &mut |id| Ok(FakeSurface::new(id, &log)),
            &mut |s, p, c| attach_fake(&log, s, p, c),
        )
        .unwrap();
        assert_eq!(
            *log.borrow(),
            vec![
                Op::DestroySubsurface {
                    child: c1,
                    parent: p1,
                },
                Op::GetSubsurface {
                    child: c1,
                    parent: p2,
                },
            ]
        );
        assert_eq!(surfaces[&c1].subsurface_parent(), Some(p2));
        assert_eq!(surfaces[&p1].z_ordered_children, vec![pos(p1), pos(c2)]);
        assert_eq!(surfaces[&p2].z_ordered_children, vec![pos(p2), pos(c1)]);

        // Removing `c2` from `p1`'s `z_ordered_children` detaches `c2` while
        // keeping its `wl_surface` alive.
        log.borrow_mut().clear();
        populate_subsurfaces_in(
            p1,
            &[pos(p1)],
            &mut surfaces,
            &mut |id| Ok(FakeSurface::new(id, &log)),
            &mut |s, p, c| attach_fake(&log, s, p, c),
        )
        .unwrap();
        assert_eq!(
            *log.borrow(),
            vec![Op::DestroySubsurface {
                child: c2,
                parent: p1,
            }]
        );
        assert_eq!(surfaces[&c2].subsurface_parent(), None);
        assert!(surfaces[&c2].local_surface.is_some());
        assert_eq!(surfaces[&p1].z_ordered_children, vec![pos(p1)]);

        // Direct `role: None` detach removes `c1` from `p2`.
        log.borrow_mut().clear();
        detach_subsurface_in(c1, &mut surfaces);
        assert_eq!(
            *log.borrow(),
            vec![Op::DestroySubsurface {
                child: c1,
                parent: p2,
            }]
        );
        assert_eq!(surfaces[&c1].subsurface_parent(), None);
        assert_eq!(surfaces[&p2].z_ordered_children, vec![pos(p2)]);
    }

    #[test]
    fn test_subsurface_cycles_and_disconnect_drop_order() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut surfaces = HashMap::new();
        let a = WlSurfaceId(1);
        let b = WlSurfaceId(2);
        let c = WlSurfaceId(3);
        surfaces.insert(a, FakeSurface::new(a, &log));
        surfaces.insert(b, FakeSurface::new(b, &log));
        surfaces.insert(c, FakeSurface::new(c, &log));

        // Self-parenting is rejected.
        assert!(
            set_subsurface_role_in(
                a,
                a,
                &mut surfaces,
                &mut |id| Ok(FakeSurface::new(id, &log)),
                &mut |s, p, ch| attach_fake(&log, s, p, ch),
            )
            .is_err()
        );

        // `a -> b -> c` succeeds; closing the cycle `c -> a` is rejected without
        // mutating `a` or `c`.
        set_subsurface_role_in(
            a,
            b,
            &mut surfaces,
            &mut |id| Ok(FakeSurface::new(id, &log)),
            &mut |s, p, ch| attach_fake(&log, s, p, ch),
        )
        .unwrap();
        set_subsurface_role_in(
            b,
            c,
            &mut surfaces,
            &mut |id| Ok(FakeSurface::new(id, &log)),
            &mut |s, p, ch| attach_fake(&log, s, p, ch),
        )
        .unwrap();
        assert!(
            set_subsurface_role_in(
                c,
                a,
                &mut surfaces,
                &mut |id| Ok(FakeSurface::new(id, &log)),
                &mut |s, p, ch| attach_fake(&log, s, p, ch),
            )
            .is_err()
        );
        assert_eq!(surfaces[&a].subsurface_parent(), None);
        assert_eq!(surfaces[&b].subsurface_parent(), Some(a));
        assert_eq!(surfaces[&c].subsurface_parent(), Some(b));

        // On client disconnect, detaching all subsurfaces before dropping
        // `surfaces` guarantees every `DestroySubsurface` happens before every
        // `DestroySurface` regardless of `HashMap` drop order.
        log.borrow_mut().clear();
        detach_all_subsurfaces_in(&mut surfaces);
        drop(surfaces);
        let ops = log.borrow();
        let last_sub_destroy = ops
            .iter()
            .rposition(|op| matches!(op, Op::DestroySubsurface { .. }))
            .unwrap();
        let first_surf_destroy = ops
            .iter()
            .position(|op| matches!(op, Op::DestroySurface(_)))
            .unwrap();
        assert!(
            last_sub_destroy < first_surf_destroy,
            "subsurfaces must all be destroyed before any parent surface: {ops:?}"
        );
    }
}
