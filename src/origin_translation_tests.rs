//! Rigid-body origin shifts preserve pending motion and solver history.
use crate::alloc_prelude::vec;
use crate::prelude::*;

fn v(x: Real, y: Real, z: Real) -> Vector {
    #[cfg(feature = "dim2")]
    {
        let _ = z;
        Vector::new(x, y)
    }
    #[cfg(feature = "dim3")]
    {
        Vector::new(x, y, z)
    }
}

fn close(a: Vector, b: Vector) {
    assert!((a - b).length() < 0.002, "{a:?} != {b:?}");
}

// A rigid-only transaction. A real integration must also translate application,
// interpolation and other world-space state before exposing the new frame.
fn shift(world: &mut PhysicsWorld, delta: Vector) {
    assert!(world.soft_bodies.is_empty());
    world.bodies.translate_origin(delta);
    world.colliders.translate_origin(delta);
    world.multibody_joints.translate_origin(delta);
    world.narrow_phase.translate_origin(delta);
    for (handle, collider) in world.colliders.iter() {
        if collider.is_enabled() {
            world.broad_phase.set_aabb(
                &world.integration_parameters,
                handle,
                collider.compute_aabb(),
            );
        }
    }
}

#[test]
fn origin_translation_preserves_flags_forces_disabled_poses_and_kinematic_targets() {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let (sleeping, attached) = world.insert(
        RigidBodyBuilder::dynamic().translation(v(3., 4., 5.)),
        ColliderBuilder::ball(0.5).translation(v(0.25, 0.5, 0.)),
    );
    let disabled = world.insert_body(RigidBodyBuilder::dynamic().translation(v(-3., 2., 1.)));
    world.bodies[disabled].set_enabled(false);
    let standalone =
        world.insert_collider(ColliderBuilder::ball(0.5).translation(v(-2., 3., 4.)), None);
    world.colliders[standalone].set_enabled(false);
    let moving = world.insert_body(
        RigidBodyBuilder::dynamic()
            .linvel(v(1., 2., 3.))
            .additional_mass(1.),
    );
    let kinematic =
        world.insert_body(RigidBodyBuilder::kinematic_position_based().translation(v(4., 0., 0.)));
    world.step();
    world.bodies[sleeping].sleep();
    world.bodies[sleeping].add_force(v(1., 2., 0.), false);
    world.bodies[kinematic].set_next_kinematic_translation(v(4.5, 0., 0.));
    let body_state = [sleeping, disabled, moving, kinematic].map(|h| {
        let b = &world.bodies[h];
        (
            h,
            *b.position(),
            *b.next_position(),
            b.center_of_mass(),
            b.changes,
            b.is_sleeping(),
            b.is_enabled(),
            b.linvel(),
            b.user_force(),
        )
    });
    let collider_state = [attached, standalone].map(|h| {
        let c = &world.colliders[h];
        (
            h,
            *c.position(),
            c.position_wrt_parent().copied(),
            c.changes,
        )
    });
    let delta = v(128., -64., 256.);
    shift(&mut world, delta);
    for (h, pose, next, com, flags, sleeping, enabled, velocity, force) in body_state {
        let b = &world.bodies[h];
        close(b.translation(), pose.translation + delta);
        close(b.next_position().translation, next.translation + delta);
        close(b.center_of_mass(), com + delta);
        assert_eq!(*b.rotation(), pose.rotation);
        assert_eq!(b.changes, flags);
        assert_eq!(b.is_sleeping(), sleeping);
        assert_eq!(b.is_enabled(), enabled);
        assert_eq!(b.linvel(), velocity);
        assert_eq!(b.user_force(), force);
    }
    for (h, pose, parent, flags) in collider_state {
        let c = &world.colliders[h];
        close(c.translation(), pose.translation + delta);
        assert_eq!(c.position_wrt_parent().copied(), parent);
        assert_eq!(c.changes, flags);
    }
    world.step();
    close(
        world.bodies[kinematic].translation(),
        v(4.5, 0., 0.) + delta,
    );
    close(
        world.bodies[kinematic].linvel(),
        v(0.5 / world.integration_parameters.dt, 0., 0.),
    );
    assert!(world.bodies[sleeping].is_sleeping());
}

#[test]
fn origin_translation_preserves_sleep_contacts_local_anchors_and_warmstart_history() {
    check_contact_shift(false);
}

#[test]
#[cfg(feature = "dim3")]
fn origin_translation_preserves_current_and_previous_cluster_contact_caches() {
    check_contact_shift(true);
}

#[test]
#[cfg(feature = "dim2")]
fn origin_translation_preserves_multiple_2d_manifolds_without_solver_clustering() {
    check_contact_shift(true);
}

fn check_contact_shift(clustered: bool) {
    let mut world = PhysicsWorld::new();
    world.gravity = v(0., -9.81, 0.);
    world.integration_parameters.contact_clustering = clustered;
    world.integration_parameters.contact_recycling = false;
    let ground_shape = if clustered {
        ColliderBuilder::compound(vec![
            (
                Pose::from_translation(v(-0.125, -10., 0.)),
                SharedShape::ball(10.),
            ),
            (
                Pose::from_translation(v(0.125, -10., 0.)),
                SharedShape::ball(10.),
            ),
        ])
    } else {
        ColliderBuilder::ball(10.).translation(v(0., -10., 0.))
    };
    let ground = world.insert_collider(ground_shape, None);
    let (body, collider) = world.insert(
        RigidBodyBuilder::dynamic().translation(v(0., 0.5, 0.)),
        ColliderBuilder::ball(0.5),
    );
    for _ in 0..360 {
        world.step();
    }
    assert!(world.bodies[body].is_sleeping());
    let pair_before = world
        .narrow_phase
        .contact_pair(ground, collider)
        .unwrap()
        .clone();
    let before = pair_before.rigid().unwrap().clone();
    assert!(before.has_any_active_contact());
    assert!(
        before
            .manifolds
            .iter()
            .chain(&before.solver_clusters)
            .flat_map(|m| &m.points)
            .any(|p| p.data.impulse > 0.)
    );
    if clustered && cfg!(feature = "dim3") {
        assert!(
            !before.solver_clusters.is_empty(),
            "fixture did not cluster contacts"
        );
        assert!(
            !before.solver_clusters_prev.is_empty(),
            "fixture lacks previous clusters"
        );
    }
    let delta = v(128., 0., -256.);
    shift(&mut world, delta);
    let pair_after = world.narrow_phase.contact_pair(ground, collider).unwrap();
    assert_eq!(pair_before.collider1, pair_after.collider1);
    assert_eq!(pair_before.collider2, pair_after.collider2);
    assert_eq!(pair_before.event_status, pair_after.event_status);
    let after = pair_after.rigid().unwrap();
    assert_eq!(before.manifolds.len(), after.manifolds.len());
    assert_eq!(before.solver_clusters.len(), after.solver_clusters.len());
    assert_eq!(
        before.solver_clusters_prev.len(),
        after.solver_clusters_prev.len()
    );
    let mut local = 0;
    let mut world_attached = 0;
    for (a, b) in before
        .manifolds
        .iter()
        .chain(&before.solver_clusters)
        .chain(&before.solver_clusters_prev)
        .zip(
            after
                .manifolds
                .iter()
                .chain(&after.solver_clusters)
                .chain(&after.solver_clusters_prev),
        )
    {
        assert_eq!(a.data.normal, b.data.normal);
        assert_eq!(a.points.len(), b.points.len());
        for (p, q) in a.points.iter().zip(&b.points) {
            assert_eq!(p.local_p1, q.local_p1);
            assert_eq!(p.local_p2, q.local_p2);
            assert_eq!(p.data.impulse, q.data.impulse);
            assert_eq!(p.data.warmstart_impulse, q.data.warmstart_impulse);
            assert_eq!(p.data.tangent_impulse, q.data.tangent_impulse);
            assert_eq!(
                p.data.warmstart_tangent_impulse,
                q.data.warmstart_tangent_impulse
            );
        }
        assert_eq!(a.data.solver_contacts.len(), b.data.solver_contacts.len());
        for (p, q) in a.data.solver_contacts.iter().zip(&b.data.solver_contacts) {
            for (index, (p, q)) in [(p.anchor1, q.anchor1), (p.anchor2, q.anchor2)]
                .into_iter()
                .enumerate()
            {
                if a.data.solver_body_ids[index] == u32::MAX {
                    close(q, p + delta);
                    world_attached += 1;
                } else {
                    assert_eq!(q, p);
                    local += 1;
                }
            }
            assert_eq!(p.dist, q.dist);
        }
    }
    assert!(local > 0 && world_attached > 0);
    let ray = Ray::new(v(0., 5., 0.) + delta, v(0., -1., 0.));
    assert_eq!(
        world
            .cast_ray(&ray, 6., true, QueryFilter::default())
            .unwrap()
            .0,
        collider
    );
    for _ in 0..20 {
        world.step();
    }
    assert!(
        world.bodies[body].is_sleeping(),
        "origin shift caused a synthetic wake"
    );
}

#[test]
fn origin_translation_multibody_roots_and_future_motion_match_unshifted_control() {
    for fixed_root in [false, true] {
        let build = || {
            let mut w = PhysicsWorld::new();
            w.gravity = Vector::ZERO;
            let builder = if fixed_root {
                RigidBodyBuilder::fixed()
            } else {
                RigidBodyBuilder::dynamic().linvel(v(1., 0., 0.))
            };
            let root = w.insert_body(builder.translation(v(3., 2., 0.)).additional_mass(1.));
            let child = w.insert_body(
                RigidBodyBuilder::dynamic()
                    .translation(v(4., 2., 0.))
                    .additional_mass(1.),
            );
            let joint = w
                .insert_multibody_joint(
                    root,
                    child,
                    FixedJointBuilder::new().local_anchor1(v(1., 0., 0.)),
                )
                .unwrap();
            for _ in 0..10 {
                w.step();
            }
            (w, root, child, joint)
        };
        let (mut moved, root, child, joint) = build();
        let (mut control, _, _, _) = build();
        let before = moved.multibody_joints.get(joint).unwrap().0.clone();
        let delta = v(128., -64., 256.);
        shift(&mut moved, delta);
        let after = moved.multibody_joints.get(joint).unwrap().0;
        for (index, (a, b)) in before.links().zip(after.links()).enumerate() {
            close(
                b.local_to_world().translation,
                a.local_to_world().translation + delta,
            );
            assert_eq!(a.local_to_world().rotation, b.local_to_world().rotation);
            if index > 0 {
                assert_eq!(a.local_to_parent(), b.local_to_parent());
                assert_eq!(a.joint.data.local_frame1, b.joint.data.local_frame1);
                assert_eq!(a.joint.data.local_frame2, b.joint.data.local_frame2);
            }
        }
        for _ in 0..40 {
            moved.step();
            control.step();
        }
        for h in [root, child] {
            close(
                moved.bodies[h].translation(),
                control.bodies[h].translation() + delta,
            );
            close(moved.bodies[h].linvel(), control.bodies[h].linvel());
        }
        shift(&mut moved, -delta);
        for h in [root, child] {
            close(
                moved.bodies[h].translation(),
                control.bodies[h].translation(),
            );
        }
    }
}
